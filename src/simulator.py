from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
import logging
from src.network import CANNetwork
from src.update_strategies import UpdateStrategy
torch.backends.cudnn.benchmark = True


class _StateHistoryWriter:
    """Chunked writer for generation-by-generation state snapshots."""

    def __init__(self, path: Path, num_generations: int, num_neurons: int, chunk_size: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.num_generations = num_generations
        self.num_neurons = num_neurons
        self.chunk_size = min(chunk_size, num_generations)

        self._mmap = np.lib.format.open_memmap(
            self.path,
            mode="w+",
            dtype=np.uint8,
            shape=(num_generations, num_neurons),
        )
        self._buffer = np.empty((self.chunk_size, num_neurons), dtype=np.uint8)
        self._buffer_count = 0
        self._next_row = 0

    def append(self, state: torch.Tensor) -> None:
        row = state.detach().to(dtype=torch.uint8, device="cpu").numpy()
        self._buffer[self._buffer_count, :] = row
        self._buffer_count += 1
        if self._buffer_count == self.chunk_size:
            self._flush_buffer()

    def finalize(self) -> None:
        self._flush_buffer()
        self._mmap.flush()
        del self._mmap

    def _flush_buffer(self) -> None:
        if self._buffer_count == 0:
            return
        end = self._next_row + self._buffer_count
        self._mmap[self._next_row:end, :] = self._buffer[:self._buffer_count, :]
        self._next_row = end
        self._buffer_count = 0


def simulate(
    network_params: dict,
    num_generations: int,
    update_strategy,
    progress_callback: Optional[Callable[[int], None]] = None,
    state_history_path: Optional[Path] = None,
    state_write_chunk: int = 256,
    keep_state_history_in_memory: bool = False,
    validate_pools_debug: bool = False,
) -> "CANNetwork":
    """Run the CANN simulation for `num_generations` sweeps."""
    if num_generations <= 0:
        raise ValueError("num_generations must be > 0")

    net = CANNetwork(**network_params)
    logging.info("Running on device: %s", net.device)

    net.initialize_weights()
    net.initialize_state()
    net.synaptic_drive = net.weights @ net.state
    net.debug_validate_pools = validate_pools_debug
    net.initialize_activity_pools()

    writer = None
    if state_history_path is not None:
        writer = _StateHistoryWriter(
            path=Path(state_history_path),
            num_generations=num_generations,
            num_neurons=net.num_neurons,
            chunk_size=state_write_chunk,
        )
        net.state_history_path = str(Path(state_history_path))
        net.state_history_dtype = "uint8"
        net.state_history_shape = (num_generations, net.num_neurons)
    else:
        net.state_history_path = None
        net.state_history_dtype = "float32"
        net.state_history_shape = None
    capture_in_memory = keep_state_history_in_memory or writer is None

    progress_stride = max(1, num_generations // 20)
    has_progress = progress_callback is not None
    has_theta_noise = net.sigma_theta_steps > 0.0
    has_syn_fail = net.syn_fail > 0.0
    has_sigma_eta = net.sigma_eta > 0.0
    block_size = net.block_size
    noise = None
    xnoise = None
    syn_ok = None
    spon = None

    try:
        # Burn-in period
        for _ in range(1000):
            active_size = net.active_pool.numel()
            inactive_size = net.inactive_pool.numel()
            idx_i_batch = torch.randint(0, active_size, (net.num_neurons,), device=net.device)
            idx_j_batch = torch.randint(0, inactive_size, (net.num_neurons,), device=net.device)
            for u in range(net.num_neurons):
                update_strategy.update(net, idx_i=idx_i_batch[u], idx_j=idx_j_batch[u])

        for gen in range(num_generations):
            if has_progress and gen % progress_stride == 0:
                progress_callback(gen)
            # refresh block-level randomness
            if gen % block_size == 0:
                if has_theta_noise:
                    z = torch.randn(net.num_neurons, device=net.device) * net.sigma_theta_steps
                    xnoise = z

            if has_syn_fail:
                syn_ok = torch.bernoulli(torch.full((net.num_neurons,), 1.0 - net.syn_fail, device=net.device))
                spon = torch.bernoulli(torch.full((net.num_neurons,), net.spon_rel, device=net.device))
            if has_sigma_eta:
                noise = torch.randn(net.num_neurons, device=net.device) * net.sigma_eta
            # one sweep = N updates in random order
            active_size = net.active_pool.numel()
            inactive_size = net.inactive_pool.numel()
            idx_i_batch = torch.randint(0, active_size, (net.num_neurons,), device=net.device)
            idx_j_batch = torch.randint(0, inactive_size, (net.num_neurons,), device=net.device)
            for u in range(net.num_neurons):
                update_strategy.update(
                    net,
                    neuron_noise=noise,
                    x_noise=xnoise,
                    synapse_noise=(syn_ok, spon),
                    idx_i=idx_i_batch[u],
                    idx_j=idx_j_batch[u],
                )

            if writer is not None:
                writer.append(net.state)
            if capture_in_memory:
                net.state_history.append(net.state.clone())
            net.generation += 1
    finally:
        if writer is not None:
            writer.finalize()

    return net
