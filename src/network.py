import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

class CANNetwork(nn.Module):
    """
    A CAN network implemented using PyTorch.
    """
    def __init__(self, 
                 num_neurons: int,
                 sigma_temp: float,
                 sigma_input: float,
                 I_str: float,
                 I_dir: float,
                 tau_ou: float,
                 sigma_ou: float,
                 syn_fail: float,
                 spon_rel: float,
                 sigma_eta: float,
                 input_resistance: float,
                 ampar_conductance: float,
                 constrict: float,
                 threshold_active_fraction: float,
                 device: Optional[torch.device] = None):
        """
        Initialize the CAN network.

        Args:
            num_neurons (int): Number of neurons.
            noise (float): Noise (temperature) parameter.
            field_width (float): Receptive field width fraction.
            syn_fail (float): Synaptic failure parameter.
            spon_rel (float): Spontaneous release parameter.
            constrict (float): Constriction scaling factor.
            fraction_active (float): Fraction of neurons active.
            I_str (array): Stimulus strength.
            I_dir (array): Directional stimulus index.
            num_updates (int): Number of updates per simulation step.
            device (Optional[torch.device]): Device to run computations.
        """
        super().__init__()
        self.device =torch.device("cpu")# device if device is not None else torch.device("cpu")
        self.num_neurons = num_neurons
        self.sigma_temp = sigma_temp
        self.sigma_input = sigma_input
        self.syn_fail = syn_fail
        self.spon_rel = spon_rel
        self.constrict = constrict
        self.threshold_active_fraction = threshold_active_fraction
        self.I_str = I_str
        self.I_dir = I_dir
        self.sigma_eta = sigma_eta
        self.input_resistance = input_resistance
        self.ampar_conductance = ampar_conductance
        self.tau_ou = tau_ou
        self.sigma_ou = sigma_ou
        self.generation = 0

        self.weights = torch.zeros((num_neurons, num_neurons), dtype=torch.float32, device=self.device)
        self.state: Optional[torch.Tensor] = None
        self.synaptic_drive: Optional[torch.Tensor] = None
        self.active_count_tensor = torch.tensor(0.0, device=self.device)

        # History tracking lists
        self.lyapunov: List[float] = []
        self.activations: List[float] = []
        self.centres: List[float] = []
        self.total_activity: List[float] = []
        self.state_history: List[torch.Tensor] = []
        self.tuning_curves: dict = {}
        self.correlations: List[float] = []
        self.input_fluctuations: List[float] = []
        self.variances: List[float] = []
        self.record_diagnostics = False


        self.A_mu   = torch.tensor(self.I_str, device=self.device)   # mean amplitude
        self.A_rho  = torch.exp(torch.tensor(-1.0/self.tau_ou, device=self.device))
        self.A_sigma= self.sigma_ou * torch.sqrt(1 - self.A_rho**2)
        self.A      = self.A_mu.clone()                        # state
        # Optional: precompute a LUT for the Gaussian over ring distances (faster)
        sigma_idx = self.sigma_input * self.num_neurons                        # width in *index* units
        d0 = torch.arange(0, (self.num_neurons//2)+1, device=self.device, dtype=torch.float32)
        self.bump_LUT = torch.exp(-0.5 * (d0 / sigma_idx)**2)  # size ≈ N/2+1
        self.num_neurons_tensor = torch.tensor(self.num_neurons, device=self.device, dtype=torch.long)
        self.center_index_tensor = torch.tensor(int(self.I_dir * self.num_neurons), device=self.device, dtype=torch.long)
        distances = torch.arange(self.num_neurons, device=self.device, dtype=torch.long)
        distances = torch.abs(distances - self.center_index_tensor)
        distances = torch.minimum(distances, self.num_neurons_tensor - distances)
        self.distance_to_center = distances
        self.input_bump_profile = self.bump_LUT[distances]
        self.target_active_tensor = torch.tensor(self.threshold_active_fraction * self.num_neurons, device=self.device)
        self.inv_num_neurons_tensor = torch.tensor(1.0 / self.num_neurons, device=self.device)

    def initialize_weights(self) -> None:
        """
        Initialize a symmetric weight matrix with periodic boundary conditions.
        """
        indices = torch.arange(self.num_neurons, device=self.device)
        i_matrix = indices.unsqueeze(1)
        j_matrix = indices.unsqueeze(0)
        diff = torch.abs(i_matrix - j_matrix)
        diff = torch.min(diff, self.num_neurons - diff)
        threshold = self.threshold_active_fraction * self.num_neurons / 2
        self.weights = (diff <= threshold).float() / self.num_neurons
        # Inhibitory connections make negative or by enforcing an average fraction of active in dynamics
        # self.weights = self.weights - 1*(diff > threshold).float()/self.num_neurons
        self.weights.fill_diagonal_(0)

        # N = self.num_neurons
        # idx = torch.arange(N, device=self.device)
        # i = idx[:, None]
        # j = idx[None, :]
        # diff = (i - j).abs()
        # diff = torch.minimum(diff, N - diff)                    # ring distance (indices)
        # radius = int(self.threshold_active_fraction * N / 2)    # top-hat half-width
        # W = (diff <= radius).float()
        # W.fill_diagonal_(0.)
        # Z = W.sum(dim=1, keepdim=True).clamp_min(1.0)           # neighbors per row
        # self.weights = W / Z                                     # row-normalized top-hat

    def initialize_state(self) -> None:
        """
        Initialize the network state as a binary vector with a fraction of neurons active.
        """
        self.state = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)
        num_active = int(self.threshold_active_fraction * self.num_neurons)
        # active_indices = torch.randperm(self.num_neurons, device=self.device)[:num_active]
        #Random index but all together
        start_index = torch.randint(0, self.num_neurons - num_active + 1, (1,),
                                     device=self.device).item()
        active_indices = torch.arange(start_index, start_index + num_active,
                                       device=self.device) % self.num_neurons
        self.state[active_indices] = 1.0
        self.active_count_tensor = self.state.sum()

    def calculate_energy(self) -> torch.Tensor:
        """
        Compute the network energy using a modified energy function.

        Returns:
            torch.Tensor: The calculated energy.
        """
        dynamic_weights = self.ampar_conductance*self.weights
        
        main_term = -0.5 * torch.dot(self.state, torch.mv(dynamic_weights, self.state))        
        center = int(round(self.I_dir * self.num_neurons)) % self.num_neurons
        idx = torch.arange(self.num_neurons, device=self.device)
        dx = torch.abs(idx - center)
        dx = torch.min(dx, self.num_neurons - dx).long()
        Iext_term = -torch.dot(self.state, self.A * self.bump_LUT[dx])

        inhibition = 0.5 * self.constrict / self.num_neurons * (torch.sum(self.state) - self.num_neurons * self.threshold_active_fraction)**2

        return self.input_resistance*(main_term + Iext_term + inhibition)

    def record_energy(self) -> None:
        """
        Record the current energy (Lyapunov value) of the network.
        """
        energy = self.calculate_energy().item()
        self.lyapunov.append(energy)

    def locate_centre(self) -> None:
        """
        Compute the centre of mass of active neurons (in circular coordinates) and normalized variance.
        """
        active_indices = (self.state == 1).nonzero(as_tuple=True)[0]
        if len(active_indices) == 0:
            return

        theta = active_indices.float() * 2 * np.pi / self.num_neurons
        x_coords = torch.cos(theta)
        y_coords = torch.sin(theta)
        theta_avg = torch.atan2(-torch.mean(y_coords), -torch.mean(x_coords)) + np.pi
        centre = (self.num_neurons * theta_avg / (2 * np.pi)) % self.num_neurons
        self.centres.append(centre.item())

        squared_diffs = []
        for i in active_indices:
            diff = abs(i.item() - centre.item())
            circular_diff = min(diff, self.num_neurons - diff) ** 2
            squared_diffs.append(circular_diff)
        diff_mean = np.sum(squared_diffs) / len(squared_diffs)
        norm_factor = (((self.threshold_active_fraction * self.num_neurons) ** 2 - 1) / 12) if (((self.threshold_active_fraction * self.num_neurons) ** 2 - 1) != 0) else 1
        self.variances.append(diff_mean / norm_factor)

    def update_state(self, update_strategy: "UpdateStrategy",
                     rand_index: Optional[torch.Tensor] = None,
                     neuron_noise: Optional[torch.Tensor] = None) -> None:
        """
        Update the network state using a provided update strategy.

        Args:
            update_strategy (UpdateStrategy): The update strategy to apply.
        """
        update_strategy.update(self, rand_index=rand_index, neuron_noise=neuron_noise)
    
    def noise_covariance(self) -> torch.Tensor:
        """
        Calculate the noise covariance matrix of the network states.

        Returns:
            torch.Tensor: The noise covariance matrix.
        """
        histories = torch.stack(self.state_history)  # (generations, neurons)
        states = histories.to(self.device, dtype=torch.float32)
        # median_state = torch.median(states, dim=0).values
        # quantile = torch.quantile(median_state, 1 - self.threshold_active_fraction)
        # mask = median_state >= quantile
        # states_sel = states[:, mask]
        # states = states - states.mean(dim=1, keepdim=True)      # remove pop-mean per time
        centered = states - states.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / (states.size(0) - 1)
        self.covariance_matrix = cov  # stays as torch.Tensor
