import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

class CANNetwork(nn.Module):
    """
    A CAN network implemented using PyTorch.
    """
    def __init__(self, 
                 num_neurons: int,
                 noise: float,
                 field_width: float,
                 syn_fail: float,
                 spon_rel: float,
                 constrict: float,
                 fraction_active: float,
                 I_str: List[float],
                 I_dir: List[float],
                 num_updates: int,
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
        self.device = device if device is not None else torch.device("cpu")
        self.num_neurons = num_neurons
        self.noise = noise
        self.field_width = field_width
        self.syn_fail = syn_fail
        self.spon_rel = spon_rel
        self.constrict = constrict
        self.fraction_active = fraction_active
        self.I_str = I_str
        self.I_dir = I_dir
        self.num_updates = num_updates
        self.generation = 0

        self.weights = torch.zeros((num_neurons, num_neurons), dtype=torch.float32, device=self.device)
        self.state: Optional[torch.Tensor] = None

        # History tracking lists
        self.lyapunov: List[float] = []
        self.activations: List[float] = []
        self.centres: List[float] = []
        self.variances: List[float] = []
        self.total_activity: List[float] = []
        self.state_history: List[torch.Tensor] = []
        self.tuning_curves: dict = {}
        self.correlations: List[float] = []

    def initialize_weights(self) -> None:
        """
        Initialize a symmetric weight matrix with periodic boundary conditions.
        """
        indices = torch.arange(self.num_neurons, device=self.device)
        i_matrix = indices.unsqueeze(1)
        j_matrix = indices.unsqueeze(0)
        diff = torch.abs(i_matrix - j_matrix)
        diff = torch.min(diff, self.num_neurons - diff)
        threshold = self.field_width * self.num_neurons / 2
        self.weights = (diff <= threshold).float() / self.num_neurons
        # Inhibitory connections make negative or by enforcing an average fraction of active in dynamics
        # self.weights = self.weights - 1*(diff > threshold).float()/self.num_neurons
        self.weights.fill_diagonal_(0)

    def initialize_state(self) -> None:
        """
        Initialize the network state as a binary vector with a fraction of neurons active.
        """
        self.state = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)
        num_active = int(self.fraction_active * self.num_neurons)
        # active_indices = torch.randperm(self.num_neurons, device=self.device)[:num_active]
        #Random index but all together
        start_index = torch.randint(0, self.num_neurons - num_active + 1, (1,),
                                     device=self.device).item()
        active_indices = torch.arange(start_index, start_index + num_active,
                                       device=self.device) % self.num_neurons
        self.state[active_indices] = 1.0

    def calculate_energy(self) -> torch.Tensor:
        """
        Compute the network energy using a modified energy function.

        Returns:
            torch.Tensor: The calculated energy.
        """
        # Do I need a threshold term for metroplis?
        # do i need  to make symmetric?
        fail_matrix = torch.tensor(np.random.choice([0, 1], p=[self.syn_fail, 1-self.syn_fail], size=(self.num_neurons, self.num_neurons)), device=self.device, dtype=torch.float32)
        symmetric_fail_matrix = (fail_matrix + fail_matrix.t()) / 2
        dynamic_weights = self.weights * symmetric_fail_matrix
        
        main_term = -0.5 * (1 - self.syn_fail) * torch.dot(self.state, torch.mv(dynamic_weights, self.state))
        spontaneous_term = -torch.sum(self.state) * self.spon_rel
        
        indices = torch.arange(self.num_neurons, device=self.device, dtype=torch.float32)
        dist = torch.abs(indices - self.I_dir[self.generation])
        dist = torch.min(dist, self.num_neurons - dist)
        sigma = self.field_width * self.num_neurons  # Adjust as appropriate
        I_ext = self.I_str[self.generation] * torch.exp(-dist**2 / (2 * sigma**2))
        Iext_term = -torch.dot(self.state, I_ext)
        
        # Iext_term = -torch.sum(self.state) * (self.I_str ** 2) * (
        #     1 - torch.exp(torch.tensor(-self.I_str * self.field_width * self.num_neurons, 
        #                                  dtype=torch.float32, device=self.device))
        # )
        return main_term + spontaneous_term + Iext_term

    def record_energy(self) -> None:
        """
        Record the current energy (Lyapunov value) of the network.
        """
        energy = self.calculate_energy().item()
        self.lyapunov.append(energy)

    def record_correlations(self) -> None:
        """
        record the noise correlation between the neurons in the network defined as the pearson correlation coefficient
        """
        # Calculate the mean of the state
        entire_state = torch.stack(self.state_history) +1e-10#gens x num_neurons
        
        # mean_state = torch.mean(entire_state, dim=0)
        # print(mean_state.shape)
        # state_centered = entire_state - mean_state
        # covariance_matrix = torch.mm(state_centered.t(), state_centered) / (entire_state.shape[0] - 1)
        # std_devs = torch.sqrt(torch.diag(covariance_matrix))
        # correlation_matrix = covariance_matrix / torch.ger(std_devs, std_devs)
        # corr_vals = correlation_matrix[torch.triu_indices(correlation_matrix.size(0), correlation_matrix.size(1), offset=1)]
        corr_matrix = torch.corrcoef(torch.transpose(entire_state,0,1))
        upper_triangle_indices = torch.triu_indices(self.num_neurons,self.num_neurons, offset=1)
        upper_triangle_values = corr_matrix[upper_triangle_indices[0,:], upper_triangle_indices[1,:]]
        self.correlations = upper_triangle_values.tolist()


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
        norm_factor = (((self.fraction_active * self.num_neurons) ** 2 - 1) / 12) if (((self.fraction_active * self.num_neurons) ** 2 - 1) != 0) else 1
        self.variances.append(diff_mean / norm_factor)

    def update_state(self, update_strategy: "UpdateStrategy") -> None:
        """
        Update the network state using a provided update strategy.

        Args:
            update_strategy (UpdateStrategy): The update strategy to apply.
        """
        update_strategy.update(self)
