from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import colors 
import datetime
import webbrowser

from src.network import CANNetwork

MAX_PLOT_ROWS = 5000


def _load_state_history_array(network: CANNetwork):
    """Load state history from disk-backed storage when available."""
    state_history_path = getattr(network, "state_history_path", None)
    if state_history_path:
        state_path = Path(state_history_path)
        if state_path.exists():
            return np.load(state_path, mmap_mode="r")

    if isinstance(network.state_history, torch.Tensor):
        return network.state_history.cpu().numpy()
    if network.state_history:
        return torch.stack([torch.as_tensor(s) for s in network.state_history]).cpu().numpy()
    raise ValueError("No state history available in memory or on disk.")


def _prepare_state_history_for_plot(state_history, max_rows: int = MAX_PLOT_ROWS):
    """Downsample only the plotted history rows for large runs."""
    num_rows = state_history.shape[0]
    if num_rows <= max_rows:
        y_axis = np.arange(num_rows)
        return np.asarray(state_history), y_axis
    sampled = np.linspace(0, num_rows - 1, num=max_rows, dtype=np.int64)
    return np.asarray(state_history[sampled]), sampled


def create_interactive_report(network: CANNetwork, output_dir: str = "reports") -> str:
    """
    Create an interactive HTML report with Plotly visualizations.
    
    Args:
        network: The CANNetwork instance containing simulation data
        output_dir: Directory to save the report
    
    Returns:
        Path to the generated HTML report file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"can_report_{timestamp}.html"
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"type": "scatter", "colspan": 1}, {"type": "scatter", "colspan": 1}]
        ],
        subplot_titles=(
            "Network State History",
            "Lyapunov Energy over Generations",
            "Centre of Active Neurons",
            "Activity Metrics"
        ),
        vertical_spacing=0.1
    )
    
    # 1. State history heatmap (downsampled for rendering only if very large)
    state_history = _load_state_history_array(network)
    state_history_plot, generation_axis = _prepare_state_history_for_plot(state_history)
    heatmap = go.Heatmap(
        z=state_history_plot,
        y=generation_axis,
        colorscale=[
            [0, 'rgba(0,0,0,0)'],  
            [1, 'rgb(210, 145, 188)']  
        ],
        showscale=False,  # Hide the colorbar initially
        name = 'State History',
        colorbar=dict(title="Activation"),
        hovertemplate='Generation: %{y}<br>Neuron: %{x}<br>Activation: %{z}<extra></extra>'
    )
    
    fig.add_trace(heatmap, row=1, col=1)

    # 2. Lyapunov Energy (middle row, spans both columns)
    generations = np.arange(len(network.lyapunov))
    lyapunov_trace = go.Scatter(
        x=generations,
        y=network.lyapunov,
        mode='lines',
        name='Lyapunov Energy',
        line=dict(color='firebrick', width=1),
        hovertemplate='Generation: %{x}<br>Energy: %{y:.4f}<extra></extra>'
    )
    
    fig.add_trace(lyapunov_trace, row=2, col=1)
    
    # 3. Centre of Activity (bottom row, left column)
    centre_trace = go.Scatter(
        x=generations[:len(network.centres)],
        y=network.centres,
        mode='lines',
        name='Centre of Activity',
        line=dict(color='blue'),
        hovertemplate='Generation: %{x}<br>Centre: %{y:.2f}<extra></extra>'
    )
    
    fig.add_trace(centre_trace, row=3, col=1)
    
    # 4. Activity Metrics: Variance and Total Activity (bottom row, right column)
    variance_trace = go.Scatter(
        x=generations[:len(network.variances)],
        y=network.variances,
        mode='lines',
        name='Variance',
        line=dict(color='green'),
        hovertemplate='Generation: %{x}<br>Variance: %{y:.2f}<extra></extra>'
    )
    
    activity_trace = go.Scatter(
        x=generations[:len(network.total_activity)],
        y=network.total_activity,
        mode='lines',
        name='Total Activity',
        line=dict(color='red'),
        hovertemplate='Generation: %{x}<br>Activity: %{y:.2f}<extra></extra>'
    )
    
    fig.add_trace(variance_trace, row=3, col=2)
    fig.add_trace(activity_trace, row=3, col=2)
    
    # Update layout
    fig.update_layout(
        title=f"Continuous Attractor Network Analysis - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        height=1000,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Neuron Index", row=1, col=1)
    fig.update_yaxes(title_text="Generation", row=1, col=1)
    
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="Energy", row=2, col=1)
    
    fig.update_xaxes(title_text="Generation", row=3, col=1)
    fig.update_yaxes(title_text="Centre Position", row=3, col=1)
    
    fig.update_xaxes(title_text="Generation", row=3, col=2)
    fig.update_yaxes(title_text="Value", row=3, col=2)
    
    # Create HTML with embedded Plotly figure
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CAN Network Analysis</title>
        <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1300px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .plot-container {{
                margin-bottom: 40px;
            }}
            .network-info {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 30px;
            }}
            .info-title {{
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #666;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Continuous Attractor Network Analysis</h1>
                <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="plot-container">
                <div id="main-plot"></div>
            </div>
            
            <div class="network-info">
                <div class="info-title">Network Information</div>
                <table>
                    <tr>
                        <th>Property</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Number of Neurons</td>
                        <td>{state_history.shape[1]}</td>
                    </tr>
                    <tr>
                        <td>Simulation Length</td>
                        <td>{len(network.lyapunov)} generations</td>
                    </tr>
                    <tr>
                        <td>Final Lyapunov Energy</td>
                        <td>{network.lyapunov[-1]:.4f}</td>
                    </tr>
                    <tr>
                        <td>Final Centre of Activity</td>
                        <td>{network.centres[-1]:.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="footer">
                <p>Continuous Attractor Neural Network Project</p>
            </div>
        </div>
        
        <script>
            var plotlyData = {fig.to_json()};
            Plotly.newPlot('main-plot', plotlyData.data, plotlyData.layout);
        </script>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    return str(output_file)

def create_visualization_report(network: CANNetwork, output_dir: str = "reports") -> str:
    """
    Create a comprehensive visualization report combining all plots.
    
    Args:
        network: The CANNetwork instance containing simulation data
        output_dir: Directory to save the report
    
    Returns:
        Path to the generated report file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"can_report_{timestamp}.pdf"
    
    # Create PDF with all plots
    with PdfPages(output_file) as pdf:
        # Plot 1: State History
        fig1 = plt.figure(figsize=(10, 8))
        state_history = _load_state_history_array(network)
        state_history_plot, _ = _prepare_state_history_for_plot(state_history)
        cmap = colors.ListedColormap(['#FFFFFF', '#D291BC'])
        plt.imshow(state_history_plot, aspect='auto', interpolation='nearest', cmap=cmap)
        plt.xlabel("Neuron Index")
        plt.ylabel("Generation")
        if state_history_plot.shape[0] < state_history.shape[0]:
            plt.title("Network State History (downsampled for plotting)")
        else:
            plt.title("Network State History")
        pdf.savefig(fig1)
        plt.close(fig1)

        # plot activity per generation
        figx = plt.figure(figsize=(10, 7))
        avg_activity = np.mean(state_history, axis=1)
        plt.plot(avg_activity, color='blue')
        plt.title("Average Network Activity Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Average Activity")
        plt.grid(True, linestyle='--', alpha=0.6)
        pdf.savefig(figx)
        plt.close(figx)
        
        # # Plot 2: Metrics (Combined)
        # fig2, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        # # Center of activity
        # axes[0].plot(network.centres, color='blue')
        # axes[0].set_title("Centre of Active Neurons")
        # axes[0].set_ylabel("Centre Position")
        # axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # # Variance
        # axes[1].plot(network.variances, color='green')
        # axes[1].set_title("Variance of Activity")
        # axes[1].set_ylabel("Variance")
        # axes[1].grid(True, linestyle='--', alpha=0.6)
        
        # # Total activity
        # axes[2].plot(network.total_activity, color='red')
        # axes[2].set_title("Total Network Activity")
        # axes[2].set_xlabel("Generation")
        # axes[2].set_ylabel("Activity Level")
        # axes[2].grid(True, linestyle='--', alpha=0.6)
        
        # plt.tight_layout()
        # pdf.savefig(fig2)
        # plt.close(fig2)
        
        # # Plot 3: Lyapunov Energy
        # fig3 = plt.figure(figsize=(10, 7))
        # generations = np.arange(len(network.lyapunov))
        # plt.scatter(generations, network.lyapunov, s=1, color='red')
        # plt.plot(generations, network.lyapunov, alpha=0.5, color='firebrick')
        # plt.xlabel("Generation")
        # plt.ylabel("Energy")
        # plt.title("Network Lyapunov Energy over Generations")
        # plt.grid(True, linestyle='--', alpha=0.6)
        # pdf.savefig(fig3)
        # plt.close(fig3)

        # Plot 4: Tuning Curves (if available)
        if network.tuning_curves:
            fig4 = plt.figure(figsize=(12, 5))
            for key, values in network.tuning_curves.items():
                plt.plot(values.cpu().numpy(), label=f"Tuning Curve {key.cpu().numpy()}")
            plt.title("Tuning Curves")
            plt.xlabel("Direction")
            plt.ylabel("Activity")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            pdf.savefig(fig4)
            plt.close(fig4)
        
        fig7 = plt.figure(figsize=(10, 7))
        plt.text(0.5, 0.5, f"Network Parameters:\n\n"
                            f"Number of Neurons: {network.num_neurons}\n"
                            f"Noise: {network.sigma_temp}\n"
                            f"Field Width: {network.sigma_input}\n"
                            f"Synaptic Failure: {network.syn_fail}\n"
                            f"Spontaneous Release: {network.spon_rel}\n"
                            f"Constriction: {network.constrict}\n"
                            f"Fraction Active: {network.threshold_active_fraction}\n"
                            f"AMPA Conductance: {network.ampar_conductance}\n"
                            f"Input Resistance: {network.input_resistance}\n",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12) 
        plt.axis('off')
        pdf.savefig(fig7)
        plt.close(fig7)

        # Plot tuning curve of dynamics, where find the average on rate per neuron.
        fig8 = plt.figure(figsize=(10, 7))
        state_history = _load_state_history_array(network)
        avg_on_rate = np.mean(state_history[30:,:], axis=0)
        #calculate width of curve at half max
        half_max = np.max(avg_on_rate) / 2
        indices_above_half = np.where(avg_on_rate >= half_max)[0]
        hwhm = (indices_above_half[-1] - indices_above_half[0]) / 2
        plt.plot(avg_on_rate, color='purple')
        plt.title(f"Average On Rate per Neuron (after 30 generations), HWHM: {hwhm:.4f} neurons")
        plt.xlabel("Neuron Index")
        plt.ylabel("Average On Rate")
        plt.grid(True, linestyle='--', alpha=0.6)
        pdf.savefig(fig8)
        plt.close(fig8)

  
        # if network.input_fluctuations is not None and len(network.input_fluctuations)>0:
        #     fig10 = plt.figure(figsize=(10, 7))
        #     plt.plot(network.input_fluctuations, color='brown')
        #     plt.title("Input Fluctuations Over Generations")
        #     plt.xlabel("Generation")
        #     plt.ylabel("Input Fluctuation (A value)")
        #     plt.grid(True, linestyle='--', alpha=0.6)
        #     pdf.savefig(fig10)
        #     plt.close(fig10)

                  #Figure of histogram of covariance matrix values flattened
        # if network.covariance_matrix is not None:
        #     fig9 = plt.figure(figsize=(10, 7))
        #     cov_matrix = network.covariance_matrix.cpu().numpy()
        #     upper_tri_indices = np.triu_indices_from(cov_matrix, k=1)
        #     cov_values = cov_matrix[upper_tri_indices]
        #     plt.hist(cov_values, bins=50, color='orange', edgecolor='black', density=True)
        #     plt.axvline(np.mean(cov_values), color='red', linestyle='--', label=f"Mean: {np.mean(cov_values):.5f}")
        #     plt.title("Histogram of Covariance Matrix Values")
        #     plt.xlabel("Covariance Value")
        #     plt.legend()
        #     plt.ylabel("Frequency")
        #     plt.grid(True, linestyle='--', alpha=0.6)
        #     pdf.savefig(fig9)
        #     plt.close(fig9)
        
        # fig11 = plt.figure(figsize=(10, 7))
        # evals, evecs = np.linalg.eigh(cov_matrix)
        # plt.hist(evals, density=True, color='teal', bins = network.num_neurons//3)
        # plt.axvline(np.mean(evals), color='red', linestyle='--', label=f"Mean Eigenvalue: {np.mean(evals):.5f}")
        # #print participation ratio
        # pr = (np.sum(evals)**2) / np.sum(evals**2)
        # plt.title(f"Eigenvalues of Covariance Matrix (Participation Ratio: {pr:.4f}, Mean: {np.mean(evals):.5f})")
        # plt.xlabel("Eigenvalue")
        # plt.ylabel("Density")
        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.6)
        # pdf.savefig(fig11)
        # plt.close(fig11)
        # # correlation matrix histogram
        # fig12 = plt.figure(figsize=(10, 7))
        # diag = np.sqrt(np.diag(cov_matrix))
        # corr_matrix = cov_matrix / np.outer(diag, diag)
        # upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
        # corr_values = corr_matrix[upper_tri_indices]
        # plt.hist(corr_values, bins=50, color='cyan', edgecolor='black', density=True)
        # plt.axvline(np.mean(corr_values), color='red', linestyle='--', label=f"Mean: {np.mean(corr_values):.5f}")
        # plt.title("Histogram of Correlation Matrix Values")
        # plt.xlabel("Correlation Value")
        # plt.legend()
        # plt.ylabel("Frequency")
        # plt.grid(True, linestyle='--', alpha=0.6)
        # pdf.savefig(fig12)
        # plt.close(fig12)
        # # evals of correlation matrix
        # fig13 = plt.figure(figsize=(10, 7))
        # corr_evals, corr_evecs = np.linalg.eigh(corr_matrix)
        # plt.hist(corr_evals, density=True, color='magenta', bins = network.num_neurons//3)
        # plt.axvline(np.mean(corr_evals), color='red', linestyle='--', label=f"Mean Eigenvalue: {np.mean(corr_evals):.5f}")
        # plt.title(f"Eigenvalues of Correlation Matrix (Mean: {np.mean(corr_evals):.5f})")
        # plt.xlabel("Eigenvalue")
        # plt.ylabel("Density")
        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.6)
        # pdf.savefig(fig13)
        # plt.close(fig13)


        
    return str(output_file)

def save_data(network: CANNetwork, generations:int, output_dir: str = "reports") -> None:
    """
    Save the full network state history (all times, runs, and trials) to a file,
    along with a reshaped version for easier analysis.
    
    Args:
        network: The CANNetwork instance containing simulation data
        number_angles: Number of angles/directions used in the simulation
        number_trials: Number of trials/runs per angle
        output_dir: Directory to save the state data
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"state_history_{timestamp}.npz"
    
    # Convert state_history to numpy array
    if isinstance(network.state_history, torch.Tensor):
        state_np = network.state_history.cpu().numpy()
    else:
        state_np = torch.stack([torch.as_tensor(s) for s in network.state_history]).cpu().numpy()
    
    # Calculate length of each run
    length_run = generations -1 #len(network.state_history) // (number_angles * number_trials)
    
    # Reshape the state history for easier analysis
    reshaped_state = np.zeros((length_run, state_np.shape[1]))
    
    for i in range(number_angles * number_trials):
        dat = state_np[i * length_run:(i + 1) * length_run, :]
        reshaped_state[:, i % number_angles, i // number_angles, :] = dat
    
    # Extract the last timepoint data and transpose for easy neuron-wise analysis
    last_timepoint_data = reshaped_state[-1, :, :, :]
    neuron_first_view = last_timepoint_data.transpose(2, 0, 1)  # Shape: (n_neurons, number_angles, number_trials)
    
    # Save all versions of the data
    np.savez_compressed(
        output_file, 
        state_history=state_np,
        neuron_centric=neuron_first_view,
        metadata=np.array([length_run,number_angles, number_trials])
    )
    print(f"Full state history saved to {output_file} with the following arrays:")
    print(f" - 'state_history': raw data, shape {state_np.shape}")
    print(f" - 'neuron_centric': (neurons, angles, trials), shape {neuron_first_view.shape}")
    print(f" - 'metadata': [length_run,number_angles, number_trials]")


def save_tuning_data(network: CANNetwork, generations:int, number_angles, number_trials: int, output_dir: str = "reports") -> None:
    """
    Save the full network state history (all times, runs, and trials) to a file,
    along with a reshaped version for easier analysis.
    
    Args:
        network: The CANNetwork instance containing simulation data
        number_angles: Number of angles/directions used in the simulation
        number_trials: Number of trials/runs per angle
        output_dir: Directory to save the state data
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"state_history_{timestamp}.npz"
    
    # Convert state_history to numpy array
    if isinstance(network.state_history, torch.Tensor):
        state_np = network.state_history.cpu().numpy()
    else:
        state_np = torch.stack([torch.as_tensor(s) for s in network.state_history]).cpu().numpy()
    
    # Calculate length of each run
    length_run = generations -1 #len(network.state_history) // (number_angles * number_trials)
    
    # Reshape the state history for easier analysis
    reshaped_state = np.zeros((length_run, number_angles, number_trials, state_np.shape[1]))
    
    for i in range(number_angles * number_trials):
        dat = state_np[i * length_run:(i + 1) * length_run, :]
        reshaped_state[:, i % number_angles, i // number_angles, :] = dat
    
    # Extract the last timepoint data and transpose for easy neuron-wise analysis
    last_timepoint_data = reshaped_state[-1, :, :, :]
    neuron_first_view = last_timepoint_data.transpose(2, 0, 1)  # Shape: (n_neurons, number_angles, number_trials)
    
    # Save all versions of the data
    np.savez_compressed(
        output_file, 
        state_history=state_np,
        neuron_centric=neuron_first_view,
        metadata=np.array([length_run,number_angles, number_trials])
    )
    print(f"Full state history saved to {output_file} with the following arrays:")
    print(f" - 'state_history': raw data, shape {state_np.shape}")
    print(f" - 'neuron_centric': (neurons, angles, trials), shape {neuron_first_view.shape}")
    print(f" - 'metadata': [length_run,number_angles, number_trials]")


# Keep the original visualization functions for backward compatibility
def plot_lyapunov(network: CANNetwork) -> None:
    """Plot the Lyapunov energy over simulation generations."""
    plt.figure("Lyapunov", figsize=(10, 7))
    plt.clf()
    generations = np.arange(len(network.lyapunov))
    plt.scatter(generations, network.lyapunov, s=1, color='red')
    plt.xlabel("Generation")
    plt.ylabel("Energy")
    plt.title("Network Lyapunov Energy over Generations")
    plt.gca().set_yticklabels([])
    plt.show()

def plot_metrics(network: CANNetwork) -> None:
    """Plot metrics: centre of activity, variance, and total network activity."""
    plt.figure("Centre")
    plt.title("Centre of Active Neurons")
    plt.plot(network.centres)
    
    plt.figure("Variance")
    plt.title("Variance")
    plt.plot(network.variances)
    
    plt.figure("Activity")
    plt.title("Total Activity")
    plt.plot(network.total_activity)
    
    plt.show()

def plot_state_history(network: CANNetwork) -> None:
    """Plot the network state history."""
    plt.figure("State History")
    state_history = _load_state_history_array(network)
    state_history_plot, _ = _prepare_state_history_for_plot(state_history)
    plt.imshow(state_history_plot, aspect='auto', interpolation='nearest')
    plt.xlabel("Neuron Index")
    plt.ylabel("Generation")
    if state_history_plot.shape[0] < state_history.shape[0]:
        plt.title("Network State History (downsampled for plotting)")
    else:
        plt.title("Network State History")
    plt.show()

def view_interactive_report(network: CANNetwork, output_dir: str = "reports") -> None:
    """
    Generate and automatically open an interactive HTML report.
    """
    report_path = create_interactive_report(network, output_dir)
    print(f"Interactive report generated: {report_path}")
    
    # Automatically open in default browser
    webbrowser.open(f"file://{Path(report_path).absolute()}")
def save_state_history(network: CANNetwork, outdir: Path) -> Path:
    """
    Persist the simulated states (and optional covariance) for later analysis.

    Args:
        network: finished CANNetwork instance.
        outdir: folder to write artifacts into.

    Returns:
        Path to the saved NumPy file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    state_history_path = getattr(network, "state_history_path", None)
    if state_history_path:
        disk_path = Path(state_history_path)
        if disk_path.exists():
            return disk_path

    if not network.state_history:
        raise ValueError("network.state_history is empty; run the simulator first.")

    states = torch.stack(network.state_history).cpu().numpy()
    state_path = outdir / "state_history.npy"
    np.save(state_path, states)

    # if hasattr(network, "covariance_matrix"):
    #     torch.save(network.covariance_matrix.cpu(), outdir / "covariance.pt")

    return state_path
