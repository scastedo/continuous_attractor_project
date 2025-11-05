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
    
    # 1. Random Data Scatter Plot (Definitive Test)
    state_history = torch.stack(network.state_history).cpu().numpy()
    heatmap = go.Heatmap(
        z=state_history,
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
        state_history = torch.stack(network.state_history).cpu().numpy()
        cmap = colors.ListedColormap(['#FFFFFF', '#D291BC'])
        plt.imshow(state_history, aspect='auto', interpolation='nearest', cmap=cmap)
        plt.xlabel("Neuron Index")
        plt.ylabel("Generation")
        plt.title("Network State History")
        pdf.savefig(fig1)
        plt.close(fig1)
        
        # Plot 2: Metrics (Combined)
        fig2, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        # Center of activity
        axes[0].plot(network.centres, color='blue')
        axes[0].set_title("Centre of Active Neurons")
        axes[0].set_ylabel("Centre Position")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # Variance
        axes[1].plot(network.variances, color='green')
        axes[1].set_title("Variance of Activity")
        axes[1].set_ylabel("Variance")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        # Total activity
        axes[2].plot(network.total_activity, color='red')
        axes[2].set_title("Total Network Activity")
        axes[2].set_xlabel("Generation")
        axes[2].set_ylabel("Activity Level")
        axes[2].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)
        
        # Plot 3: Lyapunov Energy
        fig3 = plt.figure(figsize=(10, 7))
        generations = np.arange(len(network.lyapunov))
        plt.scatter(generations, network.lyapunov, s=1, color='red')
        plt.plot(generations, network.lyapunov, alpha=0.5, color='firebrick')
        plt.xlabel("Generation")
        plt.ylabel("Energy")
        plt.title("Network Lyapunov Energy over Generations")
        plt.grid(True, linestyle='--', alpha=0.6)
        pdf.savefig(fig3)
        plt.close(fig3)

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
        # Plot 5: Correlations
        if len(network.correlations)>0:
            fig5 = plt.figure(figsize=(10, 7))
            correlations_array = network.correlations
            plt.hist(correlations_array, bins=20, color='skyblue', edgecolor='black')
            plt.grid(True, linestyle='--', alpha=0.6)
            mean_corr = np.nanmean(correlations_array)
            plt.axvline(mean_corr, color='red', linestyle='--', label=f"Mean Correlation: {mean_corr:.2f}")
            plt.text(mean_corr, 0, f"{mean_corr:.2f}", color='red', ha='right', va='bottom')
            plt.xlabel("Correlation Coefficient")
            plt.title(f"Network Correlation, Mean = {mean_corr:.2f}")

            plt.ylabel("Frequency")
            pdf.savefig(fig5)
            plt.close(fig5)
        #Print Table with parameters values:
        # "num_neurons": 100,
        # "noise": 0.01, #above 0.01 is too much
        # "field_width": 0.05,
        # "syn_fail": 0.3,
        # "spon_rel": 0.05,
        # "constrict": 1.0,
        # "fraction_active": 0.1,
        fig6 = plt.figure(figsize=(10, 7))
        plt.text(0.5, 0.5, f"Network Parameters:\n\n"
                            f"Number of Neurons: {network.num_neurons}\n"
                            f"Noise: {network.noise}\n"
                            f"Field Width: {network.field_width}\n"
                            f"Synaptic Failure: {network.syn_fail}\n"
                            f"Spontaneous Release: {network.spon_rel}\n"
                            f"Constriction: {network.constrict}\n"
                            f"Fraction Active: {network.fraction_active}\n",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12) 
        plt.axis('off')
        pdf.savefig(fig6)
        plt.close(fig6)
        # Plot tuning curve of dynamics, where find the average on rate per neuron.
        # The plot should be probability on the y axis and x axis should be neuron index (average over all time starting after 10 generations)
        fig7 = plt.figure(figsize=(10, 7))
        state_history = torch.stack(network.state_history).cpu().numpy()
        avg_on_rate = np.mean(state_history[30:,:], axis=0)
        # Smooth with a simple moving average; adjust window (odd preferable) to change smoothing amount
        # window = 3
        # kernel = np.ones(window) / window
        # smoothed = np.convolve(avg_on_rate, kernel, mode='same')
        plt.plot(avg_on_rate, color='purple')
        plt.title("Average On Rate per Neuron (after 30 generations)")
        plt.xlabel("Neuron Index")
        plt.ylabel("Average On Rate")
        plt.grid(True, linestyle='--', alpha=0.6)
        pdf.savefig(fig7)
        plt.close(fig7)


        #Figure of histogram of covariance matrix values flattened
        if network.covariance_matrix is not None:
            fig8 = plt.figure(figsize=(10, 7))
            cov_matrix = network.covariance_matrix.cpu().numpy()
            upper_tri_indices = np.triu_indices_from(cov_matrix, k=1)
            cov_values = cov_matrix[upper_tri_indices]
            plt.hist(cov_values, bins=50, color='orange', edgecolor='black', density=True)
            plt.axvline(np.mean(cov_values), color='red', linestyle='--', label=f"Mean: {np.mean(cov_values):.2f}")
            plt.title("Histogram of Covariance Matrix Values")
            plt.xlabel("Covariance Value")
            plt.legend()
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.6)
            pdf.savefig(fig8)
            plt.close(fig8)

        
    return str(output_file)

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
    state_history = torch.stack(network.state_history).cpu().numpy()
    plt.imshow(state_history, aspect='auto', interpolation='nearest')
    plt.xlabel("Neuron Index")
    plt.ylabel("Generation")
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