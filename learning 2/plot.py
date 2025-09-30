import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_data(log_file='training_log.csv', output_dir='plots'):
    """
    Reads the training log and generates a single figure with plots for 
    reward, loss, and epsilon overlaid on one another.
    """
    # Check if the log file exists
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the data using pandas
    data = pd.read_csv(log_file)
    
    # Calculate a moving average to smooth the curves (optional but recommended)
    window_size = 100
    data['Smoothed Reward'] = data['Total Reward'].rolling(window=window_size, min_periods=1).mean()
    loss_data = data[data['Loss'] > 0].copy() # Use .copy() to avoid SettingWithCopyWarning
    loss_data['Smoothed Loss'] = loss_data['Loss'].rolling(window=window_size, min_periods=1).mean() * 10

    # --- Create a single figure and a primary axis ---
    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.suptitle('Training Progress Overview', fontsize=16)

    # --- Plot 1: Smoothed Reward (Primary Y-axis, left) ---
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Smoothed Reward', color=color)
    ax1.plot(data['Epoch'], data['Smoothed Reward'], color=color, label='Smoothed Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Create a secondary axis that shares the same x-axis ---
    ax2 = ax1.twinx()  

    # --- Plot 2: Smoothed Loss (Secondary Y-axis, right) ---
    color = 'tab:orange'
    ax2.set_ylabel('Loss / Epsilon') # Shared label for the right axis
    ax2.plot(loss_data['Epoch'], loss_data['Smoothed Loss'], color=color, label='Smoothed Loss')
    ax2.tick_params(axis='y')

    # --- Plot 3: Epsilon Decay (Secondary Y-axis, right) ---
    color = 'tab:green'
    ax2.plot(data['Epoch'], data['Epsilon'], color=color, label='Epsilon')

    # --- Create a unified legend ---
    # To avoid overlapping legends, we manually combine them.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Adjust layout to prevent titles and labels from overlapping
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the combined figure
    output_path = os.path.join(output_dir, 'overlaid_training_plot.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"📈 Overlaid plot has been saved to '{output_path}'.")

if __name__ == '__main__':
    plot_training_data()

