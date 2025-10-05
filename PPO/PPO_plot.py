import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_rewards(csv_path='training_log.csv', save_path='reward_plot.png', window_size=1000):
    """
    Reads reward data from a CSV file and generates a plot with a smoothed reward curve.

    Args:
        csv_path (str): The path to the input CSV file.
        save_path (str): The path to save the output plot image.
        window_size (int): The window size for the moving average calculation.
    """
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # Read the data from the CSV file using pandas
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Calculate the moving average for the 'Total Reward' to smooth the curve
    # The 'center=True' and 'min_periods=1' arguments help handle the edges of the data gracefully.
    data['Smoothed Reward'] = data['Average Reward'].rolling(window=window_size, center=True, min_periods=1).mean()

    # Create a new figure and axes for the plot
    plt.figure(figsize=(10, 6))

    # Plot the original 'Average Reward' with some transparency
    plt.plot(data['episodes'], data['Average Reward'], marker='o', linestyle='-', color='lightblue', alpha=0.7, label='Actual Reward')

    # Plot the smoothed reward line
    plt.plot(data['episodes'], data['Smoothed Reward'], linestyle='-', color='b', linewidth=2, label=f'Smoothed Reward (window={window_size})')

    # Add a title and labels for clarity
    plt.title('Average and Smoothed Reward per Episode')
    plt.xlabel('episodes')
    plt.ylabel('Average Reward')

    # Add a legend to distinguish the lines
    plt.legend()

    # Add a grid for better readability
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(save_path)
    print(f"Plot saved successfully to '{save_path}'")


if __name__ == '__main__':
    # The script will look for 'rewards.csv' in the same directory
    # and save the plot as 'reward_plot.png'
    plot_rewards()

