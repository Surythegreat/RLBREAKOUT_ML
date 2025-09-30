import matplotlib.pyplot as plt
import os

class LivePlot():
    def __init__(self):
        # Initialize the figure and axes, storing the axes in self.ax
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.set_title('Training Progress')

        self.data = None
        self.epochs = 0 

    def update(self, stats):
        self.data = stats['avgreturns']
        self.eps_data = stats['epsilon']
        self.epochs = len(self.data)

        # Use self.ax, which was defined in __init__
        self.ax.clear() 
        self.ax.set_xlim(0, self.epochs)

        self.ax.plot(self.data, label='Average Returns')
        self.ax.plot(self.eps_data, 'r-', label='Epsilon')

        self.ax.legend(loc='upper left')

        # Create directory if it doesn't exist and save the plot
        if not os.path.exists('plots'):
            os.makedirs('plots')
        self.fig.savefig('plots/training_progress.png')