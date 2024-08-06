import pandas as pd
import matplotlib.pyplot as plt

trainnumber = 1
testnumber = 1
trainingFile = 'H:\\data\\softSensor\\merged{}.csv'.format(trainnumber)
testFile = 'H:\\data\\softSensor\\merged{}.csv'.format(testnumber)
data = pd.read_csv("H:\\data\\softSensor\\ft404.csv")
# Drop rows where all force and torque columns are zero
data.dropna(how="all", inplace=True)
# Create a figure and subplots
fig, axes = plt.subplots(6, 1, figsize=(8, 15), sharex=True)  # Share x-axis

# Define a list of colors for each subplot
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# Plot each force and torque component as line plots with different colors
axes[0].plot(data["ForceX"], '-', linewidth=1, color=colors[0])
axes[0].set_ylabel(r'$F_{x}(N)$')
axes[1].plot(data["ForceY"], '-', linewidth=1, color=colors[1])
axes[1].set_ylabel(r'$F_{y}(N)$')
axes[2].plot(data["ForceZ"], '-', linewidth=1, color=colors[2])
axes[2].set_ylabel(r'$F_{z}(N)$')

axes[3].plot(data["TorqueX"], '-', linewidth=1, color=colors[3])
axes[3].set_ylabel(r'$T_{x}(Nm)$')
axes[4].plot(data["TorqueY"], '-', linewidth=1, color=colors[4])
axes[4].set_ylabel(r'$T_{y}(Nm)$')
axes[5].plot(data["TorqueZ"], '-', linewidth=1, color=colors[5])
axes[5].set_ylabel(r'$T_{z}(Nm)$')
for ax in axes:
    ax.yaxis.set_label_coords(-0.04, 0.5)

# Add labels and adjust layout
for ax in axes.flat:
    ax.grid(True)
    ax.yaxis.set_ticks_position('left')  # Set y-axis ticks on the left side
    ax.spines['right'].set_visible(False)  # Hide the right spine
    ax.spines['top'].set_visible(False)  # Hide the top spine
axes[-1].set_xlabel("Sequence")  # Only label the bottom x-axis

plt.tight_layout()
plt.show()