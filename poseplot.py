import pandas as pd
import matplotlib.pyplot as plt

i=5

# dicsoftelec = "H:\\data\\brushSoftSensor\\" #这是在传感器上放了个柔性传感器再测
dicsoftelec = "H:\\data\\softSensor\\"
File= dicsoftelec+"pose{}.csv".format(i)
OutFile= dicsoftelec+"pose{}_mean_centered.csv".format(i)
MergedFile= dicsoftelec+"merged{}.csv".format(i)

data = pd.read_csv(File)
# data = pd.read_csv(File)
# data = pd.read_csv(MergedFile)

min_timestamp = data["Timestamp"].min()
data["RelativeTime"] = data["Timestamp"] - min_timestamp
# Create a figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

size=0.5
# Plot each force and torque component as scatter plots
axes[0, 0].plot(data["RelativeTime"], data["Alpha"], 'o', markersize=size)  # Modified line
axes[0, 0].set_title("Alpha")
axes[0, 1].plot(data["RelativeTime"], data["Beta"], 'o', markersize=size)  # Modified line
axes[0, 1].set_title("Beta")
axes[0, 2].plot(data["RelativeTime"], data["Gama"], 'o', markersize=size)  # Modified line
axes[0, 2].set_title("Gama")

axes[1, 0].plot(data["RelativeTime"], data["X"], 'o', markersize=size)  # Modified line
axes[1, 0].set_title("X")
axes[1, 1].plot(data["RelativeTime"], data["Y"], 'o', markersize=size)  # Modified line
axes[1, 1].set_title("Y")
axes[1, 2].plot(data["RelativeTime"], data["Z"], 'o', markersize=size)  # Modified line
axes[1, 2].set_title("Z")



# Add labels and adjust layout
for ax in axes.flat:
    ax.set_xlabel("Time")
    ax.set_ylabel("Pose")
    ax.grid(True)

plt.tight_layout()
plt.show()