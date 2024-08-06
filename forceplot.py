import pandas as pd
import matplotlib.pyplot as plt
from removezero import removezero
from datapro import interpolate_csv_by_index,interpolate_csv_by_index_with_average

i=4
# File= "H:\\data\\brushFTarea\\240703\\ft{}.csv".format(i)
# OutFile= "H:\\data\\brushFTarea\\240703\\ft{}_processed.csv".format(i)


# File= "H:\\data\\brushFTarea\\electric\\ft{}.csv".format(i)
# OutFile= "H:\\data\\brushFTarea\\electric\\ft{}_processed.csv".format(i)
# InterFile="H:\\data\\brushFTarea\\electric\\ft{}_interpolated.csv".format(i)

# dicsoftelec="H:\\data\\brushFTarea\\softelectric\\" #这是直接使用传感器测刷牙区域分类
dicsoftelec = "H:\\data\\brushSoftSensor\\" #这是在传感器上放了个柔性传感器再测
File= dicsoftelec+"ft{}.csv".format(i)
OutFile= dicsoftelec+"ft{}_processed.csv".format(i)
InterFile=dicsoftelec+"ft{}_interpolated.csv".format(i)

data = pd.read_csv(InterFile)
# data = pd.read_csv(File)
# data = pd.read_csv("H:\\data\\brushFTarea\\tester\\ft11_interpolated.csv")

min_timestamp = data["Timestamp"].min()
data["RelativeTime"] = data["Timestamp"] - min_timestamp
# Create a figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

size=0.5
# Plot each force and torque component as scatter plots
axes[0, 0].plot(data["RelativeTime"], data["ForceX"], 'o', markersize=size)  # Modified line
axes[0, 0].set_title("Force X")
axes[0, 1].plot(data["RelativeTime"], data["ForceY"], 'o', markersize=size)  # Modified line
axes[0, 1].set_title("Force Y")
axes[0, 2].plot(data["RelativeTime"], data["ForceZ"], 'o', markersize=size)  # Modified line
axes[0, 2].set_title("Force Z")

axes[1, 0].plot(data["RelativeTime"], data["TorqueX"], 'o', markersize=size)  # Modified line
axes[1, 0].set_title("Torque X")
axes[1, 1].plot(data["RelativeTime"], data["TorqueY"], 'o', markersize=size)  # Modified line
axes[1, 1].set_title("Torque Y")
axes[1, 2].plot(data["RelativeTime"], data["TorqueZ"], 'o', markersize=size)  # Modified line
axes[1, 2].set_title("Torque Z")



# Add labels and adjust layout
for ax in axes.flat:
    ax.set_xlabel("Time")
    ax.set_ylabel("Force/Torque")
    ax.grid(True)

plt.tight_layout()
plt.show()