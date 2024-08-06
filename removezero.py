import pandas as pd
import numpy as np

def removezero(File, OutFile, threshold):
    # Load the CSV file
    data = pd.read_csv(File)

    # 1. Remove rows where Index is 0
    data = data[data["Index"] != 0]

    # 2. Calculate resultant force
    data["Resultant Force"] = np.sqrt(data["ForceX"]**2 + data["ForceY"]**2 + data["ForceZ"]**2)

    # 3. Remove rows where resultant force is less than 1N
    data = data[data["Resultant Force"] >= threshold]

    # 4. Increase Torque values by 100 times
    data['TorqueX'] = data['TorqueX'] * 10
    data['TorqueY'] = data['TorqueY'] * 10
    data['TorqueZ'] = data['TorqueZ'] * 10

    # Now you have the filtered DataFrame 'data'
    # You can save it to a new CSV if needed:
    data.to_csv(OutFile, index=False)