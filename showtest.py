import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_csv_data(file_path):
    # Assume the function reads data and returns a NumPy array
    return pd.read_csv(file_path).values

# Read data from dose.csv
dose_data = read_csv_data('C:/Users/DELL/Documents/Python/Project II/provided-data/validation-pats/pt_201/ct.csv')
# dose_data = read_csv_data('C:/Users/DELL/Documents/Python/Project II/provided-data/validation-pats/pt_201/dose.csv')

# Convert data to a 3D array
dose_3d = np.zeros((128, 128, 128), dtype=float)  # Use float for dose values
dose_indices, dose_values = dose_data[:, 0].astype(int), dose_data[:, 1].astype(float)  # Convert to integer and float

# Filter out invalid values (e.g., values below a certain threshold)
valid_dose_values = dose_values[dose_values >= 0]  # Adjust the condition as needed
valid_dose_indices = dose_indices[dose_values >= 0]

# Assign valid values to the 3D array
dose_3d.ravel()[valid_dose_indices] = valid_dose_values

dose_3d = np.clip(dose_3d, 0, 4095)

# Extract 32 evenly spaced slices along the Z-axis (XY-plane)
num_slices = 32
z_indices = np.linspace(0, 127, num_slices, dtype=int)
ct_slices = [dose_3d[:, :, z] for z in z_indices]

# Display all 32 CT slices in a single figure
fig, axes = plt.subplots(4, 8, figsize=(16, 8))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(ct_slices[i], origin='lower')


plt.tight_layout()
plt.show()