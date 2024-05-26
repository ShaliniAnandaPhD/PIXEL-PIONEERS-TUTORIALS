# File name: heart_rate_variability_analysis_python_healthcare.py
# File library: Pandas, NumPy, Matplotlib
# Use case: Healthcare - Heart Rate Variability Analysis

import pandas as pd   # Importing pandas for data manipulation
import numpy as np    # Importing numpy for numerical calculations
import matplotlib.pyplot as plt   # Importing matplotlib for plotting
from scipy.signal import welch   # Importing welch method from scipy for frequency domain analysis

# Load heart rate data from a CSV file
data = pd.read_csv('heart_rate_data.csv')  # Read the CSV file containing heart rate data

# Extract the RR intervals (in milliseconds)
rr_intervals = data['RR_Interval'].values  # Extract the RR intervals from the data

# Convert RR intervals to seconds
rr_intervals_sec = rr_intervals / 1000.0  # Convert RR intervals from milliseconds to seconds

# Calculate heart rate variability metrics
mean_rr = np.mean(rr_intervals_sec)  # Calculate the mean RR interval
sdnn = np.std(rr_intervals_sec)  # Calculate the standard deviation of RR intervals (SDNN)
rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals_sec))))  # Calculate the root mean square of successive differences (RMSSD)

# Print the results
print("Mean RR Interval (sec):", mean_rr)  # Print the mean RR interval
print("Standard Deviation of RR Intervals (SDNN):", sdnn)  # Print the SDNN
print("Root Mean Square of Successive Differences (RMSSD):", rmssd)  # Print the RMSSD

# Plot the RR intervals
plt.figure(figsize=(10, 6))  # Create a figure for the RR interval plot
plt.plot(rr_intervals_sec, marker='o', linestyle='-', color='b')  # Plot the RR intervals
plt.xlabel('Beat Number')  # Set the x-axis label
plt.ylabel('RR Interval (sec)')  # Set the y-axis label
plt.title('RR Intervals')  # Set the title of the plot
plt.grid(True)  # Enable the grid
plt.show()  # Show the plot

# Plot the Poincaré plot
plt.figure(figsize=(8, 8))  # Create a figure for the Poincaré plot
plt.plot(rr_intervals_sec[:-1], rr_intervals_sec[1:], 'b.')  # Plot the Poincaré plot
plt.xlabel('RR Interval (i) (sec)')  # Set the x-axis label
plt.ylabel('RR Interval (i+1) (sec)')  # Set the y-axis label
plt.title('Poincaré Plot')  # Set the title of the plot
plt.grid(True)  # Enable the grid
plt.show()  # Show the plot

# Perform frequency domain analysis using Welch's method
fs = 4.0  # Define the sampling frequency (adjust according to your data)

# Compute the power spectral density (PSD) using Welch's method
frequencies, psd = welch(rr_intervals_sec, fs=fs)  # Calculate the PSD using Welch's method

# Plot the PSD
plt.figure(figsize=(10, 6))  # Create a figure for the PSD plot
plt.plot(frequencies, psd)  # Plot the PSD
plt.xlabel('Frequency (Hz)')  # Set the x-axis label
plt.ylabel('Power Spectral Density (s^2/Hz)')  # Set the y-axis label
plt.title('Power Spectral Density (PSD)')  # Set the title of the plot
plt.grid(True)  # Enable the grid
plt.show()  # Show the plot

# Compute the LF and HF power
lf_band = (0.04, 0.15)  # Define the low-frequency band
hf_band = (0.15, 0.4)  # Define the high-frequency band

lf_mask = (frequencies >= lf_band[0]) & (frequencies < lf_band[1])  # Create a mask for the LF band
hf_mask = (frequencies >= hf_band[0]) & (frequencies < hf_band[1])  # Create a mask for the HF band

lf_power = np.trapz(psd[lf_mask], frequencies[lf_mask])  # Calculate the LF power by integrating the PSD within the LF band
hf_power = np.trapz(psd[hf_mask], frequencies[hf_mask])  # Calculate the HF power by integrating the PSD within the HF band

# Compute the LF/HF ratio
lf_hf_ratio = lf_power / hf_power  # Calculate the LF/HF ratio

# Print the LF and HF power and the LF/HF ratio
print("LF Power:", lf_power)  # Print the LF power
print("HF Power:", hf_power)  # Print the HF power
print("LF/HF Ratio:", lf_hf_ratio)  # Print the LF/HF ratio
