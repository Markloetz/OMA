import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the SDOF frequency response function model with damping ratio
def sdoF_FRF_zeta(f, m, k, zeta):
    omega = 2 * np.pi * f
    H = 1 / (-m * omega**2 + 1j * 2 * np.pi * f * zeta * m + k)
    return np.abs(H)

# Sample frequency response data (frequency, magnitude)
# Replace with your actual data
frequency_data = np.array([0.1, 0.5, 1, 2, 5, 10, 20])  # Frequency data
magnitude_data = np.array([0.05, 0.2, 0.6, 0.8, 0.4, 0.15, 0.05])  # Corresponding magnitude data

# Initial guess for parameters (m, k, zeta)
initial_guess = (1.0, 1.0, 0.1)

# Perform curve fitting
popt, pcov = curve_fit(sdoF_FRF_zeta, frequency_data, magnitude_data, p0=initial_guess)

# Extract the fitted parameters
m_fit, k_fit, zeta_fit = popt

# Generate fitted curve using the fitted parameters
fitted_curve = sdoF_FRF_zeta(frequency_data, m_fit, k_fit, zeta_fit)

# Plot the original data and the fitted curve
plt.plot(frequency_data, magnitude_data, 'bo', label='Original Data')
plt.plot(frequency_data, fitted_curve, 'r-', label='Fitted Curve')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.legend()
plt.title('Fitting SDOF Frequency Response Function with Damping Ratio')
plt.show()

print("Fitted parameters: m =", m_fit, ", k =", k_fit, ", zeta =", zeta_fit)
