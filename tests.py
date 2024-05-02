import numpy as np
import matplotlib.pyplot as plt

# Example data (replace these with your actual data)
fSDOF = np.random.rand(10, 5)  # Example fSDOF array
sSDOF = np.random.rand(10, 5)  # Example sSDOF array
fPeaks = np.random.rand(5)  # Example fPeaks array
Peaks = np.random.rand(5)  # Example Peaks array
nPeaks = 5  # Example value of nPeaks

# Plotting the singular values
for i in range(nPeaks):
    fSDOF_temp_1 = fSDOF[:, i]
    sSDOF_temp_1 = sSDOF[:, i]
    fSDOF_temp_2 = fSDOF_temp_1[~np.isnan(fSDOF_temp_1)]
    sSDOF_temp_2 = sSDOF_temp_1[~np.isnan(sSDOF_temp_1)]
    color = ((nPeaks - i) / nPeaks, (i + 1) / nPeaks, 0.5, 1)
    plt.plot(fSDOF_temp_2, sSDOF_temp_2, color=color)

plt.plot(fPeaks, Peaks, marker='o', linestyle='none')
plt.xlabel('Frequency')
plt.ylabel('Singular Values')
plt.title('Singular Value Plot')
plt.grid(True)
plt.show()
