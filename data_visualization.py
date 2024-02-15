import numpy as np 
import matplotlib.pyplot as plt

datafile = 'data.csv' # File to load 

data = np.loadtxt(datafile, delimiter=',') 

# Plot runtimes 
plt.loglog(data[:,0], data[:,1], label='GPU')
plt.loglog(data[:,0], data[:,2], label='CPU')
plt.xlabel("Matrix Leading Dimension")
plt.ylabel("Runtime (Sec)")
plt.legend() 
plt.title("GPU and CPU Runtimes vs. Matrix Dimension")
plt.savefig('runtimes_vs_mat_dims.png')