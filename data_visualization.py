import numpy as np 
import matplotlib.pyplot as plt

datafile_sp_sg = 'data.csv' # Single precision single GPU file
datafile_dp_sg = 'data_double_precision.csv' # Double precision single GPU file
datafile_sp_mg = 'data_mg_sp.csv' # Single precision multi GPU file
datafile_dp_mg = 'data_mg_dp.csv' # Single precision multi GPU file

data_sp_sg_str = np.genfromtxt(datafile_sp_sg, delimiter=',', dtype=str, skip_header=1) 
data_dp_sg_str = np.genfromtxt(datafile_dp_sg, delimiter=',', dtype=str, skip_header=1) 
data_sp_mg_str = np.genfromtxt(datafile_sp_mg, delimiter=',', dtype=str, skip_header=1) 
data_dp_mg_str = np.genfromtxt(datafile_dp_mg, delimiter=',', dtype=str, skip_header=1) 

data_sp_sg = data_sp_sg_str.astype(float)
data_dp_sg = data_dp_sg_str.astype(float)
data_sp_mg = data_sp_mg_str.astype(float)
data_dp_mg = data_dp_mg_str.astype(float)

# Plot runtimes 
plt.loglog(data_sp_sg[:,0], data_sp_sg[:,2], label='CPU (Single Precision)')
plt.loglog(data_sp_sg[:,0], data_sp_sg[:,1], label='GPU (Single Precision)')
plt.loglog(data_sp_mg[:,0], data_sp_mg[:,1], label='2-GPUs (Single Precision)')
plt.loglog(data_dp_sg[:,0], data_dp_sg[:,2], label='CPU (Double Precision)')
plt.loglog(data_dp_sg[:,0], data_dp_sg[:,1], label='GPU (Double Precision)')
plt.loglog(data_dp_mg[:,0], data_dp_mg[:,1], label='2-GPUs (Double Precision)')
plt.xlabel("Matrix Leading Dimension")
plt.ylabel("Runtime (Sec)")
plt.legend() 
plt.grid()
plt.title("Runtime vs. Matrix Dimension for Different Setups")
plt.savefig('Plots/all.png')