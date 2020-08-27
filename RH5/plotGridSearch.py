import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/julian/Dev/crocoddyl/RH5/'
csv_file = 'GridResults.csv'
PDim, DDim = 12, 5

data = pd.read_csv(path+csv_file)
E = data['minFeetError']
P = data['PGain']
D = data['DGain']

plt.figure(figsize=(16,9))
DGains = [20, 30, 40, 50, 60]
[plt.plot(P[d*PDim:(d+1)*PDim], E[d*PDim:(d+1)*PDim], label='DGain='+str(DGains[d]), marker='x') for d in range(DDim)]
plt.xlabel('PGain')
plt.ylabel('Max Foot Drift [m]')
plt.legend()
plt.savefig(path + 'GridResults.png', dpi = 300)