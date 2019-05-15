import numpy as np

outfile = "temp_data.npz"
npzfile = np.load(outfile)
x=npzfile['x']
y=npzfile['y']
print("")
