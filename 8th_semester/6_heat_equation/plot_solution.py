import matplotlib.pyplot as plt
import numpy as np

matrix = np.loadtxt("solution_fin.txt", dtype='f', delimiter=',').T

fig, ax = plt.subplots()
ax.imshow(matrix)
#ax.set_xticks(np.arange(0,102,1))
#ax.set_yticks(np.arange(0,102,1))
#ax.grid()
plt.show()
