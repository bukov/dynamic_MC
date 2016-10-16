from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import pickle
from car_mountain_SARSA import *


xmin, xmax = -1.2, 0.5
vmin, vmax = -0.07, 0.07
N_lintiles = 9
N_tilings = 10


pkl_file = open('data/SARSA428.pkl', 'rb')
t1,til1 = pickle.load(pkl_file)

data=[]
dx=(xmax-xmin)/20.0
dv=(vmax-vmin)/20.0

#real_to_tiling(state_real,tiling,tile_per_dim,nb_of_tiling):


for x in np.arange(xmin,xmax,dx):
    for v in np.arange(vmin,vmax,dv):
        data.append([x,v,np.min(np.sum(t1[real_to_tiling((x,v),til1,N_lintiles,N_tilings),:],axis=0))])

data=np.array(data)

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(data[:,0], data[:,1], abs(data[:,2]), linewidth=0.2,cmap=cm.jet)

plt.show()


ax.set_xlabel(r'Position')
ax.set_ylabel(r'Velocity')
ax.set_zlabel(r'max_a{Q(x,v,a)}')

plt.show()

#print(t1==Theta)
#print(til1)
#print(tiling)
pkl_file.close()