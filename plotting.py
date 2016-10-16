from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import pickle
from car_mountain_SARSA import *
from matplotlib import rc


#===============================================================================
# x=np.arange(0,5,0.1)
# y=np.arange(0,5,0.1)
# plt.xlabel(r'Position')
# plt.ylabel(r'Velocity$e^2$')
# plt.scatter(x,y)
# plt.show()
# 
# 
# exit()
#  
#===============================================================================
def plot_surface_action_max(Theta,tiling,rangex,rangev):
    data=[]
    xmin,xmax,dx=rangex
    vmin,vmax,dv=rangev
    for x in np.arange(xmin,xmax,dx):
        for v in np.arange(vmin,vmax,dv):
            data.append([x,v,np.min(np.sum(Theta[real_to_tiling((x,v),tiling,N_lintiles,N_tilings),:],axis=0))])
    
    data=np.array(data)
            
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.set_xlabel(r'Position')
    ax.set_ylabel(r'Velocity')
    ax.set_zlabel(r'$max_a{Q(x,v,a)}$')


    ax.plot_trisurf(data[:,0], data[:,1], abs(data[:,2]), linewidth=0.2,cmap=cm.jet)

    plt.show()

        
def plot_trajectory(trajectory):
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.scatter(range(np.shape(trajectory)[0]),trajectory[:,0], s=10, c='b', marker="s", label='Position')
    ax1.scatter(range(np.shape(trajectory)[0]),trajectory[:,2], s=10, c='r', marker="s", label='Action')
    plt.legend(loc='upper left');
    plt.show()
    
    

xmin, xmax = -1.2, 0.5
vmin, vmax = -0.07, 0.07
N_lintiles = 9
N_tilings = 10


pkl_file = open('data/SARSA428.pkl', 'rb')
Theta,tiling = pickle.load(pkl_file)
pkl_file.close()

dx=(xmax-xmin)/20.0
dv=(vmax-vmin)/20.0

trajectory=np.array(compute_trajectory((0.3,0.0),Theta,tiling))

plot_trajectory(trajectory)
#print(trajectory)
#real_to_tiling(state_real,tiling,tile_per_dim,nb_of_tiling):
#plot_surface_action_max(Theta, tiling,(xmin,xmax,dx),(vmin,vmax,dv))
