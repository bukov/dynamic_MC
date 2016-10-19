from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import pickle
from car_mountain_SARSA import *
from matplotlib import rc    


def plot_surface_action_max(Theta,tiling,rangex,rangev,N_lintiles,N_tilings):
    data=[]
    xmin,xmax,dx=rangex
    vmin,vmax,dv=rangev
    for x in np.arange(xmin,xmax,dx):
        for v in np.arange(vmin,vmax,dv):
            data.append([x,v,np.max(np.sum(Theta[real_to_tiling((x,v),tiling,[N_lintiles,N_lintiles],N_tilings),:],axis=0))])
    
    data=np.array(data)
            
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.set_xlabel(r'Position')
    ax.set_ylabel(r'Velocity')
    ax.set_zlabel(r'$max_a{Q(x,v,a)}$')


    ax.plot_trisurf(data[:,0], data[:,1], abs(data[:,2]), linewidth=0.2,cmap=cm.jet)

    plt.show()

def plot_surface_action_max_LT(Qtable,tiling,rangex,rangev):
    data=[]
    xmin,xmax,dx=rangex
    vmin,vmax,dv=rangev
    for x in np.arange(xmin,xmax,dx):
        for v in np.arange(vmin,vmax,dv):
            indQ=real_to_discrete((x,v),tiling)
            data.append([x,v,abs(np.max(Qtable[indQ[0],indQ[1],:]))])
    
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
    
    
def plot_trajectory_time(Theta,tiling):
    trajectory=np.array(compute_trajectory_T((0.3,0.,0),Theta,tiling))
    #print(trajectory)
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.scatter(range(np.shape(trajectory)[0]),trajectory[:,0], s=10, c='b', marker="s", label='Position')
    ax1.scatter(range(np.shape(trajectory)[0]),trajectory[:,2], s=10, c='r', marker="s", label='Action')
    plt.legend(loc='upper left');
    plt.show()
    
def plot_epoch():
    '''
    Performance evaluation
    
    '''
    
    state_i_test=[(-0.9,0.01),(-0.5,0.0),(0.3,0.01),(-0.5,0.07),(-1.15,0.01)]
    N_sample=100
    
    mean_reward=np.zeros((np.arange(1,1000,40).shape[0],2))
    
    for n in range(N_sample):
        pos=0
        for episode in np.arange(1,1000,40):
            Theta,tiling=Q_learning(episode,Theta,tiling)
            for si in state_i_test:
                compute_trajectory(si,Theta,tilng)
                mean_reward[pos,1]+=len(compute_trajectory)
            pos+=1   

    mean_reward[:,0]=np.arange(1,1000,40)
    return mean_reward/(N_sample*len(state_i_test))    
    

    
    
#===============================================================================
# xmin, xmax = -1.2, 0.5
# vmin, vmax = -0.07, 0.07
# N_lintiles = 9
# N_tilings = 10
# 
# 
# pkl_file = open('data/SARSA428.pkl', 'rb')
# Theta,tiling = pickle.load(pkl_file)
# pkl_file.close()
# 
# dx=(xmax-xmin)/20.0
# dv=(vmax-vmin)/20.0
# 
# trajectory=np.array(compute_trajectory((0.3,0.0),Theta,tiling))
# 
# plot_trajectory(trajectory)
#===============================================================================
#print(trajectory)
#real_to_tiling(state_real,tiling,tile_per_dim,nb_of_tiling):
#plot_surface_action_max(Theta, tiling,(xmin,xmax,dx),(vmin,vmax,dv))
