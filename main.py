import pickle
import numpy as np
from car_mountain_SARSA import *
from plotting import *

action_set=[-1,0,1]
state_i = (0.3,0.0) #(-0.523599,0.0)

xmin, xmax = -1.2, 0.5
vmin, vmax = -0.07, 0.07

N_vars=2
N_lintiles = 9
N_tilings = 10
N_actions = 3

nb_episode=1000
alpha=0.05
eps=0.0
gamma=1.0
lmbda=0.0

params={}
params["nb_episode"]=nb_episode
params["alpha"]=alpha
params["eps"]=eps
params["gamma"]=gamma
params["lmbda"]=lmbda
params["N_vars"]=N_vars
params["N_lintiles"]=N_lintiles
params["N_tilings"]=N_tilings
params["N_actions"]=N_actions
params["xmin"]=xmin
params["xmax"]=xmax
params["vmin"]=vmin
params["vmax"]=vmax
params["action_set"]=action_set
params["state_i"]=state_i
      
#===============================================================================
# 
# Theta,tiling=Q_learning(params)
# pkl_file=open('data/SARSA_1000.pkl','wb')
# pickle.dump([Theta,tiling],pkl_file)
# exit()
#===============================================================================

#===============================================================================
# Theta,tiling=Q_learning_t(params)
# pkl_file=open('data/SARSA_t_1000.pkl','wb')
# pickle.dump([Theta,tiling],pkl_file)
#  
#  
#  
# exit()
#===============================================================================

#===============================================================================
#  Theta,tiling=Q_learning_t(params)
#  pkl_file=open('data/SARSA_t_1000.pkl','wb')
#  pickle.dump([Theta,tiling],pkl_file)
#  exit(0)
# Q,tiling=Q_learning_v2(params)
# print(Q)
# print(tiling)
#  Saving data
#  N_lintiles = 9
#  N_tilings = 10
#===============================================================================

#===============================================================================
# 
# pkl_file=open('data/SARSA_t_1000.pkl','rb')
# #pickle.dump([Theta,tiling],pkl_file)
# Theta,tiling=pickle.load(pkl_file)
# plot_trajectory_time(Theta,tiling)
#===============================================================================

#Theta,tiling=pickle.load(pkl_file)
#plot_surface_action_max(Theta,tiling,(-1.2,0.5,0.03),(-0.07,0.07,0.004),N_lintiles,N_tilings)

exit()
#exit()

#Qtable,tiling=pickle.load(pkl_file)

plot_surface_action_max_LT(Qtable,tiling,(-1.2,0.5,0.02),(-0.07,0.07,0.002))

#trajectory=np.array(compute_trajectory_LT((0.3,0.),Qtable,tiling))#
#print(trajectory)
#plot_trajectory(trajectory)
#print(trajectory)
#print(len(trajectory))

