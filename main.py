import pickle
import numpy as np
from car_mountain_SARSA import *

action_set=[-1,0,1]
state_i = (0.3,0.0) #(-0.523599,0.0)

xmin, xmax = -1.2, 0.5
vmin, vmax = -0.07, 0.07

N_vars=2
N_lintiles = 9
N_tilings = 10
N_actions = 3

nb_episode=428
alpha=0.05
eps=0.0
gamma=1.0
lmbda=0.9

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
      

Theta,tiling=Q_learning(params)
print(tiling)
# Saving data
pkl_file=open('data/SARSA428.pkl','wb')
pickle.dump([Theta,tiling],pkl_file)
