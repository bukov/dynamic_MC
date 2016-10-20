import pickle
import numpy as np
from func_approx import *
from plotting import *

action_set=[-1,0,1]
state_i = (0.4,0.0) #(-0.523599,0.0)

xmin, xmax = -1.2, 0.5
vmin, vmax = -0.07, 0.07

N_vars=1
N_lintiles = 40 # 40
N_tilings = 200 # 200
N_actions = 3

max_t_steps = 30 # 30

nb_episode=2500 # 2000
alpha=0.5/N_tilings
eps=0.0
gamma=1.0
lmbda=1.0

params={}

phys_params={"nb_episode":nb_episode,"alpha":alpha,"eps":eps,"gamma":gamma,"lmbda":lmbda,
			 "N_vars":N_vars,"N_lintiles":N_lintiles,"N_tilings":N_tilings,"N_actions":N_actions,
			 "action_set":action_set,"state_i":state_i,"max_t_steps":max_t_steps}
RL_params={"xmin":xmin,"xmax":xmax,"vmin":vmin,"vmax":vmax}

params.update(phys_params)
params.update(RL_params)

Theta,tiling=RL_QL_time(params)

exit()

#===============================================================================
# 
# nb_episode=10
# Theta,tiling=RL_SARSA(params)
# print(real_to_tiling((-0.9,0.01),tiling,[N_lintiles,N_lintiles],N_tilings))
# exit()
#===============================================================================

#epoch_SARSA=plot_epoch(params,method="RL_SARSA")
#pkl_file=open('data/epoch1.pkl','wb')
#pickle.dump(epoch_SARSA,pkl_file)

pkl_file=open('data/epoch1.pkl','rb')
data=pickle.load(pkl_file)


plt.plot(data)
plt.show()



print(data)
exit()
 
 
 
 
Theta,tiling=RL_QL_time(params,TO=False)

#plot_surface_action_max(Theta,tiling,[xmin,xmax,0.03],[vmin,vmax,0.004],N_lintiles,N_tilings)

exit()

pkl_file=open('data/SARSA_1000.pkl','wb')
pickle.dump([Theta,tiling],pkl_file)
exit()

## EPOCH plots ###



Theta,tiling=RL_SARSA(params)
params['Theta']=Theta
params['tiling']=tiling














#===============================================================================
# Theta,tiling=SARSA_t(params)
# pkl_file=open('data/SARSA_t_1000.pkl','wb')
# pickle.dump([Theta,tiling],pkl_file)
#  
#  
#  
# exit()
#===============================================================================

#===============================================================================
#  Theta,tiling=SARSA_t(params)
#  pkl_file=open('data/SARSA_t_1000.pkl','wb')
#  pickle.dump([Theta,tiling],pkl_file)
#  exit(0)
# Q,tiling=SARSA_v2(params)
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

#plot_surface_action_max_LT(Qtable,tiling,(-1.2,0.5,0.02),(-0.07,0.07,0.002))

#trajectory=np.array(compute_trajectory_LT((0.3,0.),Qtable,tiling))#
#print(trajectory)
#plot_trajectory(trajectory)
#print(trajectory)
#print(len(trajectory))

