"""
TO-DO-LIST:

3. Merge first dimension in Theta
4. see the motion of the car
5. keet track of best-encountered protocol by saving the best_actionss
"""

import numpy as np
import numpy.random as random
import pickle

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

#random.seed(0)


state_i = (0.25,0.0) #(-0.523599,0.0)

max_t_steps=10

xmin, xmax = -1.2, 0.5
#vmin, vmax = -0.07, 0.07

N_vars=1
N_lintiles = 20
N_tilings = 10


global action_set
N_actions = 3
action_set=[-1,0,1]



def construct_tiling(N_dim,tile_per_dim,nb_of_tiling,range_per_dim):
    '''
    Constructs tiling for uniform grids. 
    N_dim: number of dimensions
    tile_per_dim: number of tiles per dimension (passed as a N_dim dimensional list)
    nb_of_tiling: number of tiling to generate
    range_per_dim: range per dimension given as (N_dim,2) dimensions list
    
    Returns a nb_of_tiling dimensional list of N_dim dimensional list. Each of the latter list is a N_dim dimensional list
    of numpy arrays representing the centers of the tiling along the various dimensions
    '''
    
    all_tiling=[]    
    for i in range(nb_of_tiling):
    
        tiling_centers=[]
    
        for d in range(N_dim):
            d1=(range_per_dim[d][1]-range_per_dim[d][0])/tile_per_dim[d]
            tiling_centers_dim1=np.arange(range_per_dim[d][0],range_per_dim[d][1],d1)+d1/2
            tiling_centers_dim1+=(np.random.rand()*d1-d1*0.5)
            tiling_centers.append(tiling_centers_dim1)
        all_tiling.append(tiling_centers)
    
    all_tiling2=[]
    
    for d in range(N_dim):
        tmp=np.zeros((nb_of_tiling,tile_per_dim[d]))
        for i in range(nb_of_tiling):
            tmp[i]=all_tiling[i][d]
        all_tiling2.append(tmp)
    
    
    # Returned object : first index is dimension, second index is tiling
    return all_tiling2


#print(construct_tiling(2,[4,4],3,[[-1,1],[3,4]]))
#exit()

def find_closest_index(value,tiling):
    pos=np.searchsorted(tiling,value)
    if pos==0:
        return 0
    elif pos==len(tiling):
        return len(tiling)-1
    else:
        return pos-1+np.argmin([abs(tiling[pos-1]-value),abs(tiling[pos]-value)])


def real_to_tiling(state_real,tiling,tile_per_dim,nb_of_tiling):
    dim=0
    ind_state=[]
    
    tile_begin=0
    tmp=[]
    l=len(tiling[dim])
    for tile in range(l):
        ind=find_closest_index(state_real[0],tiling[dim][tile])+tile_begin
        tmp.append(ind)
        tile_begin+=len(tiling[dim][tile])
    ind_state.append(tmp)

    
    return np.array(ind_state[0])
    #return ind_state

#tiling=construct_tiling(2,[4,4],3,[[-1,1],[3,4]])
#print(real_to_tiling((-0.2,0.01),tiling,4,3))
#exit()
#===============================================================================
# tiling=construct_tiling(2,[3,3],10,[[-1.2,0.5],[-0.07,0.07]])
# print(tiling)
# print(real_to_tiling([0.1,-0.03],tiling))
# exit()
#===============================================================================


def update_state(current_state,action,old_v):
    
    terminate=False
    old_pos,old_time=current_state
    
    # Updating velocity:
    new_velocity=old_v+0.001*action_set[action]-0.0025*np.cos(3.*old_pos)
    
    # Updating position:
    new_pos=old_pos+new_velocity
    
    if new_pos < xmin:
        new_pos=xmin
    elif new_pos > xmax:
        terminate=True

    # compute reward
    R=-0.0
    if current_state[1]==max_t_steps-1:
        R += -(xmax - new_pos)
    
    return (new_pos,old_time+1),terminate,R
 
  
def Q_learning(nb_episode=100,alpha=0.05,eps=0.1,gamma=1.0,lmbda=0.9,Theta=None):
    ''' Q_learning
    alpha is the learning rate (for gradient descent)
    eps is the exploration rate
    gamma is the discount factor
    lmbda is the trace decay rate    
     '''

    tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax]])
    if Theta is None:
        Theta=np.zeros((N_vars*N_lintiles*N_tilings,max_t_steps,N_actions),dtype=np.float32)
    else:
        eps=0.0
    
    for Ep in range(nb_episode):
        action_taken=[]
        state_taken=[]
        #print("Episode",Ep)
        
        trace=np.zeros(Theta.shape,dtype=np.float32)
        current_state_real=state_i

        old_v=0.0

        t_step=0
        while True:
            
            state_taken.append(current_state_real[0])

            indTheta=real_to_tiling(current_state_real,tiling,N_lintiles,N_tilings)
            Q=np.sum(Theta[indTheta,t_step,:],axis=0)

            action_star=np.argmax(Q) #np.random.randint(0,N_actions)
            
            if random.uniform() < eps: #/(Ep+2.0):
                action = random.choice(range(N_actions))
            else:
                action = action_star

            if action != action_star:
                trace *= 0.0

            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action,old_v)
            action_taken.append(action)

            if t_step==0:
                print("Ep,a,Q:", [Ep,action,np.sum(Theta[indTheta,t_step,action]) ])
        

            delta = R-Q[action]
            
            trace[indTheta,t_step,action]=1.0 # use replacing traces !
            
            # Terminate here
            if terminate or t_step==max_t_steps-1:
                Theta+=alpha*delta*trace
                break

            indTheta_new=real_to_tiling(new_state_real,tiling,N_lintiles,N_tilings)
            Q=np.sum(Theta[indTheta_new,t_step+1,:],axis=0)
            
            delta+=gamma*max(Q)
            Theta+=alpha*delta*trace
            trace*=gamma*lmbda

            """
            if user_input=='y':
                print current_state_real, new_state_real
                print alpha*delta
            """
            current_state_real=new_state_real 

            #print current_state_real  
            
            old_v = new_state_real[0]-current_state_real[0] 
            t_step+=1
    
        print("Episode Length:\t",t_step,R)
        #print(Theta)
    
    #print(Theta)
    return Theta,tiling,action_taken,state_taken

#===============================================================================
# random.seed(10)       
# Theta,tiling=Q_learning(nb_episode=9000,alpha=0.05,eps=0.1,gamma=1.0,lmbda=0.9)
# resultfile = open('res9000.pkl', 'wb')
# 
# pickle.dump((Theta,tiling),resultfile)
# resultfile.close()
#===============================================================================
# TRAINING

Ep=500
Theta,_,_,_=Q_learning(nb_episode=Ep,alpha=0.06,eps=0.0,gamma=1.0,lmbda=0.6)

print('GREEDY')

Theta,tiling,action_taken,state_taken=Q_learning(nb_episode=1,alpha=0.06,eps=0.0,gamma=1.0,lmbda=0.6,Theta=Theta)

fig = plt.figure()
plt.plot(range(len(action_taken)),action_taken)
plt.plot(range(len(state_taken)),state_taken)
plt.show()


exit()

current_state_real=state_i
while True:
    indTheta=real_to_tiling(current_state_real,tiling,N_lintiles,N_tilings)
    Q=np.sum(Theta[indTheta,:],axis=0)
    action_star=np.argmax(Q) #np.random.randint(0,N_actions)






qvalue=[]
for Ep in [4000]:
    Theta,tiling=Q_learning(nb_episode=Ep,alpha=0.06,eps=0.0,gamma=1.0,lmbda=0.6)
    qvalue.append([Ep,select_action(Theta,real_to_tiling(stati_i,tiling),0.0)[1]])
    print(qvalue)

qvalue=np.array(qvalue)
np.savetxt("convergence.txt",qvalue)

exit()



######### PLOTTINGGGGGGGGGGGGGG ###############
 
pkl_file = open('res1000.pkl', 'rb')
t1,til1 = pickle.load(pkl_file)

data=[]
dx=(xmax-xmin)/50.0
dv=(vmax-vmin)/50.0

for x in np.arange(xmin,xmax,dx):
    for v in np.arange(vmin,vmax,dv):
        data.append([x,v,select_action(t1,real_to_tiling((x,v),til1),0.0)[1]])

data=np.array(data)

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(data[:,0], data[:,1], abs(data[:,2]), cmap=cm.jet, linewidth=0.2)

plt.show()


ax.set_xlabel(r'Position')
ax.set_ylabel(r'Velocity')
ax.set_zlabel(r'max_a{Q(x,v,a)}')

plt.show()

#print(t1==Theta)
#print(til1)
#print(tiling)
pkl_file.close()



# Check if things make sense !
# Try to reproduce book plots !
#perform_greedy(initpos,Theta,tiling)






#tiling=construct_tiling(2,[9,9],10,[[-1.2,0.5],[-0.07,0.07]])


#===============================================================================
# print(tiling[0][0])
# print(find_closest_index(0.129,tiling[0][0]))
# print(np.argmin([0,1,2,3]))    
#===============================================================================
#===============================================================================
# def find_tiling_index(state,tiling):
#     
#     nb_tiling=len(tiling)
#     for i in range(tiling):
#         for 
#     
#===============================================================================


#===============================================================================
# print(tiling[0])
#===============================================================================
#print(construct_tiling(2,[9,9],1,[[-1.2,0.5],[-0.07,0.07]]))
#===============================================================================
# print(tuple([2,2]+[2]))
# a=0,1
# print(a)
# np.arange(a[0])
# print(a[0])
#===============================================================================

#print(np.arange(0.,1,0.05))
#construct_tiling(2,[10,10],1,(1.0,1.0))
        
