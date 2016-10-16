"""
TO-DO-LIST:

3. Merge first dimension in Theta
4. see the motion of the car
5. keet track of best-encountered protocol by saving the best_actionss
"""

import numpy as np
import numpy.random as random
#random.seed(0)

def construct_tiling(N_dim,tile_per_dim,nb_of_tiling,range_per_dim):
    
    '''
    Constructs tiling for uniform grids. 
    N_dim: number of dimensions
    tile_per_dim: number of tiles per dimension (passed as a N_dim dimensional list)
    nb_of_tiling: number of tiling to generate (same for all dimensions)
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
    ind=0
    for s in state_real:
        tile_begin=0
        tmp=[]
        l=len(tiling[dim])
        for tile in range(l):
            ind=find_closest_index(s,tiling[dim][tile])+tile_begin
            tmp.append(ind)
            tile_begin+=len(tiling[dim][tile])
        ind_state.append(tmp)
        dim+=1
        ind=0
    
    return(np.append(np.array(ind_state[0]),np.array(ind_state[1])+tile_per_dim*nb_of_tiling))


def update_state(current_state,action):
    
    terminate=False
    old_pos,old_v=current_state
    
    # Updating velocity:
    new_velocity=old_v+0.001*(action-1)-0.0025*np.cos(3.*old_pos)
    
    # Maximum velocity ! (presence of friction)
    if new_velocity < vmin:
        new_velocity=vmin
    elif new_velocity > vmax:
        new_velocity=vmax
    
    # Updating position:
    new_pos=old_pos+new_velocity
    
    if new_pos < xmin:
        new_pos=xmin
        new_velocity=0.0
    elif new_pos > xmax:
        terminate=True

    # compute reward
    R = -1.0
    
    return (new_pos,new_velocity),terminate,R
 
 
def select_action(Theta,indTheta,eps):
    ''' 
     epsilon-greedy function for selecting action
     Q is the state action value function
     indTheta are the non-zero feature for the current state
     eps is the exploration rate
    '''
    if random.uniform() < eps:
        #Explore
        new_action=random.randint(0,N_actions)
        
    else: 
        #Greedy
        set_action=np.array([np.sum(Theta[indTheta,a]) for a in range(N_actions)])
        new_action=np.argmax(set_action)
        #print(set_action,new_action)
        
    Q_new_action=np.sum(Theta[indTheta,new_action])

    return new_action,Q_new_action
    
        
#print("voila")
  
def Q_learning(params,nb_episode=100,alpha=0.05,eps=0.1,gamma=1.0,lmbda=0.9):
    ''' SARSA
    alpha is the learning rate (for gradient descent)
    eps is the exploration rate
    gamma is the discount factor
    lmbda is the trace decay rate    
     '''
    
    nb_episode=params['nb_episode']
    alpha=params['alpha']
    eps=params['eps']
    gamma=params['gamma']
    lmbda=params['lmbda']
    global xmin,xmax,vmin,vmax,N_actions
    xmin=params['xmin']
    xmax=params['xmax']
    vmin=params['vmin']
    vmax=params['vmax']
    N_vars=params['N_vars']
    N_lintiles=params['N_lintiles']
    N_tilings=params['N_tilings']
    N_actions=params['N_actions']
    state_i=params['state_i']
    action_set=params['action_set']
    
    
    tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax],[vmin,vmax]])
    Theta=np.zeros((N_vars*N_lintiles*N_tilings,N_actions),dtype=np.float32)
    
    #print("Tiling\n",tiling)
    #print("Tiling,shape:\t",np.shape(tiling))
    #print("Theta, shape:\t",np.shape(Theta))
    #theta=np.empty((2,100),dtype=np.int8) 
    
    #indTheta=real_to_tiling(init_state_real,tiling)
    for Ep in range(nb_episode):
        #print("Episode",Ep)
        trace=np.zeros(Theta.shape,dtype=np.float32)
        current_state_real=state_i #(random.uniform(-1.2,0.5),random.uniform(-0.07,0.07))

        #print("Q:", [Ep,select_action(Theta,real_to_tiling(state_i,tiling,N_lintiles,N_tilings),0.0)[1]])
        
        indTheta=real_to_tiling(current_state_real,tiling,N_lintiles,N_tilings)
        
        action=0 #np.random.randint(0,N_actions)
        #print("trace\t",trace)
        #print("current_state_real\t",current_state_real)
        #print("indTheta\t",indTheta)
        #print("action\t",action)
        
        i=0
        while True:
            #print("ITERATION NO:\t",i)
            #if(i>5): break
            
            ### UNWRAP this a bit .... divergence.
            trace[indTheta,action]=1. # use replacing traces !
            #print("TRACE:\n",trace) 
            #if (i+1)%1000==0: print(i)#"\t",current_state_real)
            #if i>100: break
            #i+=1
            
            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action)
            #print("new_state_real\t",new_state_real)
        
            indTheta_new=real_to_tiling(new_state_real,tiling,N_lintiles,N_tilings)
            Q=np.sum(Theta[indTheta,action])
            #print("Q estimate:\t",Q)
            
            delta=R-Q
            
            # Terminate here
            if terminate:
                Theta+=alpha*delta*trace
                break
            
            new_action,Q_new_action=select_action(Theta,indTheta_new,eps/np.log2(Ep+2.0))
            
            delta+=gamma*Q_new_action
            Theta+=alpha*delta*trace
            trace*=gamma*lmbda
            
            current_state_real=new_state_real   
            indTheta=indTheta_new
            action=new_action
            
            i+=1
    
        #print("Episode Length:\t",i)
        #print(Theta)
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",i)
    #print(Theta)
    return Theta,tiling
#===============================================================================
# random.seed(10)       
# Theta,tiling=Q_learning(nb_episode=9000,alpha=0.05,eps=0.1,gamma=1.0,lmbda=0.9)
# resultfile = open('res9000.pkl', 'wb')
# 
# pickle.dump((Theta,tiling),resultfile)
# resultfile.close()
#===============================================================================
#===============================================================================
# 
# qvalue=[]
# for Ep in [2000]:
#     Theta,tiling=Q_learning(nb_episode=Ep,alpha=0.05,eps=0.1,gamma=1.0,lmbda=0.9)
#     qvalue.append([Ep,select_action(Theta,real_to_tiling((-0.523599,0.0),tiling),0.0)[1]])
#     print(qvalue)
# 
# qvalue=np.array(qvalue)
# np.savetxt("convergence.txt",qvalue)
#===============================================================================


