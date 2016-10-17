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



def compute_trajectory(state_i,Theta,tiling):
    state=state_i
    tile_per_dim,nb_of_tiling=np.shape(tiling[0])
    trajectory=[]
    max_it=1000
    it=0
    global xmin,xmax,vmin,vmax,N_actions
    xmin=-1.2
    xmax=0.5
    vmin=-0.07
    vmax=0.07
    N_actions=3
    
    
    while True:
        indTheta=real_to_tiling(state,tiling,tile_per_dim,nb_of_tiling)
        Q=np.sum(Theta[indTheta,:],axis=0)
        action=np.argmax(Q)
        
        
        trajectory.append([state[0],state[1],action,np.max(Q)])
        new_state,terminate,_=update_state(state,action)
        if terminate: break
        state=new_state
        it+=1
        if it>max_it: break
    
    return trajectory
 
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
    
    #indTheta=real_to_tiling(init_state_real,tiling)
    for Ep in range(nb_episode):
        #print("Episode",Ep)
        trace=np.zeros(Theta.shape,dtype=np.float32)
        current_state_real=(random.uniform(-1.2,0.5),random.uniform(-0.07,0.07))

        #print("Q:", [Ep,select_action(Theta,real_to_tiling(state_i,tiling,N_lintiles,N_tilings),0.0)[1]])
        
        indTheta=real_to_tiling(current_state_real,tiling,N_lintiles,N_tilings)
        
        action=np.random.randint(0,N_actions)
        
        i=0
        while True:
            trace[indTheta,action]=1. # use replacing traces !

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
            
            new_action,Q_new_action=select_action(Theta,indTheta_new,eps)
            
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


# Used only for Q_learning_v2
def real_to_discrete(state,tiling):
    tx,tv=tiling
    x,v=state
    return np.argmin(abs(tx-x)),np.argmin(abs(tv-v))

def Q_learning_v2(params,nb_episode=100,alpha=0.05,eps=0.1,gamma=1.0,lmbda=0.9):
    ''' SARSA with look up table
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
    
    
    #tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax],[vmin,vmax]])
    Qtable=np.zeros((100,100,3),dtype=np.float32)
    tiling=(np.linspace(-1.2,0.5,100),np.linspace(-0.07,0.07,100))
    
    #indTheta=real_to_tiling(init_state_real,tiling)
    for Ep in range(nb_episode):
        #print("Episode",Ep)
        trace=np.zeros(Qtable.shape,dtype=np.float32)
        current_state_real=(random.uniform(-1.2,0.5),random.uniform(-0.07,0.07))
        action=np.random.randint(0,N_actions)
        indQ=real_to_discrete(current_state_real,tiling)
        
        #=======================================================================
        # print(current_state_real)
        # print(action)
        # print(indQ)
        # 
        # exit()
        #=======================================================================
        
        #print("Q:", [Ep,select_action(Theta,real_to_tiling(state_i,tiling,N_lintiles,N_tilings),0.0)[1]])
        
        #indTheta=real_to_tiling(current_state_real,tiling,N_lintiles,N_tilings)
        
        
        i=0
       # print("initial state",current_state_real," ",indQ)
       # print("initial action",action)
        while True:
            trace[indQ[0],indQ[1],action]=1. # use replacing traces !
        
            #print("Q estimate:\t",Q)
    
            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action)
            
            delta=R-Qtable[indQ[0],indQ[1],action]
            #print("new_state_real\t",new_state_real)
        
            indQ_new=real_to_discrete(new_state_real,tiling)
            
            # Terminate here
            if terminate:
                Qtable+=alpha*delta*trace
                break
            if i>500:
                Qtable+=alpha*delta*trace
                break
            
            if np.random.uniform() < eps:
                new_action=np.random.randint(0,N_actions)
            else:
                new_action=np.argmax(Qtable[indQ_new[0],indQ_new[1],:])
            
            
            Q_new=Qtable[indQ_new[0],indQ_new[1],new_action]
            
            delta+=gamma*Q_new
            Qtable+=alpha*delta*trace
            trace*=gamma*lmbda
            
            current_state_real=new_state_real
            indQ=indQ_new   
            action=new_action
            
            i+=1
    
        #print("Episode Length:\t",i)
        #print(Theta)
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",i)
    #print(Theta)
    return Qtable,tiling
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


