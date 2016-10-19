import numpy as np
import numpy.random as random
import pickle
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
    tiling=[]
    for d in range(N_dim):
        tilings_dim=[]
        for nth_tiling in range(nb_of_tiling):
            #tiling_tmp=np.linspace(range_per_dim[d][0],range_per_dim[d][1],tile_per_dim[d])
            tilings_dim.append(np.linspace(range_per_dim[d][0],range_per_dim[d][1],tile_per_dim[d]))
            delta=(range_per_dim[d][1]-range_per_dim[d][0])/tile_per_dim[d]
            random_shift=np.random.uniform(-delta/2.,delta/2.)
            tilings_dim[-1]+=random_shift
        tiling.append(tilings_dim)
        
    return tiling

def find_closest_index(value,tiling):
    pos=np.searchsorted(tiling,value)
    if pos==0:
        return 0
    elif pos==len(tiling):
        return len(tiling)-1
    else:
        return pos-1+np.argmin([abs(tiling[pos-1]-value),abs(tiling[pos]-value)])

def real_to_tiling(state_real,tiling,tile_per_dim,nb_of_tiling):
    # This should be optimized !
    
    n_dim=np.shape(state_real)[0]

    indTheta=[]
    current_pos=0
    for tile in range(nb_of_tiling):
        coordinate=[]
        for dim in range(n_dim):
            index=find_closest_index(state_real[dim],tiling[dim][tile])
            coordinate.append(index)
        
        pos=0
        i=0
        for c in coordinate:
            pos+=c*pow(tile_per_dim[dim],i)
            i+=1    
        
        indTheta.append(pos+current_pos)
        current_pos+=np.prod(tile_per_dim)
    return np.array(indTheta)

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
    nb_of_tiling,tile_per_dim=np.shape(tiling[0])
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
        indTheta=real_to_tiling(state,tiling,[tile_per_dim,tile_per_dim],nb_of_tiling)
        #=======================================================================
        # print(state)
        # print(tiling)
        # print('N_lintiles',tile_per_dim)
        # print('N_tilings',nb_of_tiling)
        # print(indTheta)
        # exit(0)
        #=======================================================================
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

def RL_SARSA(params,TO=False):
    """
    functional approximation RL algorithm:
    TO: True <--> use True-Online Learning
    
    """
    
    nb_episode=params['nb_episode']
    max_t_steps=params['max_t_steps']
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
    
    
    
    if 'Theta' in params:
        Theta=params['Theta']
        tiling=params['tiling']
    else:        
        tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax],[vmin,vmax]])
        Theta=np.zeros((N_lintiles*N_lintiles*N_tilings,N_actions),dtype=np.float32)    

    for Ep in range(nb_episode):
        trace=np.zeros(Theta.shape,dtype=np.float32)
        current_state_real=(random.uniform(-1.2,0.5),random.uniform(-0.07,0.07))

        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        
        action=np.random.randint(0,N_actions)
        
        t_step=0
        while True:            

            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action)
            
            Q=np.sum(Theta[indTheta,action])
            # define delta
            delta=R-Q

            trace[indTheta,action]=1.0 # use replacing traces ! 
            # Terminate here
            if terminate:
                Theta+=alpha*delta*trace
                break
            
            # compute feature indices
            indTheta_new=real_to_tiling(new_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            
            if random.uniform() < eps:
            	#Explore
            	new_action=random.randint(0,N_actions)
            else: 
            	#Greedy
            	new_action=np.argmax(np.sum(Theta[indTheta_new,:],axis=0))
            
            Q_new_action=np.sum(Theta[indTheta_new,new_action])
            delta+=gamma*Q_new_action
	        #print(alpha,delta,trace)
            Theta+=alpha*delta*trace
            trace*=gamma*lmbda
            
            current_state_real=new_state_real   
            indTheta=indTheta_new
            action=new_action
            
            t_step+=1
    
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",t_step)
    return Theta,tiling


def RL_QL(params,TO=False):
    """
    functional approximation RL algorithm:
    SARSA: False <--> use Q-Learning
    TO: True <--> use True-Online Learning
    """
    
    nb_episode=params['nb_episode']
    max_t_steps=params['max_t_steps']
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
    Theta=np.zeros((N_lintiles*N_lintiles*N_tilings,N_actions),dtype=np.float32)    

    for Ep in range(nb_episode):
        trace=np.zeros(Theta.shape,dtype=np.float32)
        current_state_real=(random.uniform(-1.2,0.5),random.uniform(-0.07,0.07))
    
        t_step=0
        while True:    
        	indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)

        	Q=np.sum(Theta[indTheta,:],axis=0)
        	actions_star=np.argmax(Q)

        	if random.uniform() < eps:
        		action = random.randint(0,N_actions)
        	else:
        		action = actions_star

        	if action==actions_star:
        		trace*=0.0
        	# Take action
        	new_state_real,terminate,R=update_state(current_state_real,action)

        	# define delta
        	delta=R-Q[action]
        	trace[indTheta,action]=1.0 # use replacing traces !
        	# Terminate here

        	if terminate:
        		Theta+=alpha*delta*trace
        		break
        	# compute feature indices
        	indTheta_new=real_to_tiling(new_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        	Q_new=np.sum(Theta[indTheta_new,:],axis=0)
        	delta+=gamma*max(Q_new)
        	#print(alpha,delta,trace)
        	Theta+=alpha*delta*trace
        	trace*=gamma*lmbda

        	current_state_real=new_state_real
        	t_step+=1
    
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",t_step)
    return Theta,tiling


def compute_mean_Q(Theta,tiling,rangex,rangev):
    '''
    
    Computes mean value of Q(s,a_max) over a grid
    range is given in the following format [x_min,x_max,dx]

    '''
    
    nb_of_tiling,tile_per_dim=np.shape(tiling[0])
    mean=0.
    n=0

    for x in np.arange(rangex[0],rangex[1],rangex[2]):
        for v in np.arange(rangev[0],rangev[1],rangev[2]):
            indT=real_to_tiling((x,v), tiling, tile_per_dim, nb_of_tiling)
            mean+=np.max(Theta[indT,:])
            n+=1            
    
    return mean/n

def compute_performance_metrics(_params,method="RL_SARSA",file_save_pkl=None):
    '''
    
    Performance evaluation

    '''
    
    state_i_test=[(-0.9,0.01),(-0.5,0.0),(0.3,0.01),(-0.5,0.06),(-1.15,0.01)]
    N_sample=10
    
    #params=_params
    
    mean_reward=np.zeros(20)
    mean_Q=np.zeros(20)
    
    for n in range(N_sample):
        print('Sample number %d'%n)
        pos=0
        params={}
        params.update(_params)
        
        for nb_episode in np.full(20,10,dtype=np.int):
            params['nb_episode']=nb_episode
            params['Theta'],params['tiling']=eval("RL_SARSA")(params)
            for si in state_i_test:
                trajectory=compute_trajectory(si,params['Theta'],params['tiling'])
                mean_reward[pos]+=len(trajectory)
            pos+=1   
            mean_Q[pos]+=compute_mean_Q(params['Theta'],params['tiling'],[-1.2,0.5,0.02],[-0.07,0.07,0.01])
    
    metric_info=[mean_reward/(N_sample*len(state_i_test)),mean_Q]
    
    if file_save_pkl is not None:
        pkl_file=open(file_save_pkl,'wb')    
        pickle.dump(metric_info,pkl_file)
            
    return metric_info



