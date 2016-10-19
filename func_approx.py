import numpy as np
import numpy.random as random
random.seed(0)

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


def update_state_time(current_state,action,old_v):
    
    terminate=False
    old_pos,old_time=current_state
    
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
    
    return (new_pos,old_time+1),terminate,R

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
        indTheta=real_to_tiling(state,tiling,[tile_per_dim,tile_per_dim],nb_of_tiling)
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


def RL_QL_time(params,TO=False):
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
    N_vars=1 #params['N_vars']
    N_lintiles=params['N_lintiles']
    N_tilings=params['N_tilings']
    N_actions=params['N_actions']
    state_i=params['state_i']
    action_set=params['action_set']
    

    tiling=construct_tiling(N_vars,[N_lintiles],N_tilings,[[xmin,xmax]])
    Theta=np.zeros((N_lintiles*N_tilings,max_t_steps,N_actions),dtype=np.float32)    

    for Ep in range(nb_episode):
        trace=np.zeros(Theta.shape,dtype=np.float32)
        current_state_real=state_i

        indTheta=real_to_tiling([current_state_real[0]],tiling,[N_lintiles],N_tilings)
       
        if random.uniform() < eps:
    		action = random.randint(0,N_actions)
    	else:
    		action = np.argmax( np.sum(Theta[indTheta,0,:],axis=0) )

        Q_old=[0.0,0.0,0.0]
    
        t_step=0
        v_old=0.0
        #print np.sum(Theta[real_to_tiling([current_state_real[0]],tiling,[N_lintiles],N_tilings),0,:],axis=0)
        while t_step<max_t_steps or terminate:    
        	
        	# Take action
        	new_state_real,terminate,R=update_state_time(current_state_real,action,v_old)
        	indTheta_new=real_to_tiling([new_state_real[0]],tiling,[N_lintiles],N_tilings)

        	Q = np.sum(Theta[indTheta,t_step,:],axis=0)
        	new_Q = np.sum(Theta[indTheta_new,t_step,:],axis=0)

        	if random.uniform() < eps:
	    		new_action = random.randint(0,N_actions)
	    	else:
	    		new_action = np.argmax( np.sum(Theta[indTheta_new,t_step,:],axis=0) )
        	
        	action_star = np.argmax( np.sum(Theta[indTheta_new,t_step,:],axis=0) )

        	if new_Q[new_action]==new_Q[action_star]:
        		action_star=new_action

        	if terminate:
        		new_Q = [0.0,0.0,0.0]

        	delta = R + gamma*new_Q[action_star] - Q[action]

        	trace[indTheta,t_step,action]-=alpha*np.sum(trace[indTheta,t_step,action])
        	trace*=gamma*lmbda
        	trace+=1.0

        	print action, new_action, action_star
        	print new_Q[action_star], Q[action]
        	print delta

        	Theta += alpha*(delta + Q[action] - Q_old[action])*trace
        	Theta[indTheta,t_step,action] -= alpha*(Q[action]-Q_old[action])

        	if new_action != action_star:
        		trace*=0.0

        	
        	v_old = new_state_real[0]-current_state_real[0]
        	Q_old = new_Q
        	indTheta = indTheta_new
        	action = new_action
        	current_state_real = new_state_real


        	t_step+=1
    	#exit()
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",t_step, " final position:", current_state_real[0])
    return Theta,tiling
