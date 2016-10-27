import numpy as np
import numpy.random as random
import pickle
random.seed(1)

def allmax(iterable, key=None):
    "Return a list of all items equal to the max of the iterable."
    result, maxval = [], None
    key = key or (lambda x: x)
    for x in iterable:
        xval = key(x)
        if not result or xval > maxval:
            result, maxval = [x], xval
        elif xval == maxval:
            result.append(x)
    return result

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


def update_state(current_state,action,old_v=None,max_t_steps=None):
    
    terminate=False


    # compute reward
    R = -1.0

    if max_t_steps is None:
        old_pos,old_v=current_state
    else:
        old_pos,old_time=current_state
    

    # Updating velocity:
    new_v=old_v+0.005*(action-1)-0.0125*np.cos(3.*old_pos)
    
    # Maximum velocity ! (presence of friction)
    if new_v < vmin:
        new_v=vmin
    elif new_v > vmax:
        new_v=vmax
    
    # Updating position:
    new_pos=old_pos+new_v
    
    if new_pos < xmin:
        new_pos=xmin
        new_v=0.0
    elif new_pos > xmax:
        terminate=True
   
    if max_t_steps is None:
        return (new_pos,new_v),terminate,R
    else:
        # update reward
        '''
        if old_time+1==max_t_steps:
            R += -max_t_steps*(xmax - new_pos)
            #print 'here', R
        '''
        return (new_pos,old_time+1),terminate,R,new_v


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


def RL_SARSA(params):
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
    
    alpha/=N_tilings
    
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

def RL_Q(params):
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
    N_vars=2 #params['N_vars']
    N_lintiles=params['N_lintiles']
    N_tilings=params['N_tilings']
    N_actions=params['N_actions']
    state_i=params['state_i']
    action_set=params['action_set']
    
    alpha/=N_tilings
    
    if 'Theta' in params:
        Theta=params['Theta']
        tiling=params['tiling']
    else:        
        tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax],[vmin,vmax]])
        Theta=np.zeros((N_lintiles*N_lintiles*N_tilings,N_actions),dtype=np.float32)    

    trace=np.zeros(Theta.shape,dtype=np.float32)
    for Ep in range(nb_episode):
        trace*=0.0
        #current_state_real=(random.uniform(xmin,xmax),random.uniform(vmin,vmax))
        current_state_real=state_i 
        t_step=0
        old_v=0
        #E=0.0
        actions_taken=[]
        while True:

            indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,:],axis=0) )

            #print allmax(Q)             
            action_star=Q.index(random.choice(allmax(Q)))

            if t_step==0:
                print "init Q:",list( np.sum(Theta[real_to_tiling(state_i,tiling,[N_lintiles,N_lintiles],N_tilings),:],axis=0) )

            if random.uniform() < eps:
                action = random.choice( list(set(range(N_actions)) - set([action_star])  ) )
            else:
                action = action_star

            if action!=action_star:
                trace*=0.0
            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action)
            
            #print np.around(current_state_real,4), action, np.around(new_state_real,4)
            
            # define delta
            delta=R-Q[action]
            
            trace[indTheta,action] = alpha*1.0 #( - gamma*lmbda*E) 

            # Terminate here
            if terminate:
                Theta+=delta*trace
                break
            # compute feature indices
            #print new_state_real
            indTheta=real_to_tiling(new_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,:],axis=0) )

            delta+=gamma*max(Q)
            Theta+=delta*trace

            trace*=gamma*lmbda
            
            current_state_real=new_state_real
            t_step+=1

            actions_taken.append(action)
        #exit()
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",t_step)

    #"""
    print 'deltas', delta
    print "reward is", R*t_step
    current_state_real = state_i
    t_step=0
    terminate=False
    while not terminate:
        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,:],axis=0) )
        a = Q.index(random.choice(allmax(Q)))
        print t_step, np.round( np.sum(Theta[indTheta,:],axis=0), 3), [current_state_real[0]], a
        current_state_real,terminate,_=update_state(current_state_real,a)
        t_step+=1
    print '_____________'
    #"""
    return Theta,tiling


def RL_Q_U(params):
    """
    functional approximation RL algorithm:
    TO: True <--> use True-Online Learning
    
    """
    
    nb_episode=params['nb_episode']
    max_t_steps=params['max_t_steps']
    alpha_0=params['alpha_0']
    eta=params['eta']
    eps=params['eps']
    gamma=params['gamma']
    lmbda=params['lmbda']
    global xmin,xmax,vmin,vmax,N_actions
    xmin=params['xmin']
    xmax=params['xmax']
    vmin=params['vmin']
    vmax=params['vmax']
    N_vars=2 #params['N_vars']
    N_lintiles=params['N_lintiles']
    N_tilings=params['N_tilings']
    N_actions=params['N_actions']
    state_i=params['state_i']
    action_set=params['action_set']
    
    N_tiles = N_lintiles**N_vars
    
    
    if 'Theta' in params: 
        Theta=params['Theta']
        tiling=params['tiling']
        u0 = 1.0/np.inf*np.ones((N_tiles*N_tilings,), dtype=np.float64)
    else:        
        tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax],[vmin,vmax]])
        Theta=np.zeros((N_tiles*N_tilings,N_actions),dtype=np.float32)
        u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,), dtype=np.float64)    

    trace=np.zeros(Theta.shape,dtype=np.float32)
    for Ep in range(nb_episode):
        # nullify traces
        trace*=0.0
        # set initial usage vector
        u=u0.copy()



        #current_state_real=(random.uniform(xmin,xmax),random.uniform(vmin,vmax))
        current_state_real=state_i 
        t_step=0
        old_v=0
        #E=0.0
        actions_taken=[]
        while True:

            indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,:],axis=0) )

            #print allmax(Q)             
            action_star=Q.index(random.choice(allmax(Q)))

            if t_step==0:
                print "init Q:",list( np.sum(Theta[real_to_tiling(state_i,tiling,[N_lintiles,N_lintiles],N_tilings),:],axis=0) )

            if random.uniform() < eps:
                action = random.choice( list(set(range(N_actions)) - set([action_star])  ) )
            else:
                action = action_star

            if action!=action_star:
                trace*=0.0
            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action)


            # calculate usage and alpha vectors
            u[indTheta] *= (1.0-eta)
            u[indTheta] += 1.0 
            
            with np.errstate(divide='ignore'):
                # calculate alpha only at the feature indices (rest not needed)
                alpha = 1.0/(N_tilings*u[indTheta])
                alpha[u[indTheta]<1E-12] = 1.0
            
            #print np.around(current_state_real,4), action, np.around(new_state_real,4)
            
            # define delta
            delta=R-Q[action]
            

            trace[indTheta,action] = alpha*1.0
    
            

            # Terminate here
            if terminate:
                Theta+=delta*trace
                break
            # compute feature indices
            #print new_state_real
            indTheta=real_to_tiling(new_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,:],axis=0) )

            delta+=gamma*max(Q)
            Theta+=delta*trace

            trace*=gamma*lmbda
            
            current_state_real=new_state_real
            t_step+=1

            actions_taken.append(action)
        #exit()
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",t_step)

    #"""
    print 'deltas', delta
    print "reward is", R*t_step
    current_state_real = state_i
    t_step=0
    terminate=False
    while not terminate:
        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,:],axis=0) )
        a = Q.index(random.choice(allmax(Q)))
        print t_step, np.round( np.sum(Theta[indTheta,:],axis=0), 3), [current_state_real[0]], a
        current_state_real,terminate,_=update_state(current_state_real,a)
        t_step+=1
    print '_____________'
    #"""
    return Theta,tiling


def RL_Q_TO(params):
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
    N_vars=2 #params['N_vars']
    N_lintiles=params['N_lintiles']
    N_tilings=params['N_tilings']
    N_actions=params['N_actions']
    state_i=params['state_i']
    action_set=params['action_set']
    
    alpha/=N_tilings
    
    if 'Theta' in params:
        Theta=params['Theta']
        tiling=params['tiling']
    else:        
        tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax],[vmin,vmax]])
        Theta=np.zeros((N_lintiles*N_lintiles*N_tilings,N_actions),dtype=np.float32)    

    trace=np.zeros(Theta.shape,dtype=np.float32)
    for Ep in range(nb_episode):
        trace*=0.0
        #current_state_real=(random.uniform(xmin,xmax),random.uniform(vmin,vmax))
        current_state_real=state_i 
        t_step=0
        old_v=0

        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,:],axis=0) )

        actions_taken=[]
        while True:

            #print allmax(Q)             
            action_star=Q.index(random.choice(allmax(Q)))

            if t_step==0:
                print "init Q:",list( np.sum(Theta[real_to_tiling(state_i,tiling,[N_lintiles,N_lintiles],N_tilings),:],axis=0) )

            if random.uniform() < eps:
                action = random.choice( list(set(range(N_actions)) - set([action_star])  ) )
            else:
                action = action_star

            if action!=action_star:
                trace*=0.0
            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action)
            
            #print np.around(current_state_real,4), action, np.around(new_state_real,4)
            
            # define delta
            delta=R-Q[action]
            delta_TO = Q[action] - np.sum(Theta[indTheta,action],axis=0)
            
            trace[indTheta,action] = alpha*1.0
    
            
            Theta[indTheta,action]+=alpha*delta_TO
            # Terminate here
            if terminate:
                Theta+=delta*trace       
                break
            # compute feature indices
            indTheta=real_to_tiling(new_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,:],axis=0) )

            delta+=gamma*max(Q)
            Theta+=delta*trace

            trace[indTheta,action]-=alpha*np.sum(trace[indTheta,action])
            trace*=gamma*lmbda
            
            current_state_real=new_state_real
            t_step+=1

            print 'deltas', delta, delta_TO

            actions_taken.append(action)
        #exit()
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",t_step)

    #"""
    print 'deltas', delta
    print "reward is", R*t_step
    current_state_real = state_i
    t_step=0
    terminate=False
    while not terminate:
        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,:],axis=0) )
        a = Q.index(random.choice(allmax(Q)))
        print t_step, np.round( np.sum(Theta[indTheta,:],axis=0), 3), [current_state_real[0]], a
        current_state_real,terminate,_=update_state(current_state_real,a)
        t_step+=1
    print '_____________'
    #"""
    return Theta,tiling


def RL_Q_U_TO(params):
    """
    functional approximation RL algorithm:
    TO: True <--> use True-Online Learning
    
    """
    
    nb_episode=params['nb_episode']
    max_t_steps=params['max_t_steps']
    alpha_0=params['alpha_0']
    eta=params['eta']
    eps=params['eps']
    gamma=params['gamma']
    lmbda=params['lmbda']
    global xmin,xmax,vmin,vmax,N_actions
    xmin=params['xmin']
    xmax=params['xmax']
    vmin=params['vmin']
    vmax=params['vmax']
    N_vars=2 #params['N_vars']
    N_lintiles=params['N_lintiles']
    N_tilings=params['N_tilings']
    N_actions=params['N_actions']
    state_i=params['state_i']
    action_set=params['action_set']
    
    N_tiles = N_lintiles**N_vars
    
    
    if 'Theta' in params: 
        Theta=params['Theta']
        tiling=params['tiling']
        u0 = 1.0/np.inf*np.ones((N_tiles*N_tilings,), dtype=np.float64)
    else:        
        tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax],[vmin,vmax]])
        Theta=np.zeros((N_tiles*N_tilings,N_actions),dtype=np.float32)
        u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,), dtype=np.float64)    

    trace=np.zeros(Theta.shape,dtype=np.float32)
    for Ep in range(nb_episode):
        # nullify traces
        trace*=0.0
        # set initial usage vector
        u=u0.copy()

        #current_state_real=(random.uniform(xmin,xmax),random.uniform(vmin,vmax))
        current_state_real=state_i 

        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,:],axis=0) )


        t_step=0
        old_v=0
        #E=0.0
        actions_taken=[]
        while True:

            #print allmax(Q)             
            action_star=Q.index(random.choice(allmax(Q)))

            if t_step==0:
                print "init Q:",list( np.sum(Theta[real_to_tiling(state_i,tiling,[N_lintiles,N_lintiles],N_tilings),:],axis=0) )

            if random.uniform() < eps:
                action = random.choice( list(set(range(N_actions)) - set([action_star])  ) )
            else:
                action = action_star

            if action!=action_star:
                trace*=0.0
            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action)


            # calculate usage and alpha vectors
            u[indTheta] *= (1.0-eta)
            u[indTheta] += 1.0 
            
            with np.errstate(divide='ignore'):
                # calculate alpha only at the feature indices (rest not needed)
                alpha = 1.0/(N_tilings*u[indTheta])
                alpha[u[indTheta]<1E-12] = 1.0
            
            #print np.around(current_state_real,4), action, np.around(new_state_real,4)
            
            # define delta
            delta=R-Q[action]
            delta_TO = Q[action] - np.sum(Theta[indTheta,action],axis=0)
            
            # update traces
            trace[indTheta,action] = alpha*1.0
            
            # do delta_TO update
            Theta[indTheta,action]+=alpha*delta_TO

            # Terminate here
            if terminate:
                Theta+=delta*trace
                break
            # compute feature indices
            indTheta=real_to_tiling(new_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,:],axis=0) )
            # update deta
            delta+=gamma*max(Q)
            # update theta
            Theta+=delta*trace
            # decay traces
            trace[indTheta,action]-=alpha*np.sum(trace[indTheta,action])
            trace*=gamma*lmbda
            
            # update episode variables
            current_state_real=new_state_real
            t_step+=1

            # store action taken
            actions_taken.append(action)
        #exit()
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",t_step)

    #"""
    print 'deltas', delta
    print "reward is", R*t_step
    current_state_real = state_i
    t_step=0
    terminate=False
    while not terminate:
        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,:],axis=0) )
        a = Q.index(random.choice(allmax(Q)))
        print t_step, np.round( np.sum(Theta[indTheta,:],axis=0), 3), [current_state_real[0]], a
        current_state_real,terminate,_=update_state(current_state_real,a)
        t_step+=1
    print '_____________'
    #"""
    return Theta,tiling


def RL_Q_WIS_TO_GTD(params):
    """
    functional approximation RL algorithm:
    TO: True <--> use True-Online Learning
    
    """
    
    nb_episode=params['nb_episode']
    max_t_steps=params['max_t_steps']
    alpha_0=params['alpha_0']
    beta=params['beta']
    eta=params['eta']
    eps=params['eps']
    gamma=params['gamma']
    lmbda=params['lmbda']
    global xmin,xmax,vmin,vmax,N_actions
    xmin=params['xmin']
    xmax=params['xmax']
    vmin=params['vmin']
    vmax=params['vmax']
    N_vars=2 #params['N_vars']
    N_lintiles=params['N_lintiles']
    N_tilings=params['N_tilings']
    N_actions=params['N_actions']
    state_i=params['state_i']
    action_set=params['action_set']
    
    N_tiles = N_lintiles**N_vars
    beta/=N_lintiles
    
    
    if 'Theta' in params: 
        Theta=params['Theta']
        tiling=params['tiling']
        u0 = 1.0/np.inf*np.ones((N_tiles*N_tilings,), dtype=np.float64)
    else:        
        tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax],[vmin,vmax]])
        Theta=np.zeros((N_tiles*N_tilings,N_actions),dtype=np.float64)
        u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,), dtype=np.float64)    
    v0 = np.zeros(u0.shape, dtype=np.float64)

    w = np.zeros(Theta.shape,dtype=np.float64)

    trace=np.zeros(Theta.shape,dtype=np.float64)
    trace_t=trace.copy()
    trace_w=trace.copy()


    for Ep in range(nb_episode):
        # nullify traces
        trace*=0.0
        trace_t*=0.0
        trace_w*=0.0
        # set initial usage vector
        u=u0.copy()
        # set initial aux v vector
        v = v0.copy()

        #current_state_real=(random.uniform(xmin,xmax),random.uniform(vmin,vmax))
        current_state_real=state_i 

        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,:],axis=0) )


        t_step=0
        old_v=0
        rho_prime=0.0
        #E=0.0
        actions_taken=[]
        Return=0.0
        if Ep==0:
            L_ba = max_t_steps
        else:
            L_ba = len(best_actions)
        while t_step<min(max_t_steps-1,L_ba):

            #print allmax(Q)
            # take action acc to best_enc policy
            if Ep<=20:
                action_star=Q.index(random.choice(allmax(Q)))
            else:
                try:
                    action_star = best_actions[t_step]
                except(IndexError):
                    print 'EXITING'
                    


            if t_step==0:
                print "init Q:",list( np.sum(Theta[real_to_tiling(state_i,tiling,[N_lintiles,N_lintiles],N_tilings),:],axis=0) )


            if random.uniform() < eps:
                action = random.choice( list(set(range(N_actions)) - set([action_star])  ) )
            else:
                action = action_star

            # reset traces and calculate probability under beh policy
            if action!=action_star:
                trace*=0.0
                trace_t*=0.0
                trace_w*=0.0
                beh_policy = eps/N_actions
            else:
                beh_policy = eps/N_actions + 1.0-eps

            # calculate probability of taking A under target policy
            if action!=Q.index(random.choice(allmax(Q))):
                tgt_policy = eps/N_actions
            else:
                tgt_policy = eps/N_actions + 1.0-eps

            # calculate importance sampling ratio
            rho = tgt_policy/beh_policy
            #rho = min(1.0,tgt_policy/beh_policy)

            #print rho

            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action)


            # store action taken
            actions_taken.append(action)
            Return+=R
            
            #rho=1.0

            # calculate usage and alpha vectors
            u[indTheta] *= (1.0-eta)
            u += (rho-1.0)*gamma*lmbda*v
            u[indTheta] += rho - (rho-1.0)*gamma*lmbda*eta*v[indTheta]
            # update aux v vector
            v *= gamma*lmbda*rho
            v[indTheta] *= 1.0-eta
            v[indTheta] += rho 
            
            with np.errstate(divide='ignore'):
                # calculate alpha only at the feature indices (rest not needed)
                alpha = 1.0/(N_tilings*u[indTheta])
                alpha[u[indTheta]<1E-12] = 1.0
            

            #print np.around(current_state_real,4), action, np.around(new_state_real,4)
            
            # define delta
            delta=R - Q[action]
            delta_TO = Q[action] - np.sum(Theta[indTheta,action],axis=0)
            #delta_w = R - np.sum(Theta[indTheta,action],axis=0)
            
            # update traces
            trace[indTheta,action]=rho*alpha
            #trace_t[indTheta,action]=rho
            #trace_w[indTheta,action]=beta

            
            # do delta_TO update
            Theta[indTheta,action]+=rho*alpha*delta_TO
            #w[indTheta,action]-=beta*np.sum(w[indTheta,action]) 

            # Terminate here
            if terminate:
                # update theta
                Theta+=delta*trace
                # update w
                #w+=rho*delta_w*trace_w
                break
            # compute feature indices
            indTheta=real_to_tiling(new_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,:],axis=0) )
            # update deta
            delta+=gamma*max(Q)
            #delta_w+=gamma*max(Q)
            # update theta
            Theta+=delta*trace 
            #Theta[indTheta,action]=-alpha*gamma*(1.0-lmbda)*(np.multiply(w,trace_t).sum())
            # update w
            #w+=rho*delta_w*trace_w
            

            # decay traces
            trace[indTheta,action]-=rho*alpha*np.sum(trace_w[indTheta,action])
            trace*=gamma*lmbda*rho

            #trace_t*=gamma*lmbda*rho

            #trace_w[indTheta,action]-=rho_prime*np.sum(trace[indTheta,action])
            #trace_w*=gamma*lmbda*rho_prime

            #print 'deltas', delta, delta_TO
            #print 'traces', max(trace.ravel()), max(trace_t.ravel()), max(trace_w.ravel())
            
            #print max(w.ravel())

            # update episode variables
            current_state_real=new_state_real
            t_step+=1
            #rho_prime=rho

        # if zeroth episode or if improvement encountered, assign encoutered to best
        if Ep==0 or Return>best_Return or (Return==best_Return and len(actions_taken)<len(best_actions) ):
            best_actions=actions_taken
            best_Return=Return

        print '------------'
        print best_Return
        print best_actions
        print '------------'

        #exit()
        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",t_step)

    #"""
    print 'deltas', delta
    print "reward is", R*t_step
    current_state_real = state_i
    t_step=0
    terminate=False
    while not terminate:
        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,:],axis=0) )
        a = Q.index(random.choice(allmax(Q)))
        print t_step, np.round( np.sum(Theta[indTheta,:],axis=0), 3), [current_state_real[0]], a
        current_state_real,terminate,_=update_state(current_state_real,a)
        t_step+=1
    print '_____________'
    #"""
    return Theta,tiling


def RL_ABQ(params):
    """
    functional approximation RL algorithm:
    TO: True <--> use True-Online Learning
    
    """
    
    nb_episode=params['nb_episode']
    max_t_steps=params['max_t_steps']
    alpha=params['alpha']
    beta=params['beta']
    zeta=params['zeta']
    eps=params['eps']
    gamma=params['gamma']
    lmbda=params['lmbda']
    global xmin,xmax,vmin,vmax,N_actions
    xmin=params['xmin']
    xmax=params['xmax']
    vmin=params['vmin']
    vmax=params['vmax']
    N_vars=2 #params['N_vars']
    N_lintiles=params['N_lintiles']
    N_tilings=params['N_tilings']
    N_actions=params['N_actions']
    state_i=params['state_i']
    action_set=params['action_set']
    
    alpha/=N_tilings
    beta/=N_tilings
    
    if 'Theta' in params:
        Theta=params['Theta']
        tiling=params['tiling']
    else:        
        tiling=construct_tiling(N_vars,[N_lintiles,N_lintiles],N_tilings,[[xmin,xmax],[vmin,vmax]])
        Theta=np.zeros((N_lintiles*N_lintiles*N_tilings,N_actions),dtype=np.float32)    

    trace=np.zeros(Theta.shape,dtype=np.float32)
    h=np.zeros(Theta.shape,dtype=np.float32)


    best_actions=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

    for Ep in range(nb_episode):
        trace*=0.0
        #current_state_real=(random.uniform(xmin,xmax),random.uniform(vmin,vmax))
        current_state_real=state_i 

        print "init Q:",list( np.sum(Theta[real_to_tiling(state_i,tiling,[N_lintiles,N_lintiles],N_tilings),:],axis=0) )


        t_step=0
        old_v=0
        #E=0.0
        actions_taken=[]
        while True:
            
            indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,:],axis=0) )

            #print allmax(Q)
            action_star=Q.index(random.choice(allmax(Q)))
            """
            try:             
                action_star= best_actions[t_step]
            except(IndexError):
                action_star=Q.index(random.choice(allmax(Q)))
            """    

            if random.uniform() < eps:
                action = random.choice( list(set(range(N_actions)) - set([action_star])  ) )
            else:
                action = action_star

            # reset traces and calculate probability under beh policy
            if action!=action_star:
                #trace*=0.0
                beh_policy = eps/N_actions
            else:
                beh_policy = eps/N_actions + 1.0-eps

            # calculate probability of taking A under target policy
            if action!=Q.index(random.choice(allmax(Q))):
                tgt_policy = eps/N_actions
            else:
                tgt_policy = eps/N_actions + 1.0-eps  

            nu = min(zeta, 1.0/max(tgt_policy,beh_policy))


            # Take action
            new_state_real,terminate,R=update_state(current_state_real,action)
            
            #print np.around(current_state_real,4), action, np.around(new_state_real,4)
            
            # define delta
            delta=R-Q[action]
            #print 'predelta', delta
            
            trace[indTheta,action] = 1.0 

            # Terminate here
            if terminate or t_step>2E3:
                Theta+=alpha*delta*trace
                h+=beta*delta*trace
                h[indTheta,action]-=beta*np.sum(h[indTheta,action])
                break

            he = np.multiply(h,trace).sum()
            h[indTheta,action]-=beta*np.sum(h[indTheta,action])

            # compute feature indices
            #print new_state_real
            indTheta=real_to_tiling(new_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,:],axis=0) )


            #print allmax(Q)
            action_star_prime=Q.index(random.choice(allmax(Q)))
            """             
            try:             
                action_star_prime= best_actions[t_step+1]
            except(IndexError):
                action_star_prime=Q.index(random.choice(allmax(Q))) 
            """

            
            if random.uniform() < eps:
                action_prime = random.choice( list(set(range(N_actions)) - set([action_star_prime])  ) )
            else:
                action_prime = action_star_prime

            # reset traces and calculate probability under beh policy
            if action_prime!=action_star_prime:
                beh_policy_prime = eps/N_actions
            else:
                beh_policy_prime= eps/N_actions + 1.0-eps

            # calculate probability of taking A under target policy
            if action_prime!=Q.index(random.choice(allmax(Q))):
                tgt_policy_prime = eps/N_actions
            else:
                tgt_policy_prime = eps/N_actions + 1.0-eps

            
            delta+=gamma*(eps/N_actions*sum(Q) + (1.0-eps)*Q[action])
            Theta+=alpha*delta*trace

            h+=beta*delta*trace
            trace*=gamma*nu*tgt_policy


            nu_prime = min(zeta, 1.0/max(tgt_policy_prime,beh_policy_prime))

            
            Theta[indTheta,:]-=alpha*gamma*(1.0 - nu_prime*beh_policy_prime)*he*eps/N_actions
            Theta[indTheta,action]-=alpha*gamma*(1.0 - nu_prime*beh_policy_prime)*he*(1.0-eps)

            #print 'traces', max(trace.ravel())
            print 'delta', delta
            
            current_state_real=new_state_real
            t_step+=1

            actions_taken.append(action)
        #print 'steps:',t_step, (eps/N_actions*sum(Q) + (1.0-eps)*Q[action]), Q[action]

        if Ep%10 ==0 :
            print("Episode:",Ep," Length:",t_step)
            #exit()

    #"""
    print 'deltas', delta
    print "reward is", R*t_step
    current_state_real = state_i
    t_step=0
    terminate=False
    while not terminate:
        indTheta=real_to_tiling(current_state_real,tiling,[N_lintiles,N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,:],axis=0) )
        a = Q.index(random.choice(allmax(Q)))
        print t_step, np.round( np.sum(Theta[indTheta,:],axis=0), 3), [current_state_real[0]], a
        current_state_real,terminate,_=update_state(current_state_real,a)
        t_step+=1
    print '_____________'
    #"""
    return Theta,tiling

###############

def RL_Q_time(params,TO=False):
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

    alpha/=N_tilings
    

    tiling=construct_tiling(N_vars,[N_lintiles],N_tilings,[[xmin,xmax]])
    Theta=np.zeros((N_lintiles*N_tilings,max_t_steps,N_actions),dtype=np.float32)    
    trace=np.zeros(Theta.shape,dtype=np.float32)

    for Ep in range(nb_episode):
        #trace*=trace
        #current_state_real=(random.uniform(xmin,xmax), 0.0)    
        current_state_real=state_i 
    
        indTheta=real_to_tiling([current_state_real[0]],tiling,[N_lintiles],N_tilings)
        Q=list( np.sum(Theta[indTheta,0,:],axis=0) )


        t_step=0
        old_v=0
        #E=0.0
        actions_taken=[]
        while True:
 
            action_star=Q.index(random.choice(allmax(Q)))

            if t_step==0:
                print "init Q:",np.sum(Theta[real_to_tiling([state_i[0]],tiling,[N_lintiles],N_tilings),0,:],axis=0), np.around(eps/np.sqrt(Ep+1.0),4)

            if random.uniform() < eps/np.sqrt(Ep+1.0):
                action = random.choice( list(set(range(N_actions)) - set([action_star])  ) )
            else:
                action = action_star

            if action!=action_star:
                trace*=0.0
            # Take action
            new_state_real,terminate,R,old_v=update_state(current_state_real,action,old_v=old_v,max_t_steps=max_t_steps)

            #print current_state_real, action, new_state_real

            # define delta
            delta=R-Q[action]
            #delta_TO = Q[action] - np.sum(Theta[indTheta,t_step,action],axis=0)
            #Theta[indTheta,t_step,action] += alpha*delta_TO
            
            trace[indTheta,t_step,action] = alpha*1.0 #( - gamma*lmbda*E) 
    
            

            # Terminate here
            if terminate or t_step==max_t_steps-1:
                Theta+=delta*trace
                break
            # compute feature indices
            indTheta=real_to_tiling([new_state_real[0]],tiling,[N_lintiles],N_tilings)
            Q=list( np.sum(Theta[indTheta,t_step+1,:],axis=0) )

            delta+=gamma*max(Q)
            Theta+=delta*trace

            trace*=gamma*lmbda
            #E = np.sum(trace[indTheta,t_step,action],axis=0)

            #print 'deltas', delta, delta_TO

            #print '___________'
            #print np.round( np.sum(Theta[indTheta,t_step,:],axis=0), 3), current_state_real, action
            #print '______________'

            #old_v = new_state_real[0]-current_state_real[0]

            current_state_real=new_state_real
            t_step+=1

            actions_taken.append(action)
    
        print("Episode:",Ep," Length:",t_step, "pos:", [state_i[0], current_state_real[0]])
    #"""
    print 'deltas', delta
    print "reward is", R*t_step
    current_state_real = state_i
    t_step=0
    old_v=0.0
    for a in actions_taken:
        indTheta=real_to_tiling([current_state_real[0]],tiling,[N_lintiles],N_tilings)
        print t_step, np.round( np.sum(Theta[indTheta,t_step,:],axis=0), 3), [current_state_real[0]], a
        current_state_real,_,_,old_v=update_state(current_state_real,a,old_v=old_v,max_t_steps=max_t_steps)
        t_step+=1
    print '_____________'
    #"""
    return Theta,tiling

