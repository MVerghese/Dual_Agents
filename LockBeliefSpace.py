import numpy as np
import time 
from matplotlib import pyplot as plt
import warnings

gamma = .99
np.set_printoptions(suppress=True)
# np.seterr(all='raise')

# C1: Handle Lock
# C2: Handle
# C3: Deadbolt
# C4: Privacy
# C5: Door

def generate_lock_sequence(num_components,num_states, state_config = None, prior_config = None,entropy = 0):
	if np.any(prior_config == None):
		if np.any(state_config == None):
			state_config = np.zeros(num_components-1,dtype=int)*(num_states-1)
		# print(state_config)
		config = np.zeros((num_components ,num_states, num_components))
		order = np.arange(0,num_components-1,1)
		np.random.shuffle(order)
		print(order)
		# config[order[0],num_components] = 1
		for i in range(1,order.shape[0]):
			config[order[i-1],state_config[i],order[i]] = 1.
		config[order[-1],state_config[-1],num_components-1] = 1.
		# config[num_components-1,num_components] = 1
	else:
		config = np.zeros((num_components,num_components+1))



	return(config)

def generate_door(num_components,num_states, state_config = None, prior_config = None,entropy = 0):
	if np.any(prior_config == None):
		if np.any(state_config == None):
			state_config = np.zeros(num_components,dtype=int)*(num_states-1)
		# print(state_config)
		config = np.zeros((num_components ,num_states, num_components))
		
		order = np.array([1,4,4,4])
		# config[order[0],num_components] = 1
		for i in range(0,order.shape[0]):
			config[i,state_config[i],order[i]] = 1.
		
		# config[num_components-1,num_components] = 1
	else:
		config = np.zeros((num_components,num_components+1))

	return(config)

def compute_entropy(P):
	Ptemp = P[np.where(P>0)]
	return(-1*np.sum(np.multiply(Ptemp,np.log(Ptemp))))



def KL_Divergence(P,Q):
	Ptemp = P[np.where(P>0)]
	Qtemp = Q[np.where(Q>0)]

	# print(Ptemp,np.sum(Ptemp))
	# print(Qtemp,np.sum(Qtemp))


	assert(np.allclose(np.sum(Ptemp),1.))
	assert(np.allclose(np.sum(Qtemp),1.))
	return(np.sum(Ptemp*np.log(Ptemp)-Ptemp*np.log(Qtemp)))

def reg_KL_Divergence(Ptemp,Qtemp):


	assert(np.allclose(np.sum(Ptemp),1.))
	assert(np.allclose(np.sum(Qtemp),1.))
	return(np.sum(Ptemp*np.log(Ptemp)-Ptemp*np.log(Qtemp)))

def compute_ce(P,Q):
	if P>0 and Q>0 and P<1 and Q<1:

		return(-P*np.log(P)+P*np.log(P)-P*np.log(Q)+(1-P)*np.log(1-P)-(1-P)*np.log(1-Q))
	else:
		return(-P*np.log(P))


def JSD(P,Q):
	M = (P+Q)/2
	return(KL_Divergence(P,M)/2+KL_Divergence(Q,M)/2)

def belief_state_KL_Divergence(P,Q,enforce_sum = True):

	Ptemp = P[np.where(P>0)]
	Qtemp = Q[np.where(Q>0)]

	Ptemp = Ptemp[np.where(Ptemp<1)]
	Qtemp = Qtemp[np.where(Qtemp<1)]


	Pcomp = 1-Ptemp
	Qcomp = 1-Qtemp

	
	Ptotal = np.vstack((Ptemp,Pcomp))
	Qtotal = np.vstack((Qtemp,Qcomp))



	# print(Ptotal,np.sum(Ptotal,axis=0))
	# print(Qtotal,np.sum(Qtotal,axis=0))

	if enforce_sum:
		assert(np.allclose(np.sum(Ptotal,axis=0),1.))
		assert(np.allclose(np.sum(Qtotal,axis=0),1.))
	return(np.sum(Ptotal*np.log(Ptotal)-Ptotal*np.log(Qtotal)))


def bayesian_inference(ph,peh,penoth):
	try:
		out = 1/(1+(1/ph - 1)*penoth/peh)
	except FloatingPointError:
		out = 0
		print(ph,peh,penoth)
		1/0
	return(out)

def p_or(X,Y):
	assert(X <= 1)
	assert(Y <= 1)
	return(1-((1-X)*(1-Y)))

def normalize_belief_state(belief_state):
	for b in range(belief_state.shape[2]-1):
		z = 1 - np.prod(1-belief_state[:-1,:,b]) + belief_state[-1,0,b]
		belief_state[-1,0,b] /= z
		# print((1 - np.prod(1-belief_state[:-1,:,b]))/z+belief_state[-1,0,b])
		z1 = np.prod(1-belief_state[:-1,:,b])/(1 - 1/z + np.prod(1-belief_state[:-1,:,b])/z)
		# print('z1',z1)
		# print(1 - np.prod(1-belief_state[:-1,:,b])/z1+belief_state[-1,0,b])
		# print(1-belief_state[:-1,:,b])
		z2 = z1**(1/(belief_state.shape[0]-2)/2)
		# print('z2',z2)
		# print(1-np.prod((1-belief_state[:-1,:,b])/z2) + belief_state[-1,0,b])
		belief_state[:-1,:,b] = 1 - 1/z2 + belief_state[:-1,:,b]/z2
		belief_state[b,:,b] = 0.
		if np.any(belief_state[:-1,:,b] < 0):
			belief_state[:-1,:,b] = 0.
			belief_state[-1,:,b] = 1.
	return(belief_state)

def print_belief_state(belief_state,label=''):
	print(label)
	print(np.around(belief_state[:,0,:],5))
	print(np.around(belief_state[:,1,:],5))



def compute_posterior_divergence(belief_state, current_state, action, hypothesis):
	# print(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action])
	k = current_state[hypothesis[0]]
	ph = belief_state[hypothesis[0],k,hypothesis[1]]
	if ph == 0.:
		return(0.)
	pe = np.prod(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action])

	total_divergence = 0

	total_divergence += compute_ce(ph,0)*pe

	temp_belief_state = belief_state.copy()
	temp_belief_state[hypothesis[0],k,hypothesis[1]] = 0
	penoth = 1 - np.prod(1 - temp_belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action])
	peh = 1
	# print(bayesian_inference(ph,peh,penoth))
	# print(ph)
	# print(pe)
	# print(penoth)
	# 1/0

	total_divergence += (1-pe)*compute_ce(ph,bayesian_inference(ph,peh,penoth))
	# posterior = pe*ph
	# print(pe)
	# sum_prob += .5*bayesian_inference(ph,peh,penoth)


	return(total_divergence)


def compute_posterior_actual(belief_state, current_state, action, hypothesis, success):
	k = current_state[hypothesis[0]]
	ph = belief_state[hypothesis[0],k,hypothesis[1]]
	if success or ph == 0:
		return(0.)
	pe = np.prod(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action])

	temp_belief_state = belief_state.copy()
	temp_belief_state[hypothesis[0],k,hypothesis[1]] = 0
	penoth = 1 - np.prod(1 - temp_belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action])
	peh = 1

	return(bayesian_inference(ph,peh,penoth))


def update_belief_state(belief_state, current_state, action, success):
	new_belief_state = np.copy(belief_state)
	for i in range(belief_state.shape[0]):
		if i != action:

			new_belief_state[i,current_state[i],action] = compute_posterior_actual(belief_state,current_state,action,(i,action),success)
	# for a in range(belief_state.shape[0]):
	# 	new_belief_state[a,:,:] /= np.sum(new_belief_state[a,:,:])
	# print_belief_state(new_belief_state,"Pre Norm")
	# new_belief_state = normalize_belief_state(new_belief_state)
	# print_belief_state(new_belief_state,"Post Norm")
	return(new_belief_state)



def compute_info_gain(belief_state,current_state):
	info_gain = np.zeros(belief_state.shape[2])
	for b in range(belief_state.shape[2]):
		for a in range(belief_state.shape[0]):
			# print(compute_posterior(belief_state,(a,b)))
			if a != b:
				info_gain[b] += compute_posterior_divergence(belief_state, current_state, b, (a,b))

			# expected_posterior[a,:,:] /= np.sum(expected_posterior[a,:,:])
			# print(np.sum(expected_posterior[a,:,:]),expected_posterior[a,0,-1])
		# print('before',1 - np.prod(1-expected_posterior[:-1,:,b]) + expected_posterior[-1,0,b])
		# expected_posterior = normalize_belief_state(expected_posterior)
		# print('after',1 - np.prod(1-expected_posterior[:-1,:,b]) + expected_posterior[-1,0,b])
		# 1/0
		# print_belief_state(belief_state,'belief_state')
		# print_belief_state(expected_posterior,'expected_posterior')
		# print_belief_state(belief_state,'prior')
		# print_belief_state(expected_posterior,'posterior')



		# info_gain[b] = belief_state_KL_Divergence(belief_state,expected_posterior)
			# print(a,b,KL_Divergence(belief_state[a,:,:],expected_posterior[a,:,:]))


	# print(info_gain)
	# 1/0

	return(info_gain)

class LockEnv:
	def __init__(self,config,current_state):
		self.config = config
		self.current_state = current_state

	def reset(self):
		current_state = np.zeros((self.current_state.shape[0]))
		return(False,self.current_state,0)

	def step(self,action):
		# print(self.config)
		success = np.all(self.config[np.arange(0,self.current_state.shape[0]),self.current_state,np.ones(self.current_state.shape[0],dtype=int)*action] != 1)
		if success:
			self.current_state[action] = self.current_state[action]*-1 + 1
		return(success,self.current_state,self.current_state[-1])


def argmax(arr):
	if np.sum(np.max(arr) == arr) <= 1:
		return(np.argmax(arr))
	else:
		idxs = np.where(np.max(arr) == arr)[0]
		# print(idxs)
		idx = np.random.choice(idxs)
		return(idx)

def hash_state(state):
	weights = np.array([2**i for i in range(0,len(state))])
	return(np.dot(state,weights))

def info_max_policy(belief_state,current_state,state_hash,**kwargs):
	info_gain = compute_info_gain(belief_state, current_state)
	# print(info_gain)
	# print(action_prob)
	action = argmax(info_gain)
	# print(action_prob[action])
	return(action,info_gain)

def random_policy(belief_state,current_state,state_hash,**kwargs):
	action_prob = np.random.uniform(0,1,size = current_state.shape[0])
	action = argmax(action_prob)
	return(action,1/np.sum(current_state==0))

def greedy_policy(belief_state,current_state,state_hash,**kwargs):

	lock_probs = np.multiply(belief_state,1-current_state.reshape((current_state.shape[0],1)))
	unlock_probs = 1 - lock_probs
	prob_unlocked =  np.prod(unlock_probs,axis=0)
	action_prob = prob_unlocked+current_state*-100
	action = np.argmax(action_prob)
	return(action,action_prob)

def balanced_policy(belief_state,current_state,state_hash,**kwargs):
	alpha = .5
	if "alpha" in kwargs:
		alpha = kwargs["alpha"]

	info_action,info_action_probs = info_max_policy(belief_state,current_state,**kwargs)
	plan_action,plan_action_probs = greedy_policy(belief_state,current_state,**kwargs)
	action_scores = alpha*info_action_probs + (1-alpha)*plan_action_probs

	action = np.argmax(action_scores)
	return(action,action_scores)

def constant_policy(belief_state,current_state,state_hash,**kwargs):
	return(0,1)

def novelty_policy(belief_state,current_state,state_hash,**kwargs):
	actuation_prob = np.zeros(current_state.shape)
	for action in range(current_state.shape[0]):
		actuation_prob[action] = np.prod(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action])

	action_scores = np.zeros(current_state.shape)
	for action in range(current_state.shape[0]):
		proposed_state = current_state.copy()
		proposed_state[action] = 1 - proposed_state[action]
		action_scores[action] = (1 - state_hash[hash_state(proposed_state)])*actuation_prob[action]
	action = argmax(action_scores)
	return(action,action_scores)

def compute_actuation_prob(belief_state,current_state):
	actuation_prob = np.zeros(current_state.shape)
	for action in range(current_state.shape[0]):
		actuation_prob[action] = np.prod(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action])
	return(actuation_prob)

def exploration_policy(belief_state,current_state,state_hash,**kwargs):
	alpha = .5
	if "alpha" in kwargs:
		alpha = kwargs["alpha"]

	info_action,info_action_scores = info_max_policy(belief_state,current_state,state_hash,**kwargs)
	novl_action,novl_action_scores = novelty_policy(belief_state,current_state,state_hash,**kwargs)

	action_probs = info_action_scores*(1-alpha)+novl_action*(alpha)
	return(argmax(action_probs),action_probs)

class search_node:
	def __init__(self, state, bf, value = 0.):
		self.state = state
		self.connections = [None]*bf
		self.ancestor = None
		self.probabilities = np.zeros(bf)
		self.value = value

class dijkstra_search:
	def __init__(self,start_state,bf,belief_state):
		self.bf = bf
		self.nodes = [None]*2**bf
		start_node = search_node(start_state,bf,value=1.)
		self.nodes[hash_state(start_state)] = start_node
		self.start_state = start_state
		self.open_set = []
		self.open_set.append(start_node)
		self.closed_set = []
		self.belief_state = belief_state

	def expand_node(self,node):
		current_state = node.state
		actuation_probs = compute_actuation_prob(self.belief_state,current_state)
		node.probabilities = actuation_probs
		for action in range(current_state.shape[0]):
			proposed_state = current_state.copy()
			proposed_state[action] = 1 - proposed_state[action]
			if self.nodes[hash_state(proposed_state)]:
				node.connections[action] = self.nodes[hash_state(proposed_state)]
				if node.value*node.probabilities[action] > node.connections[action].value:
					node.connections[action].value = node.value*node.probabilities[action]
					node.connections[action] = ancestor
					if node.connections[action] not in self.open_set:
						self.open_set.append(node.connections[action])
					if node.connections[action] in self.open_set:
						self.closed_set.remove(node.connections[action])

			else:
				new_node = search_node(proposed_state,self.bf,node.value*node.probabilities[action])
				node.connections[action] = new_node
				new_node.ancestor = node
				self.open_set.append(new_node)

		self.open_set.remove(node)
		self.closed_set.append(node)

	def return_best_node(self,nodes):
		best_node = nodes[0]
		for node in nodes:
			if node.value > best_node.value:
				best_node = node
		return(best_node)

	def is_goal_state(self,state,goal_states):
		ret = False
		for goal_state in goal_states:
			if np.all(state == goal_state):
				ret = True
				break
		return(ret)


	def search(self,goal_states):
		goal = None
		counter = 0
		while len(self.open_set) > 0:
			node = self.return_best_node(self.open_set)
			# print(node.value)
			if self.is_goal_state(node.state, goal_states):
				goal = node
				print(goal.value)
				break
			else:
				counter += 1
				self.expand_node(node)
			if counter > 1000:
				for node in self.closed_set:
					print(node.state)
				1/0
		print(counter)

		if goal == None:
			print("search failed")
			return(False,[],[])
		else:
			action_sequence = []
			state_sequence = [goal.state]
			final_cost = goal.value
			cur_node = goal
			while not self.is_goal_state(cur_node.state, [self.start_state]):
				optimal = -1
				for action in range(self.bf):
					if cur_node.ancestor.connections[action] == cur_node:
						optimal = action
				action_sequence = [optimal] + action_sequence
				cur_node = cur_node.ancestor
				state_sequence = [cur_node.state] + state_sequence
			return(True,final_cost, state_sequence,action_sequence)

def compute_q(state,belief_state):
	n = state.shape[0]
	goals = []
	for i in range((n-1)**2,n**2):
		chars = [c for c in bin(i)[2:]]
		goals.append(np.flip(np.array(chars).astype(int)))
	actuation_probs = compute_actuation_prob(belief_state,state)
	q_vals = np.zeros(n)
	for action in range(n):
		proposed_state = state.copy()
		proposed_state[action] = 1 - proposed_state[action]
		search_obj = dijkstra_search(proposed_state,n,belief_state)
		res, final_cost, state_sequence,action_sequence = search_obj.search(goals)
		q_vals[action] = actuation_probs[action]*final_cost

	return(q_vals)





def eval_policy(policy,size,prior=None,runs = 1000,timeout=100,stop_on_completion = True, policy_args={},verbose=False):
	total = 0
	distribution_divergences = np.zeros((runs,timeout))

	for run in range(runs):
		config = generate_door(size,2)
		# print_belief_state(config)
		# 1/0
		if verbose:
			plt.imshow(np.hstack((config[:,0,:],config[:,1,:])),vmin=0,vmax=1)
			plt.show()
		current_state = np.zeros(size,dtype=int)
		if np.any(prior == None):
			belief_state = np.ones((size,config.shape[1],size))*.1
			for i in range(size):
				belief_state[i,:,i] = 0.
		else:
			belief_state = prior

		env = LockEnv(config,current_state)
		success,current_state,reward = env.reset()
		counter = 0
		# fig = plt.figure()
		# ax = fig.gca()
		state_hash = np.zeros(2**size,dtype=int)
		state_hash[hash_state(current_state)] = 1
		
		if verbose:
			bs = plt.imshow(np.hstack((belief_state[:,0,:],belief_state[:,1,:])),vmin=0,vmax=1)
			# print(np.around(belief_state[:,0,:],5))
			# print(np.around(belief_state[:,1,:],5))
			plt.draw()
			plt.pause(0.05)
		completed = False
		while reward < 1 and counter < timeout:
			if verbose:
				print("Iterations: ", counter)
			# print(policy_args)
			action, confidence = policy(belief_state,current_state,state_hash,**policy_args)
			if verbose:
				# pass
				print(action)
				print(confidence)
			success,current_state,reward = env.step(action)
			state_hash[hash_state(current_state)] = 1
			if verbose:
				# pass
				print(success)
				print(current_state)
			belief_state = update_belief_state(belief_state, current_state, action, success)
			if verbose:
				pass
				# print(belief_state)

				print_belief_state(belief_state,'Sim Loop')

			distribution_divergence = belief_state_KL_Divergence(np.clip(belief_state[:size,:,:size],0.01,0.99),np.clip(config,0.01,0.99))
			# distribution_divergence = compute_entropy(np.clip(belief_state[:size,:,:size],0.01,0.99))
			distribution_divergences[run,counter] = distribution_divergence
			if verbose:
				print("Distribution Error:", distribution_divergence)
			if verbose:
				bs.set_data(np.hstack((belief_state[:,0,:],belief_state[:,1,:])))
				plt.draw()
				plt.pause(0.05)

			counter += 1
			if not stop_on_completion:
				if reward == 1 and not completed:
					print("COMPLETED")
					completed = True
				reward = 0

			if verbose:
				time.sleep(.001)
		print(counter)



		total += counter

	divergence_over_t_mean = []
	divergence_over_t_var = []
	if verbose:
		bs.set_data(np.hstack((belief_state[:,0,:],belief_state[:,1,:])))
		plt.show()


	for i in range(timeout):
		if np.all(distribution_divergences[:,i] == 0):
			break
		divergence_over_t_mean.append(np.mean(distribution_divergences[np.where(distribution_divergences[:,i] > 0)[0],i]))
		divergence_over_t_var.append(np.var(distribution_divergences[np.where(distribution_divergences[:,i] > 0)[0],i]))
	return(total/runs, np.array(divergence_over_t_mean),np.array(divergence_over_t_var))

def calculate_mean_var(data):
	divergence_over_t_mean = []
	divergence_over_t_var = []
	for i in range(data.shape[1]):
		if np.all(data[:,i] == 0):
			break
		divergence_over_t_mean.append(np.mean(data[np.where(data[:,i] > 0)[0],i]))
		divergence_over_t_var.append(np.var(data[np.where(data[:,i] > 0)[0],i]))
	return(total/runs, np.array(divergence_over_t_mean),np.array(divergence_over_t_var))



def main():
	# config = generate_lock_sequence(5,1)
	# print(config)
	# 1/0
	# # belief_state = np.copy(config)
	# # for a in range(belief_state.shape[0]):
	# # 	for b in range(belief_state.shape[1]):
	# # 		if belief_state[a,b] == 0:
	# # 			belief_state[a,b] += np.random.uniform(0.,.5)
	# # 		else:
	# # 			belief_state[a,b] -= np.random.uniform(0.,.5)
	# # print(belief_state)
	# # current_state = np.array([0,1,1,0,0])
	# # hypothesis = (2,3)
	# # action = 3
	# # print(belief_state[hypothesis])
	# # print(compute_posterior(belief_state, current_state, action, hypothesis))
	# # info_gain = compute_info_gain(belief_state, current_state)
	# # print(np.round(info_gain,3))
	# # print(np.sum(info_gain,axis=0))
	# current_state = np.zeros(5)
	# belief_state = np.ones((5,5))/5
	# env = LockEnv(config,current_state)
	# success,current_state,reward = env.reset()
	# counter = 0
	# while reward < 1:
	# 	action, confidence = balanced_policy(belief_state,current_state,alpha=1.)
	# 	print(action)
	# 	print(confidence)
	# 	success,current_state,reward = env.step(action)
	# 	print(current_state)
	# 	belief_state = update_belief_state(belief_state, current_state, action, success)
	# 	print(belief_state)
	# 	counter += 1
	# 	time.sleep(.1)

	# print(counter)
	prior = np.ones((5,2,5))*0.05
	for i in range(5):
		prior[i,:,i] = 0.

	prior[0,0,2]= .7
	prior[1,0,4]= .7
	prior[2,0,4]= .7
	prior[3,0,4]= .7

	# prior = np.ones((5,2,5))*0.
	# prior[0,0,1]= 1.
	# prior[1,0,3]= 1.
	# prior[3,0,2]= 1.
	# prior[2,0,4]= 1.


	

	# plt.imshow(np.hstack((prior[:,0,:],prior[:,1,:])),vmin=0,vmax=1)
	# plt.show()

	# search_obj = dijkstra_search(np.array([0,0,0,0,0]),5,prior)
	# goals = []
	# for i in range(16,32):
	# 	chars = [c for c in bin(i)[2:]]
	# 	goals.append(np.flip(np.array(chars).astype(int)))
	# print(goals)
	# # goals = [np.array([1,1,1,1,1])]
	# # print(goals)
	# res, final_cost, state_sequence,action_sequence = search_obj.search(goals)
	# print(res,final_cost,state_sequence,action_sequence)
	# print(compute_entropy(prior))
	# print(compute_q(np.array([0,0,0,0,0]),prior))
	# 1/0





	info_avg_actions, info_divergence_mean, info_divergence_var = eval_policy(info_max_policy,5,prior=None,runs=1,timeout=200,stop_on_completion = True, verbose=True,policy_args={"alpha":.5})
	print(info_avg_actions)
	1/0

	info_avg_actions, info_divergence_mean, info_divergence_var = eval_policy(info_max_policy,5,stop_on_completion = True, runs=100,timeout=100,verbose=False)
	rand_avg_actions, rand_divergence_mean, rand_divergence_var = eval_policy(random_policy,5,stop_on_completion = True, runs=100,timeout=100,verbose=False)
	novl_avg_actions, novl_divergence_mean, novl_divergence_var = eval_policy(novelty_policy,5,stop_on_completion = True, runs=100,timeout=100,verbose=False,policy_args={"alpha":.9})

	print(info_avg_actions,rand_avg_actions, novl_avg_actions)

	info_divergence_var /= 3
	rand_divergence_var /= 3
	novl_divergence_var /= 3


	fig = plt.figure(figsize=(12,8))

	plt.plot(np.arange(len(info_divergence_mean)),info_divergence_mean,c='b',label="Info_Max")
	plt.plot(np.arange(len(info_divergence_mean)),info_divergence_mean-info_divergence_var,c='b',alpha=.5)
	plt.plot(np.arange(len(info_divergence_mean)),info_divergence_mean+info_divergence_var,c='b',alpha=.5)
	plt.fill_between(np.arange(len(info_divergence_mean)),info_divergence_mean-info_divergence_var,info_divergence_mean+info_divergence_var,facecolor='b',alpha=.5)
	
	plt.plot(np.arange(len(rand_divergence_mean)),rand_divergence_mean,c='r',label="Random")
	plt.plot(np.arange(len(rand_divergence_mean)),rand_divergence_mean-rand_divergence_var,c='r',alpha=.5)
	plt.plot(np.arange(len(rand_divergence_mean)),rand_divergence_mean+rand_divergence_var,c='r',alpha=.5)
	plt.fill_between(np.arange(len(rand_divergence_mean)),rand_divergence_mean-rand_divergence_var,rand_divergence_mean+rand_divergence_var,facecolor='r',alpha=.5)

	plt.plot(np.arange(len(novl_divergence_mean)),novl_divergence_mean,c='g',label="Novelty")
	plt.plot(np.arange(len(novl_divergence_mean)),novl_divergence_mean-novl_divergence_var,c='g',alpha=.5)
	plt.plot(np.arange(len(novl_divergence_mean)),novl_divergence_mean+novl_divergence_var,c='g',alpha=.5)
	plt.fill_between(np.arange(len(novl_divergence_mean)),novl_divergence_mean-novl_divergence_var,novl_divergence_mean+novl_divergence_var,facecolor='g',alpha=.5)

	plt.legend()
	plt.xlabel("Actions")
	plt.ylabel("KL Divergence")
	plt.show()


	

	1/0

	sizes = np.arange(3,10,1)
	info_max = []
	random = []
	balanced_3 = []
	balanced_5 = []
	balanced_7 = []
	balanced_9 = []



	for size in sizes:
		print(size)
		info_max.append(eval_policy(info_max_policy,size,runs=1000))
		random.append(eval_policy(random_policy,size,runs=1000))
		print("Info max improvement: ",float(info_max[-1])/float(random[-1]))
		print("Optimal improvement: ",float(size)/float(random[-1]))

		balanced_3.append(eval_policy(balanced_policy,size,policy_args={"alpha":.3}))
		balanced_5.append(eval_policy(balanced_policy,size,policy_args={"alpha":.5}))
		balanced_7.append(eval_policy(balanced_policy,size,policy_args={"alpha":.7}))
		balanced_9.append(eval_policy(balanced_policy,size,policy_args={"alpha":.9}))
		print("balanced_3 improvement: ",float(balanced_3[-1])/float(random[-1]))
		print("balanced_5 improvement: ",float(balanced_5[-1])/float(random[-1]))
		print("balanced_7 improvement: ",float(balanced_7[-1])/float(random[-1]))
		print("balanced_9 improvement: ",float(balanced_9[-1])/float(random[-1]))


	fig = plt.figure(figsize=(12,8))
	plt.plot(sizes,np.array(info_max),c='b',label='info_max')
	plt.plot(sizes,np.array(random),c='r',label='random')
	plt.plot(sizes,np.array(balanced_3),c='c',label='balanced .3')
	plt.plot(sizes,np.array(balanced_5),c='m',label='balanced .5')
	plt.plot(sizes,np.array(balanced_7),c='y',label='balanced .7')
	plt.plot(sizes,np.array(balanced_9),c='k',label='balanced .9')


	plt.plot(sizes,sizes,c='g',label='optimal')

	plt.legend()
	plt.xlabel("Number of locks")
	plt.ylabel("Average actions")
	plt.show()








if __name__ == '__main__':
	main()
