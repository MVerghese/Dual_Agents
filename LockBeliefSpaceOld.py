import numpy as np
import time 
from matplotlib import pyplot as plt
import warnings

gamma = .9
np.set_printoptions(suppress=True)
# np.seterr(all='raise')

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


def JSD(P,Q):
	M = (P+Q)/2
	return(KL_Divergence(P,M)/2+KL_Divergence(Q,M)/2)

def belief_state_KL_Divergence(P,Q,enforce_sum = True):

	Ptemp = P[np.where(P>0)]
	Qtemp = Q[np.where(Q>0)]

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



def compute_expected_posterior(belief_state, current_state, action, hypothesis):
	# print(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action])
	if hypothesis[0] >= current_state.shape[0]:
		k = 0
		ph = belief_state[-1,k,hypothesis[1]]
	else:
		k = current_state[hypothesis[0]]
		ph = belief_state[hypothesis[0],k,hypothesis[1]]
	if ph == 0:
		print(np.around(belief_state[:,0,:],5))
		print(np.around(belief_state[:,1,:],5))
		print(current_state)
		print(hypothesis)
		ph = 1-gamma
	pe = p_or(np.prod(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action]),
			  belief_state[-1,k,hypothesis[1]])

	

	sum_prob = 0
	peh = 1 - gamma
	temp_belief_state = belief_state.copy()
	temp_belief_state[hypothesis[0],k,hypothesis[1]] = 0
	penoth = p_or(np.prod(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action]),
				  belief_state[-1,k,hypothesis[1]])*gamma
	sum_prob += pe*bayesian_inference(ph,peh,penoth)
	# sum_prob += .5*bayesian_inference(ph,peh,penoth)


	peh = 1 - peh
	penoth = 1 - penoth
	sum_prob += (1-pe)*bayesian_inference(ph,peh,penoth)
	# sum_prob += .5*bayesian_inference(ph,peh,penoth)


	return(sum_prob)


def compute_posterior_actual(belief_state, current_state, action, hypothesis, success):
	if hypothesis[0] >= current_state.shape[0]:
		k = 0
		ph = belief_state[-1,k,hypothesis[1]]
	else:
		k = current_state[hypothesis[0]]
		ph = belief_state[hypothesis[0],k,hypothesis[1]]
	pe = p_or(np.prod(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action]),
			  belief_state[-1,k,hypothesis[1]])

	peh = 1 - gamma
	temp_belief_state = belief_state.copy()
	temp_belief_state[hypothesis[0],k,hypothesis[1]] = 0
	penoth = p_or(np.prod(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action]),
				  belief_state[-1,k,hypothesis[1]])*gamma
	if (not success and hypothesis[0] < current_state.shape[0]) or (success and hypothesis[0] >= current_state.shape[0]):
		peh = 1 - peh
		penoth = 1- penoth

	return(bayesian_inference(ph,peh,penoth))


def update_belief_state(belief_state, current_state, action, success):
	new_belief_state = np.copy(belief_state)
	for i in range(belief_state.shape[0]):
		if i != action:
			if i >= current_state.shape[0]:
				k = 0
			else:
				k = current_state[i]
			new_belief_state[i,k,action] = compute_posterior_actual(belief_state,current_state,action,(i,action),success)
	# for a in range(belief_state.shape[0]):
	# 	new_belief_state[a,:,:] /= np.sum(new_belief_state[a,:,:])
	print_belief_state(new_belief_state,"Pre Norm")
	new_belief_state = normalize_belief_state(new_belief_state)
	print_belief_state(new_belief_state,"Post Norm")
	return(new_belief_state)



def compute_info_gain(belief_state,current_state):
	info_gain = np.zeros(belief_state.shape[2]-1)
	for b in range(belief_state.shape[2]-1):
		expected_posterior = np.copy(belief_state)
		for a in range(belief_state.shape[0]):
			# print(compute_posterior(belief_state,(a,b)))
			if a != b:
				if a >= current_state.shape[0]:
					k = 0
				else:
					k = current_state[a]
				print(a,b)
				expected_posterior[a,k,b] = compute_expected_posterior(belief_state, current_state, b, (a,b))

			# expected_posterior[a,:,:] /= np.sum(expected_posterior[a,:,:])
			# print(np.sum(expected_posterior[a,:,:]),expected_posterior[a,0,-1])
		# print('before',1 - np.prod(1-expected_posterior[:-1,:,b]) + expected_posterior[-1,0,b])
		expected_posterior = normalize_belief_state(expected_posterior)
		# print('after',1 - np.prod(1-expected_posterior[:-1,:,b]) + expected_posterior[-1,0,b])
		# 1/0
		# print_belief_state(belief_state,'belief_state')
		# print_belief_state(expected_posterior,'expected_posterior')

		info_gain[b] = belief_state_KL_Divergence(belief_state,expected_posterior)
			# print(a,b,KL_Divergence(belief_state[a,:,:],expected_posterior[a,:,:]))


	# print(info_gain)

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

def info_max_policy(belief_state,current_state,**kwargs):
	info_gain = compute_info_gain(belief_state, current_state)
	# print(info_gain)
	# print(action_prob)
	action = argmax(info_gain)
	# print(action_prob[action])
	return(action,info_gain)

def random_policy(belief_state,current_state,**kwargs):
	action_prob = np.random.uniform(0,1,size = current_state.shape[0])
	action = argmax(action_prob)
	return(action,1/np.sum(current_state==0))

def greedy_policy(belief_state,current_state,**kwargs):

	lock_probs = np.multiply(belief_state,1-current_state.reshape((current_state.shape[0],1)))
	unlock_probs = 1 - lock_probs
	prob_unlocked =  np.prod(unlock_probs,axis=0)
	action_prob = prob_unlocked+current_state*-100
	action = np.argmax(action_prob)
	return(action,action_prob)

def balanced_policy(belief_state,current_state,**kwargs):
	alpha = .5
	if "alpha" in kwargs:
		alpha = kwargs["alpha"]

	info_action,info_action_probs = info_max_policy(belief_state,current_state,**kwargs)
	plan_action,plan_action_probs = greedy_policy(belief_state,current_state,**kwargs)
	action_scores = alpha*info_action_probs + (1-alpha)*plan_action_probs

	action = np.argmax(action_scores)
	return(action,action_scores)

def constant_policy(belief_state,current_state,**kwargs):
	return(0,1)


def eval_policy(policy,size,runs = 1000,timeout=100,policy_args={},verbose=False):
	total = 0
	distribution_divergences = np.zeros((runs,timeout))

	for run in range(runs):
		config = generate_lock_sequence(size,2)
		current_state = np.zeros(size,dtype=int)
		belief_state = np.ones((size+1,config.shape[1],size+1))
		belief_state[:-1,:,:] /= ((size-1)*(config.shape[1])+1)
		belief_state[:,1:,-1] = 0.
		belief_state[-1,:,:] /= (size)
		belief_state[-1,1:,:] = 0.
		for i in range(size+1):
			belief_state[i,:,i] = 0.

		env = LockEnv(config,current_state)
		success,current_state,reward = env.reset()
		counter = 0
		# fig = plt.figure()
		# ax = fig.gca()
		
		if verbose:
			bs = plt.imshow(np.hstack((belief_state[:,0,:],belief_state[:,1,:])),vmin=0,vmax=1)
			print(np.around(belief_state[:,0,:],5))
			print(np.around(belief_state[:,1,:],5))
			plt.draw()
			plt.pause(0.05)
		while reward < 1 and counter < timeout:
			action, confidence = policy(belief_state,current_state,**policy_args)
			if verbose:
				print(action)
				print(confidence)
			success,current_state,reward = env.step(action)
			if verbose:
				print(success)
				print(current_state)
			belief_state = update_belief_state(belief_state, current_state, action, success)
			if verbose:
				# print(belief_state)

				print_belief_state(belief_state,'Sim Loop')

			distribution_divergence = belief_state_KL_Divergence(np.clip(belief_state[:size,:,:size],0.01,0.99),np.clip(config,0.01,0.99))
			distribution_divergences[run,counter] = distribution_divergence
			if verbose:
				bs.set_data(np.hstack((belief_state[:,0,:],belief_state[:,1,:])))
				plt.draw()
				plt.pause(0.05)

			counter += 1
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



	info_avg_actions, info_divergence_mean, info_divergence_var = eval_policy(info_max_policy,5,runs=1,timeout=200,verbose=True)
	print(info_avg_actions)
	1/0

	info_avg_actions, info_divergence_mean, info_divergence_var = eval_policy(info_max_policy,5,runs=10,timeout=200,verbose=False)
	rand_avg_actions, rand_divergence_mean, rand_divergence_var = eval_policy(random_policy,5,runs=10,timeout=200,verbose=False)
	print(info_avg_actions,rand_avg_actions)

	info_divergence_var /= 1
	rand_divergence_var /= 1

	fig = plt.figure(figsize=(12,8))

	plt.plot(np.arange(len(info_divergence_mean)),info_divergence_mean,c='b',label="Info_Max")
	plt.plot(np.arange(len(info_divergence_mean)),info_divergence_mean-info_divergence_var,c='b',alpha=.5)
	plt.plot(np.arange(len(info_divergence_mean)),info_divergence_mean+info_divergence_var,c='b',alpha=.5)
	plt.fill_between(np.arange(len(info_divergence_mean)),info_divergence_mean-info_divergence_var,info_divergence_mean+info_divergence_var,facecolor='b',alpha=.5)
	
	plt.plot(np.arange(len(rand_divergence_mean)),rand_divergence_mean,c='r',label="Random")
	plt.plot(np.arange(len(rand_divergence_mean)),rand_divergence_mean-rand_divergence_var,c='r',alpha=.5)
	plt.plot(np.arange(len(rand_divergence_mean)),rand_divergence_mean+rand_divergence_var,c='r',alpha=.5)
	plt.fill_between(np.arange(len(rand_divergence_mean)),rand_divergence_mean-rand_divergence_var,rand_divergence_mean+rand_divergence_var,facecolor='r',alpha=.5)

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
