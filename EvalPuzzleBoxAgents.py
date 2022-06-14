import LockBeliefSpaceFixed
import PuzzleBoxDQNAgent2
import PuzzleBoxModelBasedEnv
import PuzzleBoxEnv

import numpy as np

from matplotlib import pyplot as plt

def eval_agent(agent,agent_type,env_type,env_index,runs,rl_agent_eps = 0,structured_agent_max_ent = 15.):
	if agent_type == 'mb_agent':
		scores = np.zeros(runs)
		for i in range(runs):
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index,return_state_mode='mb',randomize_config=True)
			_,state,_,_ = env.reset()
			score = 0
			for t in range(200):
				action = agent.act(state,env.get_goal_state(),rl_agent_eps)
				_,next_state,reward,done = env.step(action)

				# agent.step(state,action,reward,next_state,done)
				## above step decides whether we will train(learn) the network
				## actor (local_qnetwork) or we will fill the replay buffer
				## if len replay buffer is equal to the batch size then we will
				## train the network or otherwise we will add experience tuple in our 
				## replay buffer.
				state = next_state
				score += reward
				if done:
					break
			scores[i] = score
		return(scores)

	if agent_type == 'mf_agent':
		scores = np.zeros(runs)
		for i in range(runs):
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index,return_state_mode='mf',randomize_config=True)
			_,state,_,_ = env.reset()
			score = 0
			# print(env.config[:,0,:,0])
			for t in range(200):
				action = agent.act(state,rl_agent_eps)
				# print(state[:35].reshape(7,5))
				# print(state[35:].reshape(20,4))
				# print(action)
				_,next_state,reward,done = env.step(action)

				# agent.step(state,action,reward,next_state,done)
				## above step decides whether we will train(learn) the network
				## actor (local_qnetwork) or we will fill the replay buffer
				## if len replay buffer is equal to the batch size then we will
				## train the network or otherwise we will add experience tuple in our 
				## replay buffer.
				state = next_state
				score += reward
				if done:
					break
			scores[i] = score
		return(scores)

	elif agent_type == 'dual_agent':
		scores = np.zeros(runs)
		for i in range(runs):
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index)
			# print(env.get_object_ids())
			agent.init_policy(5,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = True)
			success, state, reward, done = env.reset()
			score = 0
			while not done:
				action = agent.act(state,ent_max = structured_agent_max_ent)
				success, new_state, reward, done = env.step(action)
				agent.update_policy_info(state,success)
				state = new_state
				score += reward
			scores[i] = score
			# agent.save_graph()
		return(scores)

	elif agent_type == 'dual_agent_default':
		scores = np.zeros(runs)
		for i in range(runs):
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index)
			# print(env.get_object_ids())
			agent.init_policy(5,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = False)
			success, state, reward, done = env.reset()
			score = 0
			while not done:
				action = agent.act(state,ent_max = structured_agent_max_ent)
				success, new_state, reward, done = env.step(action)
				agent.update_policy_info(state,success)
				state = new_state
				score += reward
			scores[i] = score
			# agent.save_graph()
		return(scores)

	elif agent_type == 'random_agent':
		scores = np.zeros(runs)
		for i in range(runs):
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index)
			# print(env.get_object_ids())
			success, state, reward, done = env.reset()
			score = 0
			t = 0
			while not done and t < 200:
				action, confidences = LockBeliefSpaceFixed.random_policy(np.array([]),state,np.array([]),-1)
				success, new_state, reward, done = env.step(action)
				state = new_state
				score += reward
				t += 1
			scores[i] = score
			# agent.save_graph()
		return(scores)


def load_mb_agent(num_train_envs,directory='models/',ep_num = None):
	if ep_num:
		loadpath = directory + "PreconditionModel"+str(num_train_envs)+str(ep_num)+".pth"
	else:
		loadpath = directory + "PreconditionModel"+str(num_train_envs)+".pth"
	mb_agent = PuzzleBoxModelBasedEnv.Agent(state_size=30,action_size=5,seed=0,load=True,loadpath = loadpath)
	return(mb_agent)

def load_mf_agent(num_train_envs,directory='models/',ep_num = None):
	if ep_num:
		loadpath = directory + "DQNModel"+str(num_train_envs)+str(ep_num)+".pth"
	else:
		loadpath = directory + "DQNModel"+str(num_train_envs)+".pth"
	mf_agent = PuzzleBoxDQNAgent2.Agent(state_size=30,action_size=5,seed=0,load=True,loadpath = loadpath)
	return(mf_agent)

def load_and_train_dual_agent(num_train_envs,eps_per_env = 10):
	dual_agent = LockBeliefSpaceFixed.Structured_Agent_Dist(3)
	counts = LockBeliefSpaceFixed.train_agent(dual_agent,num_train_envs,eps_per_env = eps_per_env,env_type='train')
	return(dual_agent)

def test_env(agent_type,env_type,num_train_envs,num_test_envs,runs):
	means = []
	variances = []
	for i in range(num_train_envs+1):
		print(i)
		all_scores = np.zeros((num_test_envs[i],runs))
		if i >= 0 and agent_type == 'dual_agent':
			agent = load_and_train_dual_agent(i)
			if i == 0:
				for j in range(num_test_envs[i]):
					all_scores[j,:] = eval_agent(agent,'dual_agent_default',env_type,j,runs)
			else:
				for j in range(num_test_envs[i]):
					all_scores[j,:] = eval_agent(agent,agent_type,env_type,j,runs,structured_agent_max_ent = 200.)
		elif i > 0 and agent_type == 'mb_agent':
			agent = load_mb_agent(i)
			for j in range(num_test_envs[i]):
				all_scores[j,:] = eval_agent(agent,agent_type,env_type,j,runs,rl_agent_eps = .1)
		elif i > 0 and agent_type == 'mf_agent':
			agent = load_mf_agent(i)
			for j in range(num_test_envs[i]):
				all_scores[j,:] = eval_agent(agent,agent_type,env_type,j,runs,rl_agent_eps = .1)


		means.append(np.mean(all_scores))
		print(np.mean(all_scores))
		variances.append(np.var(all_scores))

	return(np.array(means),np.array(variances))


def test_ep_gen(agent_type, runs, dual_agent_max_tests = 2000):
	means = []
	variances = []
	eps = []
	all_scores = np.zeros((3,runs))
	if agent_type == 'dual_agent':
		
		for i in range(1,int(dual_agent_max_tests/9)+1):
			print("eps: ",i*9)
			dual_agent = load_and_train_dual_agent(9,eps_per_env = i)
			for j in range(3):
				all_scores[j,:] = eval_agent(dual_agent,agent_type,'test',j,runs,structured_agent_max_ent = 200.)

			means.append(np.mean(all_scores))
			print(np.mean(all_scores))
			variances.append(np.var(all_scores))
			eps.append(i*9)
	else:
		for i in range(1,int(2000/50)+1):
			print("eps: ",i*50)
			if agent_type == 'mb_agent':
				agent = load_mb_agent(9,ep_num = i*50)
				for j in range(3):
					all_scores[j,:] = eval_agent(agent,agent_type,'test',j,runs,rl_agent_eps = .1)
			elif agent_type == 'mf_agent':
				agent = load_mf_agent(9,ep_num = i*50)
				for j in range(3):
					all_scores[j,:] = eval_agent(agent,agent_type,'test',j,runs,rl_agent_eps = .1)
			means.append(np.mean(all_scores))
			print(np.mean(all_scores))
			variances.append(np.var(all_scores))
			eps.append(i*50)

	return(np.array(means),np.array(variances),np.array(eps))









def train_and_eval_agents(num_train_envs,eval_runs):

	structured_agent = LockBeliefSpaceFixed.Structured_Agent()
	counts = LockBeliefSpaceFixed.train_agent(structured_agent,num_train_envs,env_type='all')
	print(np.mean(counts))

	# rl_agent = PuzzleBoxDQNAgent2.Agent(state_size=35,action_size=5,seed=0)
	# scores = PuzzleBoxDQNAgent2.dqn(rl_agent,num_train_envs,n_episodes=1000, eps_end = .2, eps_decay = .999)
	# print(np.mean(scores[-100:]))

	structured_agent_scores = np.zeros((num_train_envs,eval_runs))

	rl_agent_scores = np.zeros((num_train_envs,eval_runs))

	structured_agent_no_train_scores = np.zeros((num_train_envs,eval_runs))

	random_agent = np.zeros((num_train_envs,eval_runs))


	for i in range(num_train_envs):
		print("Testing env: ",i)
		
		# rl_agent_scores[i,:] = eval_agent(rl_agent,'rl_agent','test',i,eval_runs,rl_agent_eps = .3)
		
		structured_agent_scores[i,:] = eval_agent(structured_agent,'structured_agent','all',i,eval_runs,structured_agent_max_ent = 200.)
		
		structured_agent_no_train_scores[i,:] = eval_agent(structured_agent,'structured_agent_no_train','all',i,eval_runs)
		
		# random_agent[i,:] = eval_agent(None,'random_agent','test',i,eval_runs)
		

	print("rl_agent_scores",np.mean(rl_agent_scores))
	print("percent envs solved",np.sum(rl_agent_scores < 10))
	print("structured_agent_scores",np.mean(structured_agent_scores))
	print("percent envs solved",np.sum(structured_agent_scores < 10))
	print("structured_agent_no_train_scores",np.mean(structured_agent_no_train_scores))
	print("percent envs solved",np.sum(structured_agent_no_train_scores < 10))
	print("random_agent",np.mean(random_agent))
	print("percent envs solved",np.sum(random_agent < 10))

	

	sa_mean = np.mean(structured_agent_scores.flatten())
	sa_var = np.var(structured_agent_scores.flatten())

	rl_mean = np.mean(rl_agent_scores.flatten())
	rl_var = np.var(rl_agent_scores.flatten())

	nt_mean = np.mean(structured_agent_no_train_scores.flatten())
	nt_var = np.var(structured_agent_no_train_scores.flatten())

	rd_mean = np.mean(random_agent.flatten())
	rd_var = np.var(random_agent.flatten())


	return(sa_mean,sa_var,rl_mean,rl_var,nt_mean,nt_var,rd_mean,rd_var)

def run_experiment(n_train_envs,eval_runs,save_data = True):
	sa_means = np.zeros(n_train_envs-1)
	sa_vars = np.zeros(n_train_envs-1)

	rl_means = np.zeros(n_train_envs-1)
	rl_vars = np.zeros(n_train_envs-1)

	nt_means = np.zeros(n_train_envs-1)
	nt_vars = np.zeros(n_train_envs-1)

	rd_means = np.zeros(n_train_envs-1)
	rd_vars = np.zeros(n_train_envs-1)

	for i in range(1,n_train_envs):
		print("num train envs: ", i)
		sa_mean,sa_var,rl_mean,rl_var,nt_mean,nt_var,rd_mean,rd_var = train_and_eval_agents(i,eval_runs)

		sa_means[i-1] = sa_mean
		sa_vars[i-1] = sa_var

		rl_means[i-1] = rl_mean
		rl_vars[i-1] = rl_var

		nt_means[i-1] = nt_mean
		nt_vars[i-1] = nt_var

		rd_means[i-1] = rd_mean
		rd_vars[i-1] = rd_var



	if save_data:
		np.save('sa_means.npy',sa_means)
		np.save('sa_vars.npy',sa_vars)

		np.save('rl_means.npy',rl_means)
		np.save('rl_vars.npy',rl_vars)

		np.save('nt_means.npy',nt_means)
		np.save('nt_vars.npy',nt_vars)

		np.save('rd_means.npy',rd_means)
		np.save('rd_vars.npy',rd_vars)
		

	fig = plt.figure(figsize=(12,8))

	plt.plot(np.arange(len(sa_means)),sa_means,c='b',label="Structured Agent (Ours)")
	plt.plot(np.arange(len(sa_means)),sa_means-sa_vars,c='b',alpha=.5)
	plt.plot(np.arange(len(sa_means)),sa_means+sa_vars,c='b',alpha=.5)
	plt.fill_between(np.arange(len(sa_means)),sa_means-sa_vars,sa_means+sa_vars,facecolor='b',alpha=.5)
	
	plt.plot(np.arange(len(rl_means)),rl_means,c='r',label="RL Agent")
	plt.plot(np.arange(len(rl_means)),rl_means-rl_vars,c='r',alpha=.5)
	plt.plot(np.arange(len(rl_means)),rl_means+rl_vars,c='r',alpha=.5)
	plt.fill_between(np.arange(len(rl_means)),rl_means-rl_vars,rl_means+rl_vars,facecolor='r',alpha=.5)

	plt.plot(np.arange(len(nt_means)),nt_means,c='g',label="Structured Agent Control")
	plt.plot(np.arange(len(nt_means)),nt_means-nt_vars,c='g',alpha=.5)
	plt.plot(np.arange(len(nt_means)),nt_means+nt_vars,c='g',alpha=.5)
	plt.fill_between(np.arange(len(nt_means)),nt_means-nt_vars,nt_means+nt_vars,facecolor='g',alpha=.5)

	plt.plot(np.arange(len(rd_means)),rd_means,c='c',label="Random Agent")
	plt.plot(np.arange(len(rd_means)),rd_means-rd_vars,c='c',alpha=.5)
	plt.plot(np.arange(len(rd_means)),rd_means+rd_vars,c='c',alpha=.5)
	plt.fill_between(np.arange(len(rd_means)),rd_means-rd_vars,rd_means+rd_vars,facecolor='c',alpha=.5)

	plt.legend()
	plt.xlabel("Number of Train Envs Seen")
	plt.ylabel("Avg Reward")
	plt.show()

def plot_data():
	sa_means = np.load('sa_means.npy')
	sa_vars = np.load('sa_vars.npy')

	rl_means = np.load('rl_means.npy')
	rl_vars = np.load('rl_vars.npy')

	nt_means = np.load('nt_means.npy')
	nt_vars = np.load('nt_vars.npy')

	rd_means = np.load('rd_means.npy')
	rd_vars = np.load('rd_vars.npy')

	rd_vars /= 1000

	rl_vars /=100


	fig = plt.figure(figsize=(12,8))

	plt.plot(np.arange(0,len(sa_means)),sa_means,c='b',label="Structured Agent (Ours)")
	plt.plot(np.arange(0,len(sa_means)),sa_means-sa_vars,c='b',alpha=.5)
	plt.plot(np.arange(0,len(sa_means)),sa_means+sa_vars,c='b',alpha=.5)
	plt.fill_between(np.arange(0,len(sa_means)),sa_means-sa_vars,sa_means+sa_vars,facecolor='b',alpha=.5)
	
	# plt.plot(np.arange(1,len(rl_means)+1),rl_means,c='r',label="RL Agent")
	# plt.plot(np.arange(1,len(rl_means)+1),rl_means-rl_vars,c='r',alpha=.5)
	# plt.plot(np.arange(1,len(rl_means)+1),rl_means+rl_vars,c='r',alpha=.5)
	# plt.fill_between(np.arange(1,len(rl_means)+1),rl_means-rl_vars,rl_means+rl_vars,facecolor='r',alpha=.5)

	# plt.plot(np.arange(1,len(nt_means)+1),nt_means,c='g',label="Structured Agent Control")
	# plt.plot(np.arange(1,len(nt_means)+1),nt_means-nt_vars,c='g',alpha=.5)
	# plt.plot(np.arange(1,len(nt_means)+1),nt_means+nt_vars,c='g',alpha=.5)
	# plt.fill_between(np.arange(1,len(nt_means)+1),nt_means-nt_vars,nt_means+nt_vars,facecolor='g',alpha=.5)

	# plt.plot(np.arange(len(rd_means)),rd_means,c='c',label="Random Agent")
	# plt.plot(np.arange(len(rd_means)),rd_means-rd_vars,c='c',alpha=.5)
	# plt.plot(np.arange(len(rd_means)),rd_means+rd_vars,c='c',alpha=.5)
	# plt.fill_between(np.arange(len(rd_means)),rd_means-rd_vars,rd_means+rd_vars,facecolor='c',alpha=.5)

	plt.legend()
	plt.xlabel("Number of Train Envs Seen")
	plt.ylabel("Avg Reward")
	plt.show()

def plot_var(x,means,varis,c,label,alpha = .5):
	plt.plot(x,means,c=c,label=label)
	plt.plot(x,means-varis,c=c,alpha=.5)
	plt.plot(x,means+varis,c=c,alpha=.5)
	plt.fill_between(x,means-varis,means+varis,facecolor=c,alpha=alpha)

def plot_ep_gen():
	dual_means = np.load("ep_gen_dual_means.npy")
	dual_variances = np.load("ep_gen_dual_variances.npy")
	dual_eps = np.load("ep_gen_dual_eps.npy")

	mf_means = np.load("ep_gen_mf_means.npy")
	mf_variances = np.load("ep_gen_mf_variances.npy")/100
	mf_eps = np.load("ep_gen_mf_eps.npy")

	mb_means = np.load("ep_gen_mb_means.npy")
	mb_variances = np.load("ep_gen_mb_variances.npy")/100
	mb_eps = np.load("ep_gen_mb_eps.npy")

	fig = plt.figure(figsize=(12,8))
	plot_var(dual_eps,dual_means,dual_variances,'b',"Dual Agent (Ours)")
	plot_var(mf_eps,mf_means,mf_variances,'g',"Model Free Agent")
	plot_var(mb_eps,mb_means,mb_variances,'r',"Model Based Agent")

	plt.ylim(-100,0)
	plt.xlim(0,750)
	# plt.xscale('log')

	plt.legend()
	plt.xlabel("Number of Training Episodes")
	plt.ylabel("Avg Reward (Unseen Environemnts)")
	plt.show()


	



def main():
	dual_means, dual_variances = test_env('dual_agent','test',9,np.ones(10,dtype=int)*3,50)
	
	# # dual_means, dual_variances, dual_eps = test_ep_gen('dual_agent',50,dual_agent_max_tests=200)
	np.save("env_gen_dual_means.npy",dual_means)
	np.save("env_gen_dual_variances.npy",dual_variances)
	# np.save("env_gen_dual_eps.npy",dual_eps)

	mf_means, mf_variances = test_env('mf_agent','test',9,np.ones(10,dtype=int)*3,50)

	
	# # mf_means, mf_variances, mf_eps = test_ep_gen('mf_agent',50,dual_agent_max_tests=90)
	np.save("env_gen_mf_means.npy",mf_means)
	np.save("env_gen_mf_variances.npy",mf_variances)
	# np.save("env_gen_mf_eps.npy",mf_eps)

	
	mb_means, mb_variances = test_env('mb_agent','test',9,np.ones(10,dtype=int)*3,50)


	# # mb_means, mb_variances, mb_eps = test_ep_gen('mb_agent',50,dual_agent_max_tests=90)
	np.save("env_gen_mb_means.npy",mb_means)
	np.save("env_gen_mb_variances.npy",mb_variances)
	# np.save("env_gen_mb_eps.npy",mb_eps)

	# plot_ep_gen()


	# run_experiment(12,100,save_data = True)
	# plot_data()
	# agent = load_mb_agent(9)
	# scores = eval_agent(agent,'mb_agent','test',1,1,rl_agent_eps = .1)
	# print(scores)

if __name__ == "__main__":
	main()






