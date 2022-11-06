import LockBeliefSpaceFixed
import PuzzleBoxDQNAgent2
import DisassemblyDQNAgent
import DisassemblyMBAgent
import PuzzleBoxModelBasedEnv
import PuzzleBoxEnv

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
import seaborn as sns

real_dual_eps = np.array([0,.5,1])
real_dual_scores = np.array([26.0,10.0,8.8])*-1

real_mf_eps = np.array([0,1,2,3,4,5])
real_mf_scores = np.array([30,30,27.4,30,24.0,30])*-1

real_mb_eps = np.array([0,1,2,3,4,5])
real_mb_scores = np.array([30,28.6,27.8,25.8,27.8,28.8])*-1




def eval_agent(agent,agent_type,env_type,env_index,env_permutation,runs,rl_agent_eps = 0,structured_agent_max_ent = 15.,corrupt_cids = False,add_pos_noise = False):
	np.random.seed(0)
	if agent_type == 'mb_agent':
		scores = np.zeros(runs)
		for i in range(runs):
			if i%10 == 0:
				print("run: ",i)
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index,env_permutation=permutation,return_state_mode='mb',randomize_config=True)
			if corrupt_cids:
				env.corrupt_cids()
			if add_pos_noise:
				env.add_location_noise()
			_,state,_,_ = env.reset()
			score = 0
			for t in range(20):
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
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index,env_permutation=permutation,return_state_mode='mf',randomize_config=True)
			if corrupt_cids:
				env.corrupt_cids()
			if add_pos_noise:
				env.add_location_noise()
			_,state,_,_ = env.reset()
			score = 0
			# print(env.config[:,0,:,0])
			for t in range(20):
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
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index,env_permutation=permutation)
			if corrupt_cids:
				env.corrupt_cids()
			if add_pos_noise:
				env.add_location_noise()
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
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index,env_permutation=permutation)
			if corrupt_cids:
				env.corrupt_cids()
			if add_pos_noise:
				env.add_location_noise()
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
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index,env_permutation=permutation)
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

def eval_comp_agent(agent,agent_type,env_type,env_index,env_permutation,runs,rl_agent_eps = 0,structured_agent_max_ent = 15.,corrupt_cids = False,add_pos_noise = False):
	if agent_type == 'mb_agent':
		scores = np.zeros(runs)
		for i in range(runs):
			# if i%10 == 0:
			print("run: ",i)
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.CompEnv(env_index = env_index,jamming=0.0,return_state_mode = 'mb',timeout=200,randomize_config = True)
			if corrupt_cids:
				env.corrupt_cids()
			if add_pos_noise:
				env.add_location_noise()
			_,state,_,_ = env.reset()
			score = 0
			for t in range(20):
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
			scores[i] = t+1
		return(scores)

	if agent_type == 'mf_agent':
		scores = np.zeros(runs)
		for i in range(runs):
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.CompEnv(env_index = env_index,jamming=0.0,return_state_mode = 'mf',timeout=200,randomize_config = True)
			if corrupt_cids:
				env.corrupt_cids()
			if add_pos_noise:
				env.add_location_noise()
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
			# print(t+1)
			scores[i] = t+1
		return(scores)

	elif agent_type == 'dual_agent':
		scores = np.zeros(runs)
		for i in range(runs):
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.CompEnv(env_index = env_index,jamming=0.0,return_state_mode = 'dual',timeout=200)
			if corrupt_cids:
				env.corrupt_cids()
			if add_pos_noise:
				env.add_location_noise()
			# print(env.get_object_ids())
			agent.init_policy(8,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = True)
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
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index,env_permutation=permutation)
			if corrupt_cids:
				env.corrupt_cids()
			if add_pos_noise:
				env.add_location_noise()
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
			permutation = np.random.choice(4)
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = env_index,env_permutation=permutation)
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

def load_mb_agent(directory='models/',ep_num = None,num_train_envs=None):
	if ep_num:
		loadpath = directory + "PreconditionModelMTOPT9"+str(ep_num)+".pth"
	else:
		loadpath = directory + "PreconditionModel"+str(num_train_envs)+".pth"
	mb_agent = PuzzleBoxModelBasedEnv.Agent(state_size=5,action_size=5,seed=17,load=True,loadpath = loadpath)
	return(mb_agent)

def load_mf_agent(directory='models/',ep_num = None,num_train_envs=None):
	if ep_num:
		loadpath = directory + "DQNModel_MTOPT9"+str(ep_num)+".pth"
	else:
		loadpath = directory + "DQNModel"+str(num_train_envs)+".pth"
	mf_agent = PuzzleBoxDQNAgent2.Agent(state_size=5,action_size=5,seed=17,load=True,loadpath = loadpath)
	return(mf_agent)

def load_and_train_dual_agent(num_train_envs,eps_per_env = 10):
	dual_agent = LockBeliefSpaceFixed.Structured_Agent_Dist(7,dim=3)
	counts = LockBeliefSpaceFixed.train_agent(dual_agent,num_train_envs,eps_per_env = eps_per_env,env_type='train')
	return(dual_agent)

def test_env(agent_type,env_type,num_train_envs,num_test_envs,runs,corrupt_cids=False,add_pos_noise=False):
	# means = []
	# variances = []
	all_scores = np.zeros((len(num_train_envs),runs*num_test_envs))
	for ind in range(len(num_train_envs)):
		i = num_train_envs[ind]
		print(i)
		if i >= 0 and agent_type == 'dual_agent':
			agent = load_and_train_dual_agent(i)
			if i == 0:
				for j in range(num_test_envs):
					all_scores[ind,j*runs:(j+1)*runs] = eval_agent(agent,'dual_agent_default',env_type,j,0,runs,corrupt_cids=corrupt_cids,add_pos_noise=add_pos_noise)
			else:
				for j in range(num_test_envs):
					all_scores[ind,j*runs:(j+1)*runs] = eval_agent(agent,agent_type,env_type,j,0,runs,structured_agent_max_ent = 200.,corrupt_cids=corrupt_cids,add_pos_noise=add_pos_noise)
		elif i > 0 and agent_type == 'mb_agent':
			agent = load_mb_agent(i)
			for j in range(num_test_envs):
				print("J = ",j )
				all_scores[ind,j*runs:(j+1)*runs] = eval_agent(agent,agent_type,env_type,j,0,runs,rl_agent_eps = .1,corrupt_cids=corrupt_cids,add_pos_noise=add_pos_noise)
		elif i > 0 and agent_type == 'mf_agent':
			agent = load_mf_agent(i)
			for j in range(num_test_envs):
				all_scores[ind,j*runs:(j+1)*runs] = eval_agent(agent,agent_type,env_type,j,0,runs,rl_agent_eps = .1,corrupt_cids=corrupt_cids,add_pos_noise=add_pos_noise)


		# means.append(np.mean(all_scores))
		print(np.mean(all_scores[ind,:]))
		# variances.append(np.var(all_scores))

	return(all_scores)


def test_ep_gen(agent_type, runs, dual_agent_max_tests = 2000):
	# means = []
	# variances = []
	eps = []
	all_scores = []
	scores = np.zeros(runs*3)
	
	if agent_type == 'dual_agent':
		
		
		for i in range(1,dual_agent_max_tests):
			print("eps: ",i)
			print("Training")
			dual_agent = load_and_train_dual_agent(1,eps_per_env = i)
			print("Testing")
			scores = eval_comp_agent(dual_agent,agent_type,'test',1,0,runs,structured_agent_max_ent = 200.)

			# means.append(np.mean(all_scores))
			
			# variances.append(np.var(all_scores))
			all_scores.append(scores)
			print(np.mean(all_scores[-1]))
			eps.append(i)
	else:
		for i in range(1,int(2000/50)+1):
			scores = np.zeros(runs*3)
			print("eps: ",i*50)
			if agent_type == 'mb_agent':
				agent = load_mb_agent(ep_num = i*50,directory='models/PreconditionModelMTOPT/')
				for j in range(3):
					scores[j*50:(j+1)*50] = eval_agent(agent,agent_type,'test',j,0,runs,rl_agent_eps = .1)
			elif agent_type == 'mf_agent':
				# print("Loading model: " + str(i*50))
				agent = load_mf_agent(ep_num = i*50,directory='models/DQNModel_MTOPT/')
				for j in range(3):
					scores[j*50:(j+1)*50] = eval_agent(agent,agent_type,'test',j,0,runs,rl_agent_eps = .1)
			# means.append(np.mean(all_scores))
			
			# variances.append(np.var(all_scores))
			all_scores.append(np.copy(scores))
			print(np.mean(all_scores[-1]))
			eps.append(i*50)

	return(np.array(all_scores),np.array(eps))

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

def plot_var(x,means,varis,c,label,N = 50,alpha = .5):
	# rc('text', usetex=True)
	# rc('axes', linewidth=2)
	# rc('font', weight='bold')
	# rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
	rcParams["legend.loc"] = 'lower right'
	std_err = varis / np.sqrt(N)
	plt.plot(x,means,c=c,label=label,lw=3)
	plt.plot(x,means-std_err,c=c,alpha=.5)
	plt.plot(x,means+std_err,c=c,alpha=.5)
	plt.fill_between(x,means-std_err,means+std_err,facecolor=c,alpha=alpha)

def plot_reg(x,means,c,label,N = 50,alpha = .5):
	# rc('text', usetex=True)
	# rc('axes', linewidth=2)
	# rc('font', weight='bold')
	# rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
	rcParams["legend.loc"] = 'upper right'
	std_err = varis / np.sqrt(N)
	plt.plot(x,means,c=c,label=label,lw=3)
	# plt.plot(x,means-std_err,c=c,alpha=.5)
	# plt.plot(x,means+std_err,c=c,alpha=.5)
	# plt.fill_between(x,means-std_err,means+std_err,facecolor=c,alpha=alpha)

def plot_bar(ind,val,c,label,N=50,width=1):
	# rc('text', usetex=True)
	# rc('axes', linewidth=2)
	# rc('font', weight='bold')
	# rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
	rcParams["legend.loc"] = 'upper right'
	ax = plt.gca()
	# std_err = varis / np.sqrt(N)
	# print(ind)
	# print(val)
	# print(std_err)
	ax.bar(ind, val, width, label=label,color=c,edgecolor='black')



def plot_ep_gen(N = 50,alpha = .5,thresh = 10):

	font_settings = {
    "text.usetex": True,
    "font.serif": ["Times New Roman"],
    "font.family":"serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "figure.titlesize":8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "legend.title_fontsize":6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
	}
	sns.set(context="paper",style="whitegrid",font="serif",rc=font_settings)

	dual_scores = np.load("ep_gen_dual_scores.npy")*-1
	dual_thresh = np.sum(dual_scores < thresh,axis=1).flatten()/150
	dual_eps = np.load("ep_gen_dual_eps.npy")

	mf_scores = np.load("ep_gen_mf_scores.npy")*-1
	mf_thresh = np.sum(mf_scores < thresh,axis=1).flatten()/150
	mf_eps = np.load("ep_gen_mf_eps.npy")

	mb_scores = np.load("ep_gen_mb_scores.npy")*-1
	mb_thresh = np.sum(mb_scores < thresh,axis=1).flatten()/150
	mb_eps = np.load("ep_gen_mb_eps.npy")

	mf_mtopt_scores = np.load("MTOPT_ep_gen_mf_scores.npy")*-1
	mf_mtopt_thresh = np.sum(mf_mtopt_scores < thresh,axis=1).flatten()/150
	mf_mtopt_eps = np.load("MTOPT_ep_gen_mf_eps.npy")

	mb_mtopt_scores = np.load("MTOPT_ep_gen_mb_scores.npy")*-1
	mb_mtopt_thresh = np.sum(mb_mtopt_scores < thresh,axis=1).flatten()/150
	mb_mtopt_eps = np.load("MTOPT_ep_gen_mb_eps.npy")

	fig = plt.figure(figsize=(1.8*2,1.5*2))
	plt.plot(np.insert(dual_eps,0,0),np.insert(dual_thresh,0,0),c='cornflowerblue',label=r'\textbf{Memory-Based (Ours)}')
	plt.plot(np.insert(mf_eps,0,0),np.insert(mf_thresh,0,0),c='lightgreen',label=r'\textbf{Deep Model-Free}')
	plt.plot(np.insert(mb_eps,0,0),np.insert(mb_thresh,0,0),c='lightcoral',label=r'\textbf{Deep Model-Based}')
	plt.plot(np.insert(mf_mtopt_eps,0,0),np.insert(mf_mtopt_thresh,0,0),c='seagreen',label=r'\textbf{Deep Model-Free MTOPT}')
	plt.plot(np.insert(mb_mtopt_eps,0,0),np.insert(mb_mtopt_thresh,0,0),c='darkred',label=r'\textbf{Deep Model-Based MTOPT}')


	plt.ylim(0,1.1)
	plt.xlim(0,750)
	# plt.xscale('log')

	plt.legend()
	plt.title(r'\textbf{Performance vs Training Episodes}')
	plt.xlabel(r'\textbf{Number of Training Episodes}')
	plt.ylabel(r'\textbf{Fraction of Environments Solved}')
	plt.tight_layout(pad=0)
	plt.savefig('Ep_Gen.pdf')
	plt.show()

def plot_comp_ep_gen(N = 50,alpha = .5,thresh = 10):

	font_settings = {
    "text.usetex": True,
    "font.serif": ["Times New Roman"],
    "font.family":"serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "figure.titlesize":8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "legend.title_fontsize":6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
	}
	sns.set(context="paper",style="whitegrid",font="serif",rc=font_settings)

	dual_scores = np.load("dis_ep_gen_dual_scores.npy")*-1
	dual_thresh = np.sum(dual_scores < thresh,axis=1).flatten()/50
	dual_eps = np.load("dis_ep_gen_dual_eps.npy")
	print(thresh)
	print(dual_thresh)

	mf_scores = np.load("dis_ep_gen_mf_scores.npy")
	mf_thresh = np.sum(mf_scores < thresh,axis=1).flatten()/50
	mf_eps = np.load("dis_ep_gen_mf_eps.npy")

	mb_scores = np.load("dis_ep_gen_mb_scores.npy")
	mb_scores[np.where(mb_scores == 20)] = 25
	mb_thresh = np.sum(mb_scores < thresh,axis=1).flatten()/10
	mb_eps = np.load("dis_ep_gen_mb_eps.npy")

	fig = plt.figure(figsize=(1.8*2,1.5*2))
	plt.plot(np.insert(dual_eps,0,0),np.insert(dual_thresh,0,0),lw=3,c='cornflowerblue',label=r'\textbf{Memory-Based (Ours)}')
	plt.plot(np.insert(mf_eps,0,0),np.insert(mf_thresh,0,0),lw=3,c='lightgreen',label=r'\textbf{Deep Model-Free}')
	plt.plot(np.insert(mb_eps,0,0),np.insert(mb_thresh,0,0),lw=3,c='lightcoral',label=r'\textbf{Deep Model-Based}')



	plt.ylim(0,1.1)
	plt.xlim(0,750)
	# plt.xscale('log')

	plt.legend(loc=4)
	plt.title(r'\textbf{Performance vs Training Episodes}')
	plt.xlabel(r'\textbf{Number of Training Episodes}')
	plt.ylabel(r'\textbf{Fraction of Environments Solved}')
	plt.tight_layout(pad=0)
	plt.savefig('Dis_Ep_Gen.pdf')
	plt.show()


def plot_ep_gen_og(N = 50,alpha = .5,thresh = 10):

	font_settings = {
    "text.usetex": True,
    "font.serif": ["Times New Roman"],
    "font.family":"serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "figure.titlesize":8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "legend.title_fontsize":6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
	}
	sns.set(context="paper",style="whitegrid",font="serif",rc=font_settings)

	dual_means = np.load("ep_gen_dual_means.npy")
	dual_vars = np.load("ep_gen_dual_variances.npy")
	dual_eps = np.load("ep_gen_dual_eps.npy")

	mb_means = np.load("ep_gen_mb_means.npy")
	mb_vars = np.load("ep_gen_mb_variances.npy")
	mb_eps = np.load("ep_gen_mb_eps.npy")

	mf_means = np.load("ep_gen_mf_means.npy")
	mf_vars = np.load("ep_gen_mf_variances.npy")
	mf_eps = np.load("ep_gen_mf_eps.npy")

	fig = plt.figure(figsize=(1.8*2,1.5*2))
	plot_var(dual_eps,dual_means,dual_vars,'cornflowerblue',r'\textbf{Memory-Based (Ours)}',N=N,alpha=alpha)
	plot_var(mb_eps,mb_means,mb_vars,'lightgreen',r'\textbf{Deep Model-Free}',N=N,alpha=alpha)
	plot_var(mf_eps,mf_means,mf_vars,'lightcoral',r'\textbf{Deep Model-Based}',N=N,alpha=alpha)

	# plt.plot(dual_eps,dual_thresh,c='cornflowerblue',label=r'\textbf{Memory-Based (Ours)}')
	# plt.plot(mf_eps,mf_thresh,c='lightgreen',label=r'\textbf{Deep Model-Free}')
	# plt.plot(mb_eps,mb_thresh,c='lightcoral',label=r'\textbf{Deep Model-Based}')

	plt.ylim(-100,0)
	plt.xlim(1,750)
	# plt.xscale('log')

	plt.legend()
	plt.title(r'\textbf{Performance vs Training Episodes}')
	plt.xlabel(r'\textbf{Number of Training Episodes}')
	plt.ylabel(r'\textbf{Performance on Unseen Environments}')
	plt.tight_layout(pad=0)
	plt.savefig('Ep_Gen_og.pdf')
	plt.show()

def plot_real(N = 50,alpha = .5,thresh = 10):

	font_settings = {
    "text.usetex": True,
    "font.serif": ["Times New Roman"],
    "font.family":"serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "figure.titlesize":8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "legend.title_fontsize":6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
	}
	sns.set(context="paper",style="whitegrid",font="serif",rc=font_settings)


	fig = plt.figure(figsize=(1.8*2,1.5*2))
	plt.plot(real_dual_eps,real_dual_scores,c='cornflowerblue',label=r'\textbf{Memory-Based (Ours)}')
	plt.plot(real_mf_eps,real_mf_scores,c='lightgreen',label=r'\textbf{Deep Model-Free}')
	plt.plot(real_mb_eps,real_mb_scores,c='lightcoral',label=r'\textbf{Deep Model-Based}')
	# plt.plot(dual_eps,dual_thresh,c='cornflowerblue',label=r'\textbf{Memory-Based (Ours)}')
	# plt.plot(mf_eps,mf_thresh,c='lightgreen',label=r'\textbf{Deep Model-Free}')
	# plt.plot(mb_eps,mb_thresh,c='lightcoral',label=r'\textbf{Deep Model-Based}')

	plt.ylim(-30,0)
	plt.xlim(0,5)
	# plt.xscale('log')

	plt.legend()
	plt.title(r'\textbf{Performance vs Training Hours}')
	plt.xlabel(r'\textbf{Training Hours}')
	plt.ylabel(r'\textbf{Performance on Unseen Environment}')
	plt.tight_layout(pad=0)
	plt.savefig('Real_gen.pdf')
	plt.show()


def plot_env_gen(N = 50,alpha = .5,thresh=13):

	font_settings = {
    "text.usetex": True,
    "font.serif": ["Times New Roman"],
    "font.family":"serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "figure.titlesize":8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "legend.title_fontsize":6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
	}
	sns.set(context="paper",style="whitegrid",font="serif",rc=font_settings)

	dual_scores = np.load("env_dual_scores.npy")*-1
	dual_thresh = np.sum(dual_scores <= thresh,axis=1).flatten()/150

	mf_scores = np.load("env_mf_scores.npy")*-1
	mf_thresh = np.sum(mf_scores <= thresh,axis=1).flatten()/150

	mb_scores = np.load("env_mb_scores.npy")*-1
	mb_thresh = np.sum(mb_scores <= thresh,axis=1).flatten()/150
	print(mb_thresh)

	fig = plt.figure(figsize=(1.8*2,1.5*2))
	plt.plot(np.arange(10),dual_thresh,c='cornflowerblue',label=r'\textbf{Memory-Based (Ours)}')
	plt.plot(np.arange(1,10),mf_thresh,c='lightgreen',label=r'\textbf{Deep Model-Free}')
	plt.plot(np.arange(1,10),mb_thresh,c='lightcoral',label=r'\textbf{Deep Model-Based}')


	plt.ylim(0,1.1)
	# plt.xlim(0,750)

	plt.legend()
	plt.title(r'\textbf{Performance vs Training Environments}')
	plt.xlabel(r'\textbf{Number of Training Environments}')
	plt.ylabel(r'\textbf{Fraction of Environments Solved}')
	plt.tight_layout(pad=0)

	plt.savefig('Env_Gen.pdf')
	plt.show()


def plot_env_gen_og(N = 50,alpha = .5,thresh=13):

	font_settings = {
    "text.usetex": True,
    "font.serif": ["Times New Roman"],
    "font.family":"serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "figure.titlesize":8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "legend.title_fontsize":6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
	}
	sns.set(context="paper",style="whitegrid",font="serif",rc=font_settings)

	dual_means = np.load("env_gen_dual_means.npy")
	dual_vars = np.load("env_gen_dual_variances.npy")

	mb_means = np.load("env_gen_mb_means.npy")
	mb_vars = np.load("env_gen_mb_variances.npy")

	mf_means = np.load("env_gen_mf_means.npy")
	mf_vars = np.load("env_gen_mf_variances.npy")
	print(mb_means)

	fig = plt.figure(figsize=(1.8*2,1.5*2))
	plot_var(np.arange(10),dual_means,dual_vars,'cornflowerblue',r'\textbf{Memory-Based (Ours)}',N=N,alpha=alpha)
	plot_var(np.arange(10),mb_means,mb_vars,'lightgreen',r'\textbf{Deep Model-Free}',N=N,alpha=alpha)
	plot_var(np.arange(10),mf_means,mf_vars,'lightcoral',r'\textbf{Deep Model-Based}',N=N,alpha=alpha)


	plt.ylim(-100,0)
	plt.xlim(1,9)

	plt.legend()
	plt.title(r'\textbf{Performance vs Num Training Environments}')
	plt.xlabel(r'\textbf{Number of Training Environments}')
	plt.ylabel(r'\textbf{Performance on Unseen Environments}')

	plt.tight_layout(pad=0)

	plt.savefig('Env_Gen_og.pdf')
	plt.show()

def plot_noise(N = 50,width=1,thresh = 12):

	font_settings = {
    "text.usetex": True,
    "font.serif": ["Times New Roman"],
    "font.family":"serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "figure.titlesize":8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "legend.title_fontsize":6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
	}
	sns.set(context="paper",style="whitegrid",font="serif",rc=font_settings)
	dual_scores = np.load("noise_dual_scores.npy")*-1
	dual_thresh = np.sum(dual_scores < thresh,axis=1).flatten()/150


	mf_scores = np.load("noise_mf_scores.npy")*-1
	mb_thresh = np.sum(mf_scores < thresh,axis=1).flatten()/150


	mb_scores = np.load("noise_mb_scores.npy")*-1
	mb_thresh = np.sum(mb_scores < thresh,axis=1).flatten()/150


	dual_inds = np.arange(0,9,4)
	mf_inds = np.arange(0,9,4)+1
	mb_inds = np.arange(0,9,4)+2

	fig = plt.figure(figsize=(1.8*2,1.5*2))

	plot_bar(dual_inds,dual_thresh,'cornflowerblue',r'\textbf{Memory-Based (Ours)}',N=N,width=width)
	plot_bar(mf_inds,mb_thresh,'lightgreen',r'\textbf{Deep Model-Free}',N=N,width=width)
	plot_bar(mb_inds,mb_thresh,'lightcoral',r'\textbf{Deep Model-Based}',N=N,width=width)

	plt.ylim(0,1.1)
	ax = plt.gca()
	ax.set_title(r'\textbf{Agent Performance on Noisy Environments}')
	ax.set_xticks(np.arange(1,10,4))
	ax.set_xticklabels([r'\textbf{Incorrect Component}', r'\textbf{Noisy Location}', r'\textbf{Both}'])

	plt.legend()
	plt.xlabel(r'\textbf{Agent}')
	plt.ylabel(r'\textbf{Fraction of Environments Solved}')
	plt.tight_layout(pad=0)
	plt.savefig('Noisy_envs.pdf')
	plt.show()

def plot_noise_og(N = 50,width=1,thresh = 12):

	font_settings = {
    "text.usetex": True,
    "font.serif": ["Times New Roman"],
    "font.family":"serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "figure.titlesize":8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "legend.title_fontsize":6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
	}
	sns.set(context="paper",style="whitegrid",font="serif",rc=font_settings)

	dual_means = np.load("noise_dual_means.npy")*-1

	mb_means = np.load("noise_mb_means.npy")*-1

	mf_means = np.load("noise_mf_means.npy")*-1


	dual_inds = np.arange(0,9,4)
	mf_inds = np.arange(0,9,4)+1
	mb_inds = np.arange(0,9,4)+2

	fig = plt.figure(figsize=(1.8*2,1.5*2))

	plot_bar(dual_inds,dual_means,'cornflowerblue',r'\textbf{Memory-Based (Ours)}',N=N,width=width)
	plot_bar(mf_inds,mb_means,'lightgreen',r'\textbf{Deep Model-Free}',N=N,width=width)
	plot_bar(mb_inds,mf_means,'lightcoral',r'\textbf{Deep Model-Based}',N=N,width=width)

	# plt.ylim(0,1.1)
	ax = plt.gca()
	ax.set_title(r'\textbf{Agent Performance on Noisy Environments}')
	ax.set_xticks(np.arange(1,10,4))
	ax.set_xticklabels([r'\textbf{Incorrect Component}', r'\textbf{Noisy Location}', r'\textbf{Both}'])

	plt.legend()
	plt.xlabel(r'\textbf{Agent}')
	plt.ylabel(r'\textbf{Number Actions to Solve (Lower is Better)}')
	plt.tight_layout(pad=0)
	plt.savefig('Noisy_envs_og.pdf')
	plt.show()








	



def main():
	# dual_scores1 = test_env('dual_agent','test',[9],3,50,corrupt_cids=True,add_pos_noise=False)
	# dual_scores2 = test_env('dual_agent','test',[9],3,50,corrupt_cids=False,add_pos_noise=True)
	# dual_scores3 = test_env('dual_agent','test',[9],3,50,corrupt_cids=True,add_pos_noise=True)

	# dual_scores = np.vstack((dual_scores1,dual_scores2,dual_scores3))
	# np.save("noise_dual_scores.npy",dual_scores)

	# mf_scores1 = test_env('mf_agent','test',[9],3,50,corrupt_cids=True,add_pos_noise=False)
	# mf_scores2 = test_env('mf_agent','test',[9],3,50,corrupt_cids=False,add_pos_noise=True)
	# mf_scores3 = test_env('mf_agent','test',[9],3,50,corrupt_cids=True,add_pos_noise=True)

	# mf_scores = np.vstack((mf_scores1,mf_scores2,mf_scores3))
	# np.save("noise_mf_scores.npy",mf_scores)

	# mb_scores1 = test_env('mb_agent','test',[9],3,50,corrupt_cids=True,add_pos_noise=False)
	# mb_scores2 = test_env('mb_agent','test',[9],3,50,corrupt_cids=False,add_pos_noise=True)
	# mb_scores3 = test_env('mb_agent','test',[9],3,50,corrupt_cids=True,add_pos_noise=True)

	# mb_scores = np.concatenate((mb_scores1,mb_scores2,mb_scores3))
	# np.save("noise_mb_scores.npy",mb_scores)

	# dual_env_scores = test_env('dual_agent','test',np.arange(10),3,50)
	# np.save("env_dual_scores.npy",dual_env_scores)
	# mf_env_scores = test_env('mf_agent','test',np.arange(1,10),3,50)
	# np.save("env_mf_scores.npy",mf_env_scores)
	# mb_env_scores = test_env('mb_agent','test',np.arange(1,10),3,50)
	# np.save("env_mb_scores.npy",mb_env_scores)





	# dual_scores, dual_eps = test_ep_gen('dual_agent',50,dual_agent_max_tests=20)
	# np.save("dis_ep_gen_dual_scores.npy",dual_scores)
	# np.save("dis_ep_gen_dual_eps.npy",dual_eps)
	
	# mf_scores, mf_eps = test_ep_gen('mf_agent',50)
	# np.save("MTOPT_ep_gen_mf_scores.npy",mf_scores)
	# np.save("MTOPT_ep_gen_mf_eps.npy",mf_eps)

	# mb_scores, mb_eps = test_ep_gen('mb_agent',50)
	# np.save("MTOPT_ep_gen_mb_scores.npy",mb_scores)
	# np.save("MTOPT_ep_gen_mb_eps.npy",mb_eps)

	# mb_scores = np.load("dis_ep_gen_dual_scores.npy")*-1
	# print(mb_scores.shape)
	# for scores in mb_scores:
	# 	print(scores)

	# print(np.sum(mb_scores <= 22,axis=1).flatten()/50)
	# # 1/0

	plot_comp_ep_gen(thresh = 24)
	# plot_ep_gen(thresh = 15)

	# plot_real()

	# plot_env_gen_og(thresh = 15)
	# plot_noise_og(thresh = 15)

	# plot_env_gen()



	# run_experiment(12,100,save_data = True)
	# plot_data()
	# agent = load_mb_agent(9)
	# scores = eval_agent(agent,'mb_agent','test',1,1,rl_agent_eps = .1)
	# print(scores)

if __name__ == "__main__":
	main()






