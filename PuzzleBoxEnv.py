import numpy as np
import time
from matplotlib import pyplot as plt

import gym
from gym import spaces

door_rad = 2
bar_rad = 4
wheel_rad = 4
wheel_locked_rad = 2

def SE2(xytheta):
	T = np.eye(3)
	R = np.array([[np.cos(xytheta[2]),-np.sin(xytheta[2])],[np.sin(xytheta[2]),np.cos(xytheta[2])]])
	T[:2,:2] = R
	T[:2,2] = xytheta[:2]
	return(T)

def generate_component_locations(cids):
	component_poses = np.zeros((cids.shape[0],3,3))
	# component_pose_ids = np.zeros(cids.shape[0])
	component_poses[0,:,:] = np.eye(3)
	for i in range(1,cids.shape[0]):
		if cids[i-1] == 0:
			if np.random.uniform() > .5:
				component_poses[i,:,:] = np.dot(component_poses[i-1,:,:],SE2(np.array([0,-door_rad-bar_rad,0])))
			else:
				component_poses[i,:,:] = np.dot(component_poses[i-1,:,:],SE2(np.array([0,door_rad+bar_rad,np.pi])))
		elif cids[i-1] == 1:
			if cids[i] == 0:
				component_poses[i,:,:] = np.dot(component_poses[i-1,:,:],SE2(np.array([0,-door_rad-bar_rad,np.pi/2])))		
			elif cids[i] == 2:
				component_poses[i,:,:] = np.dot(component_poses[i-1,:,:],SE2(np.array([0,-wheel_locked_rad-bar_rad,0])))
		else:
			if np.random.uniform() > .5:
				component_poses[i,:,:] = np.dot(component_poses[i-1,:,:],SE2(np.array([-wheel_rad-bar_rad,0,-np.pi/2])))
			else:
				component_poses[i,:,:] = np.dot(component_poses[i-1,:,:],SE2(np.array([wheel_rad+bar_rad,0,np.pi/2])))
			



	component_locations = np.zeros((cids.shape[0],4))
	for i in range(cids.shape[0]):
		# print(component_poses[i,:,:])
		# print(component_poses[i,:,:])
		component_locations[i,:] = np.concatenate((component_poses[i,:2,2],component_poses[i,1,:2]))
	return(component_locations)

config_dict = {}

original = np.zeros((5,2,5,2))
original[0,0,1,0] = 1.
original[1,1,0,1] = 1.
original[1,0,2,0] = 1.
original[2,1,1,1] = 1.
original[2,0,3,0] = 1.
original[3,1,2,1] = 1.
original[3,0,4,0] = 1.
original[4,1,3,1] = 1.

original_ids = np.array([0,1,2,3,4])
original_cids = np.array([0,1,2,1,0])
config_dict['original'] = (original,original_ids,original_cids,generate_component_locations(original_cids))

TT1L0M = np.zeros((5,2,5,2))
TT1L0M[0,0,1,0] = 1.
TT1L0M[1,1,0,1] = 1.
TT1L0M[1,0,2,0] = 1.
TT1L0M[2,1,1,1] = 1.
TT1L0M[2,0,3,0] = 1.
TT1L0M[3,1,2,1] = 1.
TT1L0M_ids = np.array([1,2,3,4])
TT1L0M_cids = np.array([1,2,1,0])
config_dict['TT1L0M'] = (TT1L0M,TT1L0M_ids,TT1L0M_cids,generate_component_locations(TT1L0M_cids))

TT1L1M = np.zeros((5,2,5,2))
TT1L1M[1,0,2,0] = 1.
TT1L1M[2,1,1,1] = 1.
TT1L1M[2,0,3,0] = 1.
TT1L1M[3,1,2,1] = 1.
TT1L1M_ids = np.array([0,2,3,4])
TT1L1M_cids = np.array([0,2,1,0])
config_dict['TT1L1M'] = (TT1L1M,TT1L1M_ids,TT1L1M_cids,generate_component_locations(TT1L1M_cids))

TT1L2M = np.zeros((5,2,5,2))
TT1L2M[0,0,1,0] = 1.
TT1L2M[1,1,0,1] = 1.
TT1L2M[2,0,3,0] = 1.
TT1L2M[3,1,2,1] = 1.
TT1L2M_ids = np.array([0,1,3,4])
TT1L2M_cids = np.array([0,1,1,0])
config_dict['TT1L2M'] = (TT1L2M,TT1L2M_ids,TT1L2M_cids,generate_component_locations(TT1L2M_cids))

TT1L3M = np.zeros((5,2,5,2))
TT1L3M[0,0,1,0] = 1.
TT1L3M[1,1,0,1] = 1.
TT1L3M[1,0,2,0] = 1.
TT1L3M[2,1,1,1] = 1.
TT1L3M_ids = np.array([0,1,2,4])
TT1L3M_cids = np.array([0,1,2,0])
config_dict['TT1L3M'] = (TT1L3M,TT1L3M_ids,TT1L3M_cids,generate_component_locations(TT1L3M_cids))

TT1s = ['TT1L0M', 'TT1L1M', 'TT1L2M', 'TT1L3M']





# TT2OG = np.zeros((5,2,5,2))
# TT2OG[0,0,1,0] = 1.
# TT2OG[1,1,0,1] = 1.
# TT2OG[1,0,2,0] = 1.
# TT2OG[2,1,1,1] = 1.
# TT2OG[2,0,3,0] = 1.
# TT2OG[3,1,2,1] = 1.
# TT2OG[3,0,4,0] = 1.
# TT2OG[4,1,3,1] = 1.

# TT2OG_ids = np.array([0,1,2,3,4])
# TT2OG_cids = np.array([0,1,2,1,0])
# config_dict['TT2OG'] = (TT2OG,TT2OG_ids,TT2OG_cids)
env_list = [[0, 1, 0, 1, 2],
			[0, 1, 0, 1, 0],  
			[0, 1, 2, 1, 0], 
			[0, 1, 2, 1, 2], 
			[1, 0, 1, 0, 1], 
			[1, 0, 1, 2, 1], 
			[1, 2, 1, 0, 1], 
			[1, 2, 1, 2, 1], 
			[2, 1, 0, 1, 0], 
			[2, 1, 0, 1, 2], 
			[2, 1, 2, 1, 0], 
			[2, 1, 2, 1, 2]]
env_list = np.array(env_list)

train_envs = env_list[[0,1,2,4,5,7,8,10,11],:]
test_envs = env_list[[3,6,9]]

def generate_puzzle_box(num_components,cids = None):
	if np.any(cids):
		num_components = cids.shape[0]
	else:
		cids = np.zeros(num_components,dtype=int)
		cids[0] = np.random.choice(np.arange(3))
		for i in range(1,num_components):
			if cids[i-1] == 0:
				choices = np.array([1])
			if cids[i-1] == 1:
				choices = np.array([0,2])
			if cids[i-1] == 2:
				choices = np.array([1])
			cids[i] = np.random.choice(choices)

	config = np.zeros((num_components,2,num_components,2))
	config[0,0,1,0] = 1.
	for i in range(1,num_components-1):
		config[i,1,i-1,1] = 1.
		config[i,0,i+1,0] = 1.
	config[num_components-1,1,num_components-2,1] = 1.

	return((config,np.arange(num_components),cids,generate_component_locations(cids)))



def SO2(theta):
	R = np.array([[np.cos(xytheta[2]),-np.sin(xytheta[2])],[np.sin(xytheta[2]).np.cos(xytheta[2])]])
	return(R)




				
			


def enumerate_envs(env_list,env):
	if len(env) == 5:
		env_list.append(env)
	else:
		if env[-1] == 0:
			new_env = env[:]
			new_env.append(1)
			enumerate_envs(env_list,new_env)
		if env[-1] == 1:
			new_env = env[:]
			new_env.append(0)
			enumerate_envs(env_list,new_env)
			new_env = env[:]
			new_env.append(2)
			enumerate_envs(env_list,new_env)
		if env[-1] == 2:
			new_env = env[:]
			new_env.append(1)
			enumerate_envs(env_list,new_env)

			


def generate_lock_sequence(num_components, num_states, mode = 'random', env_index = 0):
	if mode == 'random':
		# state_config = np.zeros(num_components-1,dtype=int)*(num_states-1)
		# config = np.zeros((num_components ,num_states, num_components, num_states))
		# order = np.arange(0,num_components-1,1)
		# np.random.shuffle(order)
		# print(order)
		# # config[order[0],num_components] = 1
		# for i in range(1,order.shape[0]):
		# 	for j in range(num_states):
		# 		config[order[i-1],state_config[i],order[i],j] = 1.
		# for j in range(num_states):
		# 	config[order[-1],state_config[-1],num_components-1,j] = 1.
		# # config[num_components-1,num_components] = 1
		environment = generate_puzzle_box(num_components)
	elif mode == 'train':
		return(generate_puzzle_box(num_components,train_envs[env_index]))
	elif mode == 'test':
		return(generate_puzzle_box(num_components,test_envs[env_index]))
	elif mode == 'all':
		return(generate_puzzle_box(num_components,env_list[env_index]))
	else:
		environment = config_dict[mode]



	return(environment)

class LockEnv():
	def __init__(self,config_mode,num_components,num_states,env_index = 0, timeout=200,return_state_mode = 'dual', randomize_config = False):
		self.config, self.ids, self.cids, self.component_locations = generate_lock_sequence(num_components,num_states,config_mode,env_index=env_index)
		# self.component_locations = self.component_locations.T
		self.num_components = num_components
		self.goal_state = num_components - 1
		if randomize_config:
			rand_sequence = np.arange(num_components)
			np.random.shuffle(rand_sequence)
			self.config = self.config[rand_sequence,:,:,:]
			self.config = self.config[:,:,rand_sequence,:]
			self.ids = self.ids[rand_sequence]
			self.cids = self.cids[rand_sequence]
			self.goal_state = np.where(rand_sequence == num_components - 1)[0][0]
			self.component_locations = self.component_locations[rand_sequence,:]

		self.goal_state_rep = np.zeros(num_components)
		self.goal_state_rep[self.goal_state] = 1

		self.current_state = np.zeros((num_components),dtype=int)
		self.counter = 0
		self.timeout = timeout
		self.return_state_mode = return_state_mode
		self.state_history = np.zeros((num_components,num_components),dtype=int)
		self.compute_component_distances()
		

	def get_object_ids(self):
		return(self.cids)

	def get_goal_state(self):
		return(self.goal_state)

	def get_component_locations(self):
		return(self.component_locations)

	def compute_component_distances(self):
		ixs = []
		for i in range(self.num_components):
			for j in range(self.num_components):
				if i != j:
					ixs.append([i, j])
		# distances = np.zeros((ixs.shape[0],4))
		ixs = np.array(ixs)
		# print(ixs.shape)
		self.distances = self.component_locations[ixs[:,0],:] - self.component_locations[ixs[:,1],:]

	def reset(self):
		self.current_state = np.zeros((self.current_state.shape[0]),dtype=int)
		self.counter = 0
		self.state_history = np.zeros((self.current_state.shape[0],self.current_state.shape[0]),dtype=int)
		if self.return_state_mode == 'mf':
			return(False,np.concatenate((np.copy(self.state_history).flatten(),self.cids,self.goal_state_rep,self.distances.flatten())),0,False)
		elif self.return_state_mode == 'mb':
			return(False,np.concatenate((np.copy(self.current_state),self.cids,self.distances.flatten())),0,False)
		else:
			return(False,np.copy(self.current_state),0,False)

	def step(self,action):
		# print(self.config)
		# print(self.current_state[action])
		rel_config = self.config[:,:,:,int(self.current_state[action])]
		success = np.all(rel_config[np.arange(0,self.current_state.shape[0]),self.current_state,np.ones(self.current_state.shape[0],dtype=int)*action] != 1)
		self.counter += 1
		if success:
			self.current_state[action] = self.current_state[action]*-1 + 1

		self.state_history[np.arange(0,self.current_state.shape[0]-1),:] = self.state_history[np.arange(1,self.current_state.shape[0]),:]
		self.state_history[-1,:] = self.current_state
		
		if bool(self.current_state[self.goal_state]):
			reward = -1
			done = True
		elif self.counter > self.timeout:
			done = True
			reward = -1 
		else:
			done = False
			reward = -1

		if self.return_state_mode == 'mf':
			return(success,np.concatenate((np.copy(self.state_history).flatten(),self.cids,self.goal_state_rep,self.distances.flatten())),reward,done)
		elif self.return_state_mode == 'mb':
			return(success,np.concatenate((np.copy(self.current_state),self.cids,self.distances.flatten())),reward,done)
		else:
			return(success,np.copy(self.current_state),reward,done)

	def render(self):
		pass

class EnvWrapper:
	def __init(self,train):
		self.action_space = spaces.Discrete(num_components)

		self.observation_space = spaces.Discrete(115)


def main():
	# env_list = []
	# enumerate_envs(env_list,[0])
	# enumerate_envs(env_list,[1])
	# enumerate_envs(env_list,[2])
	# print(env_list)

	env = LockEnv('train',5,2,env_index=0,randomize_config = True)
	# print(env.config[:,0,:,0])
	# print(env.get_goal_state())
	# print(env.cids)
	# print(env.component_locations)
	# print(env.distances)
	component_locations = generate_component_locations(np.array([1,2,1,0,1]))
	print(component_locations)
	locs = component_locations
	c = ['r','g','b','y','k']
	for i in range(5):
		plt.scatter(locs[i,0],locs[i,1],color=c[i])
	plt.show()
	# print(SE2(np.array([0,-door_rad-bar_rad,0])))

if __name__ == '__main__':
	main()