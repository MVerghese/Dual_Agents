import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import PuzzleBoxEnv


BATCH_SIZE = 16
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

	def __init__(self, capacity):
		self.memory = deque([],maxlen=capacity)

	def push(self, *args):
		"""Save a transition"""
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

# class DQN(nn.Module):

#     def __init__(self, inputs, outputs):
#         super(DQN, self).__init__()
#         self.linear1 = nn.Linear(inputs,16)
#         self.bn1 = nn.BatchNorm1d(16)
#         self.linear2 = nn.Linear(16, 32)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.linear3 = nn.Linear(32, 32)
#         self.bn3 = nn.BatchNorm1d(32)
#         self.linear4 = nn.Linear(32,outputs)

		

#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = x.to(device)
#         x = F.relu(self.bn1(self.linear1(x)))
#         x = F.relu(self.bn2(self.linear2(x)))
#         x = F.relu(self.bn3(self.linear3(x)))
#         return self.linear4(x)

class DQN(nn.Module):
	def __init__(self, inputs, outputs):
		super(DQN, self).__init__()
		self.params = nn.Parameter(torch.empty(32, 5),requires_grad=True)

	def forward(self,x):
		weights = np.array([2**i for i in range(0,5)])
		idx = np.dot(x.numpy(),weights)
		# print(self.params[idx])
		return self.params[idx]

def plot_durations(episode_durations):
	plt.figure(2)
	plt.clf()
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())
	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())

def select_action(state, policy_net, n_actions, steps_done):
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			policy_net.eval()
			state = state.view(1,5)
			# print(state)

			return (policy_net(state).max(1)[1].view(1, 1),steps_done)
	else:
		return (torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long),steps_done)


def optimize_model(memory,policy_net,target_net,optimizer):
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), device=device, dtype=torch.bool)

	non_final_next_states = torch.cat([s for s in batch.next_state
												if s is not None])
	non_final_next_states = non_final_next_states.view(-1,5)

	state_batch = torch.cat(batch.state)
	state_batch = state_batch.view(-1,5)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	# print(state_batch.shape)
	# print(state_batch.view(-1,5))

	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	criterion = nn.SmoothL1Loss()
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

def train():
	env = PuzzleBoxEnv.LockEnv('original',5,2)
	env.reset()
	n_actions = 5

	policy_net = DQN(n_actions,n_actions).to(device)
	target_net = DQN(n_actions,n_actions).to(device)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	optimizer = optim.RMSprop(policy_net.parameters())
	memory = ReplayMemory(10000)

	steps_done = 0


	episode_durations = []


	num_episodes = 2000
	for i_episode in range(num_episodes):
		# Initialize the environment and state
		
		_, state, _ = env.reset()
		# print(state)
		state = torch.from_numpy(state).float()
		eps_reward = 0
		for t in count():
			# Select and perform an action
			action, steps_done = select_action(state, policy_net, n_actions, steps_done)
			policy_net.train()
			_, state, reward, done = env.step(action.item())
			state = torch.from_numpy(state).float()
			eps_reward += reward
			reward = torch.tensor([reward], device=device)

			# Observe new state
			if not done:
				next_state = state
			else:
				next_state = None

			# Store the transition in memory
			memory.push(state, action, next_state, reward)

			# Move to the next state
			state = next_state

			# Perform one step of the optimization (on the policy network)

			optimize_model(memory,policy_net,target_net,optimizer)
			if done:
				episode_durations.append(eps_reward)
				plot_durations(episode_durations)
				break
		# print(eps_reward)
		# print(i_episode)
		if len(episode_durations) < 100:
			print(np.mean(np.array(episode_durations)))
		else:
			print(np.mean(np.array(episode_durations)[-100:]))
		# Update the target network, copying all weights and biases in DQN
		if i_episode % TARGET_UPDATE == 0:
			target_net.load_state_dict(policy_net.state_dict())
			for name, param in policy_net.named_parameters():
			    if param.requires_grad:
			        print(name, param.data)

	print('Complete')
	# env.render()
	# env.close()
	plt.ioff()
	plt.show()

if __name__ == '__main__':
	train()
	


