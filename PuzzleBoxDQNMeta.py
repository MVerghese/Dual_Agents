import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random 
from collections import namedtuple, deque 

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv, GINConv, MetaLayer
from torch_geometric.data import Data
from torch_geometric.data import Batch

import PuzzleBoxEnv
from LockBeliefSpaceFixed import compute_q, argmax

from matplotlib import pyplot as plt 

# from torch_geometric.datasets import TUDataset
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# data = dataset
# print(data)
# 1/0

#class MLP(nn.Module):
    #def __init__(self, layers):
        #self.net = nn.Sequential(*[nn.Sequential(nn.Linear(layers[i])) for i in range(1, len(layers))])

BUFFER_SIZE = int(1e5)  #replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

LR = 5e-4               # learning rate
ALPHA = 1e-3
# ALPHA = 1
# TAU = LR
UPDATE_EVERY = 2        # how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class EdgeModel(torch.nn.Module):
    def __init__(self,F_x=32,F_e=32,out_dim=64):
        super().__init__()
        self.edge_mlp = Seq(Lin(2*F_x + F_e, 64), ReLU(), Lin(64, out_dim))

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        # print('edge model',out.shape)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self,F_x=32,F_e=32,out_dim=64):
        super().__init__()
        self.node_mlp_1 = Seq(Lin(F_x + F_e, 64), ReLU(), Lin(64, 64))
        self.node_mlp_2 = Seq(Lin(F_x + 64, 64), ReLU(), Lin(64, out_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        # print(x[row].shape)
        # print(edge_attr.shape)

        out = torch.cat([x[row], edge_attr], dim=1)
        # print('node model 1',out.shape)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # print('node model 2',out.shape)
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)



class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size,action_size, seed, fc1_unit=64,
                 fc2_unit = 64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        # self.fc1= nn.Linear(state_size,fc1_unit)
        # self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        # self.fc3 = nn.Linear(fc2_unit,fc2_unit)
        # self.fc4 = nn.Linear(fc2_unit,action_size)
        # self.conv1 = GCNConv(7, 64)
        # self.conv2 = GCNConv(64, 64)
        # self.conv5 = GCNConv(64, 1)
        # self.nn1 = MLPA()
        # self.conv1 = GINConv(self.nn1, eps=2.0)
        # self.nn2 =MLPA(indim=64)
        # self.conv2 = GINConv(self.nn2, eps=2.0)
        # self.nn3 =MLPA(indim=64)
        # self.conv3 = GINConv(self.nn3, eps=2.0)
        # self.nn4 =MLPA(indim=64)
        # self.conv4 = GINConv(self.nn4, eps=2.0)
        # self.nn5 =MLPA(indim=64)
        # self.conv5 = GINConv(self.nn5, eps=2.0)
        # self.nn6 =MLPA(indim=64)
        # self.conv6 = GINConv(self.nn6, eps=2.0)
        # self.nn7 = MLPB()
        # F_x = 2, F_e = 4
        self.conv1 = MetaLayer(EdgeModel(F_x = 7, F_e = 4),NodeModel(F_x = 7, F_e = 64))
        # F_x = 64, F_e = 64
        self.conv2 = MetaLayer(EdgeModel(F_x = 64, F_e = 64),NodeModel(F_x = 64, F_e = 64))
        # F_x = 64, F_e = 64
        self.conv3 = MetaLayer(EdgeModel(F_x = 64, F_e = 64),NodeModel(F_x = 64, F_e = 64))
        self.conv4 = MetaLayer(EdgeModel(F_x = 64, F_e = 64),NodeModel(F_x = 64, F_e = 64))

        self.conv5 = MetaLayer(EdgeModel(F_x = 64, F_e = 64),NodeModel(F_x = 64, F_e = 64,out_dim=1))


        





        
    def forward(self,batch):
        
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        # print(x.shape)
        # print(edge_index.shape)
        # print(batch.num_graphs)
        # import pdb; pdb.set_trace()
        # x = state
        """
        Build a network that maps state -> action values.
        """
        #x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        # x = F.relu(self.conv4(x, edge_index))
        #x = F.relu(self.conv5(x, edge_index))

        # x = x.view(x.shape[0],x.shape[1])
        # print(x.shape)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = torch.sigmoid(self.fc4(x))
        # print(x.shape)
        # print(edge_attr.shape)
        x, edge_attr, _ = self.conv1(x, edge_index, edge_attr)
        # print(x)
        x, edge_attr, _ = self.conv2(x, edge_index, edge_attr)
        # print()
        x, edge_attr, _ = self.conv3(x, edge_index, edge_attr)

        x, edge_attr, _ = self.conv4(x, edge_index, edge_attr)

        x, edge_attr, _ = self.conv5(x, edge_index, edge_attr)


        # x = x
        # print(x)

        # import pdb; pdb.set_trace()

        # print(x)
        return x

class Agent():
    """Interacts with and learns form environment."""
    
    def __init__(self, state_size, action_size, seed, meta_learn_k = 1, num_tasks = 1, savepath='checkpoint.pth',load=False, loadpath='checkpoint.pth'):
        """Initialize an Agent object.
        
        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.seed = seed
        self.meta_learn_k = meta_learn_k
        self.num_tasks = num_tasks
        
        
        #Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        if load:
            self.qnetwork_local.load_state_dict(torch.load(loadpath,map_location = torch.device("cpu")))
            self.qnetwork_target.load_state_dict(torch.load(loadpath,map_location = torch.device("cpu")))
            self.qnetwork_local.train()
            self.qnetwork_target.train()


        xx, yy = np.meshgrid(np.arange(5),np.arange(5))
        
        edge_index = np.vstack((xx.flatten(),yy.flatten())).astype(int)

        ixs = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    ixs.append([i, j])
        ixs = torch.as_tensor(ixs, dtype=torch.long)
        ixs = ixs.t().contiguous()

        # import pdb; pdb.set_trace()

        # edge_index = np.delete(edge_index,[0,6,12,18,24],1)
        #self.edge_index = torch.from_numpy(edge_index)
        self.edge_index = ixs
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        
        # Replay memory 
        self.memory = []
        for i in range(self.num_tasks):
            self.memory.append(ReplayBuffer(action_size, BUFFER_SIZE,BATCH_SIZE,seed))
        self.ep_len = np.zeros(self.num_tasks,dtype=int)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_step, done, t,task_num):
        # Save experience in replay memory
        # if done:
        #     self.ep_len[task_num] = t
        self.memory[task_num].add(state, action, reward, next_step, done)

        # # Learn every UPDATE_EVERY time steps.
   
        self.t_step = (self.t_step+1)% UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            # if len(self.memory)>BATCH_SIZE:
                # experience = self.memory.sample()
            self.learn(GAMMA)
    def act(self, state, eps = 0):
        """Returns action for given state as per current policy

        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(self.data_graph(state))
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        =======

            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples

            gamma (float): discount factor
        """
        
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        #shape of output from the model (batch_size,action_dim) = (64,4)
        # print(self.qnetwork_local(self.data_graph(states)).shape)
        # import pdb; pdb.set_trace()
        fine_models = []
        for task_num in range(self.num_tasks):
            # print("TASK NUM " + str(task_num))
            
                # print("Targets")
                # print(states.shape)
            if len(self.memory[task_num]) < BATCH_SIZE:
                continue
            

            fine_model = QNetwork(self.state_size, self.action_size, self.seed).to(device)
            fine_model.load_state_dict(self.qnetwork_local.state_dict())
            optimizer = optim.Adam(fine_model.parameters(),lr=LR)
            for i in range(self.meta_learn_k):

                experiences = self.memory[task_num].sample()
                states, actions, rewards, next_state, dones = experiences
                batch_size = len(states)

                predicted_targets = fine_model(self.data_graph(states)).view(batch_size, -1, 1).gather(1, actions.unsqueeze(1))
                
                with torch.no_grad():
                    # import pdb; pdb.set_trace()
                    # labels_next = self.qnetwork_target(self.data_graph(next_state)).detach().max(1)[0].unsqueeze(1)
                    # print("Labels")
                    # print(next_state.shape)
                    labels_next = self.qnetwork_target(self.data_graph(next_state)).detach().view(batch_size, -1, 1).max(dim=1)[0].unsqueeze(1)
                # .detach() ->  Returns a new Tensor, detached from the current graph.

                labels = rewards.unsqueeze(1) + (gamma* labels_next*(1-dones.unsqueeze(1)))

                loss = criterion(predicted_targets,labels).to(device)
                # print(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            fine_models.append(fine_model)
        self.meta_update(self.qnetwork_local,fine_models,ALPHA)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,TAU)

    def meta_update(self, local_model, fine_models, lr):
        for model in fine_models:
            for local_param, fine_param in zip(local_model.parameters(),model.parameters()):
                local_param.data.copy_(local_param + lr/self.num_tasks*(fine_param-local_param))
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter

        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

    def data_graph(self,state):
        # import pdb; pdb.set_trace()
        # return(state)
        # print(state.shape)
        # history = state[:, :25].view(-1, 5, 5).transpose(1, 2)
        # ids = state[:, 25:30].view(-1, 5, 1)
        # goal = state[:, 30:35].view(-1, 5, 1)
        # print(state.shape)
        # print(state.shape)
        xs = state[:,:35]
        edge_attrs = state[:,35:]
        # print(edge_attrs.shape)
        # print(xs.shape)
        xs = xs.view(-1,7,5).transpose(1,2)
        edge_attrs = edge_attrs.view(-1,20,4)
        # xs = torch.cat([history, ids, goal], dim=-1)
        # import pdb;pdb.set_trace()
        # print(x.shape)
        # 1/0
        # import pdb; pdb.set_trace()
        batch = Batch.from_data_list([Data(x=xs[i], edge_index=self.edge_index, edge_attr=edge_attrs[i]) for i in range(xs.shape[0])])
        # data = Data(x=x, edge_index = self.edge_index)
        return(batch)

            
class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)

    def meta_learn_sample(self,ep_len):
        experiences = []
        for i in range(len(self.memory)-ep_len,len(self.memory)):
            experiences.append(self.memory[i])
        # print(len(self.memory))
        # experiences = self.memory[ep_len*-1:]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



def dqn(agent,n_train_envs,n_episodes= 200, max_t = 1000, eps_start=1.0, eps_end = 0.01,
       eps_decay=0.999):
    # print(n_episodes)
    """Deep Q-Learning
    
    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon 
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
        
    """
    scores = [] # list containing score from each episode
    scores_window = deque(maxlen=50) # last 100 scores
    eps = eps_start
    for i_episode in range(1, n_episodes+1):

        env = PuzzleBoxEnv.LockEnv('train',5,2,env_index = (i_episode-1)%n_train_envs,return_state_mode='mf',randomize_config=True)
        _,state,_,_ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            _,next_state,reward,done = env.step(action)
            # print("next_state")
            # print(next_state.shape)
            agent.step(state,action,reward,next_state,done,t,(i_episode-1)%n_train_envs)
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our 
            ## replay buffer.
            state = next_state
            score += reward
            if done:
                break
        # if (i_episode-1)%n_train_envs == 0 and i_episode > 1:
        #     agent.learn(GAMMA)
        scores_window.append(score) ## save the most recent score
        scores.append(score) ## save the most recent score
        eps = max(eps*eps_decay,eps_end)## decrease the epsilon
        print('\rEpisode {}\tAverage Score {:.2f}\t epsilon {:.2f}'.format(i_episode,np.mean(scores_window),eps), end="")
        if i_episode %50==0:
            print('\rEpisode {}\tAverage Score {:.2f}\t epsilon {:.2f}'.format(i_episode,np.mean(scores_window),eps))
            
            torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
                
    return scores

def main():
    num_tasks = 1
    agent = Agent(state_size=35,action_size=5,seed=0,meta_learn_k = 1,num_tasks=num_tasks)
    scores= dqn(agent,num_tasks,n_episodes=2000, eps_end = .2, eps_decay = .99)

    mean_scores = []
    for i in range(100,len(scores)):
        mean_scores.append(np.mean(np.array(scores[i-100:i])))

    #plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)),scores)
    plt.plot(np.arange(100,100+len(mean_scores)),mean_scores)
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    plt.show()

if __name__ == '__main__':
    main()