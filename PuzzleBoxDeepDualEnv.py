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
from LockBeliefSpaceFixed import Structured_Agent_Dist

from matplotlib import pyplot as plt 

BUFFER_SIZE = int(1e5)  #replay buffer size
BATCH_SIZE = 16         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 3e-4               # learning rate
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
        self.conv1 = MetaLayer(EdgeModel(F_x = 2, F_e = 4),NodeModel(F_x = 2, F_e = 64))
        # F_x = 64, F_e = 64
        self.conv2 = MetaLayer(EdgeModel(F_x = 64, F_e = 64),NodeModel(F_x = 64, F_e = 64))
        # F_x = 64, F_e = 64
        self.conv3 = MetaLayer(EdgeModel(F_x = 64, F_e = 64),NodeModel(F_x = 64, F_e = 64))

        self.conv4 = MetaLayer(EdgeModel(F_x = 64, F_e = 64),NodeModel(F_x = 64, F_e = 64))

        self.conv5 = MetaLayer(EdgeModel(F_x = 64, F_e = 64,out_dim=1),NodeModel(F_x = 64, F_e = 64))


        # self.conv3 = MetaLayer(EdgeModel(indim=4),NodeModel(indim=4))
        # self.conv4 = MetaLayer(EdgeModel(indim=4),NodeModel(indim=4))
        # self.conv5 = MetaLayer(EdgeModel(indim=4),NodeModel(indim=4))

        





        
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


        x = torch.sigmoid(edge_attr)
        # print(x)

        # import pdb; pdb.set_trace()

        # print(x)
        return x

class Agent():
    """Interacts with and learns form environment."""
    
    def __init__(self, state_size, action_size, seed, savepath='deepdualcheckpoint.pth',load=False, loadpath='deepdualcheckpoint.pth'):
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
        
        
        #Q- Network
        self.obj_interaction_network = QNetwork(state_size, action_size, seed).to(device)
        if load:
            self.obj_interaction_network.load_state_dict(torch.load(loadpath,map_location = torch.device("cpu")))
            self.obj_interaction_network.train()

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

        # import pdb; pdb.set_trace()

        # edge_index = np.delete(edge_index,[0,6,12,18,24],1)
        #self.edge_index = torch.from_numpy(edge_index)
        # self.edge_index = ixs
        
        self.optimizer = optim.Adam(self.obj_interaction_network.parameters(),lr=LR)
        
        # Replay memory 
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE,BATCH_SIZE,seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        sa = Structured_Agent_Dist(5,dim=2)
        
    def step(self, state, action, success):
        # Save experience in replay memory
        self.memory.add(state, action, success)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory)>BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def neural_action_probs(self,belief_state,state,full_state):
        full_state = torch.from_numpy(full_state).float().unsqueeze(0).to(device)
        self.obj_interaction_network.eval()
        with torch.no_grad():
            action_probs = self.obj_interaction_network(self.data_graph(full_state)).view(5)
        self.obj_interaction_network.train()
        return(action_probs.cpu().data.numpy())

    def act(self, state, goal_state,eps = 0):
        """Returns action for given state as per current policy

        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        """
        

        #Epsilon -greedy action selction
        obj_state = state[:5]
        if random.random() > eps:

            action_values = compute_q(obj_state,[],goal_state,self.neural_action_probs,state)
            # print(obj_state,action_values)

            return argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        =======

            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples

            gamma (float): discount factor
        """
        states, actions, successes = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.obj_interaction_network.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        #shape of output from the model (batch_size,action_dim) = (64,4)
        # print(self.qnetwork_local(self.data_graph(states)).shape)
        # import pdb; pdb.set_trace()
        predicted_targets = self.obj_interaction_network(self.data_graph(states)).view(BATCH_SIZE, -1, 1).gather(1, actions.unsqueeze(1))
        predicted_targets = predicted_targets.squeeze(1)

        # print(predicted_targets.shape)
        # print(predicted_targets)

        loss = criterion(predicted_targets,successes).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # self.soft_update(self.qnetwork_local,self.qnetwork_target,TAU)

    def data_graph(self,state):
        # import pdb; pdb.set_trace()
        # return(state)
        # print(state.shape)
        # history = state[:, :25].view(-1, 5, 5).transpose(1, 2)
        # ids = state[:, 25:30].view(-1, 5, 1)
        # goal = state[:, 30:35].view(-1, 5, 1)
        # print(state.shape)
        xs = state[:,:10]
        edge_attrs = state[:,10:]
        xs = xs.view(-1,2,5).transpose(1,2)
        edge_attrs = edge_attrs.view(-1,20,4)
        # xs = torch.cat([history, ids, goal], dim=-1)
        # import pdb;pdb.set_trace()
        # print(x.shape)
        # 1/0
        # import pdb; pdb.set_trace()
        batch = Batch.from_data_list([Data(x=xs[i], edge_index=self.edge_index, edge_attr=edge_attrs[i]) for i in range(xs.shape[0])])
        # data = Data(x=x, edge_index = self.edge_index)
        batch.to(device)
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
        self.experiences = namedtuple("Experience", field_names=["cids",
                                                               "dists",
                                                               "graph"])
        self.seed = random.seed(seed)
        
    def add(self,cids, dists, graph):
        """Add a new experience to memory."""
        e = self.experiences(cids, dists, graph)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        successes = torch.from_numpy(np.vstack([e.success for e in experiences if e is not None])).float().to(device)
        
        return (states,actions,successes)
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def train_mb_agent(agent,n_train_envs,n_episodes= 200, max_t = 1000, eps_start=1.0, eps_end = 0.01,
       eps_decay=0.99):
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
    total_actions = 0
    for i_episode in range(1, n_episodes+1):

        env = PuzzleBoxEnv.LockEnv('train',5,2,env_index = 2,return_state_mode='mb',randomize_config=True)
        locations = np.array([[24.0653,-7.0341,-1,-0.0239],
                              [18.0313,-7.2389,-1,-0.0230],
                              [10.4267,-6.9789,-1,-0.0187],
                              [11.4360,0.7323,0.0090,-1],
                              [11.4134,7.5497,0.0568,-0.9984]])
        env.set_locations(locations)
        _,state,_,_ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state,env.get_goal_state(),eps)
            success,next_state,reward,done = env.step(action)
            agent.step(state,action,int(success))
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our 
            ## replay buffer.
            state = next_state
            score += reward
            total_actions += 1
            if total_actions %120==0:
                quick_eval(agent,8,eps)
            if done:
                break
            if total_actions > 120*5:
                break
        if total_actions > 120*5:
            break

        scores_window.append(score) ## save the most recent score
        scores.append(score) ## save the most recent score
        eps = max(eps*eps_decay,eps_end)## decrease the epsilon
        print('\rEpisode {}\tAverage Score {:.2f}\t epsilon {:.2f}'.format(i_episode,np.mean(scores_window),eps), end="")
        if i_episode %50==0:
            print('\rEpisode {}\tAverage Score {:.2f}\t epsilon {:.2f}'.format(i_episode,np.mean(scores_window),eps))
            _,state,_,_ = env.reset()
            print(env.get_goal_state())
            state[:5] = 1
            state[env.get_goal_state()] = 0
            print(agent.neural_action_probs([],[],state))
            
            torch.save(agent.obj_interaction_network.state_dict(),'mb_checkpoint.pth')
                
    return scores

def quick_eval(agent,eps,env_index = 8, locations = None):
    total_score = 0
    if not np.any(locations):
        locations = np.array([[23.0993,-0.1519,0.6930,0.7210],
                              [16.5110,-4.7502,0.6710,-0.7415],
                              [12.8176,-9.8099,0.0359,-0.9993],
                              [10.6856,-4.1045,0.0253,-1.0000],
                              [11.1199,1.9743,0.0763,-0.9971],])
    for i in range(5):
        env = PuzzleBoxEnv.LockEnv('train',5,2,env_index = 8,return_state_mode='mb',randomize_config=False)
        
        env.set_locations(locations)
        # env = PuzzleBoxEnv.CompEnv()

        _,state,_,_ = env.reset()
        score = 0
        for t in range(30):
            action = agent.act(state,env.get_goal_state(),eps)
            _,next_state,reward,done = env.step(action)
            # print("next_state")
            # print(next_state.shape)

            # agent.step(state,action,reward,next_state,done,t)
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our 
            ## replay buffer.
            state = next_state
            score += reward
            if done:
                break
        print(t+1)
        total_score += np.min([t+1,30])
    print(total_score/5)

def main():
    agent = Agent(state_size=30,action_size=5,seed=2,loadpath='models/PreconditionModel92000.pth',load=True)
    # quick_eval(agent,.1)
    locations = np.array([[24.0653,-7.0341,-1,-0.0239],
                          [18.0313,-7.2389,-1,-0.0230],
                          [10.4267,-6.9789,-1,-0.0187],
                          [11.4360,0.7323,0.0090,-1],
                          [11.4134,7.5497,0.0568,-0.9984]])
    quick_eval(agent,.1,env_index = 2,locations = locations)
    # scores= train_mb_agent(agent,1,n_episodes=1000, eps_end = .3, eps_decay = .9)

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