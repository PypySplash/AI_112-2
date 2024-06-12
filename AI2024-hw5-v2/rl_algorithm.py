import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import YOUR_CODE_HERE
import utils

class PacmanActionCNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PacmanActionCNN, self).__init__()
        # build your own CNN model
        "*** YOUR CODE HERE ***"
        
        # self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)  # 第一層卷積層，使用了範例中的通道數作為第一個參數
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)         # 第二層卷積層
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)         # 第三層卷積層
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)


        # 計算卷積後的特徵維度以連接到全連接層
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # 假設輸入圖像大小為 84x84
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        # linear_input_size = convw * convh * 64
        linear_input_size = convw * convh * 32
        
        self.head = nn.Linear(linear_input_size, action_dim)  # 全連接層
        
        # utils.raiseNotDefined()
        # this is just an example, you can modify this.
        # self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 將卷積層輸出展平
        x = self.head(x)           # 全連接層產生動作價值
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        
        return x

class ReplayBuffer:
    # referenced [TD3 official implementation](https://github.com/sfujim/TD3/blob/master/utils.py#L5).
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, state, action, reward, next_state, terminated):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        epsilon=0.9,
        epsilon_min=0.05,
        gamma=0.99,
        batch_size=64,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=10000,
    ):
        """
        DQN agent has four methods.

        - __init__() as usual
        - act() takes as input one state of np.ndarray and output actions by following epsilon-greedy policy.
        - process() method takes one transition as input and define what the agent do for each step.
        - learn() method samples a mini-batch from replay buffer and train q-network
        """
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size)
        # self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            # Random action
            action = np.random.randint(0, self.action_dim)
        else:
            # output actions by following epsilon-greedy policy
            # x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            # 修改后的代码
            x = x.float().unsqueeze(0).to(self.device)
            
            "*** YOUR CODE HERE ***"
            q_values = self.network(x)
            action = torch.argmax(q_values).item()  # 選擇具有最高Q值的動作
            # utils.raiseNotDefined()
            # get q-values from network
            # q_value = YOUR_CODE_HERE
            # get action with maximum q-value
            # action = YOUR_CODE_HERE
        
        return action
    
    def learn(self):
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        
        # sample a mini-batch from replay buffer
        state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        
        current_q_values = self.network(state).gather(1, action.long())
        next_q_values = self.target_network(next_state).max(1)[0].detach()
        td_target = reward + (1 - terminated) * self.gamma * next_q_values.unsqueeze(1)

        loss = F.smooth_l1_loss(current_q_values, td_target)
        
        # get q-values from network
        next_q = YOUR_CODE_HERE
        # td_target: if terminated, only reward, otherwise reward + gamma * max(next_q)
        td_target = YOUR_CODE_HERE
        # compute loss with td_target and q-values
        loss = YOUR_CODE_HERE
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # initialize optimizer
        # "self.optimizer.YOUR_CODE_HERE"
        # backpropagation
        # YOUR_CODE_HERE
        # update network
        # "self.optimizer.YOUR_CODE_HERE"
        
        return {'loss': loss.item()} # return dictionary for logging
    
    def process(self, transition):
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        
        self.buffer.update(*transition)
        
        result = {}
        self.total_steps += 1
        
        # update replay buffer
        # "self.buffer.YOUR_CODE_HERE"

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        if self.total_steps % self.target_update_interval == 0:
            # update target network
            # "self.target_network.YOUR_CODE_HERE"
            self.target_network.load_state_dict(self.network.state_dict())
        
        self.epsilon -= self.epsilon_decay
        return result