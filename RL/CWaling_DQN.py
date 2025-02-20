import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
class Qnet(nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        

    def forward(self, x):
        #x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络

        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        for param in self.target_q_net.parameters():
            param.requires_grad = False
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
    def state_one_hot(self, state):
        if state.size == 1:
            state_one_hot = torch.zeros((self.state_dim), dtype = torch.float32)
            state_one_hot[state] = 1
        else:
            state_one_hot = torch.zeros((state.size, self.state_dim), dtype = torch.float32)
            for i in range(state.size):
                state_one_hot[i][state[i]] = 1      
        return state_one_hot
    
    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = self.state_one_hot(np.array(state)).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = self.state_one_hot(transition_dict['states']).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = self.state_one_hot(transition_dict['next_states']).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        #print(dqn_loss)
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
        
    def best_action(self, state):  # 用于打印策略
        #Q_max = np.max(self.Q_table[state])
        state = self.state_one_hot(np.array(state)).to(self.device)
        #q_max = self.q_net(state).detach().tolist()
        q = self.q_net(state)
        q_max = q.max()
        a = [0 for _ in range(4)]
        for i in range(4):
            if(q[i] == q_max):
                a[i] = 1
        return a
        
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
            else:
                reward = 0
        else:
            #reward = 1/(abs(self.x - self.ncol + 1) + abs(self.y - 3))
            reward = -1
            done = False
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
    
def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
epsilon = 0.01
gamma = 0.99
hidden_dim = 128
state_dim = 12 * 4
action_dim = 4
lr = 0.01
target_update = 20
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
num_episodes = 2000  # 智能体在环境中运行的序列的数量
buffer_size = 1000
minimal_size = 500
batch_size = 64
replay_buffer = ReplayBuffer(buffer_size)
return_list = []  # 记录每一条序列的回报
action_meaning = ['^', 'v', '<', '>']
'''for i in range(nrow):
    for j in range(ncol):
        state = i * ncol + j
        for a in range(4):
            env.x = j
            env.y = i
            next_state, reward, done = env.step(a)
            replay_buffer.add(state, a, reward, next_state, done)
            print(f'{state}, {a}, {reward}, {next_state}, {done}')'''
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                state = next_state
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)    
                    transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                    agent.update(transition_dict)
            #print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
            return_list.append(episode_return)
            if (i_episode + 1) % 1 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 1 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-1:])
                })
            pbar.update(1)
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
        
episodes_list = list(range(len(return_list)))
'''plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format('Cliff Walking'))
plt.show()'''


print('Q-learning算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])