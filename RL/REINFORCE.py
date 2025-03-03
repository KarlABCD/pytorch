import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
from torchviz import make_dot

# 自定义悬崖漫游环境
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
            reward = -1
            done = False
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

# 策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

# REINFORCE算法实现
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def transfer_one_hot(self, input, dim):
        if input.size == 1:
            one_hot = torch.zeros((input.size, dim), dtype = torch.float32).to(self.device)
            one_hot[0][input] = 1
        else:
            one_hot = torch.zeros((input.size, dim), dtype = torch.float32).to(self.device)
            for i in range(input.size):
                one_hot[i][input[i]] = 1      
        return one_hot
    
    def take_action(self, state):  # 根据动作概率分布随机采样
        state = self.transfer_one_hot(np.array(state), self.state_dim)
        probs = self.policy_net(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def print_parameters(self):
        for index, values in enumerate(self.policy_net.named_parameters()):
            if(values[1].device.type == 'cuda'):
                model_data = values[1].cpu().data.detach().numpy()
            else:
                model_data = values[1].data.detach().numpy()
                if (values[0] == 'fc1.weight'):
                    print(f'Para Name: {values[0]} \n Model Para: {model_data[35]}')
        

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = self.transfer_one_hot(np.array(state_list[i]), self.state_dim).to(self.device)
            #action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            action = self.transfer_one_hot(np.array(action_list[i]), self.action_dim).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state)).to(self.device)
            G = self.gamma * G + reward
            loss = -G * torch.matmul(log_prob, action)    # 每一步的损失函数
            #self.print_parameters()
            loss.backward()  # 反向传播计算梯度
        #self.print_parameters()
        self.optimizer.step()  # 梯度下降
        #self.print_parameters()

    def best_action(self, state):  # 用于打印策略
        state = self.transfer_one_hot(np.array(state), self.state_dim).to(self.device)
        q = self.policy_net(state)
        q_max = q.max()
        a = [0 for _ in range(4)]
        for i in range(4):
            if(q[0][i] == q_max):
                a[i] = 1
        return a 
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
# 训练和测试
if __name__ == "__main__":
    
    #np.random.seed(0)
    torch.manual_seed(6)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(6)
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    gamma = 0.99
    hidden_dim = 128
    state_dim = 12 * 4
    action_dim = 4
    lr = 0.01
    num_episodes = 1000
    device = torch.device("cpu") if torch.cuda.is_available() else torch.device(
    "cpu")
    agent = REINFORCE(state_dim, hidden_dim, action_dim, lr, gamma, device)
    return_list = []
    action_meaning = ['^', 'v', '<', '>']
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
        print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
    print('Q-learning算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])