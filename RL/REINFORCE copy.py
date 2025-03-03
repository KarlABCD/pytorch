import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    def state_one_hot(self, state):
        if state.size == 1:
            state_one_hot = torch.zeros((state.size, 48), dtype = torch.float32)
            state_one_hot[0][state] = 1
        else:
            state_one_hot = torch.zeros((state.size, 48), dtype = torch.float32)
            for i in range(state.size):
                state_one_hot[i][state[i]] = 1      
        return state_one_hot

def best_action(agent, state):  # 用于打印策略
    state = agent.state_one_hot(np.array(state))
    q = agent(state)
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
                a = best_action(agent, i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()
# REINFORCE算法
def reinforce(env, policy_network, optimizer, num_episodes, gamma=0.99):
    all_rewards = []
    action_meaning = ['^', 'v', '<', '>']
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        next_states = []
        done = False
        # 生成一个episode
        while not done:
            state = policy_network.state_one_hot(np.array(state))
            probs = policy_network(state)
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs.squeeze(0)[action])
            log_probs.append(log_prob)
            next_state, reward, done = env.step(action)
            #next_state = policy_network.state_one_hot(np.array(next_state))
            rewards.append(reward)
            state = next_state

        # 计算总奖励
        all_rewards.append(sum(rewards))

        # 计算累积奖励
        R = 0
        optimizer.zero_grad()
        for r, log_prob in zip(reversed(rewards), reversed(log_probs)):
            R = r + gamma * R
            #loss = (-log_prob * R).unsqueeze(0)
            loss = -log_prob * R
            loss.backward()
        optimizer.step()
        # 更新策略网络
        if episode % 100 == 0:
            print(f"Episode {episode}: Average Reward = {np.mean(all_rewards[-100:])}")
            print_agent(policy_network, env, action_meaning, list(range(37, 47)), [47])
    return all_rewards

# 主函数
if __name__ == "__main__":
    ncol = 12
    nrow = 4
    torch.manual_seed(6)
    env = CliffWalkingEnv(ncol, nrow)
    input_dim = 48
    output_dim = 4
    policy_network = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy_network.parameters(), lr=0.01)
    num_episodes = 1000
    rewards = reinforce(env, policy_network, optimizer, num_episodes)