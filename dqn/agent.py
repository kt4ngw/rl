import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from model import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, device, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3,
                 eps_start=1.0, eps_end=0.01, eps_decay=500, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.steps_done = 0
        # 将模型移到指定设备（CPU 或 GPU）
        self.device = device
        self.policy_net = DQN(state_dim, action_dim).to(self.device)  # 将模型移到设备
        self.target_net = DQN(state_dim, action_dim).to(self.device)  # 将目标网络移到设备
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()



        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device) 

        if random.random() > eps:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(self.action_dim)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 转换成张量并将其移到正确的设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values * ~dones

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
