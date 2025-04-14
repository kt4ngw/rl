import torch
import random
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 确保将每个项转换为 Tensor 类型
        states = torch.stack([torch.tensor(state, dtype=torch.float32).clone().detach() for state in states]).cpu()
        actions = torch.tensor(actions, dtype=torch.long).cpu()
        rewards = torch.tensor(rewards, dtype=torch.float32).cpu()
        next_states = torch.stack([torch.tensor(next_state, dtype=torch.float32).clone().detach() for next_state in next_states]).cpu()
        dones = torch.tensor(dones, dtype=torch.bool).cpu()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
