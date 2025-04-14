import gym
from agent import DQNAgent
from utils import save_model, setup_logger, plot_rewards
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import os
import torch
from utils import set_seed  
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")





def train():
    # 设置随机种子
    seed = 42  # 你可以选择任何整数作为种子
    set_seed(seed)
    # 初始化环境和代理
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device)

    log_dir = "./runs/dqn_cartpole_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=log_dir)

    # 修改：将模型文件保存到 RL/DQN/models 下
    model_dir = './save_models'
    os.makedirs(model_dir, exist_ok=True)

    num_episodes = 1000
    all_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)  # 将下一个状态转移到 GPU
            agent.update()
            state = next_state
            total_reward += reward

        all_rewards.append(total_reward)

        # 记录每个 episode 的奖励
        writer.add_scalar("Reward/Episode", total_reward, episode)
        logger.info(f"Episode {episode}: Total Reward = {total_reward}")

        # 每100个episode保存一次模型
        if episode % 100 == 0:
            model_path = os.path.join(model_dir, f"model_{episode}.pth")
            save_model(agent.policy_net, episode, model_path)
        
        # 更新目标网络
        if episode % agent.target_update == 0:
            agent.update_target()

        # 记录每100个episode的平均奖励
        if episode % 100 == 0:
            moving_avg = np.mean(all_rewards[-100:])
            writer.add_scalar("Reward/MovingAverage", moving_avg, episode)

    # 绘制奖励曲线
    plot_rewards(all_rewards, save_path='./result_pics/reward_plot.png', window=100)

    # 关闭环境和 TensorBoard
    env.close()
    writer.close()
