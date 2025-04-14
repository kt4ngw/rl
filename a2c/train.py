import torch
import gym
import rl_utils  # 假设有一个 RL 相关的工具包，用来管理训练过程

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = 0
        transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)

            # 收集经验
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)

            state = next_state
            episode_rewards += reward

        # 更新 agent
        agent.update(transition_dict)

        return_list.append(episode_rewards)
        print(f"Episode {episode}, Reward: {episode_rewards}")

    return return_list
