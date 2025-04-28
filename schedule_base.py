import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

from schedule import async_scheduling

def main():
    env_config = {"truck_num": 12,
                "factory_num": 45,
                "algo": "baseline_5"}
    env = async_scheduling(env_config)
    
    # Get the number of agents from the environment
    num_agents = env.truck_num
    print(f"Number of agents: {num_agents}")

    episodes = 1
    for e in range(episodes):
        obs, _ = env.reset()
        acts = {}
        rews = {}
        total_rewards = {agent_id: 0 for agent_id in obs.keys()}
        done = False
        step = 0
        
        while not done:
            actions = {}
            for agent_id in obs.keys():
                rul = obs[agent_id][0]
                if rul < 5:
                    action = 50
                else:
                    action = random.randint(0, 49)
                actions[agent_id] = action
            obs, rewards, dones, _, _ = env.step(actions)
            for agent_id in rewards.keys():
                rews[agent_id] = rewards[agent_id]

            # states = next_states
            step += 1
            
            # Check if all agents are done
            done = dones['__all__']
        
        print(f"Episode: {e}/{episodes}, Steps: {step}")

if __name__ == "__main__":
    main()
