import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

from schedule import async_scheduling

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            
            # Get Q values for current state
            curr_q = self.model(state)
            
            # Get target Q values
            with torch.no_grad():
                target_q = curr_q.clone()
                if not done:
                    next_q = self.target_model(next_state).max(1)[0]
                    target = reward + self.gamma * next_q
                else:
                    target = reward
                
                target_q[0][action] = target
            
            # Optimize model
            self.optimizer.zero_grad()
            loss = self.criterion(curr_q, target_q)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def main():
    env_config = {"truck_num": 12,
                "factory_num": 45,
                "algo": "test"}
    env = async_scheduling(env_config)
    
    # Get the number of agents from the environment
    num_agents = env.truck_num
    print(f"Number of agents: {num_agents}")
    
    # Create a DQN agent for each agent in the environment
    # agents = []
    # for i in range(num_agents):
    #     # Assuming all agents have the same state and action space
    #     state_size = env.observation_space[i].shape[0]
    #     action_size = env.action_space[i].n
    #     agents.append(DQNAgent(state_size, action_size))
    #     print(f"Agent {i}: State size = {state_size}, Action size = {action_size}")3

    agents = {}
    for agent_id in env.observation_space.keys():
        state_size = env.observation_space[agent_id].shape[0]
        action_size = env.action_space[agent_id].n
        agents[agent_id] = DQNAgent(state_size, action_size)
        print(f"Agent {agent_id}: State size = {state_size}, Action size = {action_size}")
    
    # Training parameters
    episodes = 1000
    batch_size = 32
    
    for e in range(episodes):
        obs, _ = env.reset()
        acts = {}
        rews = {}
        states = obs
        total_rewards = {agent_id: 0 for agent_id in obs.keys()}
        done = False
        step = 0
        
        while not done:
            # actions = []
            # for i, agent in enumerate(agents):
            #     state = np.reshape(states[i], [1, agents[i].state_size])
            #     action = agent.act(state)
            #     actions.append(action)
            
            actions = {}
            for agent_id in obs.keys():
                state = np.reshape(states[agent_id], [1, agents[agent_id].state_size])
                action = agents[agent_id].act(state)
                actions[agent_id] = action
                acts[agent_id] = action
            
            # Step the environment
            obs, rewards, dones, _, _ = env.step(actions)
            for agent_id in rewards.keys():
                rews[agent_id] = rewards[agent_id]
            next_states = obs
            
            # Learn from experience for each agent
            # for i, agent in enumerate(agents):
            #     state = np.reshape(states[i], [1, agents[i].state_size])
            #     next_state = np.reshape(next_states[i], [1, agents[i].state_size])
            #     agent.remember(state, actions[i], rewards[i], next_state, dones[i])
            #     total_rewards[i] += rewards[i]
                
            #     # Train the agent if enough samples are available in memory
            #     if len(agent.memory) > batch_size:
            #         agent.replay(batch_size)
            
            for agent_id in next_states.keys():
                state = np.reshape(states[agent_id], [1, agents[agent_id].state_size])
                next_state = np.reshape(next_states[agent_id], [1, agents[agent_id].state_size])
                agents[agent_id].remember(state, acts[agent_id], rews[agent_id], next_state, True)
                states[agent_id] = next_states[agent_id]
                total_rewards[agent_id] += rews[agent_id]
                if len(agents[agent_id].memory) > batch_size:
                    agents[agent_id].replay(batch_size)
            
            
            # states = next_states
            step += 1
            
            # Check if all agents are done
            done = dones['__all__']
            
            # Optionally render the environment
            # env.render()
        
        # Update target model periodically
        if e % 10 == 0:
            for agent in agents.values():
                agent.update_target_model()
        
        print(f"Episode: {e}/{episodes}, Steps: {step}")

        
        # Save model checkpoints periodically
        # if e % 100 == 0:
        #     for i, agent in enumerate(agents):
        #         agent.save(f"dqn_agent_{i}_ep{e}.pt")
    
    # # Save final models
    # for i, agent in enumerate(agents):
    #     agent.save(f"dqn_agent_{i}_final.pt")
    
    # env.close()

if __name__ == "__main__":
    main()