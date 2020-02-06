import numpy as np
import random
from model import Actor
from model import Critic
from buffer import Buffer
from noise import OUStrategy
import torch
import torch.optim as optim
import torch.nn.functional as F
import gym
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
TAU = 1e-3
BUF_SIZE = 5000
BATCH_SIZE = 512
LR = 5e-4


class DDPG:

    def __init__(self, state_dim, action_dim):
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_c = copy.deepcopy(self.critic)

        self.actor = Actor(state_dim).to(device)
        self.target_a = copy.deepcopy(self.actor)

        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=LR)
        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=LR)

        self.replay_buffer = Buffer(BUF_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()
        return action

    def update(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.from_numpy(np.array(actions)).float().to(device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        dones = torch.from_numpy(np.array(dones)).to(device)

        Q_current = self.critic(states, actions)
        Q_next = self.target_c(next_states, self.target_a(next_states).detach())
        y = (rewards + GAMMA * Q_next).detach()

        ##################Update critic#######################
        loss_c = F.mse_loss(y, Q_current)
        self.optimizer_c.zero_grad()
        loss_c.backward()
        self.optimizer_c.step()

        ##################Update actor#######################
        loss_a = -self.critic.forward(states, self.actor(states)).mean()
        self.optimizer_a.zero_grad()
        loss_a.backward()
        self.optimizer_a.step()

        ##################Update targets#######################
        for target_param, param in zip(self.target_a.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

        for target_param, param in zip(self.target_c.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))


env = gym.make("MountainCarContinuous-v0")

seed = 21
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

max_episodes = 100

agent = DDPG(2, 1)
noise = OUStrategy(env.action_space)
buf = Buffer(BUF_SIZE)
step = 0

for episode in range(max_episodes):
    state = env.reset()
    total = 0
    done = False
    while True:
        action = agent.get_action(state)
        action = noise.get_action_from_raw_action(action, step)

        next_state, reward, done, _ = env.step(action)
        total += reward

        reward += 50 * abs(next_state[1])

        buf.add((state, action, reward, next_state, done))
        if len(buf) >= BATCH_SIZE:
            agent.update(transition)

        if done:
            print(f'episode {episode} done. reward: {total}')
            break

        state = next_state
        step += 1

for _ in range(50):
    state = env.reset()
    total = 0
    done = False
    while True:
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total += reward

        if done:
            print(f'{total}')
            break

        state = next_state
