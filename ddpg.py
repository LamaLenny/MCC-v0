import torch
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import copy
import random
from model import Actor
from model import Critic
from buffer import Buffer
from noise import OUStrategy

GAMMA = 0.98
TAU = 1e-3
BUF_SIZE = 4096
BATCH_SIZE = 512
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class DDPG:

    def __init__(self, state_dim, action_dim):
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_c = copy.deepcopy(self.critic)

        self.actor = Actor(state_dim).to(device)
        self.target_a = copy.deepcopy(self.actor)

        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=LR)
        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=LR)

    def act(self, state):
        state = torch.from_numpy(np.array(state)).float().to(device)
        return self.actor.forward(state).detach().squeeze(0).cpu().numpy()

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
        for target_pr, pr in zip(self.target_a.parameters(), self.actor.parameters()):
            target_pr.data.copy_(TAU * pr.data + (1 - TAU) * target_pr.data)

        for target_pr, pr in zip(self.target_c.parameters(), self.critic.parameters()):
            target_pr.data.copy_(TAU * pr.data + (1 - TAU) * target_pr.data)


episodes = 150

seed = 12
env = gym.make('MountainCarContinuous-v0')
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

agent = DDPG(2, 1)
buf = Buffer(BUF_SIZE)
noise = OUStrategy(env.action_space, min_sigma=1e-4)
updates_noise = 0
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        action = noise.get_action_from_raw_action(action, updates_noise)
        updates_noise += 1
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        buf.add((state, action, reward, next_state, done))
        if len(buf) >= BATCH_SIZE:
            agent.update(buf.sample(BATCH_SIZE))
        state = next_state
    print(f"I did {episode}th episode. Result: {total_reward}, sigma = {noise.sigma}")
# Я решила тренироваться до 150 эпизодов, хотя  с этим сидом оно крутится около 90, начиная с 30 эпизода.

# Вывод на последних 10 эпизодах:
# I did 139th episode. Result: 91.13059676792551, sigma = 0.17022727199999999
# I did 140th episode. Result: 90.62383628427916, sigma = 0.16973243699999999
# I did 141th episode. Result: 94.36829967370625, sigma = 0.16948352
# I did 142th episode. Result: 87.05158580519061, sigma = 0.168778755
# I did 143th episode. Result: 89.52206836735917, sigma = 0.16824493299999999
# I did 144th episode. Result: 92.20854623030216, sigma = 0.167951031
# I did 145th episode. Result: 92.42983013079339, sigma = 0.16767512299999998
# I did 146th episode. Result: 92.45146988594169, sigma = 0.167402214
# I did 147th episode. Result: 92.90206657516018, sigma = 0.16711431
# I did 148th episode. Result: 94.35916332740533, sigma = 0.16682340699999998
# I did 149th episode. Result: 95.03272339975832, sigma = 0.166583487
