import gym
import numpy as np
from numpy.core.numeric import Infinity

def runCartPoleEpisode(env, params, maxRange = 200):
    observation = env.reset()
    totalReward = 0
    timestepsTaken = maxRange
    for t in range(maxRange):
        env.render()
        action = 0 if np.matmul(observation, params) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            timestepsTaken = t
            break

    print("Episode finished after {} timesteps. Total reward: {}".format(timestepsTaken, totalReward))
    return totalReward

def runAcrobotEpisode(env, params, maxRange = 500):
    observation = env.reset()
    totalReward = 0
    timestepsTaken = maxRange
    for t in range(maxRange):
        env.render()
        mult = np.matmul(observation, params)
        action = None

        if mult < -0.33:
            action = -1
        elif mult < 0.33:
            action = 0
        else:
            action = 1

        observation, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            timestepsTaken = t
            break

    print("Episode finished after {} timesteps. Total reward: {}".format(timestepsTaken, totalReward))
    return totalReward


# env = gym.make('CartPole-v0')
env = gym.make('Acrobot-v1')
bestReward = -Infinity
bestParams = None

# bestReward = [-0.0635203, 0.50656531, 0.3433965, 0.54071262] # cartpole 
# bestReward = [0.61752563,  0.03856546, -0.78693767, 0.48913831, -0.02765827, -0.89143554] # acrobot
# bestReward = [-0.54884346, 0.07541038, 0.0550034, 0.60588305, -0.9719607, -0.47540891] # acrobot

# Train
for _ in range(50):
    params = np.random.rand(6) * 2 - 1
    reward = runAcrobotEpisode(env, params)
    if reward > bestReward:
        bestReward = reward
        bestParams = params

# Result
print("Found best reward: {}, params: {}".format(bestReward, bestParams))
for _ in range(10):
    runAcrobotEpisode(env, bestParams)

env.close()