import pickle
import tensorflow as tf
import sys
from normalizingData import normalizeData, normalizeMultipleDataLists, polyfitMultipleDataLists

# To read from written binary files
IS_PARSED = True

REWARD = 'episode_reward'
LOSS = 'loss/loss'

def parseData(fileName: str):
    rewardStepsList = []
    rewardsList = []

    lossStepsList = []
    lossList = []

    print('Parsing started for file: {}'.format(fileName))
    for event in tf.train.summary_iterator(fileName):
        for value in event.summary.value:
            if REWARD in value.tag and value.HasField('simple_value'):
                rewardsList.append(value.simple_value)
                rewardStepsList.append(event.step)
            if LOSS in value.tag and value.HasField('simple_value'):
                lossList.append(value.simple_value)
                lossStepsList.append(event.step)

    print('Rewards: ', len(rewardsList), len(rewardStepsList))
    print('Loss: ', len(lossList), len(lossStepsList))

    return {
        'rewards': rewardsList,
        'rewards_steps': rewardStepsList,
        'loss': lossList,
        'loss_steps': lossStepsList
    }

if __name__ == '__main__':

    if IS_PARSED:

        with open('data/cartpole.pkl', 'rb') as f: 
            parsedRewardResults, parsedLossResults = pickle.load(f)

        normalizeMultipleDataLists(parsedRewardResults, 'Steps', 'Reward', 'CartPole-v1 environment training results')
        polyfitMultipleDataLists(parsedRewardResults, 'Steps', 'Reward', 'CartPole-v1 environment training speed')

        with open('data/pendulum.pkl', 'rb') as f: 
            parsedRewardResults, parsedLossResults = pickle.load(f)

        normalizeMultipleDataLists(parsedRewardResults, 'Steps', 'Reward', 'Pendulum-v0 environment training results')
        polyfitMultipleDataLists(parsedRewardResults, 'Steps', 'Reward', 'Pendulum-v0 environment training speed')

    else:

        CartPoleResultsList = [
            ("A2C", "D:\Source\Kursinis1\A2C_CartPole-v1\A2C_12\events.out.tfevents.1610705436.ARNOLDAS", 1e7),
            ("ACER", "D:\Source\Kursinis1\ACER_CartPole-v1\ACER_4\events.out.tfevents.1610705452.ARNOLDAS", 6.2e6),
            ("ACKTR", "D:\Source\Kursinis1\ACKTR_CartPole-v1\ACKTR_5\events.out.tfevents.1610705468.ARNOLDAS", 2e7),
            # ("DQN", "D:\Source\Kursinis1\DQN_CartPole-v1\DQN_5\events.out.tfevents.1610705482.ARNOLDAS"),
            ("DQN", "D:\Source\Kursinis1\DQN_CartPole-v1\DQN_6\events.out.tfevents.1610809990.ARNOLDAS", 2e7),
            ("PPO", "D:\Source\Kursinis1\PPO_CartPole-v1\PPO2_4\events.out.tfevents.1610705526.ARNOLDAS", 3e6),
            ("TRPO", "D:\Source\Kursinis1\TRPO_CartPole-v1\TRPO_4\events.out.tfevents.1610705539.ARNOLDAS", 3e6)
        ]

        PendulumResultsList = [
            ("A2C", "D:\Source\Kursinis1\A2C_Pendulum-v0\A2C_6\events.out.tfevents.1610753361.ARNOLDAS", 6.8e6),
            ("ACKTR", "D:\Source\Kursinis1\ACKTR_Pendulum-v0\ACKTR_5\events.out.tfevents.1610753376.ARNOLDAS", 7e6),
            ("DDPG", "D:\Source\Kursinis1\DDPG_Pendulum-v0\DDPG_4\events.out.tfevents.1610753383.ARNOLDAS", 1.6e6),
            # ("PPO", "D:\Source\Kursinis1\PPO_Pendulum-v0\PPO2_4\events.out.tfevents.1610753395.ARNOLDAS"),
            ("PPO", "D:\Source\Kursinis1\PPO_Pendulum-v0\PPO2_5\events.out.tfevents.1610834351.ARNOLDAS", 1e7),
            ("SAC", "D:\Source\Kursinis1\SAC_Pendulum-v0\SAC_4\events.out.tfevents.1610753406.ARNOLDAS", 3.3e6),
            ("TD3", "D:\Source\Kursinis1\TD3_Pendulum-v0\TD3_5\events.out.tfevents.1610753414.ARNOLDAS", 1.5e6),
            ("TRPO", "D:\Source\Kursinis1\TRPO_Pendulum-v0\TRPO_4\events.out.tfevents.1610753422.ARNOLDAS", 5.1e6)
        ]

        parsedRewardResults = []
        parsedLossResults = []
        for algName, filePath, limit in PendulumResultsList:
            data = parseData(filePath)
            parsedRewardResults.append((algName, data['rewards_steps'], data['rewards'], limit))

        with open('data/pendulum.pkl', 'wb') as f:
            pickle.dump([parsedRewardResults, parsedLossResults], f)

        normalizeMultipleDataLists(parsedRewardResults, 'Steps', 'Reward', 'Pendulum-v0 environment training results')
        polyfitMultipleDataLists(parsedRewardResults, 'Steps', 'Reward', 'Pendulum-v0 environment training results')

        parsedRewardResults = []
        parsedLossResults = []
        for algName, filePath, limit in CartPoleResultsList:
            data = parseData(filePath)
            parsedRewardResults.append((algName, data['rewards_steps'], data['rewards'], limit))

        with open('data/cartpole.pkl', 'wb') as f:
            pickle.dump([parsedRewardResults, parsedLossResults], f)

        normalizeMultipleDataLists(parsedRewardResults, 'Steps', 'Reward', 'CartPole-v1 environment training results')
        polyfitMultipleDataLists(parsedRewardResults, 'Steps', 'Reward', 'CartPole-v1 environment training results')