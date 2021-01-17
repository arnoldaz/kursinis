import gym
import sys, getopt
from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO2, SAC, TD3, TRPO

STEPS = 2e7

def getTensorboardLogLocation(envName: str, algName: str):
    return './' + algName + '_' + envName + '/'

def main(argv):
    environmentName = ''
    algorithmName = ''

    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'e:a:', ['env=','alg='])
    except getopt.GetoptError:
        print ('--env <environment-name> --alg <algorithm-name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-e', '--env'):
            environmentName = arg
        elif opt in ('-a', '--alg'):
            algorithmName = arg

    # Create environment
    env = gym.make(environmentName)

    # Create model
    if algorithmName == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=getTensorboardLogLocation(environmentName, algorithmName), full_tensorboard_log=False)
    elif algorithmName == 'ACER':
        model = ACER('MlpPolicy', env, verbose=1, tensorboard_log=getTensorboardLogLocation(environmentName, algorithmName), full_tensorboard_log=False)
    elif algorithmName == 'ACKTR':
        model = ACKTR('MlpPolicy', env, verbose=1, tensorboard_log=getTensorboardLogLocation(environmentName, algorithmName), full_tensorboard_log=False)
    elif algorithmName == 'DDPG':
        model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=getTensorboardLogLocation(environmentName, algorithmName), full_tensorboard_log=False)
    elif algorithmName == 'DQN':
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=getTensorboardLogLocation(environmentName, algorithmName), full_tensorboard_log=False)
    elif algorithmName == 'PPO':
        model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log=getTensorboardLogLocation(environmentName, algorithmName), full_tensorboard_log=False)
    elif algorithmName == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=getTensorboardLogLocation(environmentName, algorithmName), full_tensorboard_log=False)
    elif algorithmName == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=getTensorboardLogLocation(environmentName, algorithmName), full_tensorboard_log=False)
    elif algorithmName == 'TRPO':
        model = TRPO('MlpPolicy', env, verbose=1, tensorboard_log=getTensorboardLogLocation(environmentName, algorithmName), full_tensorboard_log=False)
    else:
        print('Wrong algorithm')
        sys.exit(2)

    model.learn(total_timesteps=int(STEPS), log_interval=250)

    print('Trained algorithm:')
    print(environmentName, algorithmName)

if __name__ == '__main__':
   main(sys.argv[1:])