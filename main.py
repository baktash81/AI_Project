import numpy as np
import random 
import gymnasium as gym
from collections import defaultdict, deque
import math


episodes = 2100
gamma = 0.99
alpha = 0.01
highscore = -200

statebounds = [(0, math.pi),
           (-2,2),
           (-1,1),
           (-1,1),
           (0,math.pi),
           (-2,2),
           (0, math.pi),
           (-2,2),
           (0,1),
           (0, math.pi),
           (-2, 2),
           (0, math.pi),
           (-2, 2),
           (0, 1)]

actionbounds = (-1,1)



def updateQTable (Qtable, state, action, reward, nextState=None):
    global alpha
    global gamma

    current = Qtable[state][action]  
    qNext = np.max(Qtable[nextState]) if nextState is not None else 0
    target = reward + (gamma * qNext)
    new_value = current + (alpha * (target - current))
    return new_value

def getNextAction(qTable, epsilon, state):

    if random.random() < epsilon:

        action = ()
        for i in range (0, 4):
            action += (random.randint(0, 9),)

    else:
        action = np.unravel_index(np.argmax(qTable[state]), qTable[state].shape)
    return action


def discretizeState(state):

    discreteState = []

    for i in range(len(state)):
        index = int( (state[i]-statebounds[i][0])/ (statebounds[i][1]-statebounds[i][0])*19)
        discreteState.append(index)
    
    return tuple(discreteState)


def convertNextAction(nextAction):
    action = []

    for i in range(len(nextAction)):

        nextVal = nextAction[i] / 9 * 2 - 1
        action.append(nextVal)

    return tuple(action)

def runAlgorithmStep(env, i, qTable):

    global highscore


    print("Episode #: ", i)
    observation, info = env.reset(seed=42)
    state = discretizeState(observation[0:14])
    total_reward=  0
    epsilon = 1.0 / ( i * 0.004)

    while True:
        
        nextAction = convertNextAction(getNextAction(qTable, epsilon, state))
        nextActionDiscretized = getNextAction(qTable, epsilon, state)

        nextState, reward, terminated,truncated, info = env.step(nextAction)
        nextState = discretizeState(nextState[0:14])
        total_reward += reward
        qTable[state][nextActionDiscretized] = updateQTable(qTable, state, nextActionDiscretized, reward, nextState)
        state = nextState
        if terminated or truncated:
                break
    
    if total_reward > highscore:

        highscore = total_reward

    return total_reward

#,render_mode = "human"

env = gym.make("BipedalWalker-v3")

env.reset()

qTable = defaultdict( lambda: np.zeros((10, 10, 10, 10)))
# qTable = np.load('data.npy',allow_pickle=True)
np.save('data', np.array(dict(qTable)))
for i in range(1, episodes + 1):
    epScore = runAlgorithmStep(env, i, qTable)
    print("episode score : ",epScore)

print("All episodes finished. Highest score achieved: " + str(highscore))