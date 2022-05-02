import os
import socket
import numpy as np
import csv
import math

from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner

from simulation_base.env import resume_env, nb_actuations

import matplotlib.pyplot as plt
import numpy as np


example_environment = resume_env(plot=3000, dump=100, single_run=True)

deterministic = True

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

saver_restore = dict(directory=os.getcwd() + "/saver_data/")

agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=example_environment, max_episode_timesteps=nb_actuations,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=5, learning_rate=1e-1, subsampling_fraction=0.2, optimization_steps=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, estimate_terminal=True,  # ???
    # TODO: gae_lambda=0.97 doesn't currently exist
    # Critic
    critic_network=network,
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    # TensorFlow etc
    parallel_interactions=1,
    saver=saver_restore,
)

# restore_directory = './saver_data/'
# restore_file = 'model-40000'
# agent.restore(restore_directory, restore_file)
# agent.restore()
agent.initialize()



if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")

x = []
angle = []
width = []

def plt_angle_width():
    angle = np.loadtxt('angle.dat')
    plt.subplot(2,1,1)
    plt.plot(range(len(angle)), angle, linestyle = '-', marker = 'o', color = 'b')
    plt.xlim(0.0, len(angle) + 10)
    plt.ylim(0.0, 100)
    plt.xlabel('Episode')
    plt.ylabel('Angle')
    width = np.loadtxt('width.dat')
    plt.subplot(2,1,2)
    plt.plot(range(len(width)), width, linestyle = '-', marker = 'o', color = 'b')
    plt.xlim(0.0, len(width) + 10)
    plt.ylim(0.0, 0.2)
    plt.xlabel('Episode')
    plt.ylabel('Width')
    plt.savefig('angle_width.png', bbox_inches = 'tight')
    plt.close()

def one_run():
    for j in range(50):
        # Can regard taht reset only initialize the flow field to become stable, but no DRL actions
        state = example_environment.reset()
        example_environment.render = True
        example_environment.geometry_params['slit_width'] += example_environment.paras[0]
        example_environment.geometry_params['slit_width'] = max(min(example_environment.geometry_params['slit_width'], 0.2), 0.01)
        example_environment.geometry_params['slit_angle'] += example_environment.paras[1]*180/math.pi
        example_environment.geometry_params['slit_angle'] = min(max(example_environment.geometry_params['slit_angle'], 0.0), 180.0)
        sw = example_environment.geometry_params['slit_width']
        sa = example_environment.geometry_params['slit_angle']
        print("start simulation ", j, sw, sa)
    
        for k in range(nb_actuations//2):
            #environment.print_state()
            action = agent.act(state, deterministic=deterministic, independent=True)
            # After initialization, we need DRL actions
            state, terminal, reward = example_environment.execute(action)
        # just for test, too few timesteps
        # runner.run(episodes=10000, max_episode_timesteps=20, episode_finished=episode_finished)

        angle.append(example_environment.geometry_params['slit_angle'])
        width.append(example_environment.geometry_params['slit_width'])
    
        data = np.genfromtxt("saved_models/test_strategy.csv", delimiter=";")
        data = data[1:,1:]
        m_data = np.average(data[len(data)//2:], axis=0)
        nb_jets = len(m_data)-4
        # Print statistics
        print("Single Run finished. AvgDrag : {}, AvgRecircArea : {}".format(m_data[1], m_data[2]))
    
        name = "test_strategy_avg.csv"
        if(not os.path.exists("saved_models")):
            os.mkdir("saved_models")
        if(not os.path.exists("saved_models/"+name)):
            with open("saved_models/"+name, "w") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["Name", "Drag", "Lift", "RecircArea"] + ["Jet" + str(v) for v in range(nb_jets)])
                spam_writer.writerow([example_environment.simu_name] + m_data[1:].tolist())
        else:
            with open("saved_models/"+name, "a") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([example_environment.simu_name] + m_data[1:].tolist())

    with open('angle.dat', 'w', encoding = 'UTF-8') as angle_file:
        for i in range(len(angle)):
            angle_file.write(str(angle[i]))
            angle_file.write('\n')

    with open('width.dat', 'w', encoding = 'UTF-8') as width_file:
        for i in range(len(width)):
            width_file.write(str(width[i]))
            width_file.write('\n')

    if j != 0:
        plt_angle_width()


if not deterministic:
    for _ in range(10):
        one_run()

else:
    one_run()


