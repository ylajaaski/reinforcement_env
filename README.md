# Toolkit for Creating Reinforcement Learning Environments and Algorithms

This repository enables the user to create, train, and test reinforcement learning environments and algorithms. Currently, one example environment is available *collision_v0*. 

## Basics

This toolkit works in similar fashion compared to the *OpenAI Gym*. Reinforcement learning is based on the interaction between an environment and an agent. The agent observes the environment and chooses actions that in turn change the state of the agent and the environment. Further on, the agent keeps on getting rewards from the environment depending on the actions taken. 

## Setup

Clone the repository with:

`git clone https://github.com/ylajaaski/reinforcement_env.git`

Then create the required environment with:

`conda env create -f environment.yml`

You can train and test different environment and agents with `app.py`. To get some help run:

`python app.py -help`

## Creating own environments and agents
...



