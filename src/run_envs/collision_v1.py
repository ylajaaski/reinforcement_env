from src.envs.collision_v1.collision import Collision_v1
from src.envs.collision_v1.agents.posion_gradient import Agent
import numpy as np 
import pandas as pd

def collision_v1():

    if (torch.cuda.is_available()):
        print("Cuda is available")
    else:
        print("Cuda unavailable")

    episodes = 10000
    save_interval = 100
    actions = {} # dicitonary for actions taken
    game_duration = np.zeros(episodes) # game lengths for each episode
    actions[0] = np.zeros(episodes)
    actions[1] = np.zeros(episodes)
    actions[2] = np.zeros(episodes)
    actions[3] = np.zeros(episodes)
    agent = Agent(400,400,10,1)
    env = Collision_v1(400, 400, 30, agent)

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(episodes):
        reward_sum, timesteps = 0, 0
        done = False
        video = []
        # Reset the environment and observe the initial state
        observation = env.reset()
        #env.render()
        video.append(env.state)

        # Loop until the episode is over
        while not done:
            # # Get action from the agent
            # action, action_probabilities = agent.get_action(observation)
            # actions[action][episode_number] += 1
            # previous_observation = observation

            # # Perform the action on the environment, get new state and reward
            # observation, reward, done, info = env.step(action)
            # video.append(env.state)

            # # Store action's outcome (so that the agent can improve its policy)
            # agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # # Store total episode reward
            # reward_sum += reward
            # timesteps += 1 
            # Choose action
            action, log_probability, state_value, entropy = agent.get_training_action(observation)
            
            actions[action.item()][episode_number] += 1
            
            # Apply action -> get frame
            frame, reward, done, _ = env.step(action.detach())
            video.append(observation)
            
            agent.store_step(state_value, reward, log_probability, entropy)
            timesteps += 1

        game_duration[episode_number] = timesteps 
        actions[0][episode_number] /= timesteps
        actions[1][episode_number] /= timesteps
        actions[2][episode_number] /= timesteps
        actions[3][episode_number] /= timesteps 

        if episode_number % save_interval == 0:
            

            tmp = {}
            df = pd.DataFrame(data = tmp)
            df['duration']= game_duration[0:episode_number]
            df['down'] = actions[0][0:episode_number]
            df['right'] = actions[1][0:episode_number]
            df['left'] = actions[2][0:episode_number]
            df['up'] = actions[3][0:episode_number]

            df.to_csv("results/run/collision_v1/{}.csv".format(episode_number))
            np.save("results/run/collision_v1/{}.npy".format(episode_number), np.array(video))

        # Let the agent do its magic (update the policy)
        #agent.episode_finished(episode_number)
        #player_position = np.array([np.random.random()*(env.width - 8*env.ball_size)+3*env.player_size, np.random.random()*(env.height - 8*env.ball_size)+3*env.player_size])
        #agent.reset(player_position)
        l_PG, l_v, l_H = agent.update_network()
        player_position = np.array([100,100])#np.array([np.random.random()*(env.width - 2*env.ball_size)+env.player_size, np.random.random()*(env.height - 2*env.ball_size)+env.player_size])
        agent.reset(player_position)
        print("Episode:", episode_number, "Timesteps:", timesteps, "Actions: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(actions[0][episode_number], actions[1][episode_number], actions[2][episode_number], actions[3][episode_number]))