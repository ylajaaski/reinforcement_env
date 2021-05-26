from src.envs.collision_v0.collision import Collision_v0
from src.envs.collision_v0.agents.policy_gradient import Agent
import numpy as np 
import pandas as pd

def run_env(environment, agent, train, episodes, save, save_interval):

    # Saving by default
    action_space = environment.space.n
    mode = environment.mode # Currently only survival 


    actions = {} # dicitonary for actions taken
    game_duration = np.zeros(episodes) # game lengths for each episode
    for act in range(action_space):
        actions[act] = np.zeros(episodes)
    
    for episode in range(episodes):
        timestep = 0 # initialize time
        done = False
        video = []
        frame = environment.reset() # (400, 400, 3)
        video.append(frame)
        while not done:
            # Choose action
            action, log_probability, state_value, entropy = agent.get_training_action(frame)
            
            actions[action.item()][episode] += 1
            
            # Apply action -> get frame
            frame, reward, done, _ = environment.step(action.detach())
            video.append(frame)
            
            agent.store_step(state_value, reward, log_probability, entropy)
            timestep += 1

        # Normalize
        game_duration[episode] = timestep 
        actions[0][episode] /= timestep
        actions[1][episode] /= timestep
        actions[2][episode] /= timestep
        actions[3][episode] /= timestep 

        if episode % save_interval == 0:
            
            np.save("results/run/{}/{}.npy".format(environment,episode), np.array(video))

            tmp = {}
            df = pd.DataFrame(data = tmp)
            df['duration']= game_duration[0:episode]
            df['down'] = actions[0][0:episode]
            df['right'] = actions[1][0:episode]
            df['left'] = actions[2][0:episode]
            df['up'] = actions[3][0:episode]

            # Save data for plots
            df.to_csv("results/run/collision_v0/{}.csv".format(episode))


        print("Episode:", episode, "Timesteps:", timestep, "Actions: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(actions[0][episode], actions[1][episode], actions[2][episode], actions[3][episode]))

        # Update network
        l_PG, l_v, l_H = agent.update_network()
        player_position = np.array([np.random.random()*(env.width - 2*env.ball_size)+env.player_size, np.random.random()*(env.height - 2*env.ball_size)+env.player_size])
        agent.reset(player_position)