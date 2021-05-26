from src.envs.collision_v0.collision import Collision_v0
from src.envs.collision_v0.agents.policy_gradient import Agent
import numpy as np 
import pandas as pd
import torch 

def collision_v0():

    # agent = Agent(400, 400, 10, 1)
    # env = Collision_v0(400, 400, 30)#, agent)

    # for _ in range(10000):
    #     action = env.player.get_action(env.state)
    #     ob, reward, done, _ = env.step(action)
    #     env.render()
    #     if done:
    #         env.reset()
    # Command line prints for checking the excistence of a GPU
    if (torch.cuda.is_available()):
        print("Cuda is available")
    else:
        print("Cuda unavailable")

    # Train loop
    episodes = 100000
    save_interval = 100
    actions = {} # dicitonary for actions taken
    game_duration = np.zeros(episodes) # game lengths for each episode
    actions[0] = np.zeros(episodes)
    actions[1] = np.zeros(episodes)
    actions[2] = np.zeros(episodes)
    actions[3] = np.zeros(episodes)
    agent = Agent(400,400,10,1)
    env = Collision_v0(400,400,30,agent)

    
    for episode in range(episodes):
        timestep = 0 # initialize time
        done = False
        video = []
        frame = env.reset() # (400, 400, 3)
        video.append(frame)
        while not done:
            # Choose action
            action, log_probability, state_value, entropy = agent.get_training_action(frame)
            
            actions[action.item()][episode] += 1
            
            # Apply action -> get frame
            frame, reward, done, _ = env.step(action.detach())
            video.append(frame)
            
            agent.store_step(state_value, reward, log_probability, entropy)
            timestep += 1

            #if episode % 100 == 0:
            #    env.render()
        # Update wins
        # if reward > 0:
        #     num_of_wins += 1
        #     win_array[episode] = 1
        # Normalize
        game_duration[episode] = timestep 
        actions[0][episode] /= timestep
        actions[1][episode] /= timestep
        actions[2][episode] /= timestep
        actions[3][episode] /= timestep 

        if episode % save_interval == 0:
            
            np.save("results/run/collision_v0/{}.npy".format(episode), np.array(video))

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

        # policy_losses[episode] = l_PG
        # value_losses[episode] = l_v
        # entropy_losses[episode] = l_H

        # game_duration[episode] = timestep
        # begin = max(0, episode-1000)
        # end = episode+1
        # win_rate[episode] = 1 / (end-begin)*sum(win_array[begin:end])

        # # Normalize
        # actions[0][episode] /= timestep
        # actions[1][episode] /= timestep
        # actions[2][episode] /= timestep

        # current_win_rate = win_rate[episode]

        # Save models
        #if (episode % save_interval == 0 or current_win_rate > best_win_rate+0.005) and episode > 1000:
            #torch.save(player.network.state_dict(), "./network_parameters/{:.3f}_{}.pth".format(win_rate[episode],episode))
        
        # if current_win_rate > best_win_rate+0.005 and episode > 1000:
        #     best_win_rate = current_win_rate
        
        # if episode % save_interval == 0:
        

        #     tmp = {}
        #     df = pd.DataFrame(data = tmp)
        #     df['duration']= game_duration[0:episode]
        #     df['entropy_loss'] = entropy_losses[0:episode]
        #     df['value_loss']= value_losses[0:episode]
        #     df['policy_loss'] = policy_losses[0:episode]
        #     df['stay'] = actions[0][0:episode]
        #     df['up'] = actions[1][0:episode]
        #     df['down'] = actions[2][0:episode]
        #     df['win_rate'] = win_array[0:episode]

        #     # Save data for plots
        #     #df.to_csv("./data/{}.csv".format(episode))
        
        # print("Episode: {}, Wins: {}, Duration: {}, Actions: [{:.2f}, {:.2f}, {:.2f}], Winrate: {:.3f}" \
        #     .format(episode,num_of_wins,game_duration[episode],actions[0][episode],actions[1][episode],actions[2][episode], current_win_rate))