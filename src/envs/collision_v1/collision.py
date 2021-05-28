from src.core import *
from src.envs.collision_v1.entities import *
import numpy as np
import cv2 as cv
import torch

BALL_COLOR = (0,0,255) # red
BACKGROUND = (255, 230, 200) # light blue

class Collision_v1(Environment):
    '''
    Simple environment with entities (balls) bouncing around. The collisions between the balls are 
    assumed to be elastic. The goal of the agent is to avoid collisions with the balls as long as possible.
    For more information about this environment see: "doc/envs/collision_v0.pdf".

    Observation:
        The environment is fully observable by the agent:
        - numpy array shaped self.width * self.height * 3
    
    Reward:
        Reward is 1 for every survived timestep.
    '''

    def __init__(self, width, height, num_balls, player = None, ball_size = 20, player_size = 10, timestep = 1):
        assert width > 100 and height > 100
        
        # Initially the position of the agent is randomized.
        player_position = np.array([np.random.random()*(width - 8*ball_size)+3*player_size, np.random.random()*(height - 8*ball_size)+3*player_size])
        #player_position = np.array([100,100])
        if player is None:
            self.player = Debug(width, height, player_size, player_position, timestep)
        else:
            self.player = player#.reset(player_position) #Agent(width, height, player_size, player_position, timestep)

        self.width = width
        self.height = height
        self.timestep = timestep 
        self.num_balls = num_balls
        self.ball_size = ball_size
        self.player_size = player_size
        
        balls = []
        counter = 0
        # Making sure that new balls do not go on top of each other, and not on top of the agent.
        #np.random.seed(0)
        while counter < num_balls:
            pos = np.array([np.random.random()*(width - 2*ball_size)+ball_size, np.random.random()*(height - 2*ball_size)+ball_size])
            ball = Ball(width, height, pos, ball_size, BALL_COLOR, timestep)
            valid = True
            #print(self.player)
            if not np.sum((ball.position - self.player.position)**2) < (ball.radius + self.player.radius+10)**2:
                for b in balls:
                    if np.sum((b.position - ball.position)**2) < (ball.radius + b.radius)**2:
                        valid = False
            else:
                #print("Encountered invalid position!")
                valid = False 
            if valid:
                balls.append(ball)
                counter += 1
            #print(counter)
        self.balls = balls
        self.state = self.draw()
        #if not self.player.valid_state(self.player.position, self.balls):
        #    self.reset()
        
    
    def step(self, action):
        
        distances = self.distances()
        for i in range(self.num_balls):
            for j in range(i+1,self.num_balls):
                b1 = self.balls[i]
                b2 = self.balls[j]
                if distances[i][j] < (b1.radius + b2.radius)**2:
                    b1.update_speed(b2.speed, b2.position)
                    b2.update_speed(b1.speed, b1.position)
        
        for b in self.balls:
            b.move()
        
        done = not self.player.move(action, self.balls)
        self.state = self.draw()
        reward = not done
        info = None 
        positions = np.array(list(map(lambda x: x.position, self.balls))).flatten()
        player_position = self.player.position.flatten()
        a = np.array(list(map(lambda x: x.position, self.balls))) - player_position
        a = np.array(sorted(a, key=lambda x: np.sum(x**2))).flatten()
        pos_ = np.concatenate((a, player_position))


        return pos_, reward, done, info
         
    def reset(self):
        self.__init__(self.width, self.height, self.num_balls, self.player)
        positions = np.array(list(map(lambda x: x.position, self.balls))).flatten()
        player_position = self.player.position.flatten()
        pos_ = np.concatenate((positions, player_position))
        return pos_ 

    def render(self):
        cv.imshow("Blank", self.state)
        cv.waitKey(33)
    
    def draw(self):
        '''
        Draws the environment which is the observation of the agent.
        '''
        blank = np.zeros((self.width,self.height,3), dtype = "uint8")
        blank[:] = BACKGROUND

        for b in self.balls:
            cv.circle(blank, (int(round(b.position[0])), int(round(b.position[1]))), b.radius, b.color, thickness = -1, lineType=cv.LINE_AA)
        
        cv.circle(blank, (int(round(self.player.position[0])), int(round(self.player.position[1]))), self.player.radius, self.player.color, thickness = -1, lineType=cv.LINE_AA)
        return blank 

    def distances(self):
        '''
        Returns the distances between all the balls as a matrix
        '''
        positions = np.array(list(map(lambda x: x.position, self.balls)))
        stacked_positions = np.stack([positions for _ in range(positions.shape[0])], axis = 1)
        return np.sum((stacked_positions - positions)**2, axis = -1)