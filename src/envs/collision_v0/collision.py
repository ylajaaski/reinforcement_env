from src.core import *
from src.envs.collision_v0.entities import *
import numpy as np
import cv2 as cv

BALL_COLOR = (0,0,255) # red
BACKGROUND = (255, 230, 200) # light blue

class Collision(Environment):
    '''
    Simple environment with entities (balls) bouncing around. The goal of the agent is to avoid
    collisions with the balls as long as possible.
    '''

    def __init__(self, width, height, num_balls, player = None, ball_size = 20, player_size = 10, timestep = 1):
        assert width > 100 and height > 100
        
        # Initially the position of the agent is randomized.
        player_position = np.array([np.random.random()*(width - 2*ball_size)+player_size, np.random.random()*(height - 2*ball_size)+player_size])
        if player is None:
            self.player = Debug(width, height, player_size, player_position, timestep)
        else:
            self.player = player #Agent(width, height, player_size, player_position, timestep)

        self.width = width
        self.height = height
        self.timestep = timestep 
        self.num_balls = num_balls
        
        balls = []
        counter = 0
        # Making sure that new balls do not go on top of each other, and not on top of the agent.
        while counter < num_balls:
            pos = np.array([np.random.random()*(width - 2*ball_size)+ball_size, np.random.random()*(height - 2*ball_size)+ball_size])
            ball = Ball(width, height, pos, ball_size, BALL_COLOR, timestep)
            valid = True
            for b in balls:
                if np.sum((b.position - ball.position)**2) < (ball.radius + b.radius)**2 or np.sum((ball.position - self.player.position)**2) < (ball.radius + self.player.radius)**2:
                    valid = False
            if valid:
                balls.append(ball)
                counter += 1
        
        self.balls = balls
        self.state = self.draw()
        
    
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
        return self.state, reward, done, info
         
    def reset(self):
        self.__init__(self.width, self.height, self.num_balls, self.player)

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
