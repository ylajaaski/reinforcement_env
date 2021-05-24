from src.core import *
import numpy as np
import cv2 as cv

BALL_COLOR = (0,0,255) # red
PLAYER_COLOR = (31, 120, 10) # dark green
BACKGROUND = (255, 230, 200)  # light blue

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
            self.player = Agent(width, height, player_size, player_position, timestep)

        self.width = width
        self.height = height
        self.timestep = timestep 
        self.player = player
        self.num_balls = num_balls
        
        balls = []
        counter = 0
        # Making sure that new balls do not go on top of each other, and not on top of the agent.
        while counter < num_balls:
            pos = np.array([np.random.random()*(width - 2*ball_size)+ball_size, np.random.random()*(height - 2*ball_size)+ball_size])
            ball = Ball(width, height, pos, ball_size, BALL_COLOR, timestep)
            valid = True
            for b in balls:
                if np.sum((b.position - ball.position)**2) < (ball.radius + b.radius)**2 or np.sum((ball.position - player.position)**2) < (ball.radius + player.radius)**2:
                    valid = False
            if valid:
                balls.append(ball)
                counter += 1
        
        self.balls = balls
        self.state = self.draw()
        
    
    def step(self, action):
        
        # TODO: Optimize in the future 
        distances = self.distances()
        for i, b1 in enumerate(self.balls):
            for j, b2 in enumerate(self.balls):
                if distances[i][j] < (b1.radius + b2.radius)**2 and i != j:
                    b1.update_speed(b2.speed, b2.position)
        
        for b in self.balls:
            b.move()
        
        done = not self.player.move(action, self.balls)
        self.state = self.draw()
        reward = 0
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

class Ball(Entity):
    
    def __init__(self, env_width, env_height, position, radius, color, timestep):
        self.env_width = env_width
        self.env_height = env_height 
        self.radius = radius
        self.color = color
        self.timestep = timestep 
        self.position = position 
        self.speed = np.array([np.random.random()*2-1, np.random.random()*2-1])
        self.changed_speed = 0
    
    def move(self):
        
        self.speed += self.changed_speed
        self.changed_speed = 0
        delta_x = self.speed * self.timestep
        self.position += delta_x
        if (self.position[0] < self.radius and delta_x[0] < 0) or \
            (self.position[0] > self.env_width - self.radius and delta_x[0] > 0):
            self.speed[0] *= -1
        if (self.position[1] < self.radius and delta_x[1] < 0) or \
            (self.position[1] > self.env_height - self.radius and delta_x[1] > 0):
            self.speed[1] *= -1   
    
    def update_speed(self, speed, position):
        new_pos = self.position + self.speed * self.timestep 
        new_pos_ = position + speed * self.timestep 
        if np.sum((new_pos-new_pos_)**2) < np.sum((self.position - position)**2):
            if not (position == self.position).all():
                self.changed_speed -= (self.speed - speed) @ (self.position - position) / np.sum((self.position - position)**2) * (self.position - position)

class Agent(Player):
    
    def __init__(self, env_width, env_height, radius, position, timestep):
        self.env_width = env_width
        self.env_height = env_height
        self.radius = radius
        self.position = position
        self.color = PLAYER_COLOR
        self.speed = np.zeros(2)
        self.max_speed = 6
        self.timestep = timestep 
    
    def valid_state(self, position, ball_list):
        positions = np.array(list(map(lambda x: x.position, ball_list)))
        sizes = np.array(list(map(lambda x: x.radius, ball_list)))
        stacked_positions = np.stack([self.position for _ in range(positions.shape[0])], axis = 1)
        if np.any(np.sum((stacked_positions - positions.T)**2, axis = 0) - (self.radius + sizes)**2 < 0):
            return False
        elif position[0] < self.radius or position[0] > self.env_width - self.radius or \
             position[1] < self.radius or position[1] > self.env_height - self.radius:
            return False
        else:
            return True
    
    def move(self, action, ball_list):
        position = self.position + action
        if self.valid_state(position, ball_list):
            self.position += action * self.timestep 
            return True
        else:
            return False

class Debug(Player):

    def __init__(self, env_width, env_height, radius, position, timestep):
        self.env_width = env_width
        self.env_height = env_height
        self.radius = radius
        self.position = position
        self.color = PLAYER_COLOR
        self.speed = np.zeros(2)
        self.max_speed = 6
        self.timestep = timestep 
    
    def valid_state(self, position, ball_list):
        return True
    
    def move(self, action, ball_list):
        position = self.position + action
        if self.valid_state(position, ball_list):
            self.position += action * self.timestep 
            return True
        else:
            return False