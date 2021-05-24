from src.core import *
import numpy as np
import cv2 as cv

class Collision(Environment):

    def __init__(self, width, height, num_balls, player = None, ball_size = 20, player_size = 10, timestep = 1):
        
        if player is None:
            self.player = Debug(width, height, player_size, (0,0), timestep)
        else:
            self.player = Agent(width, height, player_size, (0,0), timestep)

        self.width = width
        self.height = height
        self.timestep = timestep 
        self.player = player
        self.num_balls = num_balls
        
        balls = []
        counter = 0
        while counter < num_balls:
            pos = np.array([np.random.random()*(width - 2*ball_size)+ball_size, np.random.random()*(height - 2*ball_size)+ball_size])
            ball = Ball(width, height, pos, ball_size, (0,0,255), timestep)
            valid = True
            for b in balls:
                if np.sum((b.position - ball.position)**2) < (ball.radius + b.radius)**2:
                    valid = False
            if valid:
                balls.append(ball)
                counter += 1
        
        self.balls = balls
        self.state = self._draw()
        
    
    def step(self, action):
        
        # Optimize in the future 
        distances = self._distances()
        for i, b1 in enumerate(self.balls):
            for j, b2 in enumerate(self.balls):
                if distances[i][j] < (b1.radius + b2.radius)**2 and i != j:
                    b1.update_speed(b2.speed, b2.position)

        
        for b in self.balls:
            b.move()
        
        action = np.random.random(2)*12 - 6
        done = not self.player.move(action, self.balls)
        self.state = self._draw()
        # TODO: return obseration, action, reward, done 
        if done:
            self.reset()
        
        
    def reset(self):
        self.__init__(self.width, self.height, self.player, self.num_balls)

    
    def render(self):
        cv.imshow("Blank", self.state)
        cv.waitKey(33)
    
    def _draw(self):
        blank = np.zeros((self.width,self.height,3), dtype = "uint8")
        blank[:] = (255, 230, 200) 

        for b in self.balls:
            cv.circle(blank, (int(round(b.position[0])), int(round(b.position[1]))), b.radius, (0, 0, 255), thickness = -1, lineType=cv.LINE_AA)
        
        cv.circle(blank, (int(round(self.player.position[0])), int(round(self.player.position[1]))), self.player.radius, (31, 120, 10), thickness = -1, lineType=cv.LINE_AA)
        return blank 


    
    def _distances(self):
        result = np.array(list(map(lambda x: x.position, self.balls)))
        a = np.stack([result for _ in range(result.shape[0])], axis = 1)
        return np.sum((a - result)**2, axis = -1)


class Ball(Entity):
    
    def __init__(self, env_width, env_height, position, radius, color, timestep):
        self.env_width = env_width
        self.env_height = env_height 
        self.radius = radius
        self.color = color
        self.timestep = timestep 
        self.position = position #np.array([np.random.random()*755+22.5, np.random.random()*765+22.5])
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
        self.color = (31, 120, 10) 
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
        self.color = (31, 120, 10) 
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