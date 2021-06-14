import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random as rd
import matplotlib.pyplot as plt

class DQN:
    def __init__(self, inp, wp):
        self.inp = inp
        self.wp = wp
        self.lr = 0.01
        self.input_size = 11 # 
        self.output_size = 2 # 오 왼
        self.target = np.empty
        self.model = Sequential()
        self.model.add(Dense(self.input_size, input_dim=self.input_size, activation='relu'))
        self.model.add(Dense(self.output_size, activation='softmax'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        self.model.summary()
        self.rAll = 0
    
    def step(self, action, loc, speed):
        wall = [0 , 10]
        r = speed
        
        way = ['L', 'R'][action]
        points = make_circle(loc, r)
        point = []
        if way == 'L':
            for i in points:
                if i[0] <= loc[0]:
                    point.append(i)
                if i[0] <= wall[0]:
                    return -1
        else:
            for i in points:
                if i[0] >= loc[0]:
                    point.append(i)            
                if i[0] >= wall[0]:
                    return -1        
        reward = 1

        return reward
    
    def ln(self):
        loc = [5,0]
        # dis = 0.9
        speed = np.random.randint(2,7)
        print(speed)
        rAll = []
        for e in range(100):        
            cho = np.arange(self.output_size)
        
            target = self.model.predict(self.inp)
            
            if rd.random() > e/100:
                action = rd.choice(cho)
            else:
                action = np.argmax(target)
    
            reward = self.step(action, loc, speed)
            
            if reward:
                target[0][action] = reward
            else:
                target[0][action] = reward
            
            print(target[0][action])
            self.model.fit(self.inp, target, epochs=1, verbose=0)
    
            rAll.append(reward)

        return rAll


def make_circle(loc, r):
    ang = np.linspace(0, 2 * np.pi, 21)
    points = np.empty((0,2), float)
    for i in ang:
        x, y = r * np.cos(i), r * np.sin(i)
        if -1e-6 < x and x < 1e-6: x = 0
        if -1e-6 < y and y < 1e-6: y = 0

        points = np.append(points, np.array([[x + loc[0], y + loc[1]]]), axis=0)
        # points = np.append(points, np.array([[x, y]]), axis = 0)
        
    return points

def make_oval(obs_loc, r, dis): # 장애물 위치, 장애물로부터 떨어질 거리(단반경), 회피를 시작하기 위한 장애물 과의 거리(장반경)
    ang = np.linspace(0, 2 * np.pi, 21)
    points = np.empty((0,2), float)
    for i in ang:
        x, y = dis * np.cos(i), dis * np.sin(i)
        if -1e-6 < x and x < 1e-6: x = 0
        if -1e-6 < y and y < 1e-6: y = 0
        x = x / ( dis / r )
        
        points = np.append(points, np.array([[x + obs_loc[0], y + obs_loc[1]]]), axis=0)
    
    return points


inp = np.array([[0,0,0,0,0,1,0,0,0,0,0]])
P = DQN(inp, 1)
rALL = P.ln()

plt.figure()
plt.plot(rALL)
plt.show()
