import time
import yaml
import gym
import numpy as np
import tensorflow as tf
import random

from purpursuit import PurePursuitPlanner
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers, initializers
from argparse import Namespace
from collections import deque

def processing(obs_dict):
    ego_idx = float(obs_dict['ego_idx'])
    scans = obs_dict['scans']
    poses_x = obs_dict['poses_x'][0]
    poses_y = obs_dict['poses_y'][0]
    poses_theta = obs_dict['poses_theta'][0]
    linear_vels_x = obs_dict['linear_vels_x'][0]
    linear_vels_y = obs_dict['linear_vels_y'][0]
    ang_vels_z = obs_dict['ang_vels_z'][0]
    collisions = obs_dict['collisions'][0]
    lap_times = obs_dict['lap_times'][0]
    lap_counts = obs_dict['lap_counts'][0]

    state = np.array([ego_idx,poses_x,poses_y,poses_theta,linear_vels_x,linear_vels_y,ang_vels_z,collisions,lap_times,lap_counts])
    state = np.concatenate((state,scans),axis=None)
    state = np.reshape(state,(1,state.shape[0]))
    return state

class NN:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size

    def build_net(state_size,action_size):
        inputs = tf.keras.Input(shape=(state_size))
        x = Dense(128, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = Dense(128, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = Flatten()(x)
        predictions = Dense(action_size, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(loss='mse',optimizer=optimizers.Adam(lr=0.001))
        model.summary()
        return model

class Agent:
    def __init__(self, conf, state_size):
        self.conf = conf
        self.PurePursuitPlanner = PurePursuitPlanner(self.conf, 0.17145+0.15875)
        
        self.state_size = state_size
        self.speed = np.array([[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]) 
        self.speed_size = self.speed.shape[1]
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1

        self.q_net = NN.build_net(self.state_size,self.speed_size)
        self.target_q_net = NN.build_net(self.state_size,self.speed_size)
        self.target_q_update()

        self.batch_size = 64
        self.replay_memory = deque(maxlen= 100000)
        self.target_q_frequecy = 5

    def target_q_update(self):
        print('please wait...')
        print('target logging...')
        print()
        self.target_q_net.set_weights(self.q_net.get_weights())

    def get_speed(self,state):
        if np.random.rand() < self.epsilon:
            speed_accel = np.random.randint(self.speed_size)
            print('random_speed:',speed_accel)
            print()
        else:
            speed_accel = np.argmax(self.q_net.predict(state))
            print('true_speed:',speed_accel)
            print()
        
        return speed_accel

    def get_steer(self,pose_x,pose_y,pose_theta,tlad,vgain):
        speed, steer = self.PurePursuitPlanner.plan(pose_x, pose_y, pose_theta, tlad, vgain)
        return steer

    def append_sample(self,state,action,reward,next_state,done):
        self.replay_memory.append((state,action,reward,next_state,done))

    def train(self):

        if self.epsilon_min <= self.epsilon:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.replay_memory,self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.q_net.predict(states)
        target_val = self.target_q_net.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                temp = np.amax(target_val[i])
                target[i][actions[i]] = rewards[i] + self.gamma* (np.amax(target_val[i]))

        self.q_net.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def save_model(self):
        print('please wait...')
        print('updating weights...')
        print()
        self.q_net.save('dqn_model.h5')

    def load_model(self):
        print('please wait...')
        print('loading weights...')
        print()
        self.q_net.load_model('dqn_model.h5')

if __name__ == '__main__':
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    episode = 100
    step = 0

    #gym_enviroment 생성
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    
    #인공 신경망 만들기 위해서 processing
    state_size = processing(obs)
    state_size = state_size.shape[1]
    agent = Agent(conf, state_size)
    
    #주행 학습 
    for i in range(episode):
        #출발 관측상태
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        env.render()

        laptime = 0.0
        start = time.time()
        #주행 시작
        while not done:
            #주행 현재 관측상태
            current_obs = obs
            current_state = processing(obs)

            #agent 행동 수행
            speed = agent.get_speed(current_state)
            steer = agent.get_steer(current_obs['poses_x'][0], current_obs['poses_y'][0], current_obs['poses_theta'][0], 
                                            work['tlad'], work['vgain'])
            print('steer:',steer,'speed:',speed)

            #주행 speed와 steer를 넣고 주행한 후 다음 상태
            next_obs, step_reward, done, info = env.step(np.array([[steer,speed]]))
            next_state = processing(next_obs)
            
            #에이전트에 경험 주입
            agent.append_sample(current_state, speed, step_reward, next_state, done)
            if step >= agent.batch_size:
                agent.train()
                if step % agent.target_q_frequecy == 0:
                    agent.target_q_update()
                    agent.save_model()
            
            #현재 상태 다음상태로 변경
            obs = next_obs
            laptime += step_reward
            step += 1
            env.render(mode='human')

        print('episode:',episode,'epsilon:',agent.epsilon,'Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)