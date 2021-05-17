import time
import yaml
import gym
import numpy as np
import tensorflow as tf
import random
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

class Actor_Net(tf.keras.Model):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.actor_fc1 = Dense(128, activation='tanh')
        self.actor_fc2 = Dense(128, activation='tanh')
        self.actor_out = Dense(action_size, activation='softmax',
                               kernel_initializer=optimizers.RandomUniform(-1e-3,1e-3))

    def call(self, x):
        actor_x = self.actor_fc1(x)
        actor_x = self.actor_fc2(actor_x)
        policy = self.actor_out(actor_x)
        return policy

class Critic_Net(tf.keras.Model):
    def __init__(self):
        super(Critic_Net, self).__init__()
        self.critic_fc1 = Dense(128, activation='relu')
        self.critic_fc2 = Dense(128, activation='relu')
        self.critic_out = Dense(1,
                                kernel_initializer=optimizers.RandomUniform(-1e-3,1e-3))

    def call(self, x):
        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return value

class OUnoise():
    def __init__(self,action_size):
        self.eps_mean = 0.0
        self.eps_std = 0.1
        self.eps_dt = 0.01
        self.eps = 0
        self.action_size = action_size
        
    def make_noise(self):
        noise = self.eps + self.eps_theta*(self.eps_mean - self.eps)*self.eps_dt + self.eps_std*np.sqrt(self.eps_dt)*np.random.normal(size= self.action_size)
        return noise

class Agent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.replay_memory = deque(maxlen=100000)
        self.batch_size = 64
        self.learning_rate = 0.001
        self.target_q_frequecy = 10

        self.actor_net = Actor_net(self.action_size)
        self.target_actor_net = Actor_Net(self.action_size)
        
        self.critic_net = Critic_Net()
        self.target_critic_net = Critic_Net()

        self.actor_optimzer = optimizers.Adam(self.learning_rate)
        self.critic_optimzer = optimizers.Adam(self.learning_rate)
        self.ou_noise = OUnoise()

        self.critic_net = Critic_net.build_net(self.action_size)
        self.target_critic_net = Critic_net.build_net(self.action_size)

        self.target_q_update()

    def get_action(self,state):
        #action[0] = speed, action[1] = steer
        mu_model = self.actor_net(state)
        noise = self.ou_noise.make_noise(self.action_size)
        action = mu_model + noise
        speed = np.clip(action[0],11)
        steer = np.clip(action[1],-4,4)
        action = np.array([[speed,steer]])
        return action

    def append_sample(self,state,action,reward,next_state,done):
        self.replay_memory.append((state,aciton,next_state,done))

    def train(self):
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

        action_target = self.actor_net.predict(states)
        critic_target = self.critic_net.predict(states)

        action_target_val = self.target_actor_net.predict(next_states)
        critic_target_val = self.target_critic_net.predict(next_states)

        self.actor_net.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

        

    def save_model(self):
        self.actor_net.save_model('ddpg_actor.h5')
        self.critic_net.save_model('ddpg_critic.h5')

    def load_model(self):
        self.actor_net.load_model('ddpg_actor.h5')
        self.critic_net.load_model('ddpg_critic.h5')

if __name__ == '__main__':
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    episode = 100000
    step = 0

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    
    #인공 신경망 만들기 위해서 processing
    obs = processing(obs)
    state_size = obs.shape[1]
    agent = Agent(state_size)
    
    #주행 학습 
    for i in range(episode):
        #출발 관측상태
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        env.render()
        obs = processing(obs)

        laptime = 0.0
        start = time.time()
        #주행 시작
        while not done:
            #주행 현재 관측상태
            current_obs = obs

            action = agent.get_action(current_obs)
            steer,speed = action

            #주행 speed와 steer를 넣고 주행한 후 다음 상태
            next_obs, step_reward, done, info = env.step(np.array([[steer,speed]]))
            next_obs = processing(next_obs)
            
            #에이전트에 경험 주입
            agent.append_sample(current_obs, action, step_reward, next_obs, done)
            if step >= agent.batch_size:
                agent.train()
                if step % agent.target_q_frequecy == 0:
                    agent.target_q_update()

            obs = next_obs
            laptime += step_reward
            step += 1
            env.render(mode='human')
        #agent.save_model()
        print('episode:',epsiode,'epsilon:',agent.epsilon,'Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
    