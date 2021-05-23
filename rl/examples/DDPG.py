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

class OUnoise:
    def __init__(self):
        self.eps_mean = 0.
        self.eps_std = 0.1
        self.eps_theta = 0.1
        self.eps_dt = 0.01
        self.eps = 0

    def make_noise(self,speed_size):
        noise = self.eps + self.eps_theta*(self.eps_mean - self.eps)*self.eps_dt + self.eps_std*np.sqrt(self.eps_dt)*np.random.normal(size=speed_size)
        return noise
        
class Agent:
    def __init__(self, conf, state_size):
        self.conf = conf
        self.planner = PurePursuitPlanner(self.conf, 0.17145+0.15875)
        
        self.state_size = state_size
        self.min_speed = 1
        self.max_speed = 20
        self.action_size = 1 
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.tau = 0.02
        self.target_frequency = 10

        self.actor = self.build_actor_net(self.state_size,self.action_size)
        self.target_actor = self.build_actor_net(self.state_size,self.action_size)
        self.critic = self.build_critic_net(self.action_size)
        self.target_critic = self.build_critic_net(self.action_size)

        self.actor_optimizer = optimizers.Adam(self.learning_rate)
        self.critic_optimizer = optimizers.Adam(self.learning_rate)
        
        self.noise = OUnoise()
        self.replay_memory = deque(maxlen=100000)
        self.batch_size = 64
    
    def build_actor_net(self,state_size,action_size):
        inputs = tf.keras.Input(shape=(state_size))
        out = Dense(256, activation="relu",kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        out = Dense(256, activation="relu",kernel_initializer=initializers.VarianceScaling(scale=2.))(out)
        outputs = Dense(action_size, activation="linear", kernel_initializer=initializers.VarianceScaling(scale=2.))(out)

        actor_model = tf.keras.Model(inputs, outputs)
        actor_model.summary()
        return actor_model

    def build_critic_net(self,action_size):
        inputs = tf.keras.Input(shape=(action_size))
        out = Dense(256, activation="relu",kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        out = Dense(256, activation="relu",kernel_initializer=initializers.VarianceScaling(scale=2.))(out)
        outputs = Dense(1, activation="linear", kernel_initializer=initializers.VarianceScaling(scale=2.))(out)

        critic_model = tf.keras.Model(inputs, outputs)
        critic_model.summary()
        return critic_model

    def append_sample(self,state,action,reward,next_state,done):
        self.replay_memory.append((state,action,reward,next_state,done))

    def get_speed(self,state):
        speed = self.actor.predict(state)
        noise = self.noise.make_noise(self.action_size)
        speed = speed + noise
        speed = np.clip(speed, self.min_speed, self.max_speed)[0][0]
        print('b_speed:',speed)
        
        if np.isnan(speed):
            speed = np.mean([i for i in range(1,21)])
        print('a_speed:',speed)
        
        return speed

    def get_steer(self,pose_x,pose_y,pose_theta,tlad,vgain):
        _, steer = self.planner.plan(pose_x, pose_y, pose_theta, tlad, vgain)
        return steer

    @tf.function
    def train(self):
        mini_batch = random.sample(self.replay_memory,self.batch_size)

        states = np.array([x[0][0] for x in mini_batch])
        actions = np.array([x[1] for x in mini_batch])
        rewards = np.array([x[2] for x in mini_batch])
        next_states = np.array([x[3][0] for x in mini_batch])
        dones = np.array([x[4] for x in mini_batch])

        critic_trainable_params = self.critic.trainable_variables
        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_value = rewards + self.gamma*(1 - dones)*self.target_critic(target_actions)

            q_value = self.critic(actions)
            critic_loss = tf.math.reduce_mean(tf.math.square(target_value - q_value))
        
        critic_grad = tape.gradient(critic_loss, critic_trainable_params)
        self.critic_optimizer.apply_gradients(zip(critic_grad, critic_trainable_params))

        actor_trainable_params = self.actor.trainable_variables
        
        with tf.GradientTape() as tape:
            action = self.actor(states)
            critic_value = self.critic(action)
            actor_loss = -tf.math.reduce_mean(tf.math.square(critic_value))

        actor_grad = tape.gradient(actor_loss, actor_trainable_params)
        self.actor_optimizer.apply_gradients(zip(actor_grad, actor_trainable_params))

    def target_update(self):
        new_weights = []
        target_variables = self.target_critic.weights
        
        for i, variable in enumerate(self.critic.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))
        
        self.target_critic.set_weights(new_weights)        

        new_weights = []
        target_variables = self.target_actor.weights
        
        for i, variable in enumerate(self.actor.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))
        
        self.target_actor.set_weights(new_weights)

    def save_model(self):
        print('saving network...')
        print('Please wait...')
        self.actor.save('actor.h5')
        self.critic.save('critic.h5')

    def load_model(self):
        print('weight loading...')
        print('Please wait...')
        self.actor.load_model('actor.h5')
        self.critic.load_model('critic.h5')

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
            #print('steer:',steer,'speed:',speed)

            #주행 speed와 steer를 넣고 주행한 후 다음 상태
            next_obs, step_reward, done, info = env.step(np.array([[steer,speed]]))
            next_state = processing(next_obs)
            #print('step_reward:',step_reward)
            
            #에이전트에 경험 주입
            agent.append_sample(current_state, speed, step_reward, next_state, done)
            if step >= agent.batch_size:
                agent.train()
                if step % agent.target_frequency == 0:
                    agent.target_update()
                    agent.save_model()
            
            #현재 상태 다음상태로 변경
            obs = next_obs
            laptime += env.timestep
            step += 1
            env.render(mode='human')

        print('episode:',episode,'Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)