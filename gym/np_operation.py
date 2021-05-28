
import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from matplotlib import pyplot as plt 

from planner.purepursuit import PurePursuitPlanner


if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('./example/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)

    COLLIDE_THRESHOLD = .05

    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta],    # Player 1
                                                        [conf.sx - 1, conf.sy + 2, conf.stheta]]))   # Player 2
    env.render()
    
    ### Planner
    planner = PurePursuitPlanner(conf, 0.17145+0.15875)

    laptime = 0.0
    start = time.time()
    speed = 1
    temp_obs = obs['scans'][0][360:720]
    
    ### init obs structure sample:
    """
        {'ego_idx': 0, 'scans': [array([0.2449455 , 0.24603289, 0.24713468, ..., 0.10711216, 0.10756797,
       0.10802973]), array([0.10802973, 0.10756797, 0.10711216, ..., 0.32490255, 0.32628515,
       0.32768581])], 'poses_x': [0.0, 0.1], 'poses_y': [0.0, 0.1], 'poses_theta': [1.37079632679, 1.37079632679], 
       'linear_vels_x': [0.0, 0.0], 'linear_vels_y': [0.0, 0.0], 'ang_vels_z': [0.0, 0.0], 
       'collisions': array([1., 1.]), 'lap_times': array([0.01, 0.01]), 'lap_counts': array([0., 0.])}
    """
    while not done:
        # player 1
        p1_speed, p1_steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        # player 2
        p2_speed, p2_steer = planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], 1-(speed*0.02), 1)
        obs, step_reward, done, info = env.step(np.array([[p1_steer, p1_speed],[p2_steer, p2_speed]])) # [player 1, player 2]
        laptime += step_reward
    
        step_obs = abs(np.average(temp_obs - obs['scans'][0][360:720]))

        if step_obs > COLLIDE_THRESHOLD:
            print(f"{round(laptime, 2)}, {step_obs}, FLAG")
    
        temp_obs = obs['scans'][0][360:720]

        # resize_0_45 = obs['scans'][0][360:540]
        # resize_45_0 = obs['scans'][0][540:720]

        # plt.subplot(1, 2, 1)
        # plt.plot(np.arange(-45, 0, 0.25), np.flip(resize_45_0))
        # plt.grid()
        # plt.xlabel('degree')
        # plt.xlim(-45,0)
        # plt.ylabel('distance')
        # plt.ylim(0,30)

        # plt.subplot(1, 2, 2)
        # plt.plot(np.arange(0, 45, 0.25), np.flip(resize_0_45))
        # plt.grid()
        # plt.xlabel('degree')
        # plt.xlim(0,45)
        # plt.ylabel('distance')
        # plt.ylim(0,30)
        # plt.pause(0.00001)
        # plt.clf()

        resize_0p = obs['scans'][0][360:720]
        resize_1p = obs['scans'][1][360:720]

        plt.subplot(2,1,1)
        plt.plot(np.arange(-45, 45, 0.25), np.flip(resize_0p))
        plt.grid()
        plt.xlabel('1P degree')
        plt.xlim(-45,45)
        plt.ylabel('distance')
        plt.ylim(0,30)

        plt.subplot(2,1,2)
        plt.plot(np.arange(-45, 45, 0.25), np.flip(resize_1p))
        plt.grid()
        plt.xlabel('2P degree')
        plt.xlim(0,45)
        plt.ylabel('distance')
        plt.ylim(0,30)
        plt.pause(0.00001)
        plt.clf()
        
        env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)