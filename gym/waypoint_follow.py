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
    episode = 1

    COLLIDE_THRESHOLD = .001

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    for i in range(episode):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

        env.render()
        planner = PurePursuitPlanner(conf, 0.17145+0.15875)

        laptime = 0.0
        start = time.time()
        speeds = [0]
        temp_obs = obs['scans'][0][360:720]

        while not done:
            
            step_obs = abs(np.average(temp_obs - obs['scans'][0][360:720]))

            if step_obs > COLLIDE_THRESHOLD or step_obs < -COLLIDE_THRESHOLD:
                print(f"{round(laptime, 2)}, {step_obs}, FLAG")
    

            temp_obs = obs['scans'][0][360:720]
            # _n = 36
            # obs_points = [
            #     np.average(obs['scans'][0][360:360+_n]), 
            #     np.average(obs['scans'][0][360+_n+1:360+(2*_n)]), 
            #     np.average(obs['scans'][0][360+(2*_n+1):360+(3*_n)]), 
            #     np.average(obs['scans'][0][360+(3*_n+1):360+(4*_n)]), 
            #     np.average(obs['scans'][0][360+(4*_n+1):360+(5*_n)]), 
            #     np.average(obs['scans'][0][360+(5*_n+1):360+(6*_n)]), 
            #     np.average(obs['scans'][0][360+(6*_n+1):360+(7*_n)]), 
            #     np.average(obs['scans'][0][360+(7*_n+1):360+(8*_n)]), 
            #     np.average(obs['scans'][0][360+(8*_n+1):360+(9*_n)]), 
            #     np.average(obs['scans'][0][360+(9*_n+1):360+(10*_n)])
            # ]
            
            
            # if min(obs_points) <= 2.:
            #     print(f"{laptime} - OBSTACLE!!!")
            #     plt.scatter(laptime, obs_points[0], c='C0')
            #     plt.scatter(laptime, obs_points[1], c='C1')
            #     plt.scatter(laptime, obs_points[2], c='C2')
            #     plt.scatter(laptime, obs_points[3], c='C3')
            #     plt.scatter(laptime, obs_points[4], c='C4')
            #     plt.scatter(laptime, obs_points[5], c='C5')
            #     plt.scatter(laptime, obs_points[6], c='C6')
            #     plt.scatter(laptime, obs_points[7], c='C7')
            #     plt.scatter(laptime, obs_points[8], c='C8')
            #     plt.scatter(laptime, obs_points[9], c='C9')



            resize_0_45 = obs['scans'][0][360:540]
            resize_45_0 = obs['scans'][0][540:720]
            resize_25_25 = obs['scans'][0][440:640]

            plt.subplot(1, 2, 1)
            plt.plot(np.arange(-45, 0, 0.25), np.flip(resize_45_0))
            plt.grid()
            plt.xlabel('degree')
            plt.xlim(-45,0)
            plt.ylabel('distance')
            plt.ylim(0,30)



            plt.subplot(1, 2, 2)
            plt.plot(np.arange(0, 45, 0.25), np.flip(resize_0_45))
            plt.grid()
            plt.xlabel('degree')
            plt.xlim(0,45)
            plt.ylabel('distance')
            plt.ylim(0,30)
            plt.pause(0.00001)
            plt.clf()
            # plt.cla()

            # if min(resize_25_25) <= 1.:
            #     plt.plot(np.arange(-25, 25, 0.25),np.flip(resize_25_25), c="r")

            speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
            speeds.append(speed)
            action = np.array([[steer, speed]])
            obs, step_reward, done, info = env.step(np.array(action))
            laptime += step_reward
            env.render(mode='human_fast')

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
