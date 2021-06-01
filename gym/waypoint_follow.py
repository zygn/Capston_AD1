import time
import yaml
import gym
import numpy as np
import logging as log
from argparse import Namespace
from matplotlib import pyplot as plt 
from planner.purepursuit import PurePursuitPlanner
from obs_avoidance import TempPath
from planner.astar import AStarPlanner




if __name__ == '__main__':

    log.basicConfig(format='[MainThread]:[%(levelname)s]:%(message)s', level=log.INFO)

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    # with open('./obs_example/config_obs.yaml') as file:
    with open('./obs_new_round/config_obs.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    episode = 1


    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    for i in range(episode):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

        env.render()
        planner = PurePursuitPlanner(conf, 0.17145+0.15875)
        local = TempPath(conf)

        laptime = 0.0
        start = time.time()
        speeds = [0]

        temp_obs = obs['scans'][0][360:720]
        log.info(f"[temp_obs]: {temp_obs}")
        while not done:
            desire_obs = list()
            for i in range(1, len(temp_obs) - 1):
                if (temp_obs[i] > 5. and temp_obs[i] < 15.) and (temp_obs[i-1] * 1.4 < temp_obs[i] or temp_obs[i-1] * 0.6 > temp_obs[i]):
                    start_idx_temp = i
                    end_idx_temp = i
                    max_idx_temp = i
                    i += 1
                    
                    while temp_obs[i] > 5. and temp_obs[i-1] * 1.1 > temp_obs[i] and temp_obs[i-1] * 0.9 < temp_obs[i] and (i+1 < len(temp_obs)):
                        if temp_obs[i] > temp_obs[max_idx_temp]:
                            max_idx_temp = i
                        i += 1
                    
                    end_idx_temp = i

                    temp_obs_step = [0]*4
                    temp_obs_step[0] = start_idx_temp
                    temp_obs_step[1] = end_idx_temp
                    temp_obs_step[2] = max_idx_temp
                    temp_obs_step[3] = temp_obs[max_idx_temp]

                    desire_obs.append(temp_obs_step)
                i += 1
            
            log.info(f"[desire_obs]: {desire_obs}")
            
            obstacles_dist = []
            obstacles_theta = []
            obs_cord = []

            for i in range(len(desire_obs)):
                obs_mid_idx = (desire_obs[i][0] + desire_obs[i][1])//2 + 360
                obs_dist = obs['scans'][0][obs_mid_idx]
                if obs_mid_idx > 540:
                    obs_theta = np.deg2rad(((720-obs_mid_idx)+360) / 4)
                else:
                    obs_theta = obs_mid_idx

                x = obs['poses_x'][0] + (obs_dist * np.cos(obs_theta))
                y = obs['poses_y'][0] + (obs_dist * np.sin(obs_theta))
                obs_cord.append([x,y])
                
            log.info(f"[obs_cord]: {obs_cord}")
            current_pose = [obs['poses_x'][0],obs['poses_y'][0]]
            current_wps = planner.current_waypoint
            log.info(f"[current_wps]: {current_wps}")
            log.info(f"[current_pose]: {current_pose}")

            if len(obs_cord) != 0 and len(current_wps) != 0 and int(laptime) == 5:
                a = AStarPlanner(conf.resolution,1)
                _obs = obs_cord
                _points = {
                    'current': {
                        'x': current_pose[0],
                        'y': current_pose[1]
                    },
                    'future': {
                        'x': current_wps[0]+10,
                        'y': current_wps[1]+10
                    }
                }
                a.plan(obstacle=_obs, waypoints=_points, conf={'show_animation': True})

            speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
            speed = 1
            # steer, speed = [0,10]
            # print(obs['poses_x'][0], obs['poses_y'][0])
            speeds.append(speed)
            action = np.array([[steer, speed]])
            obs, step_reward, done, info = env.step(np.array(action))
            laptime += step_reward
            env.render(mode='human')

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
