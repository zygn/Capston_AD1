import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from matplotlib import pyplot as plt 
import logging as log
from planner.purepursuit import PurePursuitPlanner
from planner.astar import AStarPlanner



if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    # with open('./obs_example/config_obs.yaml') as file:
    with open('./obs_new_round/config_obs.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    episode = 1
    log.basicConfig(level=log.INFO)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    for i in range(episode):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

        env.render()
        planner = PurePursuitPlanner(conf, 0.17145+0.15875)

        laptime = 0.0
        start = time.time()
        speeds = [0]

        while not done:
            desire_obs = list()

            planner.load_laser_point(obs['scans'][0])
            planner.load_poses(obs['poses_x'][0], obs['poses_y'][0])
            planner.get_obstacle_trajectory()
            current_pose = [obs['poses_x'][0],obs['poses_y'][0]]
            current_wps = planner.current_waypoint
            log.info(f"[current_wps]: {current_wps}")
            log.info(f"[current_pose]: {current_pose}")
            astar_flag = planner.find_obstacle_between_wpts()
            # astar_flag = False

            if astar_flag:
                obs_cord = planner.shortest_obs_pose 
                log.info(f"[obs_cord]: {obs_cord}")
                goal_idx = planner .set_goal(obs_cord) + 3
                goal_cord = planner.get_wpts_from_idx(goal_idx)
                log.info(f"[goal_cord]: {goal_cord}")

                a = AStarPlanner(1,1)
                _obs = {
                    'x': int(obs_cord[0] * 10),
                    'y': int(obs_cord[1] * 10)
                }
                _points = {
                    'current': {
                        'x': int(current_pose[0] * 10),
                        'y': int(current_pose[1] * 10)
                    },
                    'future': {
                        'x': int(goal_cord[0] * 10),
                        'y': int(goal_cord[1] * 10)
                    }
                }
                new_trac = a.plan(obstacle=_obs, waypoints=_points)
                if type(new_trac) == type(str):
                    log.warn(f"{new_trac}")
                else:
                    print(new_trac)


            speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
            speeds.append(speed)
            # speed = 1.5
            action = np.array([[steer, speed]])
            obs, step_reward, done, info = env.step(np.array(action))
            laptime += step_reward
            env.render(mode='human')
            # time.sleep(1000)

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
