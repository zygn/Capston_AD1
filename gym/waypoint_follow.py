import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from matplotlib import pyplot as plt 
from planner.purepursuit import PurePursuitPlanner



if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    # with open('./obs_example/config_obs.yaml') as file:
    with open('./test_map/conf.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    episode = 1


    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    for i in range(episode):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

        env.render()
        planner = PurePursuitPlanner(conf, 0.17145+0.15875)

        laptime = 0.0
        start = time.time()
        speeds = [0]
        temp_obs = obs['scans'][0][360:720]
<<<<<<< Updated upstream

        while not done:
            desire_obs = list()
            print(f"X: {obs['poses_x']}, Y: {obs['poses_y']}")
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
            
            print(desire_obs)

            temp_obs = obs['scans'][0][360:720]

=======
        # log.info(f"[temp_obs]: {temp_obs}")
        while not done:
            desire_obs = list()

            planner.load_laser_point(obs['scans'][0])
            planner.load_poses(obs['poses_x'][0], obs['poses_y'][0])
            planner.get_obstacle_trajectory()
            obs_cord = planner.expect_obs

            current_pose = [obs['poses_x'][0],obs['poses_y'][0]]
            log.info(f"[obs_cord]: {obs_cord}")
            current_wps = planner.current_waypoint
            log.info(f"[current_wps]: {current_wps}")
            log.info(f"[current_pose]: {current_pose}")
            astar_flag = planner._find_obstacle_between_wpts()

            local.input_poses(current_pose, obs_cord)
            goal_idx = local.set_goal() + 3
            goal_cord = planner.get_wpts_from_idx(goal_idx)


            if astar_flag:
                a = AStarPlanner(1,1)
                _obs = {
                    'x': int(obs_cord[1][0] * 10),
                    'y': int(obs_cord[1][1] * 10)
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
                new_trac = a.plan(obstacle=_obs, waypoints=_points, conf={'show_animation': True})
                print(new_trac)
>>>>>>> Stashed changes

            speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
            speeds.append(speed)
            action = np.array([[steer, speed]])
            obs, step_reward, done, info = env.step(np.array(action))
            laptime += step_reward
            env.render(mode='human')
            # time.sleep(1000)

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
