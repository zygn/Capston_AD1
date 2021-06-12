import time
import yaml
import gym
import numpy as np

from argparse import Namespace
from matplotlib import pyplot as plt 

import logging as log

from planner.purepursuit import PurePursuitPlanner
from planner.astar import AStarPlanner
from agent.DQN import NN,Agent,processing,make_state

if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('./obs_example/config_obs.yaml') as file:
    # with open('./obs_new_round/config_obs.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    episode = 100
    log.basicConfig(level=log.INFO)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    ex_state = processing(obs)
    state_size = ex_state.shape[1]
    agent = Agent(state_size)
    rewards = []

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
                goal_idx = planner.set_goal(obs_cord)
                log.info(f"[i, i2]: {planner.i, planner.i2}")
                log.info(f"[goal_idx]: {goal_idx}")
                goal_cord = planner.get_wpts_from_idx(planner.i2+3)
                log.info(f"[goal_cord]: {goal_cord}")
                step = 0

                a = AStarPlanner(1, 0, show_animation= False)

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

                if str(type(new_trac)) == "<class 'str'>":
                    log.warn(f"{new_trac}")
                else:
                    new_trac = np.array(new_trac)
                    new_trac = make_state(new_trac)

                #주행 현재 관측상태
                current_obs = obs
                current_state = processing(obs)

                #agent 행동 수행
                action = agent.get_action(current_state)
                print('steer:',action[0],'speed:',action[1])

                #주행 speed와 steer를 넣고 주행한 후 다음 상태
                next_obs, step_reward, done, info = env.step(np.array([action]))
                next_state = processing(next_obs)
                action_num = agent.return_action_num(action)
            
                #에이전트에 경험 주입
                agent.append_sample(current_state, action_num, step_reward, next_state, done)
                if step >= agent.batch_size:
                    agent.train()
                    if step % agent.target_q_frequecy == 0:
                        agent.target_q_update()
                        agent.save_model()
                                  
            else:
                speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
            speeds.append(speed)
            print('speed,steer:',speed,steer)
            #print('boy:',len(speeds))
            # speed = 1.5
            action = np.array([[steer, speed]])
            obs, step_reward, done, info = env.step(np.array(action))
            #print('step_reward:',step_reward)
            laptime += step_reward
            env.render(mode='human')
            # time.sleep(1000)
        rewards.append(laptime)
        print('rewards:',rewards)

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
