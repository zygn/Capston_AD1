
import numpy as np
from argparse import Namespace
import logging as log
from numba import njit

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''

    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

def nearest_point_on_trajectory_wo_jit(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''

    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

class PurePursuitPlanner:
    
    OBSTACLE_MIN_DEGREE = 360
    OBSTACLE_MAX_DEGREE = 720

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.position = []
        self.current_waypoint = []
        self.laser_point = []
        self.i, self.i2 = 0, 0


    def load_waypoints(self, conf):
        # load waypoints
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        self.global_waypoints = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
    

    def load_laser_point(self, obs):
        # load laser scans
        self.degreed_laser_point = obs[self.OBSTACLE_MIN_DEGREE:self.OBSTACLE_MAX_DEGREE]
        self.laser_point = obs

    def load_poses(self, x, y):
        self.position = [x, y]

    def get_wpts_from_idx(self, idx):
        return self.global_waypoints[idx]

    def get_obstacle_trajectory(self):
        _lsr = self.laser_point
        _dlsr = self.degreed_laser_point
        _desire_obs = []
        _expect_obs = []
        
        for i in range(1, len(_dlsr)-1):
            if (_dlsr[i] > 5.) and (_dlsr[i] < 15) and (_dlsr[i-1] * 1.4 < _dlsr[i]) or (_dlsr[i-1] * 0.6 > _dlsr[i]):
                si = i
                ei = i
                mi = i
                i += 1

                while (_dlsr[i] > 5. and (_dlsr[i-1] * 1.1 > _dlsr[i])) and (_dlsr[i-1] * 0.9 < _dlsr[i]) and i+1 < len(_dlsr):
                    if _dlsr[i] > _dlsr[mi]:
                        mi = i
                    i += 1
                
                ei = i

                tos = []
                tos.append(si)
                tos.append(ei)
                tos.append(mi)
                tos.append(_dlsr[mi])

                _desire_obs.append(tos)
            i += 1
        
        for i in range(len(_desire_obs)):
            omi = (_desire_obs[i][0] + _desire_obs[i][1]) // 2 + 360

            od = _lsr[omi]
            if omi > self.OBSTACLE_MAX_DEGREE - self.OBSTACLE_MIN_DEGREE:
                oth = np.deg2rad(((720 - omi) + 360) / 4)
            else:
                oth = omi

            x = self.position[0] + (od * np.cos(oth))
            y = self.position[1] + (od * np.sin(oth))
            _expect_obs.append([x, y])

        self.desire_obs = _desire_obs
        self.expect_obs = _expect_obs

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)

        self.i = i

        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]

            # speed
            self.i2 = i2
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]

            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None


    def find_obstacle_between_wpts(self): 
        _wpts = self.global_waypoints

        # for x in self.expect_obs:
        #     _, _, _, idx = nearest_point_on_trajectory_wo_jit(x, _wpts)
        #     _obs_idx.append(idx)
        _lowest_obs_point = []

        flag = False

        for i in self.expect_obs:
            _minimum = np.min(np.abs(self.expect_obs))
            if _minimum in i:
                _lowest_obs_point = i
        
        self.shortest_obs_pose = _lowest_obs_point

        if len(_lowest_obs_point) != 0:
            obs_x = _lowest_obs_point[0]
            obs_y = _lowest_obs_point[1]

            _between_wpts = _wpts[self.i2-5: self.i2+2]
            if len(_between_wpts) != 0:
                wps_near_x = _between_wpts[0][0]
                wps_near_y = _between_wpts[0][1]
                wps_far_x = _between_wpts[-1][0]
                wps_far_y = _between_wpts[-1][1]

                if wps_near_x <= obs_x and wps_far_x >= obs_x or wps_near_y <= obs_y and wps_far_y >= obs_y:
                    log.warn('obstacle detection')
                    flag = True

                if wps_near_x >= obs_x and wps_far_x <= obs_x or wps_near_y >= obs_y and wps_far_y <= obs_y:
                    log.warn('obstacle detection')
                    flag = True 

        return flag
        


    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        self.position = position
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        self.current_waypoint = lookahead_point
        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle
