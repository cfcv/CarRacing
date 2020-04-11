import gym
import numpy as np

SCALE       = 6.0        # Track scale
PLAYFIELD   = 2000/SCALE # Game over boundary
FPS         = 50         # Frames per second

class ObsNormalizer(gym.ObservationWrapper):

    def __init__(self, env):
        super(ObsNormalizer, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, [84, 96, 1], dtype=np.float32
        )
        self.env.verbose = 0

    def observation(self, obs):
        obs_gray = np.dot(obs[:84,:,:], [0.2989, 0.5870, 0.1140])
        return np.reshape(obs_gray/255.0, (84,96,1))  

class RewardRoute(gym.Wrapper):

    def __init__(self, env):
        super(RewardRoute, self).__init__(env)
        self.env.verbose = 0

    def on_segment(self, p, q, r):
        if((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        else:
            return False

    def orientation(self, p, q, r):
        val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))

        if(val == 0):
            return 0 #colinear
        elif(val > 0):
            return 1 #clockwise
        else:
            return 2 #counterclockwise


    def intersection(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        if(o1 != o2 and o3 != o4): 
            return True

        #// Special Cases 
        #// p1, q1 and p2 are colinear and p2 lies on segment p1q1 
        if(o1 == 0 and self.on_segment(p1, p2, q1)):
            return True 

        #// p1, q1 and p2 are colinear and q2 lies on segment p1q1 
        if(o2 == 0 and self.on_segment(p1, q2, q1)):
            return True 

        #// p2, q2 and p1 are colinear and p1 lies on segment p2q2 
        if(o3 == 0 and self.on_segment(p2, p1, q2)):
            return True 

        #// p2, q2 and q1 are colinear and q1 lies on segment p2q2 
        if(o4 == 0 and self.on_segment(p2, q1, q2)):
            return True 

        return False# // Doesn't fall in any of the above cases 


    def is_inside_polygon(self, p_car, points):
        assert (len(points) == 4), 'Polygon must have 4 vertices'

        #// Create a point for line segment from p to infinite 
        extreme = (2000, p_car[1]) 

        intersections = []
        intersections.append(self.intersection(points[0], points[1], p_car, extreme))
        intersections.append(self.intersection(points[1], points[2], p_car, extreme))
        intersections.append(self.intersection(points[2], points[3], p_car, extreme))
        intersections.append(self.intersection(points[3], points[0], p_car, extreme))

        return sum(intersections)%2

    def is_outside_road(self, p_car):
        for i, item in enumerate(self.road_poly):
            polygon = item[0]
            if(self.is_inside_polygon(p_car, polygon)):
                #print('inside:', i)
                return False
        return True
    
    def step(self, action):
        state, step_reward, done, _ = self.env.step(action)
        return state, step_reward, done, {}