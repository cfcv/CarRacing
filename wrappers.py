import gym
import numpy as np

SCALE       = 6.0        # Track scale
PLAYFIELD   = 2000/SCALE # Game over boundary
FPS         = 50         # Frames per second

 
class ObsColorNormalizer(gym.ObservationWrapper):

    def __init__(self, env):
        super(ObsColorNormalizer, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, [84, 84, 3], dtype=np.float32
        )
        self.env.verbose = 0
        print('Color ObsNormalizer')

    def observation(self, obs):
        obs = obs[:84,6:-6,:]
        return obs/255.0
    
class ObsGrayNormalizer(gym.ObservationWrapper):

    def __init__(self, env):
        super(ObsGrayNormalizer, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, [84, 84, 1], dtype=np.float32
        )
        self.env.verbose = 0
        print('Gray ObsNormalizer')

    def observation(self, obs):
        obs = obs[:84,6:-6,:]
        obs_gray = np.dot(obs[:84,:,:], [0.2989, 0.5870, 0.1140])
        return np.reshape(obs_gray/255.0, (84,84,1))
    
class StartSkip(gym.Wrapper):

    def __init__(self, env):
        super(StartSkip, self).__init__(env)
        self.env.verbose = 0
        print('StartSkip wrapper')
        
    def reset(self):
        _ = self.env.reset()
        count = 0
        while(count < 47):
            _ = self.env.step(None)
            count += 1
        return self.env.step(None)[0]

class RescaleAction(gym.ActionWrapper):
    def __init__(self, env):
        super(RescaleAction, self).__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        print('action wrapper')

    def action(self, action):
        steer = action[0]
        gas = 0
        break_ = 0
        
        if(action[1] < 0):
            break_ = abs(action[1])
        else:
            gas = action[1]

        return np.array([steer, gas, break_])


