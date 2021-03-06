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