import gym
import pyvirtualdisplay

#display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

env = gym.make('CarRacing-v0')
env.reset()
done = False

while(not done):
    action = env.action_space.sample()

    state, reward, done, _ = env.step(action)

print('DONE')
