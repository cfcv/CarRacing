import argparse
import os
import tensorflow as tf
from tqdm import tqdm
import wrappers as wp
from tf_agents.environments import suite_gym, tf_py_environment
import numpy as np
import pyvirtualdisplay

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in tqdm(range(num_episodes)):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='Directory That several different seed checkpoints were saved')
parser.add_argument('-N', help='Number of episodes')
args = parser.parse_args()

display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

print("Root directory: ", args.logdir)
subfolders = os.listdir(args.logdir)
print("Founded subfolders:", subfolders)

average = []
best_average = []

env_py = suite_gym.load('CarRacing-v0', max_episode_steps=1000, gym_env_wrappers=(wp.render_wrapper, wp.ObsGrayNormalizer, wp.StartSkip, wp.RescaleAction_clip, wp.MultInput_waypointVec,))
eval_env = tf_py_environment.TFPyEnvironment(env_py)

for subfolder in subfolders:
    path = args.logdir + subfolder
    
    policy = tf.compat.v2.saved_model.load(path+"/policy")
    avg = compute_avg_return(eval_env, policy, num_episodes=int(args.N))
    try:
        best_policy = tf.compat.v2.saved_model.load(path+"/best_policy")
        best_avg = compute_avg_return(eval_env, best_policy, num_episodes=int(args.N))
    except:
        best_avg = avg
    
    
    average.append(avg)
    best_average.append(best_avg)

mean = np.mean(average)
variance = np.var(average)
std = np.std(average)

best_mean = np.mean(best_average)
best_variance = np.var(best_average)
best_std = np.std(best_average)

print("-------------------------")
print("Mean:", mean)
print("variance:", variance)
print("Standard deviation:", std)

print("Best Mean:", best_mean)
print("Best variance:", best_variance)
print("Best Standard deviation:", best_std)
print("-------------------------")

f = open(args.logdir+"/results.txt", "w")
f.write(str(mean) + "," + str(variance) + "," + str(std) + "\n")
f.write(str(best_mean) + "," + str(best_variance) + "," + str(best_std) + "\n")

for i in range(len(average)):
    _ = f.write(subfolders[i] + "," + str(average[i]) + "," + str(best_average[i]) + "\n")
    print("-------------------------")
    print(subfolders[i], round(average[i], 3), round(best_average[i], 3))

f.close()