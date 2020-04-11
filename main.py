from SAC import SAC
import argparse
import json

import pyvirtualdisplay

parser = argparse.ArgumentParser()
parser.add_argument('logdir', help='Directory to save the checkpoints and learn summaries')
parser.add_argument('steps', help='Number of steps to train the agent', type=int)
parser.add_argument('-d', '--my-dict', type=json.loads)
args = parser.parse_args()

display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

arg_dict = {"horizon": 100, "batch_size": 128, "buffer_size": 50000, "initial_collect_steps": 2000, "critic_lr": 3e-4, "actor_lr": 3e-4, "alpha_lr": 3e-4, "tau": 0.005, "gamma": 0.99, 
"actor_layers": (256,256), "summary_interval":10000, "num_eval_episodes":20, "critic_layers":(256,256), "reward_scale":1.0}

if(args.my_dict != None):
    for key, value in args.my_dict.items():
        if(key == 'projection_std' or key == 'tau' or key == 'gamma' or key == 'reward_scale'):
            arg_dict[key] = float(value)
        else:
            arg_dict[key] = int(value)

print(arg_dict)

sac_RE_10 = SAC(env_name='CarRacing-v0', horizon=arg_dict['horizon'], batch_size=arg_dict['batch_size'], buffer_size=arg_dict['buffer_size'], initial_collect_steps=arg_dict['initial_collect_steps'], critic_lr=arg_dict['critic_lr'], actor_lr=arg_dict['actor_lr'], alpha_lr=arg_dict['alpha_lr'], log_dir=args.logdir, target_update_tau=arg_dict['tau'], target_update_period=1, gamma=arg_dict['gamma'], actor_fc_layer_params = arg_dict['actor_layers'], summary_interval=arg_dict['summary_interval'], num_eval_episodes=arg_dict['num_eval_episodes'], critic_joint_fc_layer_params=arg_dict['critic_layers'], reward_scale=arg_dict['reward_scale'])

sac_RE_10.learn(num_iterations=args.steps)


