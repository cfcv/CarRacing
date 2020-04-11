import wrappers as wp

import custom_critic_network
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import actor_distribution_network, normal_projection_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.utils import common

class SAC():
  def __init__(self, env_name, horizon, batch_size, buffer_size,
               initial_collect_steps, critic_lr, actor_lr, alpha_lr, target_update_tau, log_dir,
               target_update_period, gamma, actor_fc_layer_params, critic_joint_fc_layer_params,
               summary_interval, summaries_flush_secs=1, reward_scale=1.0, num_eval_episodes=10, 
               gradient_clipping=None, collect_steps_per_iteration=1, resume_training=False):
    
    self.env_name = env_name
    self.train_env = None
    self.eval_env = None
    self.observation_spec = None
    self.action_spec = None

    self.critic_lr = critic_lr
    self.actor_lr = actor_lr
    self.alpha_lr = alpha_lr
    self.target_update_tau = target_update_tau
    self.target_update_period = target_update_period
    self.critic_joint_fc_layer_params = critic_joint_fc_layer_params
    self.actor_fc_layer_params = actor_fc_layer_params
    self.nature_cnn = None
    self.critic_net = None
    self.actor_net = None

    self.horizon = horizon
    self.gamma = gamma
    self.batch_size = batch_size
    self.reward_scale = reward_scale
    self.buffer_size = buffer_size
    self.initial_collect_steps = initial_collect_steps
    self.num_eval_episodes = num_eval_episodes
    self.collect_steps_per_iteration = collect_steps_per_iteration
    self.summary_interval = summary_interval
    self.summaries_flush_secs = summaries_flush_secs
    self.gradient_clipping = gradient_clipping
    self.log_dir = log_dir

    self.tf_agent = None
    self.global_step = None
    self.step_metrics = None
    self.train_metrics = None
    self.resume_training = resume_training
    
    self.setup()

  def setup(self):
    self.train_env = tf_py_environment.TFPyEnvironment(self.create_env_train())
    self.eval_env = tf_py_environment.TFPyEnvironment(self.create_env())

    self.observation_spec = self.train_env.observation_spec()
    print('obs:',self.observation_spec)
    self.action_spec = self.train_env.action_spec()

    self.nature_cnn = self.pre_processing_natureCnn()
    self.critic_net = self.critic(self.nature_cnn)
    self.actor_net = self.actor(self.nature_cnn)

    self.global_step = tf.compat.v1.train.get_or_create_global_step()
    self.tf_agent = self.sac_agent()
    self.tf_agent.initialize()

    self.replay_buffer = self.create_buffer()
    self.replay_observer = [self.replay_buffer.add_batch]

    iniyial_collect_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(), 
                                                             self.train_env.action_spec())

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(self.train_env,
                                                                   iniyial_collect_policy,
                                                                   observers=self.replay_observer,
                                                                   num_steps=self.initial_collect_steps)
    
    if(not self.resume_training):
      print('------- Filling Buffer -------')
      _ = initial_collect_driver.run()
      print('------- END -------')

    self.step_metrics, self.train_metrics = self.define_metrics()

  def pre_processing_natureCnn(self):
    return tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2))),
                    tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2))),
                    tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2))),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)))
            ])
   
  def critic(self, pre_processing):
    return custom_critic_network.CriticNetwork((self.observation_spec, self.action_spec),
                                                observation_preprocessing_layers=pre_processing,
                                                observation_fc_layer_params=None,
                                                action_fc_layer_params=None,
                                                joint_fc_layer_params=self.critic_joint_fc_layer_params)

  def actor(self, pre_processing):
    return actor_distribution_network.ActorDistributionNetwork(self.observation_spec,
                                                              self.action_spec,
                                                              preprocessing_layers=pre_processing,
                                                              fc_layer_params=self.actor_fc_layer_params,
                                                              continuous_projection_net=self.normal_projection_net)


  def normal_projection_net(self, action_spec,init_means_output_factor=0.1):
    return normal_projection_network.NormalProjectionNetwork(
        self.action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)


  def sac_agent(self):
    return sac_agent.SacAgent(self.train_env.time_step_spec(),
                              self.action_spec,
                              actor_network=self.actor_net,
                              critic_network=self.critic_net,
                              actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=self.actor_lr),
                              critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=self.critic_lr),
                              alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=self.alpha_lr),
                              target_update_tau=self.target_update_tau,
                              target_update_period=self.target_update_period,
                              td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
                              gamma=self.gamma,
                              reward_scale_factor=self.reward_scale,
                              gradient_clipping=self.gradient_clipping,
                              train_step_counter=self.global_step)
 
  def create_buffer(self):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self.tf_agent.collect_data_spec,
                                                          batch_size=self.train_env.batch_size,
                                                          max_length=self.buffer_size)
  
  def create_env(self):
    return suite_gym.load(self.env_name, max_episode_steps=self.horizon, gym_env_wrappers=(wp.ObsNormalizer,))

  def create_env_train(self):
    return suite_gym.load(self.env_name, max_episode_steps=self.horizon, gym_env_wrappers=(wp.ObsNormalizer, wp.RewardRoute,))

  def define_metrics(self):
    step_metrics = [tf_metrics.NumberOfEpisodes(),
                    tf_metrics.EnvironmentSteps()]
    #train_metrics = step_metrics + ...
    train_metrics = [tf_metrics.AverageReturnMetric(buffer_size=self.num_eval_episodes, batch_size=self.train_env.batch_size),
                                    tf_metrics.AverageEpisodeLengthMetric(buffer_size=self.num_eval_episodes, batch_size=self.train_env.batch_size),]
    
    return step_metrics, train_metrics

  def compute_avg_return(self, num_episodes=5):
    total_return = 0.0
    for _ in range(num_episodes):

      time_step = self.eval_env.reset()
      episode_return = 0.0

      while not time_step.is_last():
        action_step = self.tf_agent.policy.action(time_step)
        time_step = self.eval_env.step(action_step.action)
        episode_return += time_step.reward
      total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]
  
  def learn(self, num_iterations=100000):
    dataset = self.replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=self.batch_size, num_steps=2).prefetch(3)

    iterator = iter(dataset) 

    collect_driver = dynamic_step_driver.DynamicStepDriver(self.train_env,
                                                           self.tf_agent.collect_policy,
                                                           observers=self.replay_observer + self.train_metrics,
                                                           num_steps=self.collect_steps_per_iteration)

    root_dir =  self.log_dir
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')
    checkpoint_dir = os.path.join(root_dir, 'checkpoint')
    policy_dir = os.path.join(root_dir, 'policy')

    saver_policy = policy_saver.PolicySaver(self.tf_agent.policy)
    train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir,
                                             max_to_keep=2,
                                             agent=self.tf_agent,
                                             policy=self.tf_agent.policy,
                                             replay_buffer=self.replay_buffer,
                                             global_step=self.global_step)

    if(not self.resume_training):
      train_summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=self.summaries_flush_secs*1000)
      train_summary_writer.set_as_default()
      # Reset the train step
      self.tf_agent.train_step_counter.assign(0)
    else:
      train_checkpointer.initialze_or_restore()
      self.global_step = tf.compat.v1.train.get_global_step()
      print('Resume global step: ', self.global_step)
    

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    self.tf_agent.train = common.function(self.tf_agent.train)
    collect_driver.run = common.function(collect_driver.run)


    with tf.summary.record_if(lambda: tf.math.equal(self.global_step % self.summary_interval, 0)):
      for _ in tqdm(range(num_iterations)):  
        collect_driver.run()

        experience, unused_info = next(iterator)
        train_loss = self.tf_agent.train(experience)

        for train_metric in self.train_metrics:
          train_metric.tf_summaries(train_step=self.global_step, step_metrics=self.step_metrics)

        step = self.tf_agent.train_step_counter.numpy()

        #if((step % 100000) == 0):
        #  train_checkpointer.save(self.global_step)
        #  saver_policy.save(policy_dir)

        if(step % self.summary_interval == 0):
          tf.summary.scalar('Average Reward', self.compute_avg_return(self.num_eval_episodes), step=self.global_step)
          train_checkpointer.save(self.global_step)
          saver_policy.save(policy_dir)
          #avg_return = self.compute_avg_return(self.num_eval_episodes)
          #print('step = {0}: Average Return = {1}'.format(step, avg_return))
          #policy_saver.save(policy_dir)
