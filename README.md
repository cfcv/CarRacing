# CarRacing
This project aims to use Deep Reinforcement Learning to solve the CarRacing OpenAI gym environment. Click in the image below to check out the video.

### Network architecture
The actor and critic networks have similar architecture, they only differ in the final layers. The actor has two branches: one for the actions means and one for the actions variance as the SAC method uses a stochastic policy and the critic only has a final layer with one neuron to predict the value function. As the observation space is an image, a nature CNN was used for feature extraction followed by 3 fully connected layers as shown in the image bellow:<br/>

<p align="center">
<img src="https://github.com/cfcv/CarRacing/blob/master/images/network_architecture.png">
</p>

### Modifications in the Observation and action state
1. Gray observation: Originally the environment provides a 96x96x3 RGB image, with the gray observation wrapper the observation becomes a 84x84 grayscale iamge. In that way we can store more transitions in the replay buffer. 

1. Frame Stack: As the environment only provide a color image, the agent is not aware of it's velocity. Know the current velocity is important to estimate if we need to accelerato or brake in order to successifully complete curves. In the frame Stack wrapper the last 4 observations images are stacked so we can have implicitly the velocity information.

1. Action: By default, we can our action is a vector of 3 positions [steer, gas, brake], but in this formulation we provide the agent the possibility to accelerate and brake at the same time. As we don't want this, we change the action space to a two position vector [steer, gas or brake], the value of each position of the vector rely on the interval [-1, 1] where for the second position of the vector, negative values means brake and positive values means acceleration.

### Clipped Action space and Entropy coefficient

### Table of results
For each approach, 5 agents with 5 different random seeds were trained and it's performance averaged.
The average return is computed by averaging the return(sum of reward within an episode) over N episodes, in this case N = 100. 
When we have the average return of each one of the 5 different seed agents we can compute the final average return as follows:
```
final_average = 0
N = number of seeds
avg_return = list containing the average return of each agent trained. It has N elements. 

for i in range(N):
  final_average += avg_return[i]

final_average /= N
```
This average return is computed using the last agent checkpoint(in the final iteration step) and also in the best checkpoint anywere in the learning process, these two metrics are represented in the table below as avg. Return and Best Avg. Return respectively.

Agent | Avg. Return | Best Avg. Return
------------ | ------------- | ------------- 
SAC | 278.05 +- 139.276 | 522.188 +- 103.738
SAC + gray | 0.0 | 0.0
SAC + gray + action | 0.0 | 0.0
SAC + gray + action + Frame stack | 0.0 | 0.0
SAC + gray + action + Odometrie | 0.0 | 0.0

### Possible Improvements

* Pre-train a perception encoding representation using autoencoders and then freeze this part of the network during RL training.
* ERE or PRE + ERE(Emphasizing recent experience)
* RNN
* SLAC(Soft Latent Actor Critic)
