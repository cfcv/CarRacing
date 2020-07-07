# CarRacing
This project aims to use Deep Reinforcement Learning to solve the CarRacing OpenAI gym environment. Click in the image below to check out the video.

### Network architecture
The actor and critic networks have similar architecture, they only differ in the final layers. The actor has two branches: one for the actions means and one for the actions variance as the SAC method uses a stochastic policy and the critic only has a final layer with one neuron to predict the value function. As the observation space is an image, a nature CNN was used for feature extraction followed by 3 fully connected layers as shown in the image bellow:<br/>

<p align="center">
<img src="https://github.com/cfcv/CarRacing/blob/master/images/network_architecture.png">
</p>
<br/>
### Modifications in the Observation state

### Clipped Action space and Entropy coefficient

### Table of results
Here the results were averaged in 5 different random seeds
### Improvements

* ERE or PRE + ERE(Emphasizing recent experience)
* RNN
* SLAC(Soft Latent Actor Critic)
