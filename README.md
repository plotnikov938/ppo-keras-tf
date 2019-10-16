# Implementation of PPO algorithm for reinforcement learning with Keras + tensorflow

This repository contains the code for reinforcement learning that allows to train and run an agent 
for various discrete and continuous control space tasks 
in [OpenAI gym](https://github.com/openai/gym). The Proximal Policy Optimization (PPO) algorithm implemented with Keras and tensorflow to train a policy to control an agent. 

The implementation is tested on [Pendulum-v0](https://github.com/openai/gym/wiki/Pendulum-v0), [CartPole-v0](https://github.com/openai/gym/wiki/CartPole-v0) and [LunarLander-v2](https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2) tasks. 

---
This project is the part of the datascience portfolio. Other projects can be found here:
* [Implementation of progressive groving of GANs with Keras + tensorflow](https://github.com/plotnikov938/pg-gans-keras-tf)
* [Text classification using CapsNet and multihead relative attention with Keras + tensorflow](https://github.com/plotnikov938/txt-class-keras-tf)
---

## Table of Contents
  * [Features](#features)
  * [Requirements](#requirements)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Results](#results)
  * [Resources](#resources) 
  * [Contacts](#contacts)   
  
## Features

## Installation
Make sure you use Python 3.

Clone this repo to your local machine:
```
$ git clone https://github.com/plotnikov938/ppo-keras-tf.git
```
Go into the project directory:
```
$ cd ppo-keras-tf/
```
Create a virtual environment if needed:
```
$ python3 -m venv env
```
Use the package manager pip to install requirements:
```
$ pip3 install -r requirements.txt
```

## Usage
To successfully train an agent on [Pendulum-v0](https://github.com/openai/gym/wiki/Pendulum-v0) environment run:
```
$ python3 train.py --env Pendulum-v0 --epochs 400 --gamma 0.98 --lam 0.999 --cliprange 0.2 --trans-per-epoch 12000 --critic-lr 0.018 --actor-lr 0.0006 --seed 50 --actor-train-epochs 80 --critic-train-epochs 80 --dir PPO_exp
```

For [CartPole-v0](https://github.com/openai/gym/wiki/CartPole-v0) environment run:

--- 
```
$ python3 train.py --env CartPole-v0 --epochs 4000 --gamma 0.98 --lam 0.99 --cliprange 0.1 --trans-per-epoch 400 --critic-lr 0.001 --actor-lr 0.001 --seed 50 --actor-train-epochs 2 --critic-train-epochs 2 --dir PPO_exp -c 195
```

[LunarLander-v2](https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2) environment: 
```
$ python3 train.py --env LunarLander-v2 --epochs 400 --gamma 0.98 --lam 0.999 --cliprange 0.2 --trans-per-epoch 4000 --critic-lr 0.0024 --actor-lr 0.0002 --seed 50 --actor-train-epochs 80 --critic-train-epochs 80 --dir PPO_exp -l -c 200
```

To run a trained agent on a single environment for 5 episodes and create 
a gif after experiment will be complited, simply call:
```
$ python3 run.py --env Pendulum-v0 --dir PPO_exp --episodes 5 --render --save-gif 8 --last
```
Some hyper-parameters can be modified with different arguments to train.py and run.py.

## Results

<details open>
  <summary>LunarLander-v2</summary>
 
  <pre><code id="code2">$ python3 run.py --env LunarLander-v2 --dir PPO_exp --save-gif -r -l</code></pre>

  <p align="center">
      <img id="preview" src="PPO_exp/results/LunarLander-v2.gif" width="640" name="gif" />
  </p>

</details>

<details>
  <summary>Pendulum-v0</summary>
 
  <pre><code id="code2">$ python3 run.py --env Pendulum-v0 --dir PPO_exp --save-gif -r -l</code></pre>

  <p align="center">
      <img id="preview" src="PPO_exp/results/Pendulum-v0.gif" width="640" name="gif" />
  </p>

</details>

<details>
  <summary>CartPole-v0</summary>
 
  <pre><code id="code2">$ python3 run.py --env CartPole-v0 --dir PPO_exp --save-gif -r -l</code></pre>
 
  <p align="center">
      <img id="preview" src="PPO_exp/results/CartPole-v0.gif" width="640" name="gif" />
  </p>

</details>

## Resources
1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. [OpenAI gym](https://gym.openai.com)

## Contacts
Please feel free to contact me if you have any questions:  [plotnikov.ilia938@gmail.com](mailto:plotnikov.ilia938@gmail.com)
