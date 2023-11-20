'''
Enviroment
'''
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym

'''
Additional
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from PPO import PPO
import torch as th
from torch import nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from runstats import Statistics
from networks import MarioNet

from pathlib import Path
import datetime
from pytz import timezone

# Model Param
CHECK_FREQ_NUMB = 10000
TOTAL_TIMESTEP_NUMB = 5000000
LEARNING_RATE = 0.0001
GAE = 1.0
ENT_COEF = 0.01
N_STEPS = 512
GAMMA = 0.9
BATCH_SIZE = 64
N_EPOCHS = 10
    
# Test Param
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000

STAGE_NAME = 'SuperMarioBros-1-1-v3'
MOVEMENT = [['left', 'A'], ['right', 'B'], ['right', 'A', 'B']]

import sys


def train(save_dir, 
          activation_function = th.nn.ReLU,
          orthagonal_init = False,
          grad_norm = sys.float_info.max, value_clip = None, reward_scaling = False, annealing = False):


    policy_kwargs = dict(
        features_extractor_class=MarioNet,
        features_extractor_kwargs=dict(features_dim=512),
        activation_fn = activation_function,
        squash_output = False if activation_function is th.nn.Tanh else True,
        ortho_init = orthagonal_init
    )
    
    env = gym_super_mario_bros.make(STAGE_NAME)
    env = JoypadSpace(env, MOVEMENT)
    env = CustomRewardAndDoneEnv(env)
    
    if reward_scaling:
        env = RewardScalingWrapper(env)
            
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')



    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok = True)
    reward_log_path = (save_dir / 'reward_log.csv')

    callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ_NUMB, save_path=save_dir, env = env, reward_log_path = reward_log_path)

    model = PPO('CnnPolicy', env, verbose=0, policy_kwargs=policy_kwargs, 
                tensorboard_log=save_dir, learning_rate=LEARNING_RATE, n_steps=N_STEPS,
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE, 
                ent_coef=ENT_COEF, clip_range_vf = value_clip, max_grad_norm = grad_norm, annealing = annealing)
    
    model.learn(total_timesteps=TOTAL_TIMESTEP_NUMB, callback=callback)
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame

class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += max(0, info['x_pos'] - self.max_x)
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            done = True
        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

class RewardScalingWrapper(gym.Wrapper):
    def __init__(self, env=None):
        super(RewardScalingWrapper, self).__init__(env)
        self.reward_buffer = Statistics()
        self.reward_buffer.push(0.)
        
    def reset(self, **kwargs):
        self.reward_buffer.clear()
        self.reward_buffer.push(0.)
        return self.env.reset(**kwargs)
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.reward_buffer.push(reward)
        reward /= self.reward_buffer.variance()**(0.5)
        return state, reward, done, info

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, env, verbose=1, reward_log_path = None):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.reward_log_path = reward_log_path
        self.env = env

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = (self.save_path / 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

            total_reward = [0] * EPISODE_NUMBERS
            total_time = [0] * EPISODE_NUMBERS
            best_reward = 0

            for i in range(EPISODE_NUMBERS):
                state = self.env.reset()  # reset for each new trial
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                while not done and total_time[i] < MAX_TIMESTEP_TEST:
                    action, _ = self.model.predict(state)
                    state, reward, done, info = self.env.step(action)
                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]
                    best_epoch = self.n_calls

                state = self.env.reset()  # reset for each new trial

            print('time steps:', self.n_calls, '/', TOTAL_TIMESTEP_NUMB)
            print('average reward:', (sum(total_reward) / EPISODE_NUMBERS),
                  'average time:', (sum(total_time) / EPISODE_NUMBERS),
                  'best_reward:', best_reward)
            
            if self.reward_log_path is not None:
                with open(self.reward_log_path, 'a') as f:
                    print(self.n_calls, ',', sum(total_reward) / EPISODE_NUMBERS, ',', best_reward, file=f)

        return True







