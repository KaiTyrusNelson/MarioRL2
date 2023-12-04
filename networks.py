'''
Enviroment
'''
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym
from runstats import Statistics

'''Additional'''
from torch import nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th

class MarioNet(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNet, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class MarioNet2(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNet2, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class Residual_Block(nn.Module):
    def __init__(self, num_channels, out_channels):
        super().__init__()

        self.conv_residual1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, 
                                        kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn_residual1 = nn.BatchNorm2d(num_features=num_channels)
        self.conv_residual2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, 
                                        kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn_residual2 = nn.BatchNorm2d(num_features=num_channels)
        self.relu = nn.ReLU()
        
        self.out_conv = nn.Conv2d(in_channels=num_channels, out_channels=out_channels, 
                                        kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    def forward(self, x):

        identity = x    #(1)

        x = self.conv_residual1(x)
        x = self.bn_residual1(x)
        x = self.relu(x)

        x = self.conv_residual2(x)
        x = self.bn_residual2(x)
        x = self.relu(x)

        x = x+identity   #(2)
        x = self.relu(x)

        return self.out_conv(x)
        
class MarioNet3(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNet3, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            Residual_Block(n_input_channels, 8),
            nn.ReLU(),
            Residual_Block(8, 16),
            nn.ReLU(),
            Residual_Block(16, 32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Residual_Block(32, 32),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())


    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))



