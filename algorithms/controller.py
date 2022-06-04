"""
capture_the_flag
This file constrains the Controller base class.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""

import numpy as np
from utils.acceleration_conversions import convert_angular_acceleration
from actions.joint_actions import JointActionSet
from actions.discrete_actions import DiscreteActionSet
from actions.high_level_actions import HighLevelActionSet
from actions.continuous_actions import ContinuousActionSet

from environment.reinforcement_learning_training_interface import ReinforcementLearningTrainingInterface


class Controller:
    def __init__(self, goal, team, sensor, action_set, controller_type, trainable=False):
        """Controller that works out the acceleration commands to apply to the agents.

        :param goal: what is each agent trying to do.
        :param team: which team is being controlled
        :param sensor: what is the view of the environment.
        """

        self.goal = goal
        self.team = team
        self.sensor = sensor
        self.actions = None
        self.controller_type = controller_type

        # Information for loops etc. (Note that individual agents may not know this information).
        self.n_agents = team.n
        self.target_idx = np.zeros(self.n_agents, np.int)
        self.dones = np.zeros(self.n_agents, np.int)
        self.last_action = ["None"] * self.n_agents
        self.doing_joint_actions = False

        if action_set == 'discrete':
            self.action_set = DiscreteActionSet(self.team.acceleration_limit)
            self.action_space = self.action_set.action_space
        elif action_set == 'joint':
            self.action_set = JointActionSet(self.team.acceleration_limit, self.team.n)
            self.action_space = self.action_set.action_space
            self.doing_joint_actions = True
        elif action_set == 'high_level':
            self.action_set = HighLevelActionSet(self.team.acceleration_limit)
            self.action_space = self.action_set.action_space
        elif action_set == 'continuous':
            self.action_set = ContinuousActionSet(self.team.acceleration_limit)
            self.action_space = self.action_set.action_space
        else:
            self.action_set = None

        if action_set == 'joint':
            self.trainable_agents = 1
        else:
            self.trainable_agents = self.team.n

        # Attach the model
        if controller_type == 'custom':
            self.model = None
            self.is_reinforcement_learning = False
        else:
            raise Exception("Invalid control method")

        # Attach the training environment
        self.randomise = self.sensor.env.randomise
        if trainable:
            if self.goal == 'ctf' and self.is_reinforcement_learning:
                self.training_env = ReinforcementLearningTrainingInterface(self.sensor.env, self.team, self.goal,
                                                                           self.randomise, self.doing_joint_actions,
                                                                           self.action_space,
                                                                           self.sensor.observation_space)
            else:
                raise Exception("training env does not exists")
        else:
            self.training_env = None

    def reset(self):
        """This resets any parameters associated with the controller.

        :return: none
        """
        if self.model is not None:
            self.model.reset()
        self.last_action = ["None"] * self.n_agents

    def train(self, epochs):
        if self.training_env is not None:
            self.model.train(env=self.training_env, epochs=epochs)
        else:
            print("Model doesn't require training")

    def load(self):
        """

        :return:
        """

        if self.model is not None:
            self.model.load_model()
        else:
            print("Can't load model for this controller.")

    def set_actions(self, actions):
        """Set the actions to use. (Used when training in RL)

        :param actions: actions to apply.
        :return: none
        """
        self.actions = actions

    def get_acceleration(self):
        """Get acceleration based on reinforcement learning commands

        :return: ndarray of acceleration commands.
        """

        actions = self.actions
        acceleration = np.zeros((self.n_agents, 2))
        if actions is None:
            observations = self.sensor.get_observations()
            if self.model is not None:
                actions = self.model.get_actions(observations)
            else:
                raise Exception("Model not specified")
        self.actions = None
        lateral_accelerations = self.action_set.get_lateral_acceleration(actions)

        for idx in range(self.n_agents):
            acceleration[idx] = convert_angular_acceleration(lateral_accelerations[idx],
                                                             self.sensor.get_team_azimuths()[idx])
        return acceleration
