"""
capture_the_flag
This file constrains the set of actions to some set of joint actions.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""

import itertools
import numpy as np
from gym import spaces


class JointActionSet:
    def __init__(self, acceleration_limit, n_agents):
        """ Define the set of discrete joint actions that can be taken.

        :param acceleration_limit: maximum acceleration that can be applied.
        :param n_agents: number of agents in the team.
        """
        self.acceleration_limit = acceleration_limit
        self.individual_actions = np.array([0, acceleration_limit, -1 * acceleration_limit])
        self.action_set = np.array(list(itertools.product(self.individual_actions, repeat=n_agents)))
        self.action_space = spaces.Discrete(len(self.action_set))

    def get_lateral_acceleration(self, action):
        """

        :param action:
        :return:
        """
        return self.action_set[action][0]

