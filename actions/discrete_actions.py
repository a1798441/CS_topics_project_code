"""
capture_the_flag
This file constrains the set of actions to some discrete
action set.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""

import numpy as np
from gym import spaces


class DiscreteActionSet:
    def __init__(self, acceleration_limit):
        """Defines the set of discrete acceleration commands that can be applied by an agent.

        :param acceleration_limit: The maximum acceleration that can be applied.
        """
        self.acceleration_limit = acceleration_limit
        self.action_set = np.array([0, acceleration_limit, -1 * acceleration_limit])
        self.action_space = spaces.Discrete(len(self.action_set))

    def get_lateral_acceleration(self, action):
        """Return the acceleration command that corresponds to the particular action.

        :param action: some discrete value.
        :return: the corresponding acceleration command.
        """
        return self.action_set[action]

