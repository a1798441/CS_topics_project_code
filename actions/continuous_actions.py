"""
capture_the_flag
This file constrains the set of actions to some continuous
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


class ContinuousActionSet:
    def __init__(self, acceleration_limit):
        """Defines the range of valid acceleration commands.

        :param acceleration_limit: The maximum acceleration that can be applied.
        """
        self.acceleration_limit = acceleration_limit
        self.action_space = spaces.Box(low=-acceleration_limit, high=acceleration_limit, shape=(1, 1), dtype=np.float32)

    @staticmethod
    def get_lateral_acceleration(action):
        """For continuous actions the action is the acceleration.
        :param action: action
        :return: acceleration
        """
        return action
