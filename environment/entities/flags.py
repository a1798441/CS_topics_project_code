"""
capture_the_flag
This file defines the flags in the environment.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""

from matplotlib.patches import Circle
from environment.entities.entities import Entities
import numpy as np


class Flags(Entities):
    def __init__(self, n_flags, bounds, color='blue'):
        """Represents the flags in the environment.

        :param n_flags: Number of flags in the game
        :param bounds: bounds for placement[[x_min, x_max], [y_min, y_max]]
        :param color: color of the flags.
        """
        super().__init__(n=n_flags, placement_choice="random_constraint", placement_bounds=bounds, color=color)

        # Capture info
        self.is_captured = np.zeros(self.n)  # start not captured
        self.capture_distance = 10  # how close does an entity have to be to capture

        # Set up the graphics
        for flag_idx in range(self.n):
            self.graphics.append(Circle((self.positions[flag_idx]), self.radius, color=self.color))

        # Showing the capture circle
        self.outer_circle = []
        for flag_idx in range(self.n):
            self.outer_circle.append(Circle((self.positions[flag_idx]), self.capture_distance, color=self.color,
                                            fill=False, linestyle='--'))

    def reset(self):
        """Reset the characteristics of the flags.

        :return: none
        """
        self.positions = self.get_initial_positions()

        self.is_captured = np.zeros(self.n)
        for graphic in self.graphics:
            graphic.set_alpha(1)

    def capture_flag(self, flag_idx):
        """Capture a flag.

        :param flag_idx: which flag to capture.
        :return: none
        """
        self.is_captured[flag_idx] = True
        self.graphics[flag_idx].set_alpha(0.2)

    def drop_flag(self, flag_idx):
        """Drop a flag.

        :param flag_idx: which flag to drop.
        :return: none
        """
        self.is_captured[flag_idx] = False
        self.graphics[flag_idx].set_alpha(1)

    def attempt_capture(self, agent_position, flag_idx):
        """Can capture the flag if on same spot

        :param agent_position: the position of the agent.
        :param flag_idx: which flag to attempt to capture.
        :return: True if a flag is captured else False.
        """
        # Check if captured
        if not self.is_captured[flag_idx]:
            dist = np.sqrt((agent_position[0] - self.positions[flag_idx][0]) ** 2 +
                           (agent_position[1] - self.positions[flag_idx][1]) ** 2)
            # Check if in capture range
            if dist <= self.capture_distance:
                self.capture_flag(flag_idx)
                return True
            else:
                return False
        else:
            return False
