"""
capture_the_flag
This file defines the obstacles in the environment.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""
from matplotlib.patches import Circle
from environment.entities.entities import Entities


class Obstacles(Entities):
    def __init__(self, n_obstacles, bounds, radius=10, color='black'):
        """Represents the obstacles in the environment.

        :param n_obstacles: number of obstacles in the game
        :param bounds: bounds for placement[[x_min, x_max], [y_min, y_max]]
        :param radius: size of the obstacles.
        :param color: color of the obstacles.
        """
        super().__init__(n=n_obstacles, placement_choice="random", placement_bounds=bounds, radius=radius, color=color)

        # Set up the graphics
        for obstacle_idx in range(self.n):
            self.graphics.append(Circle((self.positions[obstacle_idx]), self.radius, color=color))

    def reset(self):
        """Reset the characteristics of the obstacles.

        :return: none
        """
        self.positions = self.get_initial_positions()
