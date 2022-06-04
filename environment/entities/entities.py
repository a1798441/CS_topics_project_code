"""
capture_the_flag
This file defines the Entities base class.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""

import numpy as np
from scipy.spatial.distance import cdist


class Entities:
    def __init__(self, n, placement_choice, placement_bounds, acceleration_limit=0,
                 initial_azimuth=0, speed=0, radius=1, color='black'):
        """Entities in the game.

        :param n: Number of entities.
        :param placement_choice: where should the entities be placed.
        :param placement_bounds: Bounds on where entities can be placed.
        :param acceleration_limit: maximum acceleration that can be applied to an entity,
        :param initial_azimuth: initial azimuth of the entity.
        :param speed: speed fo the entity.
        :param radius: size of entity.
        :param color: color of entity.
        """
        # Number of entities
        self.n = n

        # Placement considerations
        self.min_placement_distance = 4.0 * radius
        self.placement_choice = placement_choice
        self.placement_bounds = placement_bounds

        # Entity characteristics
        self.initial_azimuth = initial_azimuth
        self.radius = radius
        self.speed = np.double(speed)
        self.acceleration_limit = acceleration_limit

        # Positional attributes
        self.positions = self.get_initial_positions()
        self.velocities = self.get_initial_velocities()
        self.accelerations = self.get_initial_accelerations()

        # Angles
        self.azimuths = self.get_initial_azimuths()

        # Graphics
        self.color = color
        self.graphics = []

    def reset(self):
        """Reset the characteristics of the entities. This should be individually specified for each entity.

        :return: none
        """
        raise NotImplementedError("Please implement the reset function on the entity.")

    def get_initial_positions(self):
        """Resets the position of the entities.

        :return: ndarray of positions of entities.
        """
        if self.placement_choice == "random_constraint":
            # Place the entities randomly within some bounds and a minimum distance between entities.
            return self.randomise_pos_with_constraint()
        elif self.placement_choice == "random":
            # Place the entities randomly within some bounds.
            return self.randomise_pos()
        elif self.placement_choice == "random_same":
            # Place the entities randomly within some bounds at the same location
            return self.randomise_same_pos()
        else:
            raise Exception("Placement choice is invalid.")

    def get_initial_azimuths(self):
        """Gets the initial azimuths

        :return: ndarray of azimuths
        """
        return np.array([self.initial_azimuth] * self.n)

    def get_initial_velocities(self):
        """Resets the velocities to initial conditions.

        :return: ndarray of initial velocities.
        """
        velocity_x = self.speed * np.cos(self.initial_azimuth)
        velocity_y = self.speed * np.sin(self.initial_azimuth)
        return np.array([[velocity_x, velocity_y]]*self.n)

    def get_initial_accelerations(self):
        """Entities start with zero acceleration.

        :return: ndarray of initial accelerations.
        """
        return np.zeros((self.n, 2), np.double)

    def randomise_pos(self):
        """Places all entities randomly within some bounds.

        :return: ndarray of positions of entities.
        """
        return np.random.uniform(self.placement_bounds[:, 0], self.placement_bounds[:, 1],  size=(self.n, 2))

    def randomise_same_pos(self):
        """Places all entities randomly within some bounds at the same location."""
        x = np.random.uniform(self.placement_bounds[0, 0], self.placement_bounds[0, 1])
        y = np.random.uniform(self.placement_bounds[1, 0], self.placement_bounds[1, 1])
        return np.array([[x, y]] * self.n)

    # noinspection PyArgumentList
    def randomise_pos_with_constraint(self):
        """Randomises the positions of the entities within some bounds.

        :return: ndarray of positions of entities
        """
        i = 0
        positions = np.array([])
        while len(positions) < self.n:
            x = np.random.uniform(self.placement_bounds[0, 0], self.placement_bounds[0, 1])
            y = np.random.uniform(self.placement_bounds[1, 0], self.placement_bounds[1, 1])

            if len(positions) > 0:
                dist = cdist(np.array([[x, y]]), positions).min()
                if dist > (self.min_placement_distance + 0.001):
                    positions = np.concatenate((positions, np.array([[x, y]])), axis=0)
                elif i == 10 ** 9:
                    raise Exception("Not finding a good position to place random agents")
                else:
                    i += 1
            else:
                positions = np.array([[x, y]])
        return positions
