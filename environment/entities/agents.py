"""
capture_the_flag
This file defines the agents.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""

import numpy as np
from matplotlib.patches import Wedge
from environment.entities.entities import Entities
from sensors.sensor import Sensor

from algorithms.controller import Controller
from algorithms.custom.custom_controllerR import CustomControllerR
from algorithms.custom.custom_controllerB import CustomControllerB


class Agents(Entities):
    def __init__(self, env, team_var, placement_bounds, azimuth, team_flags):
        """Represents the agents in the game.

        :param env: the environment the agents are in.
        :param team_var: parameters for the team.
        :param placement_bounds: bounds where agents can be placed.
        :param azimuth: azimuth of the agents.
        :param team_flags: flags of the team (used in placement).
        """

        self.env = env
        self.team_flags = team_flags
        self.goal = team_var["team_goal"]

        super().__init__(n=team_var["n_agents"], initial_azimuth=azimuth, speed=team_var["speed"],
                         acceleration_limit=team_var["acceleration_limit"],
                         placement_choice=team_var["placement_choice"],
                         placement_bounds=placement_bounds, color=team_var["color"])

        self.sensor, self.controller = self._add_controller_scanner(env, team_var["control"], team_var["action_set"])

        self.do_dwta = False
        self.dwta_update = 5
        self.being_trained = False
        self.action_set = team_var["action_set"]

        # State variables
        self.has_flag = np.zeros(self.n)  # If any of the drones have the flag
        self.alive = np.ones(self.n)  # Keeps track if drones are alive or dead
        self.is_tagged = np.zeros(self.n)

        self.kill_distance = 4.0
        #self.tag_distance = 4.0

        # Specify the graphics of the drone
        self.half_arc =20.0  # wedge is defined by 40 degree arc (for drawing)
        for drone_idx in range(self.n):
            ori_deg = self.azimuths[drone_idx] * 180.0 / np.pi + 180.0  # rad to deg
            theta1 = ori_deg - self.half_arc
            theta2 = ori_deg + self.half_arc
            self.graphics.append(Wedge(center=(self.positions[drone_idx]), r=self.radius,
                                       theta1=theta1, theta2=theta2, color=self.color))

    def _add_controller_scanner(self, env, control_algorithm, action_set):
        """Generates a controller and sensor based on algorithm.

        :param env: environment under consideration.
        :param control_algorithm: algorithm to control the agents.
        :return: sensor, controller objects,
        """
        if self.color == 'red':
            entity_count = {"team_n": self.env.n_red_agents,
                            "enemy_team_n": self.env.n_blue_agents,
                            "team_flag_n": self.env.n_red_flags,
                            "enemy_flag_n": self.env.n_blue_flags}
        elif self.color == 'blue':
            entity_count = {"team_n": self.env.n_blue_agents,
                            "enemy_team_n": self.env.n_red_agents,
                            "team_flag_n": self.env.n_blue_agents,
                            "enemy_flag_n": self.env.n_red_agents}
        else:
            raise Exception("Invalid team color")

        if control_algorithm == 'custom':
            sensor = Sensor(env=env, team_color=self.color)
            if self.color == 'blue':
                controller = CustomControllerB(goal=self.goal, team=self, sensor=sensor)
            elif self.color == 'red':
                controller = CustomControllerR(goal=self.goal, team=self, sensor=sensor)
            else:
                raise Exception("Invalid team color")
        else:
            raise Exception("Control Algorithm not implemented.")
        return sensor, controller

    def start_at_flag(self):
        """Place agents at flags.

        :return: none if no placements made otherwise ndarray of agent positions.
        """

        if self.n == 0:
            return None
        if self.team_flags.n == 0:
            raise Exception("Can't place at team flags since there are zero team flags.")
        elif self.team_flags.n == 1:
            return np.array([self.team_flags.positions[0]] * self.n)
        elif self.team_flags.n == self.n:
            positions = np.zeros((self.n, 2))
            for agent_idx in range(self.n):
                positions[agent_idx] = self.team_flags.positions[agent_idx]
            return positions
        else:
            raise Exception("Haven't defined which flags to place each agent at.")

    def reset(self):
        """Resets all characteristics of the drones.

        :return: none
        """

        self.positions = self.get_initial_positions()
        self.azimuths = self.get_initial_azimuths()
        self.velocities = self.get_initial_velocities()
        self.accelerations = self.get_initial_accelerations()
        self.has_flag = np.zeros(self.n)
        self.alive = np.ones(self.n)
        self.is_tagged = np.zeros(self.n)

        if self.controller is not None:
            self.controller.reset()
        for idx in range(self.n):
            self.graphics[idx].set_color(self.color)
            self.graphics[idx].set_alpha(1)

    def get_initial_positions(self):
        """Resets the position of the entities.

        :return: ndarray of positions of entities.
        """

        if self.placement_choice == 'flag':
            return self.start_at_flag()
        else:
            return super().get_initial_positions()

    def get_acceleration(self):
        """Get an acceleration command.

        :return: acceleration commands.
        """
        return self.controller.get_acceleration()

    def apply_acceleration(self, idx, acceleration, delta_time):
        """Applies an instantaneous constant acceleration for delta_time.
        https://en.wikipedia.org/wiki/Yaw_(rotation)

        :param idx: which agent.
        :param acceleration: what acceleration to apply.
        :param delta_time: how long to apply acceleration for.
        :return: none
        """
        # Get the tangential velocity
        tangential_velocity = self.speed

        # Ge the lateral acceleration (Normalise)
        lateral_acceleration = np.linalg.norm(acceleration)
        if lateral_acceleration > self.acceleration_limit:
            acceleration = acceleration / lateral_acceleration * self.acceleration_limit
            lateral_acceleration = np.linalg.norm(acceleration)

        # Work out if turning left or right and adjust lateral acceleration
        unit_vector = np.array([np.cos(self.azimuths[idx]), np.sin(self.azimuths[idx])])
        if not np.linalg.norm(acceleration) == 0:
            if np.cross(acceleration, unit_vector) < 0:
                # Turning Left
                pass
            else:
                # Turning Right
                lateral_acceleration = lateral_acceleration * -1

        # Calculate yaw velocity
        yaw_velocity = lateral_acceleration / tangential_velocity

        # Update the angle
        new_angle = self.azimuths[idx] + (yaw_velocity * delta_time)

        if not yaw_velocity == 0.0:
            # Update the x_pos
            new_x_pos = self.positions[idx][0] + \
                        tangential_velocity * (np.sin(new_angle) - np.sin(self.azimuths[idx])) / yaw_velocity

            # Update the y_pos
            new_y_pos = self.positions[idx][1] + \
                        tangential_velocity * (np.cos(self.azimuths[idx]) - np.cos(new_angle)) / yaw_velocity
        else:
            # Update the x_pos
            new_x_pos = self.positions[idx][0] + tangential_velocity * (np.cos(new_angle)) * delta_time

            # Update the y_pos
            new_y_pos = self.positions[idx][1] + tangential_velocity * (np.sin(new_angle)) * delta_time

        # Update velocities (Definition)
        self.velocities[idx][0] = tangential_velocity * np.cos(new_angle)
        self.velocities[idx][1] = tangential_velocity * np.sin(new_angle)

        # Update accelerations
        acceleration_x = self.speed * yaw_velocity * -1 * np.sin(new_angle)
        acceleration_y = self.speed * yaw_velocity * np.cos(new_angle)
        self.accelerations[idx] = np.array([acceleration_x, acceleration_y])

        # Update positions
        self.positions[idx][0] = new_x_pos
        self.positions[idx][1] = new_y_pos

        self.azimuths[idx] = np.arctan2(self.velocities[idx][1], self.velocities[idx][0])

    def kill(self, agent_idx):
        """This kills one of the agents.

        :param agent_idx: Which agent to kill.
        :return: none
        """
        self.alive[agent_idx] = False
        # Setting positions to large number to effectively remove from game
        self.positions[agent_idx][0] = float(10 ** 6)
        self.positions[agent_idx][1] = float(10 ** 6)
        self.graphics[agent_idx].set_color("white")
        self.graphics[agent_idx].set_alpha(0)

    def apply_tag(self, agent_idx):
        """Apply a tag to an agent.

        :param agent_idx: agent under consideration.
        :return: None
        """
        self.is_tagged[agent_idx] = True
        self.graphics[agent_idx].set_alpha(0.2)

    def untag(self, agent_idx):
        """Untag an agent.

        :param agent_idx: agent under consideration.
        :return: None.
        """
        self.is_tagged[agent_idx] = False
        self.graphics[agent_idx].set_alpha(1)

    def attempt_to_capture_the_flag(self, agent_idx):
        """Attempt to capture the flag.

        :param agent_idx: agent under consideration.
        :return: True if captured flag else False
        """
        if not self.has_flag[agent_idx] and not self.is_tagged[agent_idx]:
            self.has_flag[agent_idx] = self.controller.sensor.enemy_flags.attempt_capture(self.positions[agent_idx],
                                                                                          self.controller.
                                                                                          target_idx[agent_idx])
            # Agent dies after getting flag
            # if self.has_flag[agent_idx]:
            #    self.kill(agent_idx)
        return self.has_flag[agent_idx]

    def attempt_kill_enemy(self, agent_idx):
        """Kills an enemy agent.

        :param agent_idx: agent under consideration.
        :return: True if kills and enemy else False
        """
        target_idx = self.controller.target_idx[agent_idx]

        # for target_idx in range(self.controller.target.n):
        position = self.positions[agent_idx]
        target_position = self.sensor.enemy_team.positions[target_idx]
        dist = np.linalg.norm(target_position - position)

        if dist <= self.kill_distance:
            # Tag the enemy
            self.sensor.enemy_team.is_tagged[target_idx] = True

            # Kill target
            if not self.sensor.enemy_team.being_trained:
                self.sensor.enemy_team.kill(target_idx)
            # Own agent is killed as well
            self.kill(agent_idx)

    def attempt_to_deliver_flag(self, agent_idx):
        """Attempt to deliver the flag.

        :param agent_idx: agent under consideration.
        :return: True if delivered flag else False
        """
        if self.has_flag[agent_idx]:
            dist = np.linalg.norm(self.team_flags.positions[0] - self.positions[agent_idx])
            # Deliver distance is currently the same as capture distance
            if dist <= self.team_flags.capture_distance:
                self.has_flag[agent_idx] = False
                self.controller.sensor.enemy_flags.drop_flag(0)
                return True
            else:
                return False
        else:
            return False

    def take_extra_actions(self):
        """Agents will attempt to capture flags or kill enemies.

        :return: none
        """
        for agent_idx in range(self.n):
            if self.alive[agent_idx]:
                #if type(self.controller.target) is Flags:
                #    self.attempt_to_capture_the_flag(agent_idx)
                #elif type(self.controller.target) is Agents:
                self.attempt_kill_enemy(agent_idx)
                #else:
                #    raise Exception("Target Type Invalid")

