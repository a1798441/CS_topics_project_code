"""
capture_the_flag
This file defines the sensors of the agents.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""
import numpy as np


class Sensor:
    def __init__(self, env, team_color):
        """Interface between the agents and the environment.

        :param env: the environment this is an interface to.
        :param team_color: team that sensor is sensing for.
        """
        self.env = env
        self.team_color = team_color

        self.team = None
        self.team_flags = None
        self.enemy_team = None
        self.enemy_flags = None
        self.team_n = None
        self.enemy_n = None
        self.joint = False
        self.obs_dim = 0
        self.observation_space = None
        self.min_velocity = np.array([-1, -1])
        self.max_velocity = np.array([1, 1])

    def initialise_sensor_functions(self):
        if self.team_color == "red":
            self.team = self.env.red_team
            self.team_flags = self.env.red_flags
            self.enemy_team = self.env.blue_team
            self.enemy_flags = self.env.blue_flags
        elif self.team_color == "blue":
            self.team = self.env.blue_team
            self.team_flags = self.env.blue_flags
            self.enemy_team = self.env.red_team
            self.enemy_flags = self.env.red_flags
        self.team_n = self.team.n
        if self.enemy_team is None:
            self.enemy_n = 0
        else:
            self.enemy_n = self.enemy_team.n

        if self.team.action_set == 'joint':
            self.joint = True
        else:
            self.joint = False
        self.min_velocity = np.array([-1*self.team.speed, -1*self.team.speed])  # Make this generic later
        self.max_velocity = np.array([self.team.speed, self.team.speed])  # Make this generic later

    def get_team_speed(self):
        """Get the nominal speed of the team. We assume constant speed for whole team.

        :return: speed of the team.
        """
        return self.team.speed

    def get_team_positions(self):
        """Get the positions of the agents on the team.

        :return: None.
        """
        # At the moment assume perfect knowledge
        return self.team.positions

    def get_team_velocities(self):
        """Get the positions of the agents on the team.

        :return: None.
        """
        # At the moment assume perfect knowledge
        return self.team.velocities

    def get_team_azimuths(self):
        """Get the azimuths of the agents on the team.

        :return: None.
        """
        # At the moment assume perfect knowledge
        return self.team.azimuths

    def get_enemy_positions(self):
        """Get the positions of the agents on the enemy team.

        :return: None.
        """
        # At the moment assume perfect knowledge
        return self.enemy_team.positions

    def get_enemy_velocities(self):
        """Get the positions of the agents on the enemy team.

        :return: None.
        """
        # At the moment assume perfect knowledge
        return self.enemy_team.velocities

    def get_team_flag_positions(self):
        """Get the positions of the flags on the team.

        :return: None.
        """
        # At the moment assume perfect knowledge
        return self.team_flags.positions

    def get_team_flag_velocities(self):
        """Get the velocities of the flags on the team.

        :return: None.
        """
        # At the moment assume perfect knowledge
        return self.team_flags.velocities

    def get_enemy_flag_positions(self):
        """Get the positions of the flags on the enemy team.

        :return: None.
        """
        # At the moment assume perfect knowledge
        return self.enemy_flags.positions

    def get_enemy_flag_velocities(self):
        """Get the velocities of the flags on the enemy team.

        :return: None.
        """
        # At the moment assume perfect knowledge
        return self.enemy_flags.velocities
