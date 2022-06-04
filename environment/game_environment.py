"""
capture_the_flag
This file defines the environment.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""
import numpy as np
import matplotlib.pyplot as plt
import actions.high_level_actions as hla
from scipy.spatial.distance import cdist
from scipy.stats import median_absolute_deviation
from environment.entities.agents import Agents
from environment.entities.flags import Flags
from environment.entities.obstacles import Obstacles
import math


class GameEnvironment:
    def __init__(self, game_rules, red_team_var, blue_team_var, generate_graphics=True, randomise=False):
        """In this environment there are a number of blue agents, red agents, blue flags, red flags and obstacles. The
        game played can be customised.

        In the attack_defend game there are a number of red agents travelling to collect a number of blue
        flags. Optionally there are some obstacles and optionally there are some opposing blue agents attempting to
        prevent this.
        In the ctf game both sides have a flag and are attempting to capture their opponents flag and bring it to their
        home base.
        """
        # Difficulty 1: Naive red team - Defender loops circles
        # Difficulty 2: Red defender intercepts first blue agent in "enemies_in_territory" array
        # Difficulty 3: Red defender intercepts blue agent closest to red flag
        # Difficulty 4: Red attacker will aid in defending the flag
        # Difficulty 5: The red team go on pure defense
        self.difficulty = 5

        self.rules = game_rules
        self.randomise = randomise

        # Define the boundaries of the game and placement bounds [[x_min, x_max],[y_min, y_max]]
        self.game_boundary = np.array([[0.0, 160.0], [0.0, 80.0]])

        self.centre = np.array([(self.game_boundary[0, 1] - self.game_boundary[0, 0]) / 2,
                                (self.game_boundary[1, 1] - self.game_boundary[1, 0]) / 2])
        self.top = np.array([(self.game_boundary[0, 1] - self.game_boundary[0, 0]) / 2,
                             (self.game_boundary[1, 1] - self.game_boundary[1, 0]) / 4 * 3])
        self.bottom = np.array([(self.game_boundary[0, 1] - self.game_boundary[0, 0]) / 2,
                                (self.game_boundary[1, 1] - self.game_boundary[1, 0]) / 4])

        self.obstacle_bounds = np.array([[50, 110], [10, 70]])
        self.red_flags_bounds = np.array([[135, 145], [30, 50]])
        self.blue_flags_bounds = np.array([[15, 25], [30, 50]])
        self.blue_agents_bounds = np.array([[10, 30], [20, 60]])
        self.red_agents_bounds = np.array([[120, 140], [20, 60]])

        # Initial Orientation
        self.initial_blue_orientation = np.double(0)
        self.initial_red_orientation = np.pi

        # Actions of red and blue teams
        self.red_acceleration = None
        self.blue_acceleration = None

        # Scoring
        self.red_score = 0
        self.blue_score = 0

        # Graphics needs to be optional for algorithms that create multiple copies of the environment in training!
        self.generate_graphics = generate_graphics
        if self.generate_graphics:
            self.fig, self.ax = plt.subplots()
            plt.axis([self.game_boundary[0, 0], self.game_boundary[0, 1],
                      self.game_boundary[1, 0], self.game_boundary[1, 1]])
            plt.gca().set_aspect('equal', adjustable='box')
        else:
            self.fig = None
            self.ax = None

        # How often to make decisions for blue and red
        self.time_step = 0
        self.blue_delta_time = blue_team_var["delta_time"]
        self.red_delta_time = red_team_var["delta_time"]
        # Calculates self.delta_time which is how much time elapses in each time step.
        if self.blue_delta_time <= self.red_delta_time:
            modifier = (1.0/self.blue_delta_time)
            if not (modifier * self.red_delta_time) % (modifier * self.blue_delta_time) == 0:
                raise Exception("blue_delta_time does not evenly divide into red_delta_time")
            self.delta_time = self.blue_delta_time
        else:
            modifier = (1.0/self.red_delta_time)
            if not (modifier * self.blue_delta_time) % (modifier * self.red_delta_time) == 0:
                raise Exception("red_delta_time does not evenly divide into blue_delta_time")
            self.delta_time = self.red_delta_time

        # How often to make decisions in time steps.
        self.blue_time_step = np.round(self.blue_delta_time/self.delta_time, 0)
        self.red_time_step = np.round(self.red_delta_time/self.delta_time, 0)

        self.max_episode_length = int(800/self.delta_time)
        self.render_steps = int(1/self.delta_time)  # Render every 'render_steps' frames
        self.dist_matrix = None

        # Entity counts
        self.n_blue_flags = blue_team_var["n_flags"]
        self.n_red_flags = red_team_var["n_flags"]
        self.n_blue_agents = blue_team_var["n_agents"]
        self.n_red_agents = red_team_var["n_agents"]

        # Blue Flags
        if self.n_blue_flags > 0:
            self.blue_flags = Flags(n_flags=self.n_blue_flags, bounds=self.blue_flags_bounds, color='blue')
        else:
            self.blue_flags = None

        # Red flags
        if self.n_red_flags > 0:
            self.red_flags = Flags(n_flags=self.n_red_flags, bounds=self.red_flags_bounds, color='red')
        else:
            self.red_flags = None

        # Obstacles
        self.n_obstacles = 0
        if self.n_obstacles > 0:
            self.obstacles = Obstacles(n_obstacles=self.n_obstacles, bounds=self.obstacle_bounds, radius=5,
                                       color='black')
        else:
            self.obstacles = None

        # Blue agents
        if self.n_blue_agents > 0:
            self.blue_team = Agents(env=self, team_var=blue_team_var, placement_bounds=self.blue_agents_bounds,
                                    azimuth=self.initial_blue_orientation, team_flags=self.blue_flags)
        else:
            self.blue_team = None

        # Red agents
        if self.n_red_agents > 0:
            self.red_team = Agents(env=self, team_var=red_team_var, placement_bounds=self.red_agents_bounds,
                                   azimuth=self.initial_red_orientation, team_flags=self.red_flags)
        else:
            self.red_team = None

        # Initialise sensors
        if self.n_red_agents > 0:
            self.red_team.sensor.initialise_sensor_functions()
        if self.n_blue_agents > 0:
            self.blue_team.sensor.initialise_sensor_functions()

        # Collision Detection
        self.collision_safety_dist = 1  # extra factor for safety
        self.agent_collision_dist = (self.red_team.radius * 2) + self.collision_safety_dist
        if self.obstacles is not None:
            self.obstacle_collision_dist = self.red_team.radius + self.obstacles.radius + self.collision_safety_dist
        else:
            self.obstacle_collision_dist = None

        self.blue_team_override = [False] * self.n_blue_agents
        self.red_team_override = [False] * self.n_red_agents

        self.red_text = []
        self.blue_text = []
        if generate_graphics:
            if self.n_blue_agents > 0:
                for blue_idx, blue_agent_graphics in enumerate(self.blue_team.graphics):
                    self.ax.add_patch(blue_agent_graphics)
                    blue_text_string = "Agent %s" % blue_idx
                    self.blue_text.append(plt.text(x=self.blue_team.positions[blue_idx, 0],
                                                   y=self.blue_team.positions[blue_idx, 1], s=blue_text_string,
                                                   color='blue', fontsize=6))

            if self.n_red_agents > 0:
                for red_idx, red_agent_graphics in enumerate(self.red_team.graphics):
                    self.ax.add_patch(red_agent_graphics)
                    red_text_string = "Agent %s" % red_idx
                    self.red_text.append(plt.text(x=self.red_team.positions[red_idx, 0],
                                                  y=self.red_team.positions[red_idx, 1], s=red_text_string,
                                                  color='red', fontsize=6))

            if self.n_red_flags > 0:
                for red_flag_graphics in self.red_flags.graphics:
                    self.ax.add_patch(red_flag_graphics)
                for red_flag_outer_circle in self.red_flags.outer_circle:
                    self.ax.add_patch(red_flag_outer_circle)

            if self.n_blue_flags > 0:
                for blue_flag_graphics in self.blue_flags.graphics:
                    self.ax.add_patch(blue_flag_graphics)
                for blue_flag_outer_circle in self.blue_flags.outer_circle:
                    self.ax.add_patch(blue_flag_outer_circle)

            if self.n_obstacles > 0:
                for obstacle_graphics in self.obstacles.graphics:
                    self.ax.add_patch(obstacle_graphics)
            plt.vlines(x=80, ymin=0, ymax=80, colors='black', linestyles='--')

        if generate_graphics:
            if self.n_blue_agents > 0:
                for blue_agent_graphics in self.blue_team.graphics:
                    self.ax.add_patch(blue_agent_graphics)

            if self.n_red_agents > 0:
                for red_agent_graphics in self.red_team.graphics:
                    self.ax.add_patch(red_agent_graphics)

            if self.n_red_flags > 0:
                for red_flag_graphics in self.red_flags.graphics:
                    self.ax.add_patch(red_flag_graphics)

            if self.n_blue_flags > 0:
                for blue_flag_graphics in self.blue_flags.graphics:
                    self.ax.add_patch(blue_flag_graphics)

            if self. n_obstacles > 0:
                for obstacle_graphics in self.obstacles.graphics:
                    self.ax.add_patch(obstacle_graphics)

    def get_environment_state(self):
        """Returns the current state of the environment.

        :return: dictionary containing the state of the environment.
        """

        if self.red_flags is None and self.blue_team is None:
            return {"red_team_positions": self.red_team.positions.copy(),
                    "red_team_velocities": self.red_team.velocities.copy(),
                    "red_team_azimuths": self.red_team.azimuths.copy(),
                    "red_team_accelerations": self.red_team.accelerations.copy(),
                    "red_team_has_flag": self.red_team.has_flag.copy(),
                    "red_team_alive": self.red_team.alive.copy(),
                    "red_team_tag": self.red_team.is_tagged.copy(),

                    "blue_team_flag_positions": self.blue_flags.positions.copy(),
                    "blue_team_flag_is_captured": self.blue_flags.is_captured.copy()}
        else:
            return {"red_team_positions": self.red_team.positions.copy(),
                    "red_team_velocities": self.red_team.velocities.copy(),
                    "red_team_azimuths": self.red_team.azimuths.copy(),
                    "red_team_accelerations": self.red_team.accelerations.copy(),
                    "red_team_has_flag": self.red_team.has_flag.copy(),
                    "red_team_alive": self.red_team.alive.copy(),
                    "red_team_tag": self.red_team.is_tagged.copy(),

                    "red_team_flag_positions": self.red_flags.positions.copy(),
                    "red_team_flag_is_captured": self.red_flags.is_captured.copy(),

                    "blue_team_positions": self.blue_team.positions.copy(),
                    "blue_team_velocities": self.blue_team.velocities.copy(),
                    "blue_team_azimuths": self.blue_team.azimuths.copy(),
                    "blue_team_accelerations": self.blue_team.accelerations.copy(),
                    "blue_team_has_flag": self.blue_team.has_flag.copy(),
                    "blue_team_alive": self.blue_team.alive.copy(),
                    "blue_team_tag": self.blue_team.is_tagged.copy(),

                    "blue_team_flag_positions": self.blue_flags.positions.copy(),
                    "blue_team_flag_is_captured": self.blue_flags.is_captured.copy()}

    def set_environment_state(self, state):
        """Sets the environment to a specified state.

        :param state: A dictionary containing the desired state of the environment.
        :return: None
        """

        if self.red_flags is None and self.blue_team is None:
            self.red_team.positions = state["red_team_positions"]
            self.red_team.velocities = state["red_team_velocities"]
            self.red_team.azimuths = state["red_team_azimuths"]
            self.red_team.accelerations = state["red_team_accelerations"]
            self.red_team.has_flag = state["red_team_has_flag"]
            self.red_team.alive = state["red_team_alive"]
            self.red_team.is_tagged = state["red_team_tag"]

            self.blue_flags.positions = state["blue_team_flag_positions"]
            self.blue_flags.is_captured = state["blue_team_flag_is_captured"]

        else:
            self.red_team.positions = state["red_team_positions"]
            self.red_team.velocities = state["red_team_velocities"]
            self.red_team.azimuths = state["red_team_azimuths"]
            self.red_team.accelerations = state["red_team_accelerations"]
            self.red_team.has_flag = state["red_team_has_flag"]
            self.red_team.alive = state["red_team_alive"]
            self.red_team.is_tagged = state["red_team_tag"]

            self.red_flags.positions = state["red_team_flag_positions"]
            self.red_flags.is_captured = state["red_team_flag_is_captured"]

            self.blue_team.positions = state["blue_team_positions"]
            self.blue_team.velocities = state["blue_team_velocities"]
            self.blue_team.azimuths = state["blue_team_azimuths"]
            self.blue_team.accelerations = state["blue_team_accelerations"]
            self.blue_team.has_flag = state["blue_team_has_flag"]
            self.blue_team.alive = state["blue_team_alive"]
            self.blue_team.is_tagged = state["blue_team_tag"]

            self.blue_flags.positions = state["blue_team_flag_positions"]
            self.blue_flags.is_captured = state["blue_team_flag_is_captured"]

    def reset_env(self):
        """Reset the environment.

        :return: None.
        """
        # Reset the blue flags
        if self.blue_flags is not None:
            self.blue_flags.reset()

        # Reset the red flags
        if self.red_flags is not None:
            self.red_flags.reset()

        # Reset the blue team
        if self.blue_team is not None:
            self.blue_team.reset()

        # Reset the red team
        if self.red_team is not None:
            self.red_team.reset()

        # Reset the obstacles
        if self.obstacles is not None:
            self.obstacles.reset()

        self.time_step = 0  # Reset timestep
        self.red_score = 0  # Reset red score
        self.blue_score = 0  # Reset blue score

    def run_ctf(self, should_render=False, store_data=True):
        """Run an instance of the capture the flag game.

        :param should_render: whether the game should be rendered.
        :param store_data: record the game.
        :return: None
        """

        self.reset_env()
        for t in range(self.max_episode_length):
            if should_render:
                if t % self.render_steps == 0:
                    self.render()

            if store_data:
                """Currently we view this from the perspective of the red team"""
                #print(self.red_team.controller.last_action)
                pass
            self.update_environment()

            if store_data:
                pass

    def run_attack(self, should_render=False, store_data=True):
        """Run an instance of the attack_defend game.

        :param should_render: whether the game should be rendered.
        :param store_data: record the game.
        :return: None
        """

        self.reset_env()
        for t in range(self.max_episode_length):
            if should_render:
                if t % self.render_steps == 0:
                    self.render()

            if store_data:
                """Currently we view this from the perspective of the red team"""
                #print(self.red_team.controller.last_action)
                pass
            self.update_environment()

            if self.blue_flags.is_captured[0]:
                break

            if store_data:
                pass

    def update_environment(self):
        """Updates the state of the environment dependant on which game we are playing

        :return: none
        """
        if self.rules == 'attack_defend':
            self.update_environment_simultaneous_attack_defend()
        elif self.rules == 'ctf':
            self.update_environment_simultaneous_ctf()

    def update_environment_simultaneous_ctf(self):
        """Both red and blue calculate acceleration demands simultaneously and apply simultaneously.

        :return: none
        """

        # Get acceleration commands
        self.get_team_accelerations_simultaneous()

        # Apply red acceleration commands
        for idx in range(self.n_red_agents):
            self.red_team.apply_acceleration(idx, self.red_acceleration[idx], self.delta_time)

        # Attempt to tag agents
        self.attempt_tag()

        # Apply blue acceleration commands
        for idx in range(self.n_blue_agents):
            self.blue_team.apply_acceleration(idx, self.blue_acceleration[idx], self.delta_time)

        # Red (flag capture)
        if self.blue_flags.is_captured[0]:
            for agent_idx in range(self.n_red_agents):
                deliver = self.red_team.attempt_to_deliver_flag(agent_idx)
                if deliver:
                    self.red_score += 1
        else:
            for agent_idx in range(self.n_red_agents):
                self.red_team.attempt_to_capture_the_flag(agent_idx)

        # Blue
        if self.red_flags.is_captured[0]:
            for agent_idx in range(self.n_blue_agents):
                deliver = self.blue_team.attempt_to_deliver_flag(agent_idx)
                if deliver:
                    self.blue_score += 1
        else:
            for agent_idx in range(self.n_blue_agents):
                self.blue_team.attempt_to_capture_the_flag(agent_idx)

        # red/blue attempt to tag blue/red
        self.attempt_tag()

        # Untag if back at base
        self.untag_at_base()

        # Increment time step
        self.time_step += 1

    def get_team_accelerations_simultaneous(self):
        """Sets the red and blue team accelerations simultaneously.

        :return: None.
        """

        if self.red_team is not None:
            if self.time_step % self.red_time_step == 0:
                self.red_acceleration = self.red_team.get_acceleration()
            # Override actions if tagged
            self.red_team_override = [False] * self.red_team.n
            for agent_idx in range(self.n_red_agents):
                if self.red_team.is_tagged[agent_idx]:
                    self.red_team_override[agent_idx] = True
                    self.red_acceleration[agent_idx] = hla.go_to_base(self.red_team, self.red_flags, agent_idx, 0, self.delta_time)
        else:
            self.red_acceleration = None

        # Get blue acceleration commands
        if self.blue_team is not None:
            if self.time_step % self.blue_time_step == 0:
                self.blue_acceleration = self.blue_team.get_acceleration()
            # Override actions if tagged
            self.blue_team_override = [False] * self.blue_team.n
            for agent_idx in range(self.n_blue_agents):
                if self.blue_team.is_tagged[agent_idx]:
                    self.blue_team_override[agent_idx] = True
                    self.blue_acceleration[agent_idx] = hla.go_to_base(self.blue_team, self.blue_flags, agent_idx, 0, self.delta_time)
        else:
            self.blue_acceleration = None

    def attempt_tag(self):
        """To tag has to be in the corresponding territory and has to not be tagged themselves.

        :return: None
        """
        if self.red_team is None or self.blue_team is None:
            return
        dist = cdist(self.red_team.positions, self.blue_team.positions, metric='euclidean')
        indices = np.where(dist < self.red_team.kill_distance)

        for i in range(len(indices[0])):
            red_idx = indices[0][i]
            blue_idx = indices[1][i]

            if self.in_red_territory(self.red_team, red_idx) and self.in_red_territory(self.blue_team, blue_idx):
                self.blue_team.apply_tag(blue_idx)
                # If tagged then drop flag
                if self.blue_team.has_flag[blue_idx]:
                    self.blue_team.has_flag[blue_idx] = False
                    self.red_flags.drop_flag(0)

            elif self.in_blue_territory(self.red_team, red_idx) and self.in_blue_territory(self.blue_team, blue_idx):
                self.red_team.apply_tag(red_idx)
                # If tagged then drop flag
                if self.red_team.has_flag[red_idx]:
                    self.red_team.has_flag[red_idx] = False
                    self.blue_flags.drop_flag(0)

    def untag_at_base(self):
        """To tag has to be in the corresponding territory and has to not be tagged themselves.

        :return: None
        """
        if self.red_team is not None:
            dist = cdist(self.red_team.positions, self.red_flags.positions, metric='euclidean')
            indices = np.where(dist < self.red_flags.capture_distance)

            for i in range(len(indices[0])):
                red_idx = indices[0][i]
                self.red_team.untag(red_idx)

        if self.blue_team is not None:
            dist = cdist(self.blue_team.positions, self.blue_flags.positions, metric='euclidean')
            indices = np.where(dist < self.blue_flags.capture_distance)

            for i in range(len(indices[0])):
                blue_idx = indices[0][i]
                self.blue_team.untag(blue_idx)

    def has_left_boundary(self, team, agent_idx):
        """Checks if the agent has left the boundary.

        :return: True if it does leave boundary else False
        """
        return (team.positions[agent_idx][0] < self.game_boundary[0, 0]) or \
               (team.positions[agent_idx][0] > self.game_boundary[0, 1]) or \
               (team.positions[agent_idx][1] < self.game_boundary[1, 0]) or \
               (team.positions[agent_idx][1] > self.game_boundary[1, 1])
    ### a1798441 start
    def flag_holder(self, team):
        """returns the enemy holding the flag

        :param team: red or blue.
        :return: list of enemies (enemy index) in the territory.
        """

        idxholder = 0


        if team.color == 'red':
            for enemy_idx in range(self.n_blue_agents):
                if self.blue_team.has_flag[enemy_idx]:
                    idxholder = enemy_idx


        else:
            for enemy_idx in range(self.n_red_agents):
                if self.in_blue_territory(self.red_team, enemy_idx):
                    if self.red_team.has_flag[enemy_idx]:
                        idxholder = enemy_idx

        return idxholder

    def closest_to_flag(self, team):
        """returns the enemy closest to the team flag.

        :param team: red or blue.
        :return: list of enemies (enemy index) in the territory.
        """

        bdist = cdist(self.blue_team.positions, self.red_flags.positions, metric='euclidean')
        rdist = cdist(self.red_team.positions, self.blue_flags.positions, metric='euclidean')

        distclosest = 10000
        idxclosest = None



        if team.color == 'red':
            for enemy_idx in range(self.n_blue_agents):
                if bdist[enemy_idx][0] < distclosest:
                    distclosest = bdist[enemy_idx][0]
                    idxclosest = enemy_idx


        else:
            for enemy_idx in range(self.n_red_agents):
                if self.in_blue_territory(self.red_team, enemy_idx):
                    if rdist[enemy_idx][0] < distclosest:
                        distclosest = rdist[enemy_idx][0]
                        idxclosest = enemy_idx

        return idxclosest

    def top_enemy(self, team):
        """returns the enemy thats at the highest relative to the map


        :param idx: 0 or 1
        :return: the idx of the enemy on top for the team of idx 0 and the enemy on bottom for idx 1
        """
        targetidx = None
        highest = 0

        if team.color == 'red':
            for enemy_idx in range(self.n_blue_agents):
                if self.blue_team.positions[enemy_idx][1] > highest:
                    highest = self.blue_team.positions[enemy_idx][1]
                    targetidx = enemy_idx

        elif team.color == 'blue':
            for enemy_idx in range(self.n_red_agents):
                if self.red_team.positions[enemy_idx][1] > highest:
                    highest = self.red_team.positions[enemy_idx][1]
                    targetidx = enemy_idx

        return targetidx

    def bot_enemy(self, team):
        """returns the enemy thats at the lowest relative to the map


        :param team: red or blue
        :return: the idx of the enemy on top for the team of idx 0 and the enemy on bottom for idx 1
        """
        targetidx = None
        lowest = 1000

        if team.color == 'red':
            for enemy_idx in range(self.n_blue_agents):
                if self.blue_team.positions[enemy_idx][1] < lowest:
                    lowest = self.blue_team.positions[enemy_idx][1]
                    targetidx = enemy_idx

        elif team.color == 'blue':
            for enemy_idx in range(self.n_red_agents):
                if self.red_team.positions[enemy_idx][1] < lowest:
                    lowest = self.red_team.positions[enemy_idx][1]
                    targetidx = enemy_idx

        return targetidx

    def agent_zero_target(self, team):
        """returns the enemy closest to red 0


        :param team: team: red or blue.
        :return:
        """


        bdist = cdist(self.blue_team.positions, self.red_team.positions, metric='euclidean')

        closest_to_agent = 1000
        targetidx = None

        if team.color == 'red':
            for enemy_idx in range(self.n_blue_agents):
                if bdist[enemy_idx][0] < closest_to_agent:
                    closest_to_agent = bdist[enemy_idx][0]
                    targetidx = enemy_idx

        return targetidx

    def farthest_from_flag(self, team):
        """returns the enemy farthest from the team flag.

        :param team: red or blue.
        :return: idx of enemy farthest from flag.
        """

        bdist = cdist(self.blue_team.positions, self.red_flags.positions, metric='euclidean')
        rdist = cdist(self.red_team.positions, self.blue_flags.positions, metric='euclidean')

        distfarthest = 0
        idxfarthest = None



        if team.color == 'red':
            for enemy_idx in range(self.n_blue_agents):
                if bdist[enemy_idx][0] >= distfarthest:
                    distfarthest = bdist[enemy_idx][0]
                    idxfarthest = enemy_idx


        else:
            for enemy_idx in range(self.n_red_agents):
                if self.in_blue_territory(self.red_team, enemy_idx):
                    if rdist[enemy_idx][0] > distfarthest:
                        distfarthest = rdist[enemy_idx][0]
                        idxfarthest = enemy_idx

        return idxfarthest
    ### a1798441 end

    def check_for_enemies_in_territory(self, team):
        """Checks if any of the enemy agents are in a given team's territory.

        :param team: red or blue.
        :return: list of enemies (enemy index) in the territory.
        """
        enemies_in_territory = []
        if team.color == 'red':
            for enemy_idx in range(self.n_blue_agents):#for enemy_idx in range(self.n_blue_agents):
                if self.in_red_territory(self.blue_team, enemy_idx):
                    enemies_in_territory.append(enemy_idx)
        else:
            for enemy_idx in range(self.n_red_agents):#for enemy_idx in range(self.n_red_agents):
                if self.in_blue_territory(self.red_team, enemy_idx):
                    enemies_in_territory.append(enemy_idx)
        return enemies_in_territory

    def in_red_territory(self, team, agent_idx):
        """Checks if a particular agent in a particular team is in red territory.

        :param team: a particular team (Agents object).
        :param agent_idx: which agent to check.
        :return: Boolean
        """
        return (team.positions[agent_idx, 0] < self.game_boundary[0, 1]) and \
               (team.positions[agent_idx, 0] > self.game_boundary[0, 1] / 2) and \
               (team.positions[agent_idx, 1] > self.game_boundary[1, 0]) and \
               (team.positions[agent_idx, 1] < self.game_boundary[1, 1])

    def in_blue_territory(self, team, agent_idx):
        """Checks if a particular agent in a particular team is in blue territory.

        :param team: a particular team (Agents object)
        :param agent_idx: which agent to check.
        :return: Boolean
        """
        return (team.positions[agent_idx, 0] > self.game_boundary[0, 0]) and \
               (team.positions[agent_idx, 0] < self.game_boundary[0, 1]/2) and \
               (team.positions[agent_idx, 1] > self.game_boundary[1, 0]) and \
               (team.positions[agent_idx, 1] < self.game_boundary[1, 1])

    def render(self, animate=False):
        """Render the environment.

        :param animate: Boolaen to determine if this render frame will be used as part of an animation.
        :return: None.
        """
        # Update Red Flag graphics
        if self.red_flags is not None:
            for red_idx, red_flag_graphics in enumerate(self.red_flags.graphics):
                red_flag_graphics.center = (self.red_flags.positions[red_idx])

                for red_idx, red_flag_graphics in enumerate(self.red_flags.outer_circle):
                    red_flag_graphics.center = (self.red_flags.positions[red_idx])

        # Update Blue Flag graphics
        if self.blue_flags is not None:
            for blue_idx, blue_flag_graphics in enumerate(self.blue_flags.graphics):
                blue_flag_graphics.center = (self.blue_flags.positions[blue_idx])

                for blue_idx, blue_flag_graphics in enumerate(self.blue_flags.outer_circle):
                    blue_flag_graphics.center = (self.blue_flags.positions[blue_idx])

        # Update Red Agent graphics
        if self.red_team is not None:
            for red_idx, red_agent_graphics in enumerate(self.red_team.graphics):
                ori_graphics_deg = self.red_team.azimuths[red_idx] * 180 / np.pi + 180  # rad to deg
                theta1 = ori_graphics_deg - self.red_team.half_arc
                theta2 = ori_graphics_deg + self.red_team.half_arc
                red_agent_graphics.set_center((self.red_team.positions[red_idx]))
                red_agent_graphics.set_theta1(theta1)
                red_agent_graphics.set_theta2(theta2)

                # Text
                self.red_text[red_idx].set(x=self.red_team.positions[red_idx, 0],
                                           y=self.red_team.positions[red_idx, 1])
                if self.red_team.has_flag[red_idx]:
                    red_agent_graphics.set_color('green')
                else:
                    red_agent_graphics.set_color('red')

        # Update Blue Agent graphics
        if self.blue_team is not None:
            for blue_idx, blue_agent_graphics in enumerate(self.blue_team.graphics):
                ori_graphics_deg = self.blue_team.azimuths[blue_idx] * 180 / np.pi + 180  # rad to deg
                theta1 = ori_graphics_deg - self.blue_team.half_arc
                theta2 = ori_graphics_deg + self.blue_team.half_arc
                blue_agent_graphics.set_center((self.blue_team.positions[blue_idx]))
                blue_agent_graphics.set_theta1(theta1)
                blue_agent_graphics.set_theta2(theta2)

                # Text
                self.blue_text[blue_idx].set(x=self.blue_team.positions[blue_idx, 0],
                                             y=self.blue_team.positions[blue_idx, 1])
                if self.blue_team.has_flag[blue_idx]:
                    blue_agent_graphics.set_color('green')
                else:
                    blue_agent_graphics.set_color('blue')

        # Update obstacle graphics
        if self.obstacles is not None:
            for obs_idx, obstacle_graphics in enumerate(self.obstacles.graphics):
                obstacle_graphics.center = (self.obstacles.positions[obs_idx])

        if not animate:
            plt.pause(0.0000001)
            plt.draw()

    def load(self, team):
        """Loads a controller for a particular team.

        :param team: red or blue.
        :return: None.
        """
        if team == 'red':
            if self.n_red_agents > 0:
                self.red_team.controller.load()
        elif team == 'blue':
            if self.n_blue_agents > 0:
                self.blue_team.controller.load()

    def train(self, team, epochs):
        """Trains a specified team for a specified number of epochs.

        :param team: red or blue.
        :param epochs: how many times the model gets updated.
        :return: None
        """
        if team == 'red':
            if self.n_red_agents > 0:
                self.red_team.controller.train(epochs)
        elif team == 'blue':
            if self.n_blue_agents > 0:
                self.blue_team.controller.train(epochs)

    def get_obstacle_collisions(self):
        """Get the number of obstacle collisions.

        :return: int number of collisions.
        """
        if self.obstacles is not None:
            dist = cdist(self.red_team.positions, self.obstacles.positions, metric='euclidean')
            return sum(dist[dist < self.obstacle_collision_dist])
        else:
            return 0

    def get_agent_agent_collisions(self):
        """Get the number of inter agent collisions.

        :return: int number of collisions.
        """
        dist = cdist(self.red_team.positions, self.red_team.positions, metric='euclidean')
        dist[np.diag_indices(self.red_team.n)] = np.inf
        return sum(dist[dist < self.agent_collision_dist])

    def get_normalised_score(self):
        """Get the normalised score of the game.

        :return: int number of flags captured over total number of flags
        """
        return sum(self.blue_flags.is_captured)/self.n_blue_flags

    def evaluate(self, evaluation_type='ctf', evaluation_eps=500, should_render=False):
        """Runs the environment for n episodes and prints some evaluations stats.

        :return: none.
        """
        if evaluation_type == 'attack_defend':
            self.evaluate_attack_defend(evaluation_eps, should_render)
        elif evaluation_type == 'ctf':
            self.evaluate_ctf(evaluation_eps, should_render)

    def evaluate_attack_defend(self, evaluation_eps=500, should_render=False):
        """Runs a specified number of episodes and collects some statistics during the runs. Prints these statistics
        to the terminal after completing all episodes. The evaluation is on the attack_defend game.

        :param evaluation_eps: Number of episodes to run.
        :param should_render: Should the episodes be displayed as they are running.
        :return: None
        """
        n_evaluation_episodes = evaluation_eps
        total_obstacle_collisions = []
        total_agent_collisions = []
        total_score = []
        total_time = []

        for evaluation_episode in range(n_evaluation_episodes):
            if evaluation_episode % 10 == 0:
                print(evaluation_episode)

            # Reset the environment
            obstacle_collisions = 0
            agent_collisions = 0
            ep_len = 0
            self.reset_env()
            for t in range(self.max_episode_length):

                if should_render:
                    if t % self.render_steps == 0:
                        self.render()

                self.update_environment()
                # Check for obstacle collisions
                obstacle_collisions += self.get_obstacle_collisions()

                # Check for agent-agent collisions
                agent_collisions += self.get_agent_agent_collisions()

                finish = True
                ep_len = t + 1
                for idx in range(self.red_team.n):
                    if self.red_team.alive[idx]:
                        finish = False
                if finish:
                    break

            # Check the winning condition
            score = self.get_normalised_score()

            # Append to averages
            total_obstacle_collisions.append(obstacle_collisions)
            total_agent_collisions.append(agent_collisions)
            total_score.append(score)
            total_time.append(ep_len)

        total_obstacle_collisions = np.array(total_obstacle_collisions)
        total_agent_collisions = np.array(total_agent_collisions)
        total_score = np.array(total_score)
        total_time = np.array(total_time)

        print("(Mean, Standard Deviation, Median, Median Absolute Deviation")
        print("Obstacle Collisions: (%f, %f, %f, %f)" % (float(np.mean(total_obstacle_collisions)),
                                                         float(np.std(total_obstacle_collisions)),
                                                         float(np.median(total_obstacle_collisions)),
                                                         median_absolute_deviation(total_obstacle_collisions)
                                                         ))
        print("Agent Collisions: (%f, %f, %f, %f)" % (float(np.mean(total_agent_collisions)),
                                                      float(np.std(total_agent_collisions)),
                                                      float(np.median(total_agent_collisions)),
                                                      float(median_absolute_deviation(total_agent_collisions))
                                                      ))
        print("Game Time: (%f, %f, %f, %f)" % (float(np.mean(total_time)),
                                               float(np.std(total_time)),
                                               float(np.median(total_time)),
                                               float(median_absolute_deviation(total_time))
                                               ))
        print("Score: (%f, %f, %f, %f)" % (float(np.mean(total_score)),
                                           float(np.std(total_score)),
                                           float(np.median(total_score)),
                                           float(median_absolute_deviation(total_score))
                                           ))

    def evaluate_ctf(self, evaluation_eps=500, should_render=False):
        """Runs a specified number of episodes and collects some statistics during the runs. Prints these statistics
        to the terminal after completing all episodes. The evaluation is on the ctf game.

        :param evaluation_eps: Number of episodes to run.
        :param should_render: Should the episodes be displayed as they are running.
        :return: None
        """
        n_evaluation_episodes = evaluation_eps
        red_wins = []
        tags = 0
        for evaluation_episode in range(n_evaluation_episodes):
            if evaluation_episode % 10 == 0:
                print(evaluation_episode)

            # Reset the environment
            ep_len = 0
            self.reset_env()
            got_tagged = False
            for t in range(self.max_episode_length):

                if should_render:
                    if t % self.render_steps == 0:
                        self.render()

                self.update_environment()

                if self.red_team.is_tagged[0]:
                    got_tagged = True

            # Check the winning condition
            if self.red_score > self.blue_score:
                score = 1
            elif self.red_score == self.blue_score:
                score = 0
            else:
                score = -1

            if got_tagged:
                tags += 1
                print("GOT TAGGED!")
            # Append to averages
            red_wins.append(score)
        red_score = np.array(red_wins)

        print("Tags: %s" % tags)
        print("(Mean, Standard Deviation, Median, Median Absolute Deviation")
        ### a1708087 start
        # displaying score after the evaluation ends
        print("Score: (%f, %f, %f, %f)" % (float(np.mean(red_score)),
                                           float(np.std(red_score)),
                                           float(np.median(red_score)),
                                           float(median_absolute_deviation(red_score))
                                           ))
        counterB = 0
        counterD = 0
        counterR = 0
        for y in red_score:
            if y == 0:
                counterD += 1
            elif y > 0:
                counterR += 1
            else:
                counterB += 1
        print("Red team difficulty: %d" % self.difficulty)
        print("Scoreboard: B %d - %d - %d R" % (counterB, counterD, counterR))
        ### a1708087 ends

    def generate_animation(self, file_name):
        """Generates an animation and saves it to file.

        :param file_name: name of file to save to.
        :return: None.
        """
        from matplotlib.animation import FFMpegWriter
        self.reset_env()
        # movie_writer = FFMpegWriter(fps=100)
        movie_writer = FFMpegWriter(fps=50)

        with movie_writer.saving(plt.gcf(), file_name, dpi=100):
            for t in range(self.max_episode_length):
                self.render(animate=False)
                movie_writer.grab_frame()

                self.update_environment()
                finish = True
                for idx in range(self.red_team.n):
                    if self.red_team.alive[idx]:
                        finish = False
                if finish:
                    break

    def close(self):
        """Close the figure.

        :return: None.
        """
        plt.close(self.fig)

    def update_environment_simultaneous_attack_defend(self):
        """Both red and blue calculate acceleration demands simultaneously and apply simultaneously.

        :return: None.
        """

        # Dynamic target allocations
        if self.n_red_agents > 0:
            if self.red_team.do_dwta and self.time_step % self.red_team.dwta_update == 0:
                self.red_team.controller.update_target_allocations()

        if self.n_blue_agents > 0:
            if self.blue_team.do_dwta and self.time_step % self.blue_team.dwta_update == 0:
                self.blue_team.controller.update_target_allocations()

        # Get red acceleration commands
        if self.red_team.n > 0:
            if self.time_step % self.red_time_step == 0:
                self.red_acceleration = self.red_team.get_acceleration()
        else:
            self.red_acceleration = None

        # Get blue acceleration commands
        if self.n_blue_agents > 0:
            if self.time_step % self.blue_time_step == 0:
                self.blue_acceleration = self.blue_team.get_acceleration()
        else:
            self.blue_acceleration = None

        # Apply red acceleration commands
        for idx in range(self.red_team.n):
            if self.red_team.alive[idx]:
                self.red_team.apply_acceleration(idx, self.red_acceleration[idx], self.delta_time)

        # Apply blue acceleration commands
        for idx in range(self.n_blue_agents):
            if self.blue_team.alive[idx]:
                self.blue_team.apply_acceleration(idx, self.blue_acceleration[idx], self.delta_time)

        if self.n_blue_agents > 0:
            self.blue_team.take_extra_actions()
        # Swapped out from doing extra actions
        for agent_idx in range(self.n_red_agents):
            caught = self.red_team.attempt_to_capture_the_flag(agent_idx)
            if caught:
                self.red_team.kill(0)
                if self.n_blue_agents > 0:
                    self.blue_team.kill(0)

        if self.n_obstacles > 0:
            if self.check_agent_obstacle_collision(0):
                self.red_team.kill(0)

        self.time_step += 1

    def play_dfr_action(self, action, team):
        #self.render()
        # self.render()
        if team == 'red':
            # Apply red acceleration commands
            for idx in range(self.red_team.n):
                if self.red_team.alive[idx]:
                    self.red_acceleration = [self.red_team.controller.do_discrete_action(action, idx)]
                    self.red_team.apply_acceleration(idx, self.red_acceleration[idx], self.delta_time)

            # Check flag capture (red team)
            for idx in range(self.red_team.n):
                if not self.red_team.has_flag[idx]:
                    self.red_team.has_flag[idx] = self.blue_flags.attempt_capture(self.red_team.positions[idx],
                                                                                  flag_idx=0)
                if self.n_obstacles > 0:
                    if self.check_agent_obstacle_collision(idx):
                        self.red_team.kill(idx)

                # Agent dies after getting flag
                if self.red_team.has_flag[idx]:
                    self.red_team.kill(idx)

        elif team == 'blue':
            # Apply blue acceleration commands
            for idx in range(self.blue_team.n):
                if self.blue_team.alive[idx]:
                    self.blue_acceleration = [self.blue_team.controller.do_discrete_action(action, idx)]
                    self.blue_team.apply_acceleration(idx, self.blue_acceleration[idx], self.delta_time)

            # Attempt capture (blue team)
            for idx in range(self.blue_team.n):
                if not self.blue_team.has_flag[idx]:
                    self.blue_team.has_flag[idx] = self.red_flags.attempt_capture(self.blue_team.positions[idx],
                                                                                  flag_idx=0)
                # Agent dies after getting flag
                if self.blue_team.has_flag[idx]:
                    self.blue_team.kill(idx)
                # if self.blue_team.has_flag[idx]:
                #    print("Captured Flag")
        else:
            raise Exception("Invalid team")

    def check_agent_obstacle_collision(self, agent_idx):
        """TODO: fix this.

        :param agent_idx:
        :return:
        """
        dist = cdist(self.red_team.positions, self.obstacles.positions, metric='euclidean')
        if dist[0, 0] < self.obstacle_collision_dist:
            return True
        else:
            return False
