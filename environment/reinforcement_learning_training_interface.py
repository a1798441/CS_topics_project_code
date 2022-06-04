"""
capture_the_flag
This is a wrapper to run reinforcement learning algorithms.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""
import random
import numpy as np
from scipy.spatial.distance import cdist


class ReinforcementLearningTrainingInterface:
    def __init__(self, env, team, game_type='ctf', randomise=True, joint=False, action_space=None,
                 observation_space=None):
        """Provides an interface to the environment that can be used by a reinforcement learning algorithm
        during training.

        :param env: Python env object.
        :param team: Which team is using this interface.
        :param game_type: There should be a different reward structure based on what game is being played.
        :param randomise:
        :param joint: Should a joint reward be considered.
        :param action_space: What actions are available.
        :param observation_space: What can be observed in the environment.
        """

        self.env = env
        self.team = team

        if self.team.color == "red":
            self.randomise_red = True
            self.randomise_blue = False
        elif self.team.color == "blue":
            self.randomise_red = False
            self.randomise_blue = True
        else:
            raise Exception("Invalid color")

        self.randomise_both = True

        self.max_episode_length = self.env.max_episode_length
        self.current_red_score = 0
        self.current_blue_score = 0
        self.randomise = randomise
        self.game_type = game_type
        self.joint = joint  # return one observation/reward etc. or multiple

        self.action_space = action_space
        self.observation_space = observation_space

        # Reward that should be punished
        self.punish_obstacle_collisions = False
        self.punish_agent_collisions = False

    def reset(self):
        """Resets the state of the environment. If self.randomise has been set then there is some probability that the
        state will be set to a random state.

        :return: an observation of the state of the environment as given by the sensors.
        """
        # Reset environment state to an initial position
        self.env.reset_env()
        self.current_red_score = 0
        self.current_blue_score = 0

        # Randomise the state (to possibly a non-initial state)
        if self.randomise:
            random_number = random.uniform(0, 1)
            if random_number < 0.8:
                self._randomise_state()

        observations = self.team.sensor.get_observations()
        return observations

    def step(self, actions):
        """Take a step in the environment. This is used in reinforcement learning.

        :param actions: actions for each of the agents.
        :return: observations of each of the agents, rewards associated with each agent, done for each of the agents
        infos for each of the agents (currently nothing).
        """

        # Set what actions the agents should do
        self.team.controller.set_actions(actions)
        #self.env.render()
        # Update the environment (if on a faster time step then opponent then update quicker)
        if self.env.red_time_step == self.env.blue_time_step:
            self.env.update_environment()
        elif self.env.red_time_step > self.env.blue_time_step:
            for i in range(np.int(self.env.red_time_step)):
                self.env.update_environment()
        else:
            raise Exception("Haven't considered case where RL is faster than other solutions.")

        # Get observations after all agents have taken their actions
        observations = self.team.sensor.get_observations()

        if not self.joint:
            rewards, dones = self._get_reward_individual()
            infos = {'override': []}

            if self.team.color == 'red':
                for idx in range(self.team.n):
                    infos['override'].append(self.env.red_team_override[idx])
            elif self.team.color == 'blue':
                for idx in range(self.team.n):
                    infos['override'].append(self.env.blue_team_override[idx])
            else:
                raise Exception("Invalid team color")
            return observations, rewards, dones, infos
        else:
            rewards, dones = self._get_reward_joint()

            # Get infos (currently nothing)
            if self.team.color == 'red':
                if all(self.env.red_team_override):
                    infos = {'override': [True]}
                else:
                    infos = {'override': [False]}  # Unclear about override when it is joint action
            elif self.team.color == 'blue':
                if all(self.env.blue_team_override):
                    infos = {'override': [True]}
                else:
                    infos = {'override': [False]}  # Unclear about override when it is joint action
            else:
                raise Exception("Invalid team color")
            print(infos)
            return observations, [rewards], [dones], infos

    def _get_reward_joint(self):
        """Calculates the joint reward for the team. The reward value for the joint actions of the team. It also
        calculates whether the team is done (should the episode end).

        :return: the joint reward for the team and whether the team is done.
        """
        if self.game_type == 'ctf':
            return self._get_joint_reward_ctf()
        elif self.game_type == 'attack_defend':
            raise Exception("No reward function")
        else:
            raise Exception("No reward function for this game type")

    def _get_reward_individual(self):
        """This generates a list of rewards and dones for each of the agents in the team. The list of dones indicates
        whether a particular agent has terminated.

        :return: list of rewards and list of dones.
        """

        rewards = []
        dones = []
        # Get rewards and dones
        if self.game_type == 'ctf':
            team_scored = self._scored_flag(self.team.color)
            enemy_scored = self._scored_flag(self.team.sensor.enemy_team.color)
            for idx in range(self.team.n):
                reward, done = self._get_individual_reward_ctf(idx, team_scored, enemy_scored)
                rewards.append(reward)
                dones.append(done)
        elif self.game_type == 'attack_defend':
            for idx in range(self.team.n):
                reward, done = self._get_individual_reward_attack_defend(idx)
                rewards.append(reward)
                dones.append(done)
        else:
            raise Exception("No reward function for this game type")
        return rewards, dones

    def _get_joint_reward_ctf(self):
        """Calculates the joint reward and joint done for a team when playing capture the flag.

        :return: reward, done
        """
        team_scored = self._scored_flag(self.team.color)
        enemy_scored = self._scored_flag(self.team.sensor.enemy_team.color)
        if team_scored:
            return 10.0, True  # 10.0, True  # False
        elif enemy_scored:
            return 0.0, True  # -10.0, True  # False
        else:
            for agent_idx in range(self.team.n):
                if self.env.has_left_boundary(self.team, agent_idx):
                    # Penalise leaving boundaries
                    return -1.0, False
            return 0.0, False

    def _get_individual_reward_ctf(self, agent_idx, team_scored, enemy_scored):
        """Calculates the reward and done for a particular agent on a team playing capture the flag.

        :param agent_idx: agent under consideration.
        :param team_scored: whether the team scored.
        :param enemy_scored: whether the enemy team scored.
        :return: reward, done
        """

        if team_scored:
            # print("SCORED!")
            return 10.0, True
        elif enemy_scored:
            return 0.0, True  # -10.0, False
        else:
            if self.env.has_left_boundary(self.team, agent_idx):
                # Penalise leaving boundaries
                return -1.0, False
            return 0.0, False

    def _get_joint_reward_attack_defend(self):
        """Calculates the joint reward and joint done for a team when playing the attack_defend game.

        :return: reward, done
        """
        raise NotImplementedError("The _get_joint_reward_attack_defend function has not been implemented")

    def _get_individual_reward_attack_defend(self, agent_idx):
        """Get the reward and checks if done for a particular agent playing the attack_defend game.

        :param agent_idx: which agent to consider
        :return: tuple (reward, done)
        """

        if self.env.red_team.has_flag[agent_idx]:
            # Get the flag
            return 10.0, True  # Previously I had normalised rewards but maybe not great for this example.
        elif self.env.red_team.is_tagged[agent_idx]:
            # Tagged but not dead
            # self.red_team.tag[agent_idx] = False  # being tagged should only affect onc transition
            return -1.0, False
        elif not self.env.red_team.alive[agent_idx]:
            # Penalise dying
            return -1.0, True
        elif self.env.has_left_boundary(self.env.red_team, agent_idx):
            # Penalise leaving boundaries
            return -1.0, False
        else:
            # Penalise collisions with obstacles
            if self.punish_obstacle_collisions:
                dist = cdist(self.env.obstacles.positions, self.env.red_team.positions[agent_idx].reshape(1, 2),
                             metric='euclidean')
                if any(dist < self.env.obstacle_collision_dist):
                    return -1.0, False

            # Penalise collisions with other agents
            #if self.punish_agent_collisions and self.env.n_red_agents > 1:
            #    if any(np.delete(self.env.red_team.controller.sensor.dist_matrix[agent_idx], agent_idx) <
            #           self.env.agent_collision_dist):
            #        return -1.0, False

            # Default (Penalise prolonging the game)
            return -0.01, False

    def _scored_flag(self, team_color):
        """This checks to see if a given team managed to score a point by capturing the flag.

        :param team_color: can be blue or red.
        :return: boolean for whether the team scored a flag.
        """
        if team_color == 'red':
            if self.current_red_score < self.env.red_score:
                self.current_red_score += 1
                return True
            else:
                return False
        elif team_color == 'blue':
            if self.current_blue_score < self.env.blue_score:
                self.current_blue_score += 1
                return True
            else:
                return False
        else:
            raise Exception("Invalid Team")

    def _randomise_state(self):
        """This randomises the state of the environment.

        :return: None.
        """
        # Randomise red team variables
        if self.randomise_red or self.randomise_both:
            self.env.red_team.positions = np.random.uniform(self.env.game_boundary[:, 0], self.env.game_boundary[:, 1],
                                                            size=(self.env.red_team.n, 2))
            self.env.red_team.azimuths = np.random.uniform(-np.pi, np.pi, size=self.env.red_team.n)
            for i in range(self.env.red_team.n):
                self.env.red_team.velocities[i, 0] = self.env.red_team.speed * np.cos(self.env.red_team.azimuths[i])
                self.env.red_team.velocities[i, 1] = self.env.red_team.speed * np.sin(self.env.red_team.azimuths[i])

            # Randomise blue flag status
            rand_number = random.uniform(0, 1)
            if rand_number < 0.5:
                self.env.blue_flags.is_captured[0] = True
                self.env.blue_flags.graphics[0].set_alpha(0.2)
                rand_number_2 = random.randint(0, (self.env.red_team.n - 1))
                self.env.red_team.has_flag[rand_number_2] = True

        if self.randomise_blue or self.randomise_both:
            # Randomise blue team variables
            self.env.blue_team.positions = np.random.uniform(self.env.game_boundary[:, 0], self.env.game_boundary[:, 1],
                                                             size=(self.env.blue_team.n, 2))
            self.env.blue_team.azimuths = np.random.uniform(-np.pi, np.pi, size=self.env.blue_team.n)
            for i in range(self.env.blue_team.n):
                self.env.blue_team.velocities[i, 0] = self.env.blue_team.speed * np.cos(self.env.blue_team.azimuths[i])
                self.env.blue_team.velocities[i, 1] = self.env.blue_team.speed * np.sin(self.env.blue_team.azimuths[i])

            # Randomise red flag status
            rand_number = random.uniform(0, 1)
            if rand_number < 0.5:
                self.env.red_flags.is_captured[0] = True
                self.env.red_flags.graphics[0].set_alpha(0.2)
                rand_number_2 = random.randint(0, (self.env.blue_team.n - 1))
                self.env.blue_team.has_flag[rand_number_2] = True