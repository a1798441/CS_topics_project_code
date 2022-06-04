"""
capture_the_flag
This file defines the custom controller that controls the agents.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.

!!!! from a1798441
This code file was edited by a1798441


"""

##needed for calculating distance between agents
from scipy.spatial.distance import cdist
import numpy as np
from utils.utils import choices
from algorithms.controller import Controller
import actions.high_level_actions as hla


class CustomControllerR(Controller):

    def __init__(self, goal, team, sensor, model=None):

        if not goal == 'ctf':
            raise Exception("Goal is not supported by this controller")

        super().__init__(goal, team, sensor, action_set='high_level', controller_type='custom')

    def get_acceleration(self):
        """Get acceleration commands based on the proportional navigation algorithm.

        :return: ndarray of acceleration commands.
        """
        ### a1798441 start
        enemy_flag_captured = self.sensor.enemy_flags.is_captured[0]
        team_flag_captured = self.sensor.team_flags.is_captured[0]
        ### a1798441 end
        acceleration = np.zeros((self.n_agents, 2))
        Enemy_to_flags_dist = cdist(self.sensor.team_flags.positions, self.sensor.enemy_team.positions, metric='euclidean')
        for idx in range(self.n_agents):
            if self.sensor.team.is_tagged[idx]:
                # If tagged then return to base
                acceleration[idx] = hla.go_to_base(self.sensor.team, self.sensor.team_flags, idx, 0,
                                                   self.sensor.env.delta_time)
                self.last_action[idx] = "tagged"
            else:
                ### a1798441 start
                if idx == 0 or self.sensor.env.difficulty == 5:
                    # Check if any enemy agents in territory
                    enemies_in_territory = self.sensor.env.check_for_enemies_in_territory(self.team)

                    #finds the enemy closest to the flag
                    closest_flag = self.sensor.env.closest_to_flag(self.team)

                    #returns the enemy farthest from the flag Unused
                    farthest_flag = self.sensor.env.farthest_from_flag(self.team)

                    #finds enemy thats near the top of the map
                    topenemy = self.sensor.env.top_enemy(self.team)

                    #finds the enemy at the bottom of the map
                    botenemy = self.sensor.env.bot_enemy(self.team)

                ### a1798441 end
                    # Remove any tagged enemies
                    for enemy in enemies_in_territory:
                        if self.sensor.enemy_team.is_tagged[enemy]:
                            enemies_in_territory.remove(enemy)
                            ### a1798441 start
                            if len(enemies_in_territory) != 0:
                                closest_flag = enemies_in_territory[0]
                                topenemy = enemies_in_territory[0]
                                botenemy = enemies_in_territory[0]
                            ### a1798441 end

                    ### a1798441 start
                    # difficulty check
                    #defender to loop flag
                    if self.sensor.env.difficulty == 1:
                        # Defender agent
                        acceleration[idx] = hla.wait_at_team_flag(self.sensor.team, self.sensor.team_flags,
                                                                  idx, 0, self.sensor.env.delta_time)
                        self.last_action[idx] = 'wait'
                    #PN applied to defender to intercept blue agent 0
                    elif self.sensor.env.difficulty > 1:
                        if len(enemies_in_territory) == 0:
                            # Defender agent
                            acceleration[idx] = hla.wait_at_team_flag(self.sensor.team, self.sensor.team_flags,
                                                                      idx, 0, self.sensor.env.delta_time)
                            self.last_action[idx] = 'wait'
                        else:
                            # attacks id counter final if difficulty is 3
                            if self.sensor.env.difficulty == 2:
                                acceleration[idx] = hla.go_tag_agent(self.team, self.sensor.enemy_team, idx,
                                                                     enemies_in_territory[0], self.sensor.env.delta_time)
                                self.last_action[idx] = 'go_tag'
                            elif self.sensor.env.difficulty >= 3:
                                acceleration[idx] = hla.go_tag_agent(self.team, self.sensor.enemy_team, idx,
                                                                     closest_flag, self.sensor.env.delta_time)
                                if self.sensor.env.difficulty == 5:
                                    if idx == 0 :
                                        acceleration[idx] = hla.go_tag_agent(self.team, self.sensor.enemy_team, idx,
                                                                         topenemy, self.sensor.env.delta_time)
                                    if idx == 1:
                                        acceleration[idx] = hla.go_tag_agent(self.team, self.sensor.enemy_team, idx,
                                                                         botenemy, self.sensor.env.delta_time)

                    else:
                        acceleration[idx] = hla.wait_at_team_flag(self.sensor.team, self.sensor.team_flags,
                                                                  idx, 0, self.sensor.env.delta_time)
                        self.last_action[idx] = 'wait'

                elif self.sensor.env.difficulty != 5:
                    # Attacker agent

                    if enemy_flag_captured:
                        if self.sensor.team.has_flag[idx]:

                            if not (self.last_action[idx] == 'return_top' or
                                    self.last_action[idx] == 'return_bottom' or
                                    self.last_action[idx] == 'return_centre'):
                                action = choices(['return_top', 'return_bottom', 'return_centre'], [1 / 3] * 3)[0]
                                self.last_action[idx] = action

                            if self.last_action[idx] == 'return_top':
                                acceleration[idx] = hla.return_top(self.sensor.team, self.sensor.team_flags, idx, 0,
                                                                   self.sensor.env.delta_time)
                            elif self.last_action[idx] == 'return_bottom':
                                acceleration[idx] = hla.return_bottom(self.sensor.team, self.sensor.team_flags, idx, 0,
                                                                      self.sensor.env.delta_time)
                            elif self.last_action[idx] == 'return_centre':
                                acceleration[idx] = hla.return_centre(self.sensor.team, self.sensor.team_flags, idx, 0,
                                                                      self.sensor.env.delta_time)
                            else:
                                raise Exception("Invalid action")

                        else:
                            acceleration[idx] = hla.wait_at_enemy_flag(self.sensor.team, self.sensor.enemy_flags, idx,
                                                                       0, self.sensor.env.delta_time)
                    elif team_flag_captured and self.sensor.env.difficulty == 4:
                            holder = self.sensor.env.flag_holder(self.team)
                            acceleration[idx] = hla.go_tag_agent(self.team, self.sensor.enemy_team, idx,
                                        holder, self.sensor.env.delta_time)
                    elif self.sensor.env.difficulty == 4 and not team_flag_captured:
                        if not (self.last_action[idx] == 'attack_top' or self.last_action[idx] == 'attack_bottom'):
                            action = choices(['attack_top', 'attack_bottom'], [1 / 2] * 2)[0]
                            self.last_action[idx] = action

                        if self.last_action[idx] == 'attack_top':
                            acceleration[idx] = hla.attack_top(self.sensor.team, self.sensor.enemy_team,
                                                               self.sensor.enemy_flags,
                                                               idx, 0, 0, self.sensor.env.delta_time)
                        elif self.last_action[idx] == 'attack_bottom':
                            acceleration[idx] = hla.attack_bottom(self.sensor.team, self.sensor.enemy_team,
                                                                  self.sensor.enemy_flags,
                                                                  idx, 0, 0, self.sensor.env.delta_time)
                    ### a1798441 end
                    else:
                        if not (self.last_action[idx] == 'attack_top' or self.last_action[idx] == 'attack_bottom' or
                                self.last_action[idx] == 'attack_centre'):
                            action = choices(['attack_top', 'attack_bottom', 'attack_centre'], [1 / 3] * 3)[0]
                            self.last_action[idx] = action

                        if self.last_action[idx] == 'attack_top':
                            acceleration[idx] = hla.attack_top(self.sensor.team, self.sensor.enemy_team,
                                                               self.sensor.enemy_flags,
                                                               idx, 0, 0, self.sensor.env.delta_time)
                        elif self.last_action[idx] == 'attack_bottom':
                            acceleration[idx] = hla.attack_bottom(self.sensor.team, self.sensor.enemy_team,
                                                                  self.sensor.enemy_flags,
                                                                  idx, 0, 0, self.sensor.env.delta_time)
                        elif self.last_action[idx] == 'attack_centre':
                            acceleration[idx] = hla.attack_bottom(self.sensor.team, self.sensor.enemy_team,
                                                                  self.sensor.enemy_flags,
                                                                  idx, 0, 0, self.sensor.env.delta_time)
                        else:
                            raise Exception("Invalid action")




        return acceleration