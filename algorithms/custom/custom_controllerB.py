"""
capture_the_flag
This file defines the custom controller that controls the agents.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""

import numpy as np
from utils.utils import choices
from algorithms.controller import Controller
import actions.high_level_actions as hla


class CustomControllerB(Controller):

    def __init__(self, goal, team, sensor, model=None):

        if not goal == 'ctf':
            raise Exception("Goal is not supported by this controller")

        super().__init__(goal, team, sensor, action_set='high_level', controller_type='custom')

    def get_acceleration(self):
        """Get acceleration commands based on the proportional navigation algorithm.

        :return: ndarray of acceleration commands.
        """

        enemy_flag_captured = self.sensor.enemy_flags.is_captured[0]
        acceleration = np.zeros((self.n_agents, 2))
        for idx in range(self.n_agents):
            if self.sensor.team.is_tagged[idx]:
                # If tagged then return to base
                acceleration[idx] = hla.go_to_base(self.sensor.team, self.sensor.team_flags, idx, 0,
                                                   self.sensor.env.delta_time)
                self.last_action[idx] = "tagged"

            else:
                # Check if any enemy agents in territory
                enemies_in_territory = self.sensor.env.check_for_enemies_in_territory(self.team)

                ### a1708087 start
                # count how many tagged enemies in territory
                tagged_enemies_in_territory = []

                # Remove any tagged enemies
                for enemy in enemies_in_territory:
                    if self.sensor.enemy_team.is_tagged[enemy]:
                        enemies_in_territory.remove(enemy)
                        tagged_enemies_in_territory.append(enemy)

                #defender agents wait for enemy to be tagged, then switch to attack
                #check if blue agents are in blue territory
                in_allied_territory = self.sensor.env.in_blue_territory(self.team, idx)

                #skip defense phase for testing purposes?
                skip_defense = False

                #there are no red attackers for difficulty 5, so skip blue defense
                if self.sensor.env.difficulty == 5:
                    skip_defense = True

                #defends until an enemy is tagged
                if ((len(tagged_enemies_in_territory) == 0 and in_allied_territory and not enemy_flag_captured and self.sensor.env.difficulty > 1) \
                        or (idx == 0 and self.sensor.env.difficulty < 2)) and not skip_defense:
                    #print("Defending")
                    # Defender agent
                    if len(enemies_in_territory) == 0:
                        acceleration[idx] = hla.wait_at_team_flag(self.sensor.team, self.sensor.team_flags,
                                                                idx, 0, self.sensor.env.delta_time)
                        self.last_action[idx] = 'wait'
                    else:
                        acceleration[idx] = hla.go_tag_agent(self.team, self.sensor.enemy_team, idx,
                                                            enemies_in_territory[0], self.sensor.env.delta_time)
                                                            #use "idx % len(enemies_in_territory)" instead of "0". This allows multiple attackers to be targeted by different defenders.
                        self.last_action[idx] = 'go_tag'

                else:
                    # Attacker agent
                    if enemy_flag_captured:
                        if self.sensor.team.has_flag[idx]:

                            #difficulty 2+
                            if self.sensor.env.difficulty > 1:
                                acceleration[idx] = hla.return_smarter(self.sensor.team, self.sensor.enemy_team, self.sensor.team_flags, self.sensor.enemy_flags, idx, 0, 0,
                                                                self.sensor.env.delta_time)
                            else:
                                acceleration[idx] = hla.return_smart(self.sensor.team, self.sensor.enemy_team, self.sensor.team_flags, self.sensor.enemy_flags, idx, 0, 0,
                                                                 self.sensor.env.delta_time)
                        else:
                            acceleration[idx] = hla.wait_at_enemy_flag(self.sensor.team, self.sensor.enemy_flags, idx,
                                                                       0, self.sensor.env.delta_time)
                    else:
                        if self.sensor.env.difficulty == 2:
                            #print(idx, "Attacking")
                            acceleration[idx] = hla.go_to_enemy_flag_smarter(self.sensor.team, self.sensor.enemy_team, self.sensor.enemy_flags,
                                                               idx, 0, 0, self.sensor.env.delta_time)
                        elif self.sensor.env.difficulty > 2:
                            #print(idx, "Attacking")
                            acceleration[idx] = hla.go_to_enemy_flag_smartest(self.sensor.team, self.sensor.enemy_team, self.sensor.enemy_flags,
                                                               idx, 0, 0, self.sensor.env.delta_time)


                        #for difficulty 1
                        elif self.sensor.env.difficulty == 1:
                            if not (self.last_action[idx] == 'attack_top' or self.last_action[idx] == 'attack_bottom' or
                                    self.last_action[idx] == 'attack_centre'):
                                action = choices(['attack_top', 'attack_bottom', 'attack_centre'], [1 / 3] * 3)[0]
                                self.last_action[idx] = action

                            if self.last_action[idx] == 'attack_top':
                                acceleration[idx] = hla.attack_top(self.sensor.team, self.sensor.enemy_team, self.sensor.enemy_flags,
                                                                   idx, 0, 0, self.sensor.env.delta_time)
                            elif self.last_action[idx] == 'attack_bottom':
                                acceleration[idx] = hla.attack_bottom(self.sensor.team, self.sensor.enemy_team, self.sensor.enemy_flags,
                                                                   idx, 0, 0, self.sensor.env.delta_time)
                            elif self.last_action[idx] == 'attack_centre':
                                acceleration[idx] = hla.attack_centre(self.sensor.team, self.sensor.enemy_team, self.sensor.enemy_flags,
                                                                   idx, 0, 0, self.sensor.env.delta_time)
                        else:
                            raise Exception("Invalid action")

        return acceleration
        ### a1708087 end