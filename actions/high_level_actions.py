"""
capture_the_flag
This file constrains the set of actions to some set of high level actions.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""
##needed for calculating distance between agents
from scipy.spatial.distance import cdist

from guidance_laws.proportional_navigation import proportional_navigation
from guidance_laws.all_aspect_proportional_navigation import all_aspect_proportional_navigation
import numpy as np
from gym import spaces


class HighLevelActionSet:

    def __init__(self, acceleration_limit):
        """Defines the set of high level actions that a particular agent could take.

        :param acceleration_limit: The maximum acceleration that can be applied.
        """
        self.acceleration_limit = acceleration_limit

        self.action_set = np.array(["go_to_enemy_flag", "go_to_base", "wait_at_enemy_flag", "go_tag_agent",
                                    "attack_centre", "attack_bottom", "attack_top", "return_centre", "return_bottom",
                                    "return_top"])
        self.action_space = spaces.Discrete(len(self.action_set))


def go_to_enemy_flag(team, enemy_flags, agent_idx, flag_idx, delta_time):
    """Determine the acceleration commands for an agent to take the direct path to the enemy flag.

    :param team:
    :param enemy_flags:
    :param agent_idx:
    :param flag_idx:
    :param delta_time: step time.
    :return:
    """

    return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                                 team.azimuths[agent_idx], delta_time)

### a1708087 start
##adds enemy team and enemy flags as new parameters
#difficulty 1
def go_to_enemy_flag_smart(team, enemy_team, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time):
    #for the naive enemies
    """Determine the acceleration commands for an agent to take the direct path to the enemy flag, but avoiding defenders if possible.

    :param team:
    :param enemy_flags:
    :param enemy_team:
    :param agent_idx:
    :param enemy_idx:
    :param flag_idx:
    :param delta_time: step time.
    :return:
    """
    #variables for ease of control
    enemy_avoidance_radius = 40

    enemy_top_flank = [100, 70]
    enemy_bottom_flank = [100, 10]

    #defenderTail is a set distance behind the enemy defender. The blue attacker tries to navigate to this location before
    #going for the flag
    defenderTail = enemy_team.positions[0] - enemy_team.velocities[0]*8

    #distance between attackers and defenders
    dist = cdist(team.positions, enemy_team.positions, metric='euclidean')
    #attackers and flag
    distFlags = cdist(team.positions, enemy_flags.positions, metric='euclidean')
    #defenders and flag
    distFlagsE = cdist(enemy_team.positions, enemy_flags.positions, metric='euclidean')

    ##go for flag when safe
    if distFlags[1][0] < 15 and distFlagsE[0][0] > 15:
        #print("Going for flag")
        return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                            team.azimuths[agent_idx], delta_time)

    ##if enemy defender x-component velocity is strong negative (moving left), swerve attacker.
    elif enemy_team.velocities[0][0] < -0.6 and dist[1][0] < enemy_avoidance_radius:
        if enemy_team.velocities[0][1] < 0:
            #swerves north if enemy y-component is negative
            #print("Swerving North")
            return take_direct_path(team.positions[agent_idx], enemy_top_flank, team.speed,
                                team.azimuths[agent_idx], delta_time)
        else:
            #swerves south if enemy y-component is positive
            #print("Swerving South")
            return take_direct_path(team.positions[agent_idx], enemy_bottom_flank, team.speed,
                                    team.azimuths[agent_idx], delta_time)

    ##dist[1][0] should be comparing blue attacker with red defender
    ##dist[1][1] compares blue attacker with red attacker
    elif dist[1][0] < enemy_avoidance_radius:
        #print("Tailing")
        return take_direct_path(team.positions[agent_idx], defenderTail, team.speed,
                                team.azimuths[agent_idx], delta_time)

    else:
        return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                            team.azimuths[agent_idx], delta_time)

#difficulty 2
def go_to_enemy_flag_smarter(team, enemy_team, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time):
    #for enemies with PN
    """Determine the acceleration commands for an agent to take the direct path to the enemy flag, but avoiding defenders if possible.

    :param team:
    :param enemy_flags:
    :param enemy_team:
    :param agent_idx:
    :param enemy_idx:
    :param flag_idx:
    :param delta_time: step time.
    :return:
    """

    #variables for ease of control
    enemy_avoidance_radius = 50

    #ONLY FOR DIFFICULTY 2 (sending a distraction wont work for 3.)
    #The flanker attacks the wider side?
    mirror_flag_strategy = True
    #send the attacking blue agent[1] right for the flag?
    send_one_straight = True

    #flank destination
    agent_zero_flank = [130, 70]
    #where agent 0 moves to along the middle border before commencing flank in enemy territory
    agent_zero_mid = team.env.centre

    if mirror_flag_strategy:
        #agent 1 will flank the same side as the flag (north or south), leaving agent 0 to be the distraction on the opposite side
        if enemy_flags.positions[flag_idx][1] < 40: #if flag is in lower half of playing area
            agent_zero_flank = [130, 70]
            agent_zero_mid = team.env.top
        else:
            agent_zero_flank = [130, 10]
            agent_zero_mid = team.env.bottom


    #attacking angles, to ensure flanks
    if team.env.in_blue_territory(team, agent_idx):
        if agent_idx == 0:
            if mirror_flag_strategy:
                #as the distraction, goes opposite side to flag
                return take_direct_path(team.positions[agent_idx], agent_zero_mid, team.speed,
                                team.azimuths[agent_idx], delta_time)
            else:
                return take_direct_path(team.positions[agent_idx], team.env.top, team.speed,
                                        team.azimuths[agent_idx], delta_time)

        elif agent_idx == 1:
            if mirror_flag_strategy:
                #as the flag capturer, goes right for flag
                return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                                        team.azimuths[agent_idx], delta_time)
            elif send_one_straight:
                return take_direct_path(team.positions[agent_idx], team.env.centre, team.speed,
                                        team.azimuths[agent_idx], delta_time)
            else:
                return take_direct_path(team.positions[agent_idx], team.env.bottom, team.speed,
                                        team.azimuths[agent_idx], delta_time)
        else:
            return take_direct_path(team.positions[agent_idx], team.env.centre, team.speed,
                                    team.azimuths[agent_idx], delta_time)

    ##go for flag when reached flank
    elif team.positions[agent_idx][0] > 130:
        #print(agent_idx, "Going for flag")
        return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                            team.azimuths[agent_idx], delta_time)
    #take a flank to try draw the defender to one attacker
    else:
        #agent 0 will swing wider to draw the defender
        if agent_idx == 0:
            return take_direct_path(team.positions[agent_idx], agent_zero_flank, team.speed,
                                    team.azimuths[agent_idx], delta_time)
        #agent 1 will go straight for flag
        elif agent_idx == 1:
            return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                                    team.azimuths[agent_idx], delta_time)
        else:
            return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                                    team.azimuths[agent_idx], delta_time)


# difficulty 3+ only.
def go_to_enemy_flag_smartest(team, enemy_team, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time):
    # for enemies with PN
    """Determine the acceleration commands for an agent to take the direct path to the enemy flag, but avoiding defenders if possible.
    Difficulty 3 means the attackers will try remain equidistant from the flag to 'confuse' the red defender

    :param team:
    :param enemy_flags:
    :param enemy_team:
    :param agent_idx:
    :param enemy_idx:
    :param flag_idx:
    :param delta_time: step time.
    :return:
    """

    # variables for ease of control
    #default 30
    enemy_avoidance_radius = 41

    top_flank = [130, 65]
    bottom_flank = [130, 15]

    evade_top = [80, 75]
    evade_mid = [80, 40]
    evade_bottom = [80, 5]

    # distance between attackers and defenders
    dist = cdist(team.positions, enemy_team.positions, metric='euclidean')
    # attackers and flag
    distFlags = cdist(team.positions, enemy_flags.positions, metric='euclidean')
    #prevents 'zigzagging'
    dist_buffer = 5


    # attacking angles, to ensure flanks
    if team.env.in_blue_territory(team, agent_idx):
        if agent_idx == 0:
            #if agent 0 is ahead, slow down
            if distFlags[0][0] < distFlags[1][0] - dist_buffer:
                #print("0 is ahead!")
                return take_direct_path(team.positions[agent_idx], [70, 70], team.speed,
                                        team.azimuths[agent_idx], delta_time)
            else:
                return take_direct_path(team.positions[agent_idx], team.env.top, team.speed,
                                        team.azimuths[agent_idx], delta_time)
        elif agent_idx == 1:
            # if agent 1 is ahead, slow down
            if distFlags[1][0] < distFlags[0][0] - dist_buffer:
                #print("1 is ahead!")
                return take_direct_path(team.positions[agent_idx], [70, 10], team.speed,
                                        team.azimuths[agent_idx], delta_time)
            else:
                return take_direct_path(team.positions[agent_idx], team.env.bottom, team.speed,
                                        team.azimuths[agent_idx], delta_time)
        else:
            return take_direct_path(team.positions[agent_idx], team.env.centre, team.speed,
                                    team.azimuths[agent_idx], delta_time)

    ##go for flag when reached flank or if ally is tagged
    elif team.positions[agent_idx][0] > 130 or (team.is_tagged[0] or team.is_tagged[1]):
        # print(agent_idx, "Going for flag")
        return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                                team.azimuths[agent_idx], delta_time)
    # move to flanks
    else:
        # agent 0 will go north
        if agent_idx == 0:
            if dist[0][0] < enemy_avoidance_radius and distFlags[0][0] < distFlags[1][0]:
                #print("0 is evading!")
                #if agent is southbound, swerve south
                if team.positions[0][1] < enemy_team.positions[0][1]:
                    return take_direct_path(team.positions[agent_idx], evade_bottom, team.speed,
                                        team.azimuths[agent_idx], delta_time)
                else:
                    return take_direct_path(team.positions[agent_idx], evade_top, team.speed,
                                            team.azimuths[agent_idx], delta_time)
            else:
                return take_direct_path(team.positions[agent_idx], top_flank, team.speed,
                                    team.azimuths[agent_idx], delta_time)

        # agent 1 will go south
        elif agent_idx == 1:
            if dist[1][0] < enemy_avoidance_radius and distFlags[1][0] < distFlags[0][0]:
                #print("1 is evading!")
                # if agent is southbound, swerve south
                if team.positions[1][1] < enemy_team.positions[0][1]:
                    return take_direct_path(team.positions[agent_idx], evade_bottom, team.speed,
                                            team.azimuths[agent_idx], delta_time)
                else:
                    return take_direct_path(team.positions[agent_idx], evade_top, team.speed,
                                            team.azimuths[agent_idx], delta_time)
            else:
                return take_direct_path(team.positions[agent_idx], bottom_flank, team.speed,
                                        team.azimuths[agent_idx], delta_time)
        else:
            return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                                    team.azimuths[agent_idx], delta_time)


# difficulty 1
def return_smart(team, enemy_team, home_flags, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time):
    """Travel to the home flag through the centre path. Specifically the agent will travel to the centre of the map
    and then travel to the home flag. Attempts to avoid enemies.

    :param team:
    :param enemy_team
    :param home_flags:
    :param agent_idx:
    :param enemy_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """

    if team.color == 'blue':
        distFlags = cdist(team.positions, enemy_flags.positions, metric='euclidean')

        if team.env.in_red_territory(team, agent_idx):
            #finds the relative vector from blue attacker to red defender
            relativeVector = enemy_team.positions[0] - team.positions[1]
            #if defender is above attacker, return bottom
            if relativeVector[1] > 0 and distFlags[1][0] < 10:
                #print("taking bottom retreat")
                return take_direct_path(team.positions[agent_idx], [140, 10], team.speed,
                                        team.azimuths[agent_idx], delta_time)
            #if below, return top
            elif distFlags[1][0] < 10:
                #print("taking top retreat")
                return take_direct_path(team.positions[agent_idx], [140, 70], team.speed,
                                        team.azimuths[agent_idx], delta_time)
            #once clear of flag, go home
            else:
                #print("taking flag home")
                return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)

        else:
            return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)
    else:
        raise Exception("Invalid Team")

#difficulty 2+
def return_smarter(team, enemy_team, home_flags, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time):
    """Travel to the home flag through the centre path. Specifically the agent will travel to the centre of the map
    and then travel to the home flag. Attempts to avoid enemies.

    :param team:
    :param enemy_team
    :param home_flags:
    :param agent_idx:
    :param enemy_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """
    distFlags = cdist(team.positions, enemy_flags.positions, metric='euclidean')

    #how close to the edges, along middle boundary that the flag capturer escapes to
    #default: 10. Lower values should increase the success rate of escape
    edge_distance = 1

    ##smarter function steers clear of enemy defender
    if team.env.in_red_territory(team, agent_idx):
        #print(agent_idx, "checking enemy location", enemy_team.positions[0][1] - team.positions[agent_idx][1])
        #calculates y position of the defender to the attacker agent
        if enemy_team.positions[0][1] > team.positions[agent_idx][1]:
            #take bottom path if enemy is above
            #print(agent_idx, "Escaping South")
            return take_direct_path(team.positions[agent_idx], [80, edge_distance], team.speed,
                                    team.azimuths[agent_idx], delta_time)

        elif enemy_team.positions[0][1] < team.positions[agent_idx][1]:
            #take top path if enemy is below
            #print(agent_idx, "Escaping North")
            return take_direct_path(team.positions[agent_idx], [80, 80 - edge_distance], team.speed,
                                    team.azimuths[agent_idx], delta_time)
        else:
            return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)

    elif not team.env.in_red_territory(team, agent_idx):
        return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)

    else:
        raise Exception("Invalid Team")
    ### a1708087 end


def go_to_base(team, home_flags, agent_idx, flag_idx, delta_time):
    """Determine the acceleration commands for an agent to travel to their home base (their own flag).

    :param team:
    :param home_flags:
    :param agent_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """
    return take_direct_path(team.positions[agent_idx], home_flags.positions[flag_idx], team.speed,
                                 team.azimuths[agent_idx], delta_time)


def wait_at_enemy_flag(team, enemy_flags, agent_idx, flag_idx, delta_time):
    """Determine the acceleration commands for an agent to wait at the enemy flag.

    :param team:
    :param enemy_flags:
    :param agent_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """
    return take_direct_path(team.positions[agent_idx], enemy_flags.positions[flag_idx], team.speed,
                                 team.azimuths[agent_idx], delta_time)


def wait_at_team_flag(team, home_flags, agent_idx, flag_idx, delta_time):
    """Determine the acceleration commands for an agent to wait at their home flag.

    :param team:
    :param home_flags:
    :param agent_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """
    return take_direct_path(team.positions[agent_idx], home_flags.positions[flag_idx], team.speed,
                                 team.azimuths[agent_idx], delta_time)


def go_tag_agent(team, enemy_team, agent_idx, enemy_idx, delta_time, aapn=False):
    """Determine the acceleration commands for an agent to travel to an enemy to tag them.

    :param team:
    :param enemy_team:
    :param agent_idx:
    :param enemy_idx:
    :param delta_time:
    :param aapn:
    :return:
    """
    if not aapn:
        # Calculate heading error
        heading_error = get_angle_diff(team.positions[agent_idx], enemy_team.positions[enemy_idx],
                                            team.azimuths[agent_idx])
        if heading_error < np.pi/2:
            return proportional_navigation(team.positions[agent_idx], team.velocities[agent_idx],
                                           enemy_team.positions[enemy_idx], enemy_team.velocities[enemy_idx])
        else:
            return take_direct_path(team.positions[agent_idx], enemy_team.positions[enemy_idx], team.speed,
                                         team.azimuths[agent_idx], delta_time)
    else:
        return all_aspect_proportional_navigation(team.positions[agent_idx], team.velocities[agent_idx],
                                                  enemy_team.positions[enemy_idx], enemy_team.velocities[enemy_idx],
                                                  team.azimuths[agent_idx])

def attack_centre(team, enemy_team, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time):
    """Travel to the enemy flag through the centre path. Specifically the agent will travel to the centre of the map
    and then travel to the enemy flag.

    :param team:
    :param enemy_team:
    :param enemy_flags:
    :param enemy_idx:
    :param agent_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """

    if team.color == 'red':
        if team.env.in_red_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.centre, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_enemy_flag(team, enemy_flags, agent_idx, flag_idx, delta_time)
    elif team.color == 'blue':
        if team.env.in_blue_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.centre, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_enemy_flag_smart(team, enemy_team, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time)
    else:
        raise Exception("Invalid Team")


def attack_bottom(team, enemy_team, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time):
    """Travel to the enemy flag through a lower path. Specifically the agent will travel to the midpoint of the line
    that directly connects the centre of the map with the lower border of the map. The agent will then then travel to
    the enemy flag.

    :param team:
    :param enemy_team:
    :param enemy_flags:
    :param enemy_idx:
    :param agent_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """
    if team.color == 'red':
        if team.env.in_red_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.bottom, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_enemy_flag(team, enemy_flags, agent_idx, flag_idx, delta_time)
    elif team.color == 'blue':
        if team.env.in_blue_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.bottom, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_enemy_flag_smart(team, enemy_team, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time)
    else:
        raise Exception("Invalid Team")


def attack_top(team, enemy_team, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time):
    """Travel to the enemy flag through an upper path. Specifically the agent will travel to the midpoint of the line
    that directly connects the centre of the map with the upper border of the map. The agent will then then travel to
    the enemy flag.

    :param team:
    :param enemy_team:
    :param enemy_flags:
    :param enemy_idx:
    :param agent_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """

    if team.color == 'red':
        if team.env.in_red_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.top, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_enemy_flag(team, enemy_flags, agent_idx, flag_idx, delta_time)

    elif team.color == 'blue':
        if team.env.in_blue_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.top, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_enemy_flag_smart(team, enemy_team, enemy_flags, agent_idx, enemy_idx, flag_idx, delta_time)
    else:
        raise Exception("Invalid Team")


def return_centre(team, home_flags, agent_idx, flag_idx, delta_time):
    """Travel to the home flag through the centre path. Specifically the agent will travel to the centre of the map
    and then travel to the home flag.

    :param team:
    :param home_flags:
    :param agent_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """
    if team.color == 'red':
        if team.env.in_blue_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.centre, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)
    elif team.color == 'blue':
        if team.env.in_red_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.centre, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)
    else:
        raise Exception("Invalid Team")


def return_bottom(team, home_flags, agent_idx, flag_idx, delta_time):
    """Travel to the home flag through a lower path. Specifically the agent will travel to the midpoint of the line
    that directly connects the centre of the map with the lower border of the map. The agent will then then travel to
    the home flag.

    :param team:
    :param home_flags:
    :param agent_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """
    if team.color == 'red':
        if team.env.in_blue_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.bottom, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)
    elif team.color == 'blue':
        if team.env.in_red_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.bottom, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)
    else:
        raise Exception("Invalid Team")


def return_top(team, home_flags, agent_idx, flag_idx, delta_time):
    """Travel to the home flag through an upper path. Specifically the agent will travel to the midpoint of the line
    that directly connects the centre of the map with the upper border of the map. The agent will then then travel to
    the home flag.

    :param team:
    :param home_flags:
    :param agent_idx:
    :param flag_idx:
    :param delta_time:
    :return:
    """
    if team.color == 'red':
        if team.env.in_blue_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.top, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)

    elif team.color == 'blue':
        if team.env.in_red_territory(team, agent_idx):
            return take_direct_path(team.positions[agent_idx], team.env.top, team.speed,
                                         team.azimuths[agent_idx], delta_time)
        else:
            return go_to_base(team, home_flags, agent_idx, flag_idx, delta_time)
    else:
        raise Exception("Invalid Team")


def get_angle_diff(current_position, desired_position, azimuth):
    """Define a unit vector in the direction of an agent's azimuth, define a vector in the direction of the line
     of sight between the agent and some given point. Find the angle between these vectors.

     :param current_position: x, y position of agent.
     :param desired_position: x, y position of another point.
     :param azimuth: heading angle of agent.
     :return: angle between unit vectors.
     """
    unit_vector_heading = np.array([np.cos(azimuth), np.sin(azimuth)])

    y_diff = desired_position[1] - current_position[1]
    x_diff = desired_position[0] - current_position[0]
    unit_vector_diff = np.array([x_diff, y_diff])
    unit_vector_diff = unit_vector_diff / np.linalg.norm(unit_vector_diff)

    dot_product = np.dot(unit_vector_heading, unit_vector_diff)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle = np.arccos(dot_product)
    return angle


def take_direct_path(current_position, desired_position, speed, azimuth, delta_time):
    """Calculate acceleration commands to take a direct path to a location. This is by minimising the angle between
    a agents heading and the line of sight vector.

    :param current_position:
    :param desired_position:
    :param speed:
    :param azimuth:
    :param delta_time:
    :return:
    """

    unit_vector_heading = np.array([np.cos(azimuth), np.sin(azimuth)])
    los_y = desired_position[1] - current_position[1]
    los_x = desired_position[0] - current_position[0]
    unit_vector_los = np.array([los_x, los_y])
    unit_vector_los = unit_vector_los / np.linalg.norm(unit_vector_los)

    angle = get_angle_diff(current_position, desired_position, azimuth)

    lateral_acceleration = angle * speed / delta_time
    if np.cross(unit_vector_los, unit_vector_heading) < 0:
        # Turning Left
        pass
    else:
        # Turning Right
        lateral_acceleration = lateral_acceleration * -1

    acceleration_x = lateral_acceleration * -1 * np.sin(azimuth)
    acceleration_y = lateral_acceleration * np.cos(azimuth)
    return np.array([acceleration_x, acceleration_y])