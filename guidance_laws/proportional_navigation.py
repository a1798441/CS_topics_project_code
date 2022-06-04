"""
capture_the_flag
This file runs the proportional navigation algorithm.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""
import numpy as np


def proportional_navigation(agent_position, agent_velocity, target_position, target_velocity, target_acceleration=None):
    """Does the proportional navigation algorithm.

    :param agent_position: position of the agent.
    :param agent_velocity: velocity of the agent.
    :param target_position: position of the target.
    :param target_velocity: velocity of the target.
    :param target_acceleration: acceleration of the target.
    :return: nd array of acceleration commands.
    """

    proportionality_constant = 5  # 1 is actually the best to avoid going in circles.

    # Calculate instantaneous relative position and velocity
    relative_position = target_position - agent_position
    relative_velocity = target_velocity - agent_velocity

    # Calculated dot products and cross products
    dot_position = np.dot(relative_position, relative_position)
    cross_position_velocity = np.cross(relative_position, relative_velocity)

    # Calculate rotation vector
    line_of_sight_rate = cross_position_velocity / dot_position
    line_of_sight_rate = np.array([0, 0, line_of_sight_rate])

    closing_velocity = np.linalg.norm(relative_velocity) * agent_velocity / np.linalg.norm(agent_velocity)
    closing_velocity = np.array([closing_velocity[0], closing_velocity[1], 0])

    acceleration = np.cross(-1 * proportionality_constant * closing_velocity, line_of_sight_rate)
    acceleration = acceleration[0:2]

    # Augmented acceleration (This has problems when you use the instantaneous acceleration).
    if target_acceleration is not None:
        augmented_acceleration = 0.5 * proportionality_constant * target_acceleration
        acceleration = acceleration + augmented_acceleration

    return acceleration
