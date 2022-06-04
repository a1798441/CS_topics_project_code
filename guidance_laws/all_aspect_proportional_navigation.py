import numpy as np

"""
capture_the_flag
This file runs the all aspect proportional navigation algorithm.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""
from utils.acceleration_conversions import convert_angular_acceleration

# Set proportionality constant
PROPORTIONALITY_CONSTANT = 5  # 1 is actually the best to avoid going in circles.


def all_aspect_proportional_navigation(agent_position, agent_velocity, target_position, target_velocity, azimuth):
    """Does the proportional navigation algorithm.
    See https://link.springer.com/article/10.1023/A:1008395924656

    :param agent_position: position of the agent.
    :param agent_velocity: velocity of the agent.
    :param target_position: position of the target.
    :param target_velocity: velocity of the target.
    :param azimuth: current azimuth of the agent.

    :return: nd array of acceleration commands.
    """

    # Calculate instantaneous relative position and velocity
    relative_position = target_position - agent_position
    relative_velocity = target_velocity - agent_velocity
    agent_speed = np.linalg.norm(agent_velocity)
    relative_distance = np.linalg.norm(target_position - agent_position)

    unit_vector_heading = np.array([np.cos(azimuth), np.sin(azimuth)])
    unit_vector_los = relative_position / np.linalg.norm(relative_position)

    # Check this for errors
    heading_error = np.arctan2(unit_vector_heading[0] * unit_vector_los[1] -
                               unit_vector_heading[1] * unit_vector_los[0],
                               unit_vector_heading[0] * unit_vector_los[0] +
                               unit_vector_heading[1] * unit_vector_los[1])

    # Calculated dot products and cross products
    dot_position = np.dot(relative_position, relative_position)
    cross_position_velocity = np.cross(relative_position, relative_velocity)

    # Calculate rotation vector
    line_of_sight_rate = cross_position_velocity / dot_position

    function = (0.1717 * heading_error -
                0.3885 * (heading_error ** 2) +
                0.1925 * (heading_error ** 3)) * (agent_speed ** 3 / relative_distance)
    acceleration = 3 * agent_speed * line_of_sight_rate + function  # Does it need to be negative 3?

    return convert_angular_acceleration(acceleration, azimuth)


def get_angle_diff(current_position, desired_position, azimuth):
    """Gets the difference between an agent's heading and a point"""
    unit_vector_heading = np.array([np.cos(azimuth), np.sin(azimuth)])

    y_diff = desired_position[1] - current_position[1]
    x_diff = desired_position[0] - current_position[0]
    unit_vector_diff = np.array([x_diff, y_diff])
    unit_vector_diff = unit_vector_diff / np.linalg.norm(unit_vector_diff)

    dot_product = np.dot(unit_vector_heading, unit_vector_diff)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # arccos only defined for [-1, 1]
    angle = np.arccos(dot_product)

    return angle
