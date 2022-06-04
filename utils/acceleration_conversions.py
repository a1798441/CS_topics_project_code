"""
capture_the_flag
Converts a lateral acceleration to a acceleration vector.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""
import numpy as np


def convert_angular_acceleration(lateral_acceleration, azimuth):
    """Converts an lateral acceleration command into euclidean space.

    :param lateral_acceleration: what is the latax.
    :param azimuth: angle of agent.
    :return: array of acceleration commands.
    """
    acceleration_x = lateral_acceleration * -1 * np.sin(azimuth)
    acceleration_y = lateral_acceleration * np.cos(azimuth)
    return np.array([acceleration_x, acceleration_y])
