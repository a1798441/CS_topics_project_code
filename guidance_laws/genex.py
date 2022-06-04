"""
capture_the_flag
This file runs the GENEX algorithm.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""
import numpy as np


def genex(agent_position, agent_velocity, target_position, target_velocity):
    """Does the GENEX guidance law.

    :param agent_position: position of the agent.
    :param agent_velocity: velocity of the agent.
    :param target_position: position of the target.
    :param target_velocity: velocity of the target.
    :return: ndarray of acceleration commands.
    """

    n = 3  # GENEX gain
    k1 = (n + 2) * (n + 3)
    k2 = -1 * (n + 1) * (n + 2)

    relative_position = target_position - agent_position
    xtm = relative_position.conj().T / np.linalg.norm(relative_position)

    # Unit vector representing the desired final velocity vector
    vm_f = -1 * target_velocity.conj().T / np.linalg.norm(target_velocity)
    vm = agent_velocity.conj().T / np.linalg.norm(agent_velocity)

    acceleration = np.linalg.norm(agent_velocity) ** 2 / np.linalg.norm(relative_position) * \
        (k1 * (xtm - vm * np.dot(xtm, vm)) + k2 * (vm_f - vm * np.dot(vm_f, vm)))

    return acceleration
