"""
capture_the_flag
This is the main file to run the environment.

Copyright: Commonwealth of Australia 2022
Developed by: David Hubczenko CWT/WCSD/DST Group
POC: David.Hubczenko@dst.defence.gov.au
Released to be used in the project entitled "Autonomous multi-agent decision making
in Capture the Flag game" for the Advanced Topics in Computer Science course at the
University of Adelaide.
"""


from environment.game_environment import GameEnvironment

if __name__ == '__main__':

    # Should the red team be trained. If the red team is trained, should the existing neural network be updated.
    train_red = False
    train_existing_red = False

    # Should the blue team be trained. If the blue team is trained, should the existing neural network be updated.
    train_blue = False
    train_existing_blue = False

    # Runs the game multiple times and outputs some statistics on the success/failure of the teams playing the game.
    should_evaluate = True

    # Displays the game running.
    should_display = False

    # Generates an mp4 file of the game.
    should_generate_animation = False

    # Training parameters
    epochs = 10
    training_iterations = 1000000
    randomise = True

    # Goal options are attack or defend or ctf
    # Placement options are random_same, random_constraint, "random", "flag"
    # Control options are custom
    # action sets are discrete, joint, high_level, continuous
    red_team_var = {"n_agents": 2,
                    "n_flags": 1,
                    "acceleration_limit": 0.1,
                    "speed": 1.0,
                    "delta_time": 1.0,
                    "team_goal": 'ctf',
                    "placement_choice": "random_constraint",
                    "control": 'custom',
                    "action_set": "discrete",
                    "color": 'red'}

    blue_team_var = {"n_agents": 2,
                     "n_flags": 1,
                     "acceleration_limit": 0.1,
                     "speed": 1.0,
                     "delta_time": 1.0,
                     "team_goal": 'ctf',
                     "placement_choice": "random_constraint",
                     "control": 'custom',
                     "action_set": "discrete",
                     "color": 'blue'}

    # Generate the environment
    game_rules = 'ctf'  # The games can be CTF or attack_defend (red is attack and blue is defend)
    env = GameEnvironment(game_rules=game_rules, red_team_var=red_team_var, blue_team_var=blue_team_var,
                          generate_graphics=True, randomise=randomise)

    if train_red or train_blue:
        for i in range(training_iterations):
            if train_red:
                if train_existing_red:
                    env.load("red")
                env.train("red", epochs)
            else:
                env.load("red")

            if train_blue:
                if train_existing_blue:
                    env.load("blue")
                env.train("blue", epochs)
            else:
                env.load("blue")
    else:
        env.load("red")
        env.load("blue")

    if should_evaluate:
        env.evaluate(evaluation_type=game_rules, evaluation_eps=1000, should_render=False)

    if should_display:
        for i in range(10):
            env.run_ctf(should_render=True)

    if should_generate_animation:
        for i in range(10):
            file_name = 'myfile_%s.mp4' % i
            env.generate_animation(file_name)
