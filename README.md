# CAPTURE THE FLAG FRAMEWORK

## INSTALLATION
Can copy and run this anywhere without any setup so long as the following dependencies are met.

### Dependencies
- Python 3: This software was developed on Python 3.5 but other versions probably work.

### List of Python packages:
Python packages can be installed by: pip3 install "package name"
- torch
- matplotlib
- numpy
- scipy
- gym

##How to find work/contribution made by student
In the files of environment/game_environment.py and algorithms/custom/custom_controllerR.py
one can find the contributions made by student by pressing [ctrl+F] and type in "### a1798441 start"
and the end of the contribution/edit with "### a1798441 end"

##Project paper instructions
1A. If the reader would like the code to display rounds of CTF: first check within the file [main.py] that variable 
‘should_evaluate’ at line 27 is set to ‘False’ and that the variable ‘should_display’ at line 30 is set to ‘True’

1B. If the reader would like the code to evaluate a set amount of rounds in CTF: first check within the file [main.py]
that variable ‘should_evaluate’ at line 27 is set to ‘True’ and that the variable ‘should_display’ at line 30 is set to
‘False’. Then at line 91 of [main.py] within the function ‘env.evaluate’, the parameter of ‘evaluation_eps’ will be the
amount rounds run in the evaluation, as of submitting it should be at 1000 and if the reader wanted to perform the same
experiment should keep it as such

2. To change the speed of red team: Within the [main.py] file, at line 47 is the speed variable for the red team whereas
 the blue counter part is at line 58. To perform the same experiment, one must make sure the blue team speed is 1 so to 
test a red team at 50% the speed would be set to 0.5. 
 
3. To change the level/difficulty: To change the level go to file [game_environment.py] found  in the ‘environment’ folder, 
then at line 39 is ‘self.difficulty’ which can be set values 1 to 3 which corresponds the different levels as stated in the 
methodology section of the report

4. Running the code: If the proper steps are run then the code should be able to run with differing results depending the 
values used in steps 3 and 2. The Biggest changes are in 1A and 1B. If 1A was the step chosen then when running, the code 
will show a window of ctf running and will repeatedly do so for a bit. If 1B was taken the code will only run in the terminal 
continually updating with every ten games played, after it is done it will display a message showing the results for example 
“B 450 -  550 - 0 R” Meaning out of 1000 games, blue won 450, both drew at 550 and red won 0.


## Games Description
### Attack defend game
One of the games that can be played is called the attack defend game. In this game one of the teams is explicitly the 
attacking side and one of the teams is explicitly the defender side. The attacking team has to bypass the defenders and
reach some targets (called flags). The defenders have to stop the attackers by crashing into the attackers.

### Capture the flag game
Another game that can be played is capture the flag. There are two teams a red and a blue team. The code has been 
written such that the two team should be interchangeable however traditionally red has been the attacking agents and 
blue has been the defending agents. Both sides can have some number of agents, and some number of flags. 

## FILE STRUCTURE
main.py is the main file from which the parameters can be set, and the program is run.

actions folder: The action set of the teams can be continuous, discrete, high level or joint actions. Continuous means
that a particular agent can apply any acceleration command (within some limits). Discrete means that agents can apply
some discrete set of acceleration commands. High level means that there is a set of high level actions that the agents
can choose from. Joint means that actions are considered in the context of the group rather than by individual agents.

algorithms folder: This contains all the algorithms that determine the actions of the agents.

buffers folder: In reinforcement learning algorithms there is a typically a buffer that stores data that is generated
by the algorithm and then later used to update a neural network. There are different options for buffers that could be 
used.

environment folder: Contains the implementation of the environment.

guidance_laws folder: This contains some implementations of specific guidance laws.

neural_network_architectures folder: The neural network used by a particular reinforcement learning algorithm 

sensors folder: There are different sensors that could be used to model how an agent perceives the world. 

target_allocation folder: Contains some algorithms for target allocation.

utils folder: Some utility functions

## USAGE GUIDE
The program is started by running the main.py file.
python3 main.py

In main.py you can set all the parameters for the experiments you may wish to undertake.

#### Main.py usage parameters.
- train_red = True/False to indicate whether the red team should be trained. Typically, this involved iteratively
updating the parameters of a neural network.
- train_existing_red = True/False to indicate whether the red team should be trained based on updating an existing
model or trained from scratch. 

- train_blue = True/False to indicate whether the blue team should be trained. Typically, this involved iteratively
updating the parameters of a neural network.
- train_existing_blue = True/False to indicate whether the blue team should be trained based on updating an existing
model or trained from scratch. 
    
- should_evaluate = True/False. If True this runs the game multiple times and outputs some statistics on the 
success/failure of the teams playing the game.
- should_display = True/False. If True this runs the game and displays the game running graphically on the screen.

- should_generate_animation = True/False. If True this runs the game multiple times and generates and mp4 file for each
run.

#### Training parameters (within main.py)
- epochs = <integer>. Set this to some integer value. This is how many times the model will be updated in training.
- training_iterations = <integer>. Set this to some integer value. This is how many episodes of data will be generated.
- randomise = True/False. This is experimental. If True, during training the initial state will be randomised each time.

##### Team parameters (within main.py)
- n_agents = <integer>. How many agents are on the team.
- n_flags = <integer>. How many flags are on the team.
- acceleration_limit = <real number>. This is the maximum acceleration that an agent can apply.
- speed = <real number>. This is how fast the agents on the team move.
- delta_time = <real number>. This is how often the agents make decisions.
- team_goal = <string>. What goal the team is pursuing during the game. This is dependent on the game the
environment is emulating. The choices are attack, defend, or ctf.
- placement_choice = <string>. How the agents should be placed at the start of each episode. The choices are 
random_same, random_constraint, random, flag.
    - 'random_same': placed at the same random location in the starting bounds.
    - 'random_constraint': placed at random locations in the starting bounds and with a min dist between agents.
    - 'random': placed at random locations in the starting bounds.
    - 'flag': the agents should be placed at the team flag/s.
- control = <string>. Sets the control algorithm to be used by the team.
- action_set = <string>. The choices are discrete, joint, high_level, continuous.
- color = <string>. The choices are red or blue.

#### Environment parameters (within main.py)
- game_rules = <string>. The games can be CTF or attack_defend (red is attack and blue is defend)

## DEVELOPER GUIDE
### Game Environment
The game environment is implemented within environment/game_environment.py. To help run reinforcement learning 
algorithms there is a file environment/reinforcement_learning_training_interface.py. This interface provides an 
OpenAI Gym like interface to the environment. This was done as it is common practise to use the gym interface in
typical reinforcement learning. The design decision was made to keep this interface separate from the environment 
because the environment should support multiple algorithms. The main aspect of this interface that a developer may want
to alter is the reward structure. Either new functions could be defined or the existing ones altered as needed. 

### Entities
There is a base class called Entities that represents a number of entities. There are some classes that have been
defined that inherit from the base Entities class; these are Agents, Flags and Obstacles. This choice was made as it 
was thought that there might be some characteristics which all these classes share. Some of the attributes defined
by Entities is positions, velocities  and accelerations. Most of the functions implemented here are with resetting the
state of the entities to some initial value as well as different functions for the placement of the entities within the
environment.

#### Agents
This class inherits from Entities. It implements the agents that are playing the game. It mostly acts as a class that 
manages all the aspects of the agents. There are separate Controller and Sensor classes that this class connects to.
This class implements some of the functions that control certain actions of the agents such as attempting to capture
the flag. There are also some functions that are implemented that show how the agents react to various acceleration 
demands. At the moment it assumed that agents have constant speed so only a lateral acceleration can be applied.

#### Flags
This class manages the flags. In the game there will be two Flags objects instantiated; one for the red team and one 
for the blue team. This class manages the state of the flags i.e. whether they are currently captured or not. It also
implements some functions to capture or drop the flag.

#### Obstacles
This class manages obstacles. Obstacles are defined as circles. Checking for collisions is handled in the 
GameEnvironment class.

### Sensors
This is the base class from which the various sensor models can be extended. 

### Actions
Includes various sets of actions including continuous, discrete, joint and high level. These are the actions that are
available to a particular group of agents.

### Guidance Laws
This implements some guidance laws that are used as part of the set of high level actions.

## NOTES
~100 epochs with 20-20-20-20 (get to flag)

~To get past enemies I have a negative reward from the point in which the agent got killed.

## POSSIBLE ISSUES
I have run into issues with floating-point errors and differing timesteps. The current version should be OK but might 
need to check this.
