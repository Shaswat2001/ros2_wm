# ActionSequence.msg
# A sequence of actions to be imagined forward through the world model. 
# Maps to the actions list passed to wm.predict() in a loop. 

std_msgs/Header header

# Flattened action data: length = horizon * action_dim
# For discrete spaces: each action is a single int case to float32
# For continuous spaces: each action is a vector of floats
float32[] actions

# Number of timesteps in this sequence
uint32 horizon

# Dimensionality of a single action (1 for discrete, N for continuous)
uint32 action_dim

# "discrete" or "continuous"
string action_space

# Optional label for this candidate (e.g. "cem_elite_3", "random_17")
string label