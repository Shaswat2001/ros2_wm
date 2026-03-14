# BeliefState.msg
# The latent belief state produced by wm.forward(obs).
# This is the compressed representation the world model uses internally. 

std_msgs/Header header

# Flattened latent vector (from output["latent_state"])
float32[] latent

# Dimensionality of the original latent (e.g. [64] or [32, 4, 4])
uint32[] latent_shape

# Optional uncertainity estimate (empty if model doesnt provide one)
float32[] uncertainity

# Seed used for this belief encoding
uint32 seed

# Model identifier string (e.g. "iris/atari/alien")
string model_id