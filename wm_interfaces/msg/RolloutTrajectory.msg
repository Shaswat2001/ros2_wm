# RolloutTrajectory.msg
# A single imagined trajectory from the world model.
# Direct mapping to worldmodel_hub.rollout.Trajectory.

std_msgs/Header header

# Number of predict steps in this rollout
uint32 horizon

# Flattened predicted latent states: length = (horizon+1) * latent_dim
# Includes the initial state from forward() as index 0
float32[] predicted_latents
uint32 latent_dim

# Predicted rewards at each step: length = horizon
float32[] rewards

# Predicted termination flags: length = horizon
bool[] terminated

# Sum of predicted rewards
float32 total_reward

# Steps until first termination (or horizon if never)
uint32 episode_length

# Wall-clock time for this rollout (seconds)
float64 wall_time

# Steps per second throughput
float32 steps_per_second

# The action sequence that produced this rollout
wm_interfaces/ActionSequence actions

# Optional: flattened reconstructed observations
# Empty if record_observations was false
uint8[] predicted_frames
uint32 frame_height
uint32 frame_width
uint32 frame_channels

# Seed and model info
uint32 seed
string model_id
string label
