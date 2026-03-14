# WorldModelObservation.msg
# A bundled observation to feed into the world model.
# The model_server_node selects the relevant fields based on
# the loaded model's expected input format.

std_msgs/Header header

# Primary image observation (the most common input type)
sensor_msgs/Image image

# Optional additional images (e.g. multi-camera setups)
sensor_msgs/Image[] extra_images
string[] image_keys

# Joint state (for robotics world models)
sensor_msgs/JointState joint_state

# End-effector pose (for manipulation world models)
geometry_msgs/PoseStamped ee_pose

# Task description (for language-conditioned models)
string task

# Raw flattened observation vector (for state-based models)
# This is what gets passed to wm.forward() for most Atari/MuJoCo models
float32[] obs_vector
uint32[] obs_shape
