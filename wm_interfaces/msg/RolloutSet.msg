# RolloutSet.msg
# A collection of imagined trajectories with scoring.
# Produced by the imagination server after evaluating candidates.

std_msgs/Header header

# All rollout trajectories
wm_interfaces/RolloutTrajectory[] trajectories

# Index of the best trajectory (by total_reward or scorer)
uint32 best_index

# The belief state these rollouts started from
wm_interfaces/BeliefState belief
