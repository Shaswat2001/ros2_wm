# ModelStatus.msg
# Published by the model server to report its current state.

std_msgs/Header header

# Is a model loaded and ready to serve?
bool ready

# HuggingFace repo and subfolder
string repo_id
string subfolder

# Model metadata from worldmodel_hub
string model_type
string environment
string device

# Runtime stats
float32 avg_predict_ms
float32 avg_forward_ms
uint64 total_predictions
uint64 total_rollouts