from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    env_id = LaunchConfiguration("env_id")
    num_candidates = LaunchConfiguration("num_candidates")
    horizon = LaunchConfiguration("horizon")
    timer_period = LaunchConfiguration("timer_period")
    max_steps_per_episode = LaunchConfiguration("max_steps_per_episode")
    seed = LaunchConfiguration("seed")

    return LaunchDescription([
        DeclareLaunchArgument("env_id", default_value="Pendulum-v1"),
        DeclareLaunchArgument("num_candidates", default_value="32"),
        DeclareLaunchArgument("horizon", default_value="5"),
        DeclareLaunchArgument("timer_period", default_value="0.1"),
        DeclareLaunchArgument("max_steps_per_episode", default_value="200"),
        DeclareLaunchArgument("seed", default_value="0"),

        Node(
            package="wm_nodes",
            executable="imagination_server_node",
            name="imagination_server",
            output="screen",
            parameters=[{
                "backend_type": "oracle_gym",
                "env_id": env_id,
            }],
        ),

        Node(
            package="wm_env_gym",
            executable="gym_planner_node",
            name="gym_planner_node",
            output="screen",
            parameters=[{
                "env_id": env_id,
                "num_candidates": num_candidates,
                "horizon": horizon,
                "timer_period": timer_period,
                "max_steps_per_episode": max_steps_per_episode,
                "seed": seed,
            }],
        ),
    ])