from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    start_x = LaunchConfiguration("start_x")
    start_y = LaunchConfiguration("start_y")
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    num_candidates = LaunchConfiguration("num_candidates")
    horizon = LaunchConfiguration("horizon")
    timer_period = LaunchConfiguration("timer_period")

    return LaunchDescription([
        DeclareLaunchArgument("start_x", default_value="0.0"),
        DeclareLaunchArgument("start_y", default_value="0.0"),
        DeclareLaunchArgument("goal_x", default_value="5.0"),
        DeclareLaunchArgument("goal_y", default_value="5.0"),
        DeclareLaunchArgument("num_candidates", default_value="32"),
        DeclareLaunchArgument("horizon", default_value="5"),
        DeclareLaunchArgument("timer_period", default_value="0.5"),

        Node(
            package="wm_nodes",
            executable="imagination_server_node",
            name="imagination_server",
            output="screen",
        ),
        Node(
            package="wm_nodes",
            executable="planner_node",
            name="planner_node",
            output="screen",
            parameters=[{
                "start_x": start_x,
                "start_y": start_y,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "num_candidates": num_candidates,
                "horizon": horizon,
                "timer_period": timer_period,
            }],
        ),
        Node(
            package="wm_nodes",
            executable="visualizer_node",
            name="visualizer_node",
            output="screen",
        ),
    ])