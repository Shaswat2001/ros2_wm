from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
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
        ),
        Node(
            package="wm_nodes",
            executable="visualizer_node",
            name="visualizer_node",
            output="screen",
        ),
    ])