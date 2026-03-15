from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    pkg_share = FindPackageShare('wm_runtime')

    # ── Launch arguments ──────────────────────────────────────
    args = [
        DeclareLaunchArgument('repo_id', default_value='SaiResearch/sai_wm',
                              description='HuggingFace repo ID'),
        DeclareLaunchArgument('subfolder', default_value='iris/atari/alien',
                              description='Model subfolder in the repo'),
        DeclareLaunchArgument('device', default_value='cpu',
                              description='Device: cpu, cuda:0, etc.'),
        DeclareLaunchArgument('trust_remote_code', default_value='false',
                              description='Trust remote code for custom models'),
        DeclareLaunchArgument('obs_mode', default_value='dataset',
                              description='Observation mode: dataset, image, vector'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw',
                              description='Image topic (for obs_mode=image)'),
        DeclareLaunchArgument('publish_rate_hz', default_value='10.0',
                              description='Belief publish rate (Hz)'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Launch RViz'),
        DeclareLaunchArgument('rviz_config', default_value=PathJoinSubstitution([
                                  pkg_share, 'rviz', 'wm_runtime.rviz']),
                              description='Path to RViz config file'),
    ]

    # ── Config file path ──────────────────────────────────────
    config_file = PathJoinSubstitution([pkg_share, 'config', 'wm_runtime.yaml'])

    # ── Nodes ─────────────────────────────────────────────────
    model_server = Node(
        package='wm_runtime',
        executable='model_server',
        name='wm_model_server',
        parameters=[
            config_file,
            {
                'repo_id': LaunchConfiguration('repo_id'),
                'subfolder': LaunchConfiguration('subfolder'),
                'device': LaunchConfiguration('device'),
                'trust_remote_code': LaunchConfiguration('trust_remote_code'),
            },
        ],
        output='screen',
    )

    belief_publisher = Node(
        package='wm_runtime',
        executable='belief_publisher',
        name='wm_belief_publisher',
        parameters=[
            config_file,
            {
                'obs_mode': LaunchConfiguration('obs_mode'),
                'image_topic': LaunchConfiguration('image_topic'),
                'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
            },
        ],
        output='screen',
    )

    rollout_visualizer = Node(
        package='wm_runtime',
        executable='rollout_visualizer',
        name='wm_rollout_visualizer',
        parameters=[config_file],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    return LaunchDescription(args + [
        model_server,
        belief_publisher,
        rollout_visualizer,
        rviz_node,
    ])