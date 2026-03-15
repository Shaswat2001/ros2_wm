import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'wm_runtime'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ArenaX Labs',
    maintainer_email='research@competesai.com',
    description='ROS2 imagination runtime backed by worldmodel_hub',
    license='MIT',
    entry_points={
        'console_scripts': [
            'model_server = wm_runtime.model_server_node:main',
            'belief_publisher = wm_runtime.belief_publisher_node:main',
            'rollout_visualizer = wm_runtime.rollout_visualizer_node:main',
        ],
    },
)