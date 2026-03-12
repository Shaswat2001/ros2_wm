from setuptools import find_packages, setup

package_name = 'wm_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shaswatgargai',
    maintainer_email='sis_shaswat@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "imagination_server_node = wm_nodes.imagination_server_node:main",
            "planner_node = wm_nodes.planner_node:main",
            "visualizer_node = wm_nodes.visualizer_node:main",
        ],
    },
)
