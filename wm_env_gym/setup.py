from setuptools import find_packages, setup

package_name = 'wm_env_gym'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'gymnasium'],
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
        'console_scripts': [
            "gym_planner_node = wm_env_gym.gym_planner_node:main"
        ],
    },
)
