from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'person_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='didim',
    maintainer_email='you@example.com',
    description='YOLOv8 + YDLiDAR G4 사람 추적 ROS2 패키지',
    license='MIT',
    entry_points={
        'console_scripts': [
            'person_follower_node = person_follower.person_follower_node:main',
            'camera_publisher_node = person_follower.camera_publisher_node:main',
	        'visualizer_node = person_follower.visualizer_node:main',


        ],
    },
)

