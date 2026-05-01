from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('person_follower'),
        'config', 'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='person_follower',
            executable='camera_publisher_node',
            name='camera_publisher',
            parameters=[config],
            output='screen'
        ),
        Node(
            package='person_follower',
            executable='person_follower_node',
            name='person_follower',
            parameters=[config],
            output='screen'
        ),
    ])