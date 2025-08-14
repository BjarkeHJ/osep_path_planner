from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="osep_2d_map",
            executable="osep_2d_map",
            name="costmap_2d_node",
            output="screen",
            parameters=[
                {"resolution": 1.0},
                {"free_center_radius": 5.0},
                {"local_map_size": 400.0},
                {"global_map_size": 1600.0},
                {"frame_id": "base_link"},
                {"safety_distance": 10.0},          # Use only this parameter
            ]
        ),
    ])