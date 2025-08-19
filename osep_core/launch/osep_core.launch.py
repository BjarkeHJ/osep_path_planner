from launch import LaunchDescription
from launch_ros.actions import Node

FRAME_ID = "base_link"
SAFETY_DISTANCE = 10.0

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="osep_2d_map",
            executable="costmap_2d_node",
            name="costmap_2d_node",
            output="screen",
            parameters=[
                {"resolution": 1.0},
                {"free_center_radius": 5.0},
                {"local_map_size": 400.0},
                {"global_map_size": 1600.0},
                {"frame_id": FRAME_ID},
                {"safety_distance": SAFETY_DISTANCE},
            ]
        ),
        Node(
            package="osep_2d_map",
            executable="path_interpolator_node",
            name="planner",
            output="screen",
            parameters=[
                {"frame_id": FRAME_ID},
                {"interpolation_distance": 2.0},
                {"costmap_topic": "/local_costmap/costmap"},
                {"waypoints_topic": "/osep/viewpoints"},
                {"path_planner_prefix": "/planner"},
                {"ground_truth_update_interval": 8000},
                {"safety_distance": SAFETY_DISTANCE},
            ]
        ),
    ])
