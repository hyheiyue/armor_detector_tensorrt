# # Copyright 2025 Lihan Chen
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.launch_description import LaunchDescription
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    bringup_dir = get_package_share_directory("armor_detector_tensorrt")

    namespace = LaunchConfiguration("namespace")
    params_file_node = LaunchConfiguration("params_file_node")
    params_file_component = LaunchConfiguration("params_file_component")
    container_name = LaunchConfiguration("container_name")
    use_external_container = LaunchConfiguration("use_external_container")
    use_sim_time = LaunchConfiguration("use_sim_time")
    model_path = LaunchConfiguration("model_path")
    use_composition = LaunchConfiguration("use_composition")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )
    declare_use_composition_cmd = DeclareLaunchArgument(
        "use_composition",
        default_value="true",
        description="Use composition if True",
    )

    declare_namespace_cmd = DeclareLaunchArgument(
        "namespace", default_value="", description="Namespace"
    )

    declare_params_file_node_cmd = DeclareLaunchArgument(
        "params_file_node",
        default_value=os.path.join(
            bringup_dir, "config", "armor_detector_node.yaml"),
        description="Full path to the ROS2 parameters file to use for all launched nodes",
    )

    declare_params_file_component_cmd = DeclareLaunchArgument(
        "params_file_component",
        default_value=os.path.join(
            bringup_dir, "config", "armor_detector_component.yaml"),
        description="Full path to the ROS2 parameters file to use for all launched composablenodes",
    )

    declare_container_name_cmd = DeclareLaunchArgument(
        "container_name",
        default_value="armor_detector_tensorrt_container",
        description="Container name",
    )

    declare_use_external_container_cmd = DeclareLaunchArgument(
        "use_external_container",
        default_value="false",
        description="Use external container",
    )

    declare_model_path_cmd = DeclareLaunchArgument(
        "model_path",
        default_value=os.path.join(
            bringup_dir, "model", "opt-1208-001.onnx"
        ),
        description="Full path to the ROS2 parameters file to use for all launched nodes",
    )

    container_node = Node(
        name=container_name,
        package="rclcpp_components",
        executable="component_container",
        output="screen",
        condition=UnlessCondition(use_external_container),
    )

    load_detector_components = LoadComposableNodes(
        condition=UnlessCondition(PythonExpression(["'", use_composition, "' == 'false'"])),
        target_container=container_name,
        composable_node_descriptions=[
            ComposableNode(
                package="armor_detector_tensorrt",
                plugin="rm_auto_aim::ArmorDetectorTensorrtNode",
                name="armor_detector_tensorrt",
                parameters=[params_file_component, {"use_sim_time": use_sim_time},
                                         {"detector.model_path": model_path}],
                namespace=namespace,
            )
        ],
    )
    load_detector_node = Node(
        condition=UnlessCondition(use_composition),
        package="armor_detector_tensorrt",
        executable="armor_detector_tensorrt_node",
        name="armor_detector_tensorrt",
        parameters=[params_file_node, {"use_sim_time": use_sim_time},
                                 {"detector.model_path": model_path}],
        namespace=namespace,
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_params_file_node_cmd)
    ld.add_action(declare_params_file_component_cmd)
    ld.add_action(declare_container_name_cmd)
    ld.add_action(declare_use_external_container_cmd)
    ld.add_action(declare_model_path_cmd)
    ld.add_action(declare_use_composition_cmd)

    ld.add_action(container_node)
    ld.add_action(load_detector_components)
    ld.add_action(load_detector_node)

    return ld
