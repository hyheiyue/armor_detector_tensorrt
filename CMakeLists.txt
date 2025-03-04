cmake_minimum_required(VERSION 3.8)
project(armor_detector_tensorrt)

# 启用 CUDA 语言
enable_language(CUDA)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 86)

# 设置 CUDA 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86")

# 默认 C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# 设置 CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ROS 依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(image_transport_plugins REQUIRED)
find_package(tf2 REQUIRED)
find_package(auto_aim_interfaces REQUIRED)
find_package(rmoss_interfaces REQUIRED)
find_package(rmoss_util REQUIRED)

set(ros_dependencies
  rclcpp
  rclcpp_components
  geometry_msgs
  visualization_msgs
  cv_bridge
  image_transport
  image_transport_plugins
  tf2
  auto_aim_interfaces
  rmoss_util
  rmoss_interfaces
)

# set(CUDNN_INCLUDE_DIR=/usr/local/cuda-11.7/include)
# set(CUDNN_LIBRARY=/usr/local/cuda-11.7/lib)

# 第三方依赖
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED core imgproc video dnn)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
find_package(fmt REQUIRED)
find_package(CUDA REQUIRED)
# find_package(cudart REQUIRED)

# 包含路径
include_directories(${CUDA_INCLUDE_DIRS} /usr/include/x86_64-linux-gnu)
include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda-11.8/targets/x86_64-linux)
link_directories(${CUDA_LIBRARY_DIRS} /usr/local/cuda-11.8/targets/x86_64-linux)
include_directories(${TENSORRT_INCLUDE_DIRS} /home/jacket/packages/TensorRT-8.5.3.1/include)
link_directories(${TENSORRT_LIBRARY_DIRS} /home/jacket/packages/TensorRT-8.5.3.1/lib)
# include_directories(${CUDNN_INCLUDE_DIRS})
# link_directories(${CUDNN_LIBRARY_DIRS} /usr/local/cuda-11.8/lib)


include_directories(${CMAKE_SOURCE_DIR}/TRTInferSample/include)

set(other_dependencies
  Eigen3::Eigen
  ${OpenCV_LIBS}
  openvino::frontend::onnx
  openvino::runtime
  fmt::fmt
)
add_compile_options(-Wno-deprecated-declarations)  # 忽略所有弃用警告
# 构建库
add_library(${PROJECT_NAME} SHARED
  src/mono_measure_tool.cpp
  src/tensorrt_detect_node.cpp
  src/trt_module.cpp
)

# 设置 CUDA 源文件的编译选项
# set_source_files_properties(src/CUDA/preprocess.cu PROPERTIES LANGUAGE CUDA)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  # ${CUDNN_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${other_dependencies}
  nvinfer nvinfer_plugin nvonnxparser
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY_DIRS}
  # ${CUDNN_LIBRARIES}
  # cudart
)
ament_target_dependencies(${PROJECT_NAME} ${ros_dependencies})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "rm_auto_aim::TensorRTDetectorNode"
  EXECUTABLE ${PROJECT_NAME}_node
)

# 安装
install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config model
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# 导出
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(${ros_dependencies})

ament_package()