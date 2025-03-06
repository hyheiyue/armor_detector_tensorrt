cmake_minimum_required(VERSION 3.20)
project(armor_detector_tensorrt)

## Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
# add_definitions(-Wall -Werror)

## Suppress warnings for deprecated declarations
add_compile_options(-Wno-deprecated-declarations)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
ament_auto_find_build_dependencies()

list(APPEND PLUGINS nvinfer)
list(APPEND PLUGINS nvparsers)
list(APPEND PLUGINS nvonnxparser)
list(APPEND PLUGINS nvinfer_plugin)

foreach(libName ${PLUGINS})
  find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
  list(APPEND PLUGIN_LIBS "${${libName}_lib}")
endforeach()

include_directories(${CUDA_INCLUDE_DIRS})

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_link_libraries(${PROJECT_NAME} 
  ${PLUGINS}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY_DIRS}
  ${OpenCV_LIBS}
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "rm_auto_aim::ArmorDetectorTensorrtNode"
  EXECUTABLE ${PROJECT_NAME}_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
set(ament_cmake_clang_format_CONFIG_FILE "${CMAKE_SOURCE_DIR}/.clang-format")
find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_uncrustify
    ament_cmake_flake8
  )
  ament_lint_auto_find_test_dependencies()
endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
  model
)
