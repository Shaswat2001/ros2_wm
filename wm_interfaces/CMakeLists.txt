cmake_minimum_required(VERSION 3.8)
project(wm_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Core state
  "msg/BeliefState.msg"
  "msg/ActionSequence.msg"
  "msg/RolloutTrajectory.msg"
  "msg/RolloutSet.msg"

  # Observation bundle
  "msg/WorldModelObservation.msg"

  # Status / diagnostics
  "msg/ModelStatus.msg"

  # Services
  "srv/ForwardObservation.srv"
  "srv/Imagine.srv"
  "srv/PlanAction.srv"
  "srv/WhatIf.srv"
  "srv/LoadModel.srv"

  DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
    builtin_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()