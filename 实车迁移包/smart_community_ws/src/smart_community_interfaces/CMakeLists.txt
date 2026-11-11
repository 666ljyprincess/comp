cmake_minimum_required(VERSION 3.8)
project(smart_community_interfaces)

find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PerceptionBox.msg"
  "msg/PerceptionResult.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
