cmake_minimum_required(VERSION 3.8)
project(smart_community_slam)

find_package(ament_cmake REQUIRED)
find_package(slam_toolbox REQUIRED)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
