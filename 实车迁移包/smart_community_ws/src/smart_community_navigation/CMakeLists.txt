cmake_minimum_required(VERSION 3.8)
project(smart_community_navigation)

find_package(ament_cmake REQUIRED)
find_package(nav2_bringup REQUIRED)

install(DIRECTORY
  maps
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
