#!/bin/bash
# 板端 ROS 环境（每个新 shell source 一次；或写进 ~/.bashrc）
source /opt/ros/humble/setup.bash
[ -f /opt/tros/humble/local_setup.bash ] && source /opt/tros/humble/local_setup.bash
export ROS_DOMAIN_ID=42          # PC 端要设成同一个值才能互相看到话题
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /userdata/smart_community_ws/install/local_setup.bash
echo "[env_real] ROS 环境已加载: DOMAIN=$ROS_DOMAIN_ID RMW=$RMW_IMPLEMENTATION"
