#!/usr/bin/env python3
"""Capture a legally reachable Nav2 observation pose for the user to paste in waypoints.yaml."""
import argparse
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zone',choices=['A','B'],required=True)
    ap.add_argument('--index',type=int,required=True)
    ap.add_argument('--dwell',type=float,default=2.0)
    args=ap.parse_args()
    rclpy.init()
    node=Node('people_waypoint_capture')
    buf=Buffer()
    listener=TransformListener(buf,node,spin_thread=True)
    try:
        input('Use Nav2 to place the robot at a safe legal observation point; press Enter to capture...')
        t=buf.lookup_transform('map','base_link',rclpy.time.Time(),timeout=Duration(seconds=2.0))
        q=t.transform.rotation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        print(f'- {{name: people_{args.zone.lower()}_{args.index}, task: people, zone: {args.zone}, '
              f'x: {t.transform.translation.x:.3f}, y: {t.transform.translation.y:.3f}, '
              f'yaw: {yaw:.5f}, dwell: {args.dwell:.1f}}}')
    except Exception as exc:
        print('Could not capture TF:',exc)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
