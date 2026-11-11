import math

from smart_community_mission.mission_lib import (
    format_announcement,
    format_waypoint_line,
    quat_to_yaw,
    should_wait,
)


def test_format_announcement():
    assert format_announcement("traffic_light", "绿灯，请通行") == "绿灯，请通行"


def test_should_wait_red_and_yellow():
    assert should_wait("RED") is True
    assert should_wait("YELLOW") is True
    assert should_wait("GREEN") is False


def test_quat_to_yaw():
    assert quat_to_yaw(0.0, 1.0) == 0.0
    assert math.isclose(quat_to_yaw(math.sin(math.pi / 4), math.cos(math.pi / 4)), math.pi / 2)
    assert math.isclose(quat_to_yaw(1.0, 0.0), math.pi)


def test_format_waypoint_line_people():
    wp = {'name': 'wp_1', 'task': 'people', 'zone': 'A',
          'x': -0.4, 'y': 0.85, 'yaw': 1.5708, 'dwell': 3.0}
    assert format_waypoint_line(wp) == (
        "  - {name: wp_1, task: people, zone: A, x: -0.400, y: 0.850, yaw: 1.57080, dwell: 3.0}"
    )


def test_format_waypoint_line_plate():
    wp = {'name': 'plate_1', 'task': 'license_plate', 'lot': 2,
          'x': 1.55, 'y': -1.05, 'yaw': 1.57, 'dwell': 2.0}
    assert format_waypoint_line(wp) == (
        "  - {name: plate_1, task: license_plate, lot: 2, x: 1.550, y: -1.050, yaw: 1.57000, dwell: 2.0}"
    )
