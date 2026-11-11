import math


def format_announcement(task: str, summary: str, waypoint=None) -> str:
    if task == 'license_plate':
        lot = waypoint.get('lot', '?') if waypoint else '?'
        return f"{lot}号停车场车牌号为{summary}"
    if task == 'people' and waypoint and waypoint.get('zone'):
        return f"{waypoint['zone']}区{summary}"
    return summary


def should_wait(state: str) -> bool:
    return state != "GREEN"


def quat_to_yaw(z: float, w: float) -> float:
    return 2.0 * math.atan2(z, w)


def format_waypoint_line(wp: dict) -> str:
    parts = [f"name: {wp['name']}", f"task: {wp['task']}"]
    if wp.get('zone'):
        parts.append(f"zone: {wp['zone']}")
    if wp.get('lot'):
        parts.append(f"lot: {wp['lot']}")
    parts.append(f"x: {float(wp['x']):.3f}")
    parts.append(f"y: {float(wp['y']):.3f}")
    parts.append(f"yaw: {float(wp['yaw']):.5f}")
    parts.append(f"dwell: {float(wp.get('dwell', 2.0)):.1f}")
    return "  - {" + ", ".join(parts) + "}"
