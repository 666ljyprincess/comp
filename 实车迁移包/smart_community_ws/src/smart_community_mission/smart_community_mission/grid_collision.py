"""Conservative occupancy/keepout footprint checks for a 2-D Nav2 Path.

Pure Python geometry so it can be unit-tested without a ROS installation.
This is an *extra* check; Nav2 local costmap/controller remains responsible
for observing moving obstacles during execution.
"""
import math


def occupied(grid, x, y, unknown_is_obstacle=False, threshold=65):
    """grid: {data, width, height, resolution, origin_x, origin_y}."""
    res = grid['resolution']
    ix = math.floor((x - grid['origin_x']) / res)
    iy = math.floor((y - grid['origin_y']) / res)
    if not (0 <= ix < grid['width'] and 0 <= iy < grid['height']):
        return True
    cost = grid['data'][iy * grid['width'] + ix]
    return cost >= threshold or (unknown_is_obstacle and cost < 0)


def footprint_clear(grids, x, y, radius=0.075):
    """Circular overapproximation of 0.11 x 0.10 m Nav2 footprint."""
    # Filled disk sampling at <= 0.025m spacing, not just the perimeter.
    # A circle of radius 0.075m encloses the 0.11x0.10m Nav2 footprint.
    rings = max(1, math.ceil(radius / 0.025))
    samples = [(x, y)]
    for ring in range(1, rings + 1):
        r = radius * ring / rings
        count = max(12, math.ceil(2 * math.pi * r / 0.025))
        for j in range(count):
            angle = 2 * math.pi * j / count
            samples.append((x + r * math.cos(angle), y + r * math.sin(angle)))
    for px, py in samples:
        for grid in grids:
            if occupied(grid, px, py):
                return False
    return True


def verify_polyline(points, grids, radius=0.075, step=0.025):
    """Return (passed, detail). Both endpoints and every interpolated point are checked."""
    if not grids:
        return False, '未收到占据栅格地图'
    if len(points) < 2:
        return False, '路径点不足2个'
    for idx, (x, y) in enumerate(points):
        if idx == 0:
            if not footprint_clear(grids, x, y, radius):
                return False, f'起点足迹与地图/禁行区域冲突 ({x:.3f},{y:.3f})'
            continue
        x0, y0 = points[idx - 1]
        n = max(1, math.ceil(math.hypot(x - x0, y - y0) / step))
        for k in range(1, n + 1):
            t = k / n
            sx, sy = x0 + t * (x - x0), y0 + t * (y - y0)
            if not footprint_clear(grids, sx, sy, radius):
                return False, f'路径与地图/禁行区域冲突 ({sx:.3f},{sy:.3f})'
    return True, 'OK'


def as_grid(occupancy_grid_msg):
    meta = occupancy_grid_msg.info
    return dict(data=occupancy_grid_msg.data, width=meta.width, height=meta.height,
                resolution=meta.resolution, origin_x=meta.origin.position.x,
                origin_y=meta.origin.position.y)
