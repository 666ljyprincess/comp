"""Pure-Python static-landmark association for a fixed Gazebo arena.

The landmark identifiers are NOT identity labels. They prevent the same
standee being counted again when photographed from several positions.
Resident/outsider comes solely from the trained YOLO classification.
"""
import math
from collections import defaultdict


def project_landmark(x, y, robot_x, robot_y, yaw, width, height,
                     hfov=1.40, camera_forward=0.095, camera_z=0.11,
                     landmark_z=0.075):
    # ROS base_link: +x forward, +y left. Gazebo camera optical image: +u right.
    cx = robot_x + camera_forward * math.cos(yaw)
    cy = robot_y + camera_forward * math.sin(yaw)
    dx, dy = x - cx, y - cy
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    if forward <= 0.08:
        return None
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    u = 0.5 * width - fx * left / forward
    v = 0.5 * height - fy * (landmark_z - camera_z) / forward
    if u < -30 or u > width + 30 or v < -30 or v > height + 30:
        return None
    return u, v, forward


class ZoneFusion:
    def __init__(self, landmarks, min_frames=2, max_pixel_error=65.0):
        self.landmarks = landmarks
        self.min_frames = max(1, int(min_frames))
        self.max_pixel_error = max_pixel_error
        self.votes = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.frames_seen = defaultdict(int)
        self.assigned = defaultdict(int)
        self.unassigned = defaultdict(int)

    def add_frame(self, zone, detections, pose, width, height, hfov=1.40):
        """detections is [(x1,y1,x2,y2,label,score), ...]."""
        if zone not in ('A', 'B'):
            return 0
        self.frames_seen[zone] += 1
        x, y, yaw = pose
        projected = []
        for lm in self.landmarks:
            if lm['zone'] != zone:
                continue
            p = project_landmark(lm['x'], lm['y'], x, y, yaw, width, height, hfov)
            if p is not None:
                projected.append((lm['id'], p))
        matches = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, label, score = det
            if label not in ('居民', '外来人员') or score < .30:
                continue
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            for lm_id, (u, v, dist) in projected:
                error = math.hypot(cx - u, cy - v)
                # Tolerate small localization inaccuracies; strongly reject distant matches.
                threshold = min(self.max_pixel_error, max(25., .6 * abs(x2 - x1) + 25.))
                if error <= threshold:
                    matches.append((error, i, lm_id, label))
        matches.sort()
        used_det, used_lm = set(), set()
        for _, i, lm_id, label in matches:
            if i in used_det or lm_id in used_lm:
                continue
            used_det.add(i)
            used_lm.add(lm_id)
            self.votes[zone][lm_id][label] += 1
            self.assigned[zone] += 1
        self.unassigned[zone] += max(0, len(detections) - len(used_det))
        return len(used_det)

    def report(self, zone):
        resident, outsider, uncertain = 0, 0, 0
        details = {}
        for lm_id, counts in self.votes[zone].items():
            total = sum(counts.values())
            if total < self.min_frames:
                continue
            best = max(counts, key=counts.get)
            # A contested vote does not invent a confident identity.
            if counts[best] < 0.6 * total:
                uncertain += 1
                details[lm_id] = '类别待核'
                continue
            details[lm_id] = best
            if best == '居民':
                resident += 1
            else:
                outsider += 1
        return {'zone': zone, 'total': resident + outsider,
                'resident': resident, 'outsider': outsider,
                'uncertain': uncertain, 'landmarks': details,
                'frames': self.frames_seen[zone],
                'matched_observations': self.assigned[zone],
                'unmatched_detections': self.unassigned[zone]}

    def reset(self, zone):
        self.votes.pop(zone, None)
        self.frames_seen.pop(zone, None)
        self.assigned.pop(zone, None)
        self.unassigned.pop(zone, None)
