"""Pure-Python competition rules (easy to regression-test without ROS 2)."""
import math
import re
from collections import Counter


def choose_view_count(samples):
    """Choose the most frequently repeated *nonempty* validated frame count.

    Frames at the same viewpoint are evidence, not independent people.
    This counts ONE snapshot per viewpoint, not a sum of video frames.
    A validated zero is used only when no people were confirmed at all.
    """
    if not samples:
        return None
    nonempty = [tuple(x) for x in samples if sum(x) > 0]
    if not nonempty:
        return (0, 0)
    votes = Counter(nonempty)
    return max(votes, key=lambda counts: (votes[counts], sum(counts), counts[1]))


def people_consensus(samples, *, min_votes=2, allow_empty=False, min_valid_frames=2):
    """Stable viewpoint count; ignore preliminary empty frames when persons appear.

    Return None until two (configurable) positive frames agree. Empty views
    require an explicit opt-in, otherwise a YOLO warmup frame must not be
    interpreted as a valid zero-person observation.
    """
    valid = [tuple(v) for v in samples if (len(v)==2
              and all(type(n) is int and n >= 0 for n in v))]
    positive = Counter(v for v in valid if sum(v) > 0)
    if positive:
        winner, votes = max(positive.items(), key=lambda it: (it[1], sum(it[0]), it[0][1]))
        return winner if votes >= max(2, int(min_votes)) else None
    if allow_empty and len(valid) >= max(2, int(min_valid_frames)):
        return (0, 0)
    return None

def zone_totals(view_results):
    """User-required simple sum of viewpoint counts, WITHOUT cross-view dedup."""
    residents = sum(v[0] for v in view_results)
    outsiders = sum(v[1] for v in view_results)
    return residents, outsiders


def angular_error(target, actual):
    return abs(math.atan2(math.sin(target - actual), math.cos(target - actual)))


def parking_pass(x, y, yaw, target_x, target_y, target_yaw, v, w,
                 xy_tolerance=0.12, yaw_tolerance=0.20,
                 linear_tolerance=0.03, angular_tolerance=0.10):
    """True only if inside the zone, facing correctly and stationary."""
    return (math.hypot(x-target_x, y-target_y) <= xy_tolerance
            and angular_error(target_yaw, yaw) <= yaw_tolerance
            and abs(v) <= linear_tolerance
            and abs(w) <= angular_tolerance)


# Exactly the plate format validated by the existing plate detector:
# province + A-Z + 5..6 uppercase alphanumeric characters.
_PLATE_FORMAT = re.compile(r'[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][A-Z][A-Z0-9]{5,6}\Z')


def valid_plate(value):
    """Do not count a blank, unknown or malformed OCR result as a vote."""
    return isinstance(value, str) and bool(_PLATE_FORMAT.fullmatch(value))


def plate_consensus(votes, *, min_votes=2):
    """Return a uniquely supported license plate, else an empty string.

    The input counts independent camera callbacks at the *current* viewpoint,
    not character guesses from repeated transforms of one image.
    """
    positive = [(plate, count) for plate, count in votes.items()
                if valid_plate(plate) and type(count) is int and count > 0]
    if not positive:
        return ''
    positive.sort(key=lambda item: item[1], reverse=True)
    top_plate, top_count = positive[0]
    if top_count < max(2, int(min_votes)):
        return ''
    if len(positive) > 1 and positive[1][1] == top_count:
        return ''  # competing license plates with equal evidence
    return top_plate
