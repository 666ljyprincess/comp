"""Blue rear-plate detector for the 4.2 m Gazebo smart-community competition.

Pipeline:
  1) find blue *vehicle* foreground, preferably the largest nearby rear view;
  2) find a dark navy, plate-shaped patch INSIDE that vehicle;
  3) run PaddleOCR on that patch, never on the body/spoiler;
  4) emit diagnostic crops and recognizer raw text for troubleshooting.

Still dependent on adequate image resolution and legible Gazebo plate texture.
"""
import os
import re
import traceback

import cv2

from ..detector_base import Detector, Detection

_OCR = None
_OCR_INIT_DONE = False


def _get_ocr():
    global _OCR, _OCR_INIT_DONE
    if _OCR_INIT_DONE:
        return _OCR
    _OCR_INIT_DONE = True
    try:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='ch',
            enable_mkldnn=False,
        )
    except Exception as exc:
        print(f'[车牌识别] PaddleOCR 初始化失败：{exc!r}', flush=True)
        traceback.print_exc(limit=5)
        _OCR = None
    return _OCR



_LINE_REC = None
_LINE_REC_INIT_DONE = False


def _get_line_recognizer():
    """Text-line recognizer: avoid re-detecting text in a 59x20 plate crop.

    PaddleOCR 3.x official API. Reuses the already cached v6 medium_rec in the
    user's logs; the model is loaded once.  Override with env variable if needed.
    """
    global _LINE_REC, _LINE_REC_INIT_DONE
    if _LINE_REC_INIT_DONE:
        return _LINE_REC
    _LINE_REC_INIT_DONE = True
    try:
        from paddleocr import TextRecognition
        name = os.environ.get('SMART_PLATE_REC_MODEL', 'PP-OCRv6_medium_rec')
        print(f'[车牌OCR] 正在加载整行识别模型 {name}', flush=True)
        _LINE_REC = TextRecognition(model_name=name)
    except Exception as exc:
        print(f'[车牌OCR] TextRecognition 初始化失败：{exc!r}', flush=True)
        traceback.print_exc(limit=5)
        _LINE_REC = None
    return _LINE_REC


def _read_model_texts(results):
    """Handle PaddleX/PaddleOCR's {'res':{...}} and older pipeline dict shapes."""
    if results is None:
        return []
    if isinstance(results, (dict, str)):
        results = [results]
    found = []
    for item in results:
        if isinstance(item, str):
            found.append((item, 0.0))
            continue
        if not hasattr(item, 'get'):
            # Some paddle versions expose JSON via .json rather than Mapping.
            payload = getattr(item, 'json', None)
            if callable(payload):
                payload = payload()
            if not isinstance(payload, dict):
                continue
            item = payload
        data = item.get('res', item)
        if not hasattr(data, 'get'):
            continue
        single = data.get('rec_text')
        if single:
            found.append((str(single), float(data.get('rec_score') or 0.0)))
        texts = data.get('rec_texts') or []
        scores = data.get('rec_scores') or []
        for i, text in enumerate(texts):
            if text:
                score = float(scores[i]) if i < len(scores) else 0.0
                found.append((str(text), score))
    return found


def _variants(roi):
    """Simple, non-hallucinating pixel preprocessors for tiny simulation fonts."""
    h, w = roi.shape[:2]
    target_w = 320
    scale = min(8.0, max(3.0, target_w / max(w, 1)))
    up = cv2.resize(roi, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)
    yield 'color_cubic', up
    # Contrast enhancement only when plain color input did not pass validation.
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enh = clahe.apply(gray)
    yield 'gray_clahe', cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)

_PROVINCES = '京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新'
_PLATE_RE = re.compile(rf'[{_PROVINCES}][A-Z][A-Z0-9]{{5,6}}')


def normalize_plate(value):
    if not isinstance(value, str):
        return ''
    value = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', value).upper()
    match = _PLATE_RE.search(value)
    return match.group(0) if match else ''


_DEBUG_DIR = os.environ.get('SMART_PLATE_DEBUG_DIR', '/tmp/smart_community_plate_debug')
_DEBUG_SAMPLES = 0


def _save_debug(image, car_box, plate_box, mask, route):
    """Save latest debug frames even at car_2 (after car_1 used earlier slots)."""
    global _DEBUG_SAMPLES
    try:
        os.makedirs(_DEBUG_DIR, exist_ok=True)
        frame = image.copy()
        if car_box is not None:
            cx, cy, cw, ch = car_box
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 255), 1)
        x, y, w, h = plate_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi = image[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(_DEBUG_DIR, 'latest_frame_bbox.jpg'), frame)
        cv2.imwrite(os.path.join(_DEBUG_DIR, 'latest_mask.png'), mask)
        if roi.size:
            cv2.imwrite(os.path.join(_DEBUG_DIR, 'latest_roi.png'), roi)
            cv2.imwrite(os.path.join(_DEBUG_DIR, 'latest_roi_4x.png'),
                        cv2.resize(roi, None, fx=4, fy=4,
                                   interpolation=cv2.INTER_CUBIC))
        # Numbered snapshots are bounded; latest_* is always overwritten.
        if _DEBUG_SAMPLES < 30:
            prefix = f'{_DEBUG_SAMPLES:02d}'
            _DEBUG_SAMPLES += 1
            cv2.imwrite(os.path.join(_DEBUG_DIR, f'{prefix}_frame_bbox.jpg'), frame)
            if roi.size:
                cv2.imwrite(os.path.join(_DEBUG_DIR, f'{prefix}_roi.png'), roi)
        print(f'[车牌定位] 策略={route}；车身框={car_box}；车牌框={plate_box}；'
              f'原图宽高={image.shape[1]}x{image.shape[0]}；调试目录={_DEBUG_DIR}',
              flush=True)
    except Exception as exc:
        print(f'[车牌定位] 保存截图失败：{exc!r}', flush=True)


def _locate_plate(image):
    """Find blue license plates by TEXT strokes, not by blue vehicle contours.

    Why: at car_2, the painted vehicle is fragmented in HSV, so the old
    vehicle-contour prerequisite fails and its white fallback boxes the road.
    A car plate itself, however, has a dense horizontal cluster of vertical
    character strokes and a blue background; this route works without any
    detected car bounding box.

    Returns (x, y, w, h), optional car box, diagnostic binary mask, strategy.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Group vertical character strokes into a plate-shaped horizontal band.
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dx = cv2.convertScaleAbs(dx)
    _, strokes = cv2.threshold(dx, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(round(width/200))), 3))
    stroke_groups = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(stroke_groups, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / max(1.0, float(h))
        # The competition's plate has physical ratio 9.5cm / 3cm ~= 3.17.
        # Allow margin for perspective and imperfect crop.
        if not (2.25 <= ratio <= 5.2):
            continue
        if not (w >= max(24, .035 * width) and h >= max(7, .012 * height)):
            continue
        if w > .38 * width or h > .16 * height:
            continue
        if x <= 1 or y <= 1 or x + w >= width - 1:
            # Partially visible neighbors should not outrank the central car.
            continue

        region = hsv[y:y + h, x:x + w]
        # Plate backgrounds vary in brightness among different Gazebo textures;
        # do NOT require V <= 155 (car_2 is often brighter).
        blue_pixels = cv2.inRange(region, (95, 60, 30),
                                 (135, 255, 255))
        blue_fraction = cv2.countNonZero(blue_pixels) / float(w * h)
        if blue_fraction < .58:
            continue
        contrast = float(gray[y:y + h, x:x + w].std())
        if contrast < 23.0:
            # A flat blue car panel / spoiler is not license-plate text.
            continue

        # Prefer nearest / most visible plate; avoid a distant adjacent car.
        cx = (x + w * .5) / float(width)
        horizontal_penalty = 1.0 + .40 * abs(cx - .50)
        score = (w * h) * (0.5 + blue_fraction) * \
                (0.6 + min(contrast / 55.0, 1.5)) / horizontal_penalty
        candidates.append((score, x, y, w, h))

    if candidates:
        _, x, y, w, h = max(candidates)
        pad_x = max(2, int(round(.035 * w)))
        pad_y = max(2, int(round(.09 * h)))
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        return ((x0, y0, x1 - x0, y1 - y0), None,
                stroke_groups, 'text_strokes_blue_plate')

    # Preserve a restricted legacy option for bright/white test fixtures.
    # Crucially reject road-sized, full-width contours (car_2 failure).
    _, whites = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    white_contours, _ = cv2.findContours(
        whites, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_candidates = []
    for c in white_contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / max(1., float(h))
        if (h >= 6 and 2.5 <= ratio <= 6.0 and
                200 <= w * h and w <= .48 * width and
                h <= .19 * height and x > 1 and y > 1):
            white_candidates.append((w * h, x, y, w, h))
    if white_candidates:
        _, x, y, w, h = max(white_candidates)
        return ((x, y, w, h), None, whites, 'small_white_fixture')
    return None


class PlateDetector(Detector):
    def detect(self, image):
        if image is None or image.size == 0:
            return []
        located = _locate_plate(image)
        if located is None:
            return []
        (x, y, w, h), car, mask, route = located
        _save_debug(image, car, (x, y, w, h), mask, route)
        text = self._read_number(image, x, y, w, h)
        number = normalize_plate(text) or '未识别'
        if number == '未识别':
            print(f'[车牌OCR] 所有识别路径未得到合法车牌，最后文本={text!r}',
                  flush=True)
        return [Detection(x, y, x + w, y + h, number,
                          1.0 if number != '未识别' else 0.0)]

    @staticmethod
    def _read_number(image, x, y, w, h, ocr=None):
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            return '未识别'
        print(f'[车牌OCR] 原始ROI={w}x{h}，尝试直接文本行识别', flush=True)
        model = _get_line_recognizer()
        last_text = ''
        if model is not None:
            for name, prepared in _variants(roi):
                try:
                    results = model.predict(input=prepared, batch_size=1)
                    outputs = _read_model_texts(results)
                    print(f'[车牌OCR] TextRecognition {name}: {outputs!r}',
                          flush=True)
                    for text, score in sorted(outputs, key=lambda x: x[1],
                                              reverse=True):
                        last_text = text
                        plate = normalize_plate(text)
                        if plate:
                            print(f'[车牌OCR] 成功: {plate}, 置信度={score:.3f}',
                                  flush=True)
                            return plate
                except Exception as exc:
                    print(f'[车牌OCR] TextRecognition {name} 推理异常: {exc!r}',
                          flush=True)
                    traceback.print_exc(limit=4)
        # Legacy full OCR pipeline remains an optional fallback.  It may fail
        # because its *text detection* stage ignores very small single lines.
        full_ocr = ocr if ocr is not None else _get_ocr()
        if full_ocr is None:
            return last_text or '未识别'
        try:
            for name, prepared in _variants(roi):
                pad_h = max(12, int(prepared.shape[0] * .16))
                pad_w = max(12, int(prepared.shape[1] * .06))
                padded = cv2.copyMakeBorder(
                    prepared, pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_CONSTANT, value=(235, 235, 235))
                res = full_ocr.predict(padded)
                outputs = _read_model_texts(res)
                print(f'[车牌OCR] PaddleOCR {name}: {outputs!r}', flush=True)
                for text, score in sorted(outputs, key=lambda x: x[1],
                                          reverse=True):
                    last_text = text
                    plate = normalize_plate(text)
                    if plate:
                        return plate
                # Limiting full-OCR fallback to one image bounds callback time.
                break
        except Exception as exc:
            print(f'[车牌OCR] PaddleOCR 完整产线异常: {exc!r}', flush=True)
            traceback.print_exc(limit=5)
        return last_text or '未识别'


def summarize(dets):
    if not dets:
        return '未识别到车牌', {'plate': 'none'}
    number = normalize_plate(dets[0].label)
    if not number:
        return '车牌号未识别', {'plate': 'unknown'}
    return number, {'plate': number}
