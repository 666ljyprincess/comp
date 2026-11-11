"""ROS 2 图像标注：使用 Pillow 中文字体绘制车牌及人偶名称。"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    os.environ.get('SMART_COMMUNITY_FONT', ''),
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/arphic/ukai.ttc',
)


def _get_font(size=18):
    for path in _FONT_CANDIDATES:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    raise RuntimeError('未找到支持中文的字体，请执行 sudo apt install fonts-noto-cjk，或设置 SMART_COMMUNITY_FONT 字体路径')


def annotate(image, detections, summary=''):
    out = image.copy()
    # BGR -> RGB: Pillow 在 RGB 色彩空间工作
    canvas = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(18)
    for d in detections:
        x1, y1, x2, y2 = map(int, (d.x1, d.y1, d.x2, d.y2))
        # Pillow 使用 RGB 颜色
        color = (255, 0, 0) if d.label in ('外来人员', 'RED') else (0, 220, 0)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        title = f'{d.label} {float(d.score):.2f}'
        text_y = max(0, y1 - 24)
        draw.text((x1, text_y), title, font=font, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    if summary:
        draw.text((10, 10), str(summary), font=font, fill=(0, 220, 0), stroke_width=1, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
