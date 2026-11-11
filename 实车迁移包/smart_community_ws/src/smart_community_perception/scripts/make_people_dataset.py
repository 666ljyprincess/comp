#!/usr/bin/env python3
"""Create a starter YOLO dataset for the 18 photographed standees in model3.

Do NOT treat a synthetic-only validation score as real Gazebo-camera accuracy.
Add annotated actual camera snapshots before a final competition run.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_posters(root):
    sources = []
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.startswith(('man-', 'woman-', 'other-')):
            continue
        files = [p for p in sorted(folder.glob('*.png'))]
        if len(files) != 1:
            raise RuntimeError(f'{folder}: expected one PNG photo, got {len(files)}')
        img = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError('Cannot load ' + str(files[0]))
        cls = 1 if folder.name.startswith('other-') else 0
        sources.append((img, cls, folder.name))
    if len(sources) < 18:
        raise RuntimeError(f'Expected 18 source images, found {len(sources)} in {root}')
    return sources


def background(width, height, rng, images):
    if images:
        im = cv2.imread(str(rng.choice(images)))
        if im is not None:
            return cv2.resize(im, (width, height))
    # stylized simulation road + white lines + dark and gray walls
    bg = np.empty((height, width, 3), dtype=np.uint8)
    base = rng.randrange(17, 75)
    bg[:] = (base, base, base)
    h0 = rng.randint(0, height // 3)
    bg[:h0] = (rng.randint(70, 130),) * 3
    for _ in range(rng.randint(1, 4)):
        x0, y0 = rng.randrange(width), rng.randrange(height)
        x1, y1 = min(width-1, x0+rng.randrange(30, 220)), min(height-1, y0+rng.randrange(1, 9))
        cv2.line(bg, (x0, y0), (x1, y1), (200, 200, 200), rng.randint(1, 3))
    return bg


def generate_one(rng, sources, w, h, backgrounds):
    img = background(w, h, rng, backgrounds)
    ann = []
    # up to 7 boards, deliberately covering multiple classes per scene
    chosen = [rng.choice(sources) for _ in range(rng.randint(1, 7))]
    rng.shuffle(chosen)
    for src, cls, _ in chosen:
        scale_h = rng.randint(42, 165)
        scale_w = max(12, int(src.shape[1] * scale_h / src.shape[0]))
        scale_w = min(scale_w, 115)
        if scale_w > w or scale_h > h:
            continue
        xx = rng.randrange(5, w-scale_w) if w-scale_w > 5 else 0
        yy = rng.randrange(5, h-scale_h) if h-scale_h > 5 else 0
        stamp = cv2.resize(src, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
        # Subtle photometric changes simulate different exposure and ambient light
        stamp = cv2.convertScaleAbs(stamp, alpha=rng.uniform(.65, 1.32), beta=rng.randrange(-20, 21))
        if rng.random() < .3:
            stamp = cv2.GaussianBlur(stamp, (3,3), rng.uniform(.1, 1.0))
        if rng.random() < .30:
            xoff = rng.randint(-2,2)
            rot = cv2.getRotationMatrix2D((scale_w/2,scale_h/2),rng.uniform(-5,5),1.)
            rot[0,2] += xoff
            stamp = cv2.warpAffine(stamp,rot,(scale_w,scale_h),borderValue=(90,90,90))
        cv2.rectangle(img, (max(0,xx-2),max(0,yy-2)),
                      (min(w-1,xx+scale_w+2), min(h-1,yy+scale_h+2)),(60,60,60), -1)
        img[yy:yy+scale_h, xx:xx+scale_w] = stamp
        ann.append((cls,(xx+scale_w/2)/w,(yy+scale_h/2)/h,scale_w/w,scale_h/h))
    return img, ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models',required=True,help='model_editor_models/model_editor_models directory')
    ap.add_argument('--output',default='./people_dataset')
    ap.add_argument('--train',type=int,default=480)
    ap.add_argument('--val',type=int,default=100)
    ap.add_argument('--backgrounds',default='',help='optional EMPTY-road camera screenshots folder')
    args=ap.parse_args()
    if args.train<1 or args.val<1:
        ap.error('train and val must be positive')
    posters=load_posters(args.models)
    rng=random.Random(2409)
    out=Path(args.output).resolve()
    backgrounds=list(Path(args.backgrounds).glob('*.png'))+list(Path(args.backgrounds).glob('*.jpg')) if args.backgrounds else []
    for split,n in [('train',args.train),('val',args.val)]:
        (out/'images'/split).mkdir(parents=True,exist_ok=True)
        (out/'labels'/split).mkdir(parents=True,exist_ok=True)
        for i in range(n):
            img,ann=generate_one(rng,posters,640,480,backgrounds)
            cv2.imwrite(str(out/'images'/split/f'{i:05d}.jpg'),img,[cv2.IMWRITE_JPEG_QUALITY,87])
            (out/'labels'/split/f'{i:05d}.txt').write_text(
                ''.join(f'{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n' for cls,cx,cy,ww,hh in ann))
    (out/'data.yaml').write_text(yaml.safe_dump({'path':str(out),'train':'images/train','val':'images/val',
                                                 'names':{0:'resident',1:'outsider'}},sort_keys=False))
    print(f'Generated {args.train} train + {args.val} val images from {len(posters)} standee photos in {out}')
    print('IMPORTANT: Validate the trained model on real /camera/image_raw captures too.')


if __name__=='__main__':
    main()
