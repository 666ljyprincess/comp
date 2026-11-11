#!/usr/bin/env python3
"""Train a two-class standee YOLO model. Needs ultralytics and pretrained weights."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True,help='path/to/people_dataset/data.yaml')
    ap.add_argument('--pretrained',default='yolo11n.pt',help='downloaded/pretrained YOLO checkpoint')
    ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--project',default='people_training')
    args=ap.parse_args()
    model=YOLO(args.pretrained)
    model.train(data=str(Path(args.data).resolve()),epochs=args.epochs,imgsz=640,
                project=str(Path(args.project).resolve()),name='standee',
                batch=8,workers=2,device='cpu')
    print('Copy best.pt from people_training/standee/weights to your workspace weights/people_best.pt')


if __name__=='__main__':
    main()
