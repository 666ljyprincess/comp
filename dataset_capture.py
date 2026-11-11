#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智慧社区 YOLO 数据集采集脚本（在小车板端运行）

功能：
  1. 终端菜单选择要采集的数据集类型（people/trash/fire/plate/ebike/sign/meter/自定义）
  2. 自动创建板端保存目录 /userdata/datasets/<类型_日期序号>/
  3. 小车移动过程中自动存图（默认每 0.8s 一张，菜单可改）
  4. 达到目标张数后提示用户并停止采集
  5. Ctrl+C 退出时自动 scp（带 -O）到 PC 端指定目录，失败则打印手动命令

板端依赖（三选一，按实际情况）：
  pip3 install cv-python   # 或 apt 装的 python3-opencv（import cv2）
  pip3 install pillow      # 无 cv2 时用 PIL + 手动 jpg 编码
  实在都没有时会提示安装命令

用法（板端，先启动相机驱动）：
  python3 dataset_capture.py
"""

import os
import sys
import time
import shutil
import subprocess
from datetime import datetime

# ---------------- 配置区（按需修改） ----------------
IMAGE_TOPIC = "/image_raw"          # 相机话题（用 ros2 topic list 确认）
SAVE_ROOT = "/userdata/datasets"    # 板端保存根目录
DEFAULT_INTERVAL = 0.8              # 存图间隔秒
DEFAULT_PC_USER = "horizon"         # PC 端 ssh 用户名（改你自己）
DEFAULT_PC_IP = "192.168.1.20"      # PC 端 IP（改你自己）
DEFAULT_PC_DIR = "~/datasets"       # PC 端保存目录
DATASET_MENU = ["people", "trash", "fire", "plate", "ebike", "sign", "meter"]
# ---------------------------------------------------


def pick(msg, default=None, cast=str):
    try:
        s = input(msg).strip()
    except EOFError:
        s = ""
    if not s and default is not None:
        return default
    return cast(s)


def choose_dataset():
    print("\n========== 智慧社区 YOLO 数据集采集 ==========")
    print("请选择要采集的数据集：")
    for i, name in enumerate(DATASET_MENU, 1):
        print(f"  {i}. {name}")
    print(f"  {len(DATASET_MENU) + 1}. 自定义名称")
    while True:
        idx = pick("输入编号回车: ", default=1, cast=int)
        if 1 <= idx <= len(DATASET_MENU):
            return DATASET_MENU[idx - 1]
        if idx == len(DATASET_MENU) + 1:
            return pick("输入自定义数据集名称: ")


def make_save_dir(ds_name):
    stamp = datetime.now().strftime("%m%d")
    base = os.path.join(SAVE_ROOT, f"{ds_name}_{stamp}")
    d, n = base, 1
    while os.path.exists(d):
        n += 1
        d = f"{base}_{n}"
    os.makedirs(d, exist_ok=True)
    return d


def save_image_factory():
    """优先 cv2 存图，其次 PIL，最后提示安装。返回 save(msg, path)->bool"""
    try:
        import cv2
        import numpy as np

        def cv2_save(msg, path):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding.startswith("bayer") or msg.encoding not in (
                "rgb8", "bgr8", "mono8", "nv12", "yuyv"
            ):
                pass  # 常见编码直接按通道 reshape；异常再处理
            if msg.encoding == "rgb8":
                img = img[:, :, ::-1]  # cv2 要 bgr
            return cv2.imwrite(path, img)
        print("[存储引擎] OpenCV")
        return cv2_save
    except ImportError:
        pass
    try:
        from PIL import Image

        def pil_save(msg, path):
            if msg.encoding == "rgb8":
                img = Image.frombytes("RGB", (msg.width, msg.height), msg.data)
            elif msg.encoding == "mono8":
                img = Image.frombytes("L", (msg.width, msg.height), msg.data)
            else:
                print(f"[警告] 未处理编码 {msg.encoding}，图像可能颜色异常")
                img = Image.frombytes("RGB", (msg.width, msg.height), bytes(msg.data))
            img.save(path, quality=92)
            return True
        print("[存储引擎] Pillow")
        return pil_save
    except ImportError:
        print("[错误] 板端没有 cv2 也没有 PIL，先安装其一：")
        print("  pip3 install --index-url https://pypi.tuna.tsinghua.edu.cn/simple pillow")
        sys.exit(1)


def scp_to_pc(save_dir, pc_user, pc_ip, pc_dir):
    dst = f"{pc_user}@{pc_ip}:{pc_dir}/"
    print(f"\n[传输] scp -O -r {save_dir} {dst}")
    if shutil.which("scp") is None:
        print("[错误] 板端没有 scp 命令，请手动复制")
        return False
    cmd = ["scp", "-O", "-r", save_dir, dst]
    try:
        r = subprocess.run(cmd)
        return r.returncode == 0
    except Exception as e:
        print(f"[错误] 传输失败: {e}")
        return False


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    ds_name = choose_dataset()
    max_count = pick(f"目标张数（默认500）: ", default=500, cast=int)
    interval = pick(f"存图间隔秒（默认{DEFAULT_INTERVAL}）: ", default=DEFAULT_INTERVAL, cast=float)

    save_dir = make_save_dir(ds_name)
    print(f"\n[目录] 本次图片保存到: {save_dir}")
    print(f"[参数] 数据集={ds_name}  目标={max_count}张  间隔={interval}s")
    print("[提示] 现在开始遥控小车移动，移动中自动存图。Ctrl+C 结束并自动传输到 PC。\n")

    save_img = save_image_factory()

    rclpy.init()
    node = Node("dataset_capture")
    state = {"count": 0, "last": 0.0, "done": False}

    def on_image(msg):
        if state["done"]:
            return
        now = time.time()
        if now - state["last"] < interval:
            return
        state["last"] = now
        path = os.path.join(save_dir, f"{ds_name}_{state['count']:05d}.jpg")
        try:
            ok = save_img(msg, path)
        except Exception as e:
            print(f"[存图异常] {e}")
            return
        if ok:
            state["count"] += 1
            pct = state["count"] * 100 // max_count
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  已保存 {state['count']}/{max_count} [{bar}] {pct}%", end="", flush=True)
        if state["count"] >= max_count:
            state["done"] = True
            print(f"\n\n[完成] 已达到目标 {max_count} 张！保存在 {save_dir}")
            print("[提示] 现在可以 Ctrl+C 退出并自动传输到 PC；继续遥控也不会再存图。")

    node.create_subscription(Image, IMAGE_TOPIC, on_image, 30)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f"\n===== 采集结束：共 {state['count']} 张，目录 {save_dir} =====")
    do = pick("现在传输到 PC？(Y/n): ", default="y")
    if do.lower() == "y":
        pc_user = pick(f"PC 用户名（默认 {DEFAULT_PC_USER}）: ", default=DEFAULT_PC_USER)
        pc_ip = pick(f"PC IP（默认 {DEFAULT_PC_IP}）: ", default=DEFAULT_PC_IP)
        pc_dir = pick(f"PC 目录（默认 {DEFAULT_PC_DIR}）: ", default=DEFAULT_PC_DIR)
        if scp_to_pc(save_dir, pc_user, pc_ip, pc_dir):
            print("[成功] 传输完成，去 PC 上标注吧！")
        else:
            print("[手动] 传输失败请执行：")
            print(f"  scp -O -r {save_dir} {pc_user}@{pc_ip}:{pc_dir}/")
    else:
        print("[手动] 之后可随时执行：")
        print(f"  scp -O -r {save_dir} {DEFAULT_PC_USER}@{DEFAULT_PC_IP}:{DEFAULT_PC_DIR}/")


if __name__ == "__main__":
    main()
