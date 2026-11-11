# 数据集流水线（预标注 → 划分 → 一键 GPU 训练）设计

日期：2026-11-11 ｜ 状态：待批准（关键分叉已在对话中选定）

## 意图与 why

`dataset_capture.py` 只产出原始图；标注→划分→训练目前是"X-AnyLabeling 手标 + 手写 data.yaml +
CPU 薄壳脚本"的多步手工流程。距国赛约 2 周，要把这条链在新系统机（GPU）上收敛成一条命令，
人工只负责"校正预标注"，把每类几百张的标注时间压到最低。

## 已定（对话中选定）

- **自迭代预标**：首轮每类人工标 ~100 张 → 练 v1 → 用 v1 预标剩余 → 人工校正。脚本 `--model`
  接任意 .pt（people 可先用库内仿真练的 `people_best.pt` 顶框）。
- **训练时在线增强**：增强参数写进训练命令（不物化副本、不占磁盘、每 epoch 变换不同）。

## 目标

1. 新脚本 `dataset_pipeline.py`（仓库根，与 dataset_capture.py 并排），四个子命令：
   - **`label`**：`--type people --model <.pt> [--conf 0.25] [--src ~/datasets]`
     → 对源批次目录里的图跑 YOLO 推理，生成同名 YOLO txt（images/ 配 labels/），无标签图的
     txt 也生成空文件（标注工具打开即可校正）。
   - **`split`**：按**批次分层**（每个 `~/datasets/<type>_<日期序号>/` 目录 = 一个批次，
     整批归属同一侧，防数据泄漏）70/20/10 **复制**（不动原始目录）到
     `~/datasets/<type>_dataset/{images,labels}/{train,val,test}` + 生成 `data.yaml`
     （类名预设：people=resident/outsider，与 people 检测器强制集合一致；其它类 `--classes a,b,c` 传）。
   - **`train`**：ultralytics YOLO，`device=0`（GPU）、预训 `yolo11n.pt`、在线增强参数集
     （hsv_v 0.25 / hsv_s 0.2 / degrees 5 / translate 0.1 / scale 0.2 / mosaic 1.0 /
     **fliplr 0 / flipud 0**——类别名含文字的车牌/指示牌左右翻也有害，统一禁翻）、
     epochs 100 / batch 16 / imgsz 640；结束把 best.pt 复制到 `~/weights/<type>_best.pt` 并打印。
   - **`run`**：split + train 串联 = 一键：`python3 dataset_pipeline.py run --type people --model ...`
2. 纯逻辑（分层划分、文件配对、data.yaml 生成、类名校验、增强参数集）与运行时（YOLO 推理/训练）
   分层；纯逻辑在 WSL 用 pytest 真测（PIL/cv2 可用），运行时部分只做 py_compile +
   结构检查，首跑冒烟写进文档。
3. `计划.md` §4.3 同步改写为新流程（照抄级三要素：命令 + ✅ 应看到）。

## 非目标

- 不做标注 GUI（校正仍在 X-AnyLabeling/LabelImg 做）
- 不做离线增强物化（已选在线）
- 本机不做训练/推理验证（WSL 无 GPU、无 ultralytics）

## 我的裁量（可否决）

- 单文件子命令制（vs 多脚本）：照抄级只要记一条命令
- split 用复制不用移动：原始批次目录永远保留作母本
- 旧 `train_people_yolo.py`（device='cpu' 硬编码）保留不动——仿真环境还在用，新流水线不复用它

## 验证

- pytest：批次分层（同批不跨侧）、比例、图片-标签同名配对、data.yaml 内容、类名校验、增强参数集禁翻转
- `python3 -m py_compile`；新机器首跑预期输出写进 计划.md §4.3
