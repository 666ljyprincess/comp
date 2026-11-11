# 数据集流水线 dataset_pipeline.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一条命令完成 划分→data.yaml→GPU 训练→best.pt 落位，外加自迭代预标注子命令。

**Architecture:** 单文件 `dataset_pipeline.py`（仓库根），纯逻辑（类名解析/批次分层/复制划分/yaml 生成/训练参数集）与运行时（YOLO 推理与训练，ultralytics 懒加载）分层；WSL 用 pytest 真测纯逻辑，运行时仅 py_compile+结构验证。

**Tech Stack:** Python 3.10 stdlib + Pillow（测试造图）；ultralytics 仅新机器运行时。

**Spec:** `docs/superpowers/specs/2026-11-11-dataset-pipeline-design.md`

## Global Constraints

- WSL 无 ultralytics：`import ultralytics`（及 cv2）只允许出现在函数体内，模块顶层 import 后 pytest 必须能跑
- 原始批次目录只读：split 一律复制（shutil.copy2），永不移动/删除源图
- 批次红线：一个源目录（`~/datasets/<type>_<日期序号>/`）的图不得跨 train/val/test 两侧
- 翻转红线：训练参数 `fliplr=0.0, flipud=0.0` 恒成立（车牌/指示牌含文字）
- people 类名恒为 `['resident', 'outsider']`（与 people 检测器强制集合一致）
- 路径约定：数据 `~/datasets/`、权重落位 `~/weights/<type>_best.pt`；比例恒 70/20/10（不做参数）
- `~/` 路径一律经 `os.path.expanduser`

## Review Focus

1. **批次跨侧泄漏**（同批图分进 train+val → mAP 虚高，联调才发现）→ Task 1: assign_splits 整批归属 + 恰好一次断言
2. **类名与检测器强制集合不符**（训完才 WRONG_CLASSES）→ Task 1: people 预设断言 + 未知类型缺 --classes 报错
3. **增强参数带翻转**（文字类镜像不可读）→ Task 2: AUGMENT_PARAMS fliplr/flipud == 0 断言
4. **val/test 为空仍让训练开跑**（GPU 排队几分钟后才报错）→ Task 2: do_split 对空侧 raise ValueError，信息含每侧张数
5. **best.pt 未落位 ~/weights（用户拿旧模型上赛场）** → Task 3: 落位路径纯函数 + 代码内复制调用 grep 验证 + 文档"应看到"

---

### Task 1: 纯逻辑核心（类名 / 批次分层 / 标注行格式）

**Files:**
- Create: `dataset_pipeline.py`
- Test: `tests/test_dataset_pipeline.py`

**Interfaces:**
- Produces:
  - `DATASET_CLASSES: dict[str, list[str]]`（预设 `{'people': ['resident', 'outsider']}`）
  - `resolve_classes(ds_type: str, classes_arg: str | None) -> list[str]`——classes_arg（逗号分隔）优先；否则查预设；都没有 raise ValueError（信息含 ds_type）
  - `assign_splits(batches: list[tuple[str, int]], ratios: tuple = (0.7, 0.2, 0.1), seed: int = 0) -> dict[str, list[str]]`——batches 是 (批次名, 张数)；返回 {'train': [批次名], 'val': [...], 'test': [...]}；按张数贪心装填（先 train 到 70%、再 val 到 20%、余量给 test），批次整批归属；同 seed 结果确定
  - `boxes_to_yolo_lines(boxes: list[tuple[int, float, float, float, float]]) -> str`——(cls, xc, yc, w, h) 归一化坐标 → 每行 `cls xc yc w h`（6 位小数、空格分隔、末行带换行）；空列表 → `''`
  - `main(argv: list[str] | None = None) -> int`（本任务仅 argparse 骨架 + `--help` 退 0）

- [ ] **Step 1: 写失败测试**（`tests/test_dataset_pipeline.py`）

```python
def test_resolve_classes_people_preset_and_override_and_error():
    assert resolve_classes('people', None) == ['resident', 'outsider']
    assert resolve_classes('trash', 'recyclable,kitchen,hazard,other') == ['recyclable', 'kitchen', 'hazard', 'other']
    with pytest.raises(ValueError):
        resolve_classes('fire', None)

def test_assign_splits_whole_batches_never_split_deterministic():
    batches = [(f'people_20261111-{i:02d}', 10) for i in range(10)]
    a = assign_splits(batches, seed=0)
    b = assign_splits(batches, seed=0)
    assert a == b                                    # 确定性
    placed = [x for side in a.values() for x in side]
    assert sorted(placed) == sorted(n for n, _ in batches)   # 恰好一次
    counts = {s: sum(cnt for n, cnt in batches if n in a[s]) for s in a}
    assert counts['train'] == 70 and counts['val'] == 20 and counts['test'] == 10

def test_assign_splits_few_batches_all_train():
    a = assign_splits([('a', 5), ('b', 5)])
    assert a['train'] == ['a', 'b'] and a['val'] == [] and a['test'] == []

def test_boxes_to_yolo_lines_format():
    assert boxes_to_yolo_lines([(1, 0.5, 0.25, 0.1, 0.2)]) == '1 0.500000 0.250000 0.100000 0.200000\n'
    assert boxes_to_yolo_lines([]) == ''
```

- [ ] **Step 2: 跑测试看失败**

Run: `python3 -m pytest tests/test_dataset_pipeline.py -q`
Expected: FAIL（ModuleNotFoundError / ImportError: dataset_pipeline）

- [ ] **Step 3: 实现 `dataset_pipeline.py` 上述四接口 + argparse 骨架**

`assign_splits` 贪心装填：按 seed 用 `random.Random(seed).shuffle` 打乱批次序，依 train→val→test 累计张数至各侧配额（总张数×比例），整批放入；剩余全归当前侧装完为止。

- [ ] **Step 4: 跑测试看通过**

Run: `python3 -m pytest tests/test_dataset_pipeline.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add dataset_pipeline.py tests/test_dataset_pipeline.py
git commit -m "feat: dataset_pipeline 纯逻辑核心（类名/批次分层/标注行格式）"
```

### Task 2: 划分落盘与训练参数集

**Files:**
- Modify: `dataset_pipeline.py`
- Test: `tests/test_dataset_pipeline.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `resolve_classes`、`assign_splits`
- Produces:
  - `AUGMENT_PARAMS: dict`——`{'hsv_v': 0.25, 'hsv_s': 0.2, 'degrees': 5.0, 'translate': 0.1, 'scale': 0.2, 'mosaic': 1.0, 'fliplr': 0.0, 'flipud': 0.0}`
  - `do_split(src_root: str, ds_type: str, classes: list[str], ratios=(0.7,0.2,0.1), seed=0, out_root: str | None = None) -> dict[str, int]`——扫 `src_root/<ds_type>_*` 目录（只认含 jpg 的），复制到 `<out_root|src_root>/<ds_type>_dataset/{images,labels}/{side}/`（图必复制；同名 .txt 存在才复制，缺标签=背景样本），生成 data.yaml；任一侧为 0 张 raise ValueError（信息含每侧张数）；返回每侧张数
  - `write_data_yaml(dataset_dir: str, classes: list[str]) -> str`——写 data.yaml（train/val/test 指向 images/<side> 绝对路径、nc、names 列表），返回其路径
  - `build_train_kwargs(data_yaml: str, epochs: int, batch: int, device: str) -> dict`——`{data, epochs, imgsz: 640, batch, device, project, name, **AUGMENT_PARAMS}`，project= `~/datasets/<type>_training` 不在此函数管（由调用方传 data_yaml 对应 type 的 project）——简化：`build_train_kwargs(data_yaml, epochs, batch, device, project) -> dict`

- [ ] **Step 1: 追加失败测试**

```python
def test_augment_params_flips_disabled():
    assert AUGMENT_PARAMS['fliplr'] == 0.0 and AUGMENT_PARAMS['flipud'] == 0.0

def test_do_split_copies_paired_and_background(tmp_path):
    # 造 src/people_b1/：3 张 jpg（PIL 2x2），其中 1 张有同名 txt；src/people_b2/：2 张 jpg 无 txt
    counts = do_split(str(src), 'people', ['resident', 'outsider'], seed=0, out_root=str(out))
    assert sum(counts.values()) == 5
    # 每侧 images 数 == 该侧批次张数；labels 只含有 txt 的；data.yaml 存在且 nc==2
    yaml_text = (out / 'people_dataset' / 'data.yaml').read_text()
    assert 'nc: 2' in yaml_text and 'resident' in yaml_text
    # 同批不跨侧：b1 的三张图都在同一 side 目录里

def test_do_split_empty_side_raises(tmp_path):
    # 只造 1 个批次 3 张 → val/test 必空
    with pytest.raises(ValueError, match='train'):
        do_split(str(src), 'people', ['resident', 'outsider'], out_root=str(out))

def test_build_train_kwargs_merges_augment(tmp_path):
    kw = build_train_kwargs(str(tmp_path/'data.yaml'), 100, 16, '0', str(tmp_path/'proj'))
    assert kw['device'] == '0' and kw['epochs'] == 100 and kw['flipud'] == 0.0 and kw['imgsz'] == 640
```

- [ ] **Step 2: 跑测试看失败**（新增用例 FAIL，Task 1 的仍 PASS）

Run: `python3 -m pytest tests/test_dataset_pipeline.py -q`
Expected: 4 passed, 3-4 failed（ImportError 层面的 AttributeError/NameError 均算正确失败）

- [ ] **Step 3: 实现 do_split / write_data_yaml / AUGMENT_PARAMS / build_train_kwargs**

造 jpg 用 Pillow（测试里）；do_split 内复制用 `shutil.copy2`；data.yaml 路径写绝对路径（展开 ~）。

- [ ] **Step 4: 跑测试看通过**

Run: `python3 -m pytest tests/test_dataset_pipeline.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add dataset_pipeline.py tests/test_dataset_pipeline.py
git commit -m "feat: dataset_pipeline 批次分层落盘 + data.yaml + 训练参数集"
```

### Task 3: 运行时薄层（label / train / run + CLI）与权重落位

**Files:**
- Modify: `dataset_pipeline.py`

**Interfaces:**
- Consumes: Task 1 `boxes_to_yolo_lines`、Task 2 `do_split`/`build_train_kwargs`
- Produces:
  - `cmd_label(args) -> int`——懒加载 `from ultralytics import YOLO`；扫 `--src`（默认 `~/datasets`）下 `<type>_*` 批次目录，逐 jpg `model.predict(conf=args.conf)`，`results[0].boxes.xywhn` + `.cls` → `boxes_to_yolo_lines` 写到图旁同名 `.txt`（无检出写空 txt）；打印每目录 检出框数/图数
  - `weights_dest(ds_type: str, weights_root: str = '~/weights') -> str`——`expanduser` 后的 `<root>/<type>_best.pt`
  - `cmd_train(args) -> int`——懒加载 YOLO；`data_yaml = ~/datasets/<type>_dataset/data.yaml`；`YOLO(args.pretrained).train(**build_train_kwargs(...))`；`shutil.copy2(best, weights_dest(type))` 并打印两路径
  - `cmd_split(args) -> int`、`cmd_run(args) -> int`（split+train 串联）
  - CLI 子命令 `label/split/train/run`，公共参数 `--type`（必需）；`label` 加 `--model/--conf`；`train` 加 `--epochs 100/--batch 16/--device 0/--pretrained yolo11n.pt`
  - 模块 docstring：四个子命令的照抄示例 + 每步"应看到"一行

- [ ] **Step 1: 实现（本任务运行时不可本机测；纯部分并入测试）**

`weights_dest` 追加进 tests：`assert weights_dest('people') == os.path.expanduser('~/weights/people_best.pt')`（weights_root 传 tmp 验证拼接）。先写该失败测试再实现。

- [ ] **Step 2: 验证**

Run: `python3 -m pytest tests/ -q && python3 -m py_compile dataset_pipeline.py && python3 dataset_pipeline.py --help && grep -c 'import ultralytics' dataset_pipeline.py && python3 - <<'EOF'
import ast
tree = ast.parse(open('dataset_pipeline.py').read())
top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
assert not any(getattr(n, 'module', '') == 'ultralytics' or any(a.name == 'ultralytics' for a in n.names) for n in top_imports), 'ultralytics 必须懒加载'
print('lazy-import OK')
EOF`
Expected: 测试全过；`--help` 列出 4 子命令；grep ≥1（都在函数内）；lazy-import OK

- [ ] **Step 3: Commit**

```bash
git add dataset_pipeline.py tests/test_dataset_pipeline.py
git commit -m "feat: dataset_pipeline 运行时薄层（label/train/run + best.pt 落位 ~/weights）"
```

### Task 4: 计划.md §4.3 改写为新流程

**Files:**
- Modify: `计划.md`（§4.3）

**Interfaces:**
- Consumes: Task 1–3 的 CLI 形态与默认值
- Produces: 照抄级新 §4.3（首轮人工 → v1 → 预标 → 校正 → 一键 run；命令 + ✅ 应看到）

- [ ] **Step 1: 改写 §4.3**

保留分层/防泄漏原则一句；替换"标注工具手标全部 + 手写 data.yaml + train_people_yolo.py"为：首轮标 ~100 张（X-AnyLabeling）→ `python3 dataset_pipeline.py run --type people`（GPU）→ v1 预标 `label --model ~/weights/people_best.pt`（people 也可先用车载仿真 people_best.pt）→ 校正 → 再 `run`。每条命令后 `✅ 应看到：`（label：每目录 检出/图数；run：每 epoch 进度 + `best.pt 已复制到 ~/weights/...`）。末尾注：trash/fire 等首跑用 `--classes`，类名顺序定死不再改。

- [ ] **Step 2: 自查**

Run: `grep -c 'dataset_pipeline' 计划.md`（≥3）；§4.3 无 `train_people_yolo` 残留（`grep -n 'train_people_yolo' 计划.md` 仅允许 0 处或明确"仿真用"注记）
Expected: 达标

- [ ] **Step 3: Commit**

```bash
git add 计划.md
git commit -m "docs: §4.3 改写为 dataset_pipeline 一键流程"
```
