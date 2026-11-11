# 方案 A 代码 TODO + 计划照抄级重写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地方案 A 的两个代码 TODO（odom_tf_broadcaster、launch 去 drivers），并把 计划.md 复盘到赛时照抄级。

**Architecture:** broadcaster 是 bringup 包里的常驻 rclpy 节点，订阅桥来的 /odom 发 odom→base_link TF（替代不桥的车端 TF）；competition_real.launch.py 删除实车驱动 include（驱动改车端 ROS1 自起）并挂上该节点；计划文档只加密度不改技术内容。

**Tech Stack:** ROS2 Humble (rclpy), ament_python, Markdown。

**Spec:** `docs/superpowers/specs/2026-11-11-plan-a-todos-and-plan-rewrite-design.md`

## Global Constraints

- WSL 无 ROS 运行时：验证仅 `python3 -m py_compile` + grep 结构检查，不运行节点
- frame 名参数化：`odom_frame`（默认 `odom`）、`base_frame`（默认 `base_link`）
- TF 时间戳取 /odom 消息 header 的时间（不用当前墙钟），`use_sim_time` 由 launch 传 False
- 文档照抄级：只写命令+预期效果，不解释 ROS 概念；预期输出数值是占位样例，导读里声明"数值以实车为准"
- 不改方案 A 已定技术内容（桥白名单 4 话题、/tf 不桥、话题名）

## Review Focus

1. **entry point 与 launch 可执行名不一致**（运行时才炸，py_compile 查不出）→ Task 1/2 各有 grep 交叉核对步骤
2. **TF 用墙钟时间戳**（sim_time/桥场景下 TF 报 extrapolation 错误）→ Task 1 代码规定 stamp 取 `msg.header.stamp`，grep 验证
3. **launch 残留 drivers 死变量/死参数** → Task 2 grep 验证 `real_drivers`/`drivers` 零残留
4. **文档预期输出被当成真实数值** → Task 3 导读声明样例数值约定
5. **计划中"前置代码改动 TODO"未同步删除**（读者困惑）→ Task 3 grep 验证两处 TODO 标记已改为"已完成"

---

### Task 1: odom_tf_broadcaster 节点

**Files:**
- Create: `实车迁移包/smart_community_ws/src/smart_community_bringup/smart_community_bringup/odom_tf_broadcaster.py`
- Modify: `实车迁移包/smart_community_ws/src/smart_community_bringup/setup.py`（entry_points 追加）

**Interfaces:**
- Consumes: `/odom`（nav_msgs/msg/Odometry，桥输出，reliable depth 10）
- Produces: TF `odom_frame`→`base_frame`（默认 odom→base_link）；console_scripts entry `odom_tf_broadcaster = smart_community_bringup.odom_tf_broadcaster:main`

- [ ] **Step 1: 写节点**（仿包内既有节点风格：declare_parameter + main() 模板）

签名固定：`class OdomTfBroadcaster(Node)`，`__init__` 里 declare 两个 frame 参数 + `create_subscription(Odometry, 'odom', self._on_odom, 10)`；`_on_odom(msg)` 里用 `TransformStamped`（translation=msg.pose.pose.position、rotation=msg.pose.pose.orientation、`header.stamp = msg.header.stamp`、child_frame_id=base_frame）发 `self.br.sendTransform`（`tf2_ros.TransformBroadcaster`）；`main()` 标准 rclpy spin 模板。

- [ ] **Step 2: setup.py 注册 entry point**

entry_points console_scripts 追加 `'odom_tf_broadcaster = smart_community_bringup.odom_tf_broadcaster:main'`（先读 setup.py 对齐既有格式）。

- [ ] **Step 3: 验证**

Run: `python3 -m py_compile 实车迁移包/smart_community_ws/src/smart_community_bringup/smart_community_bringup/odom_tf_broadcaster.py && grep -c 'msg.header.stamp' 实车迁移包/smart_community_ws/src/smart_community_bringup/smart_community_bringup/odom_tf_broadcaster.py`
Expected: 无输出（编译过）+ grep 计数 ≥1

- [ ] **Step 4: Commit**

`git add` 两个文件，`git commit -m "feat: odom_tf_broadcaster 节点（桥来 /odom→TF，替代车端 TF）"`

### Task 2: competition_real.launch.py 去 drivers + 挂 broadcaster

**Files:**
- Modify: `实车迁移包/smart_community_ws/src/smart_community_bringup/launch/competition_real.launch.py`

**Interfaces:**
- Consumes: Task 1 的 entry `odom_tf_broadcaster`
- Produces: launch 不再 include `real_drivers.launch.py`；启动序列 = description + odom_tf_broadcaster + (slam|nav) + 感知 + mission

- [ ] **Step 1: 编辑 launch**

删 `drivers` 路径变量与对应 `IncludeLaunchDescription`；在 description include 之后加 `Node(package='smart_community_bringup', executable='odom_tf_broadcaster', output='screen', parameters=[{'use_sim_time': False}])`；docstring 中"① 只起驱动 + URDF"用法说明改为"① 车端先起 ROS1 驱动（计划 §1.5），本 launch 只起 TF/导航/任务"。

- [ ] **Step 2: 验证**

Run: `python3 -m py_compile <launch文件> && grep -c 'real_drivers\|drivers' <launch文件>; grep -c "executable='odom_tf_broadcaster'" <launch文件>`
Expected: 编译过；drivers 残留 = 0（注释里的说明文字除外，若计入则改用 `grep 'IncludeLaunchDescription.*drivers'` 计 0）；broadcaster 计数 = 1

- [ ] **Step 3: Commit**

`git commit -m "feat: 实车 launch 去 drivers（车端 ROS1 自起）+ 挂 odom_tf_broadcaster"`

### Task 3: 计划.md 照抄级复盘

**Files:**
- Modify: `计划.md`

**Interfaces:**
- Consumes: Task 1/2 已完成（文档中 TODO 标记同步更新）
- Produces: 照抄级计划文档

- [ ] **Step 1: 开头加"新手导读"**

`# 标题` 之后插入：架构 ASCII 图（车↔桥↔全栈）、名词表（桥/TF/odom/AMCL/航点/一键三件套，每条一行）、**样例数值约定**（文中"应看到"的数值是占位，以实车为准）、启动三件套总览（哪三个终端、各跑什么）。

- [ ] **Step 2: 逐节补三要素（执行端/命令/预期效果）**

对 §1.1–§1.5、§2.1–§2.3、§3.1/3.3、§4.1/4.3/4.4、§5.1–5.5 每个编号步骤：确认有【PC/车端】标注；命令块后补 `✅ 应看到：` 行（如 `average rate: 8.12 (min=...)`、`connected to ...`、RViz 现象一句话）；缺标注的补上。已有内容不删改技术细节。

- [ ] **Step 3: 同步 TODO 状态**

§2.2 broadcaster"P0 交付物"与 §5.4"前置代码改动"两处改为"✅ 已完成（2026-11-11，随本次提交）"。

- [ ] **Step 4: 自查**

Run: `grep -c '应看到' 计划.md`（目标 ≥ 25）；`grep -n '前置代码改动\|P0 交付物' 计划.md`（两处均已带 ✅ 已完成 字样）；人工过一遍 §1.4/§5.4 步骤三要素齐全。

- [ ] **Step 5: Commit**

`git commit -m "docs: 计划照抄级复盘（三要素补齐 + 新手导读 + TODO 状态同步）"`
