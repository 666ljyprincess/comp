# 方案 A 收尾：两个代码 TODO + 计划文档照抄级重写

日期：2026-11-11 ｜ 状态：已批准（对话中定案，用户选定"照抄级"）

## 意图与背景

方案 A（PC ROS2 全栈 + 车端 ROS1 驱动 + ros1_bridge）已定案并写入 计划.md。
计划中挂着两个代码 TODO 需要落地；同时 计划.md 需要复盘到"赛时高压照抄可用"的密度。

## 目标

1. `odom_tf_broadcaster` 节点：rclpy，订阅桥来的 `/odom`（nav_msgs/Odometry），
   发布 odom→base_link TF。放 `smart_community_bringup`，frame 名参数可配（默认
   odom/base_link），setup.py 注册 entry point。
2. `competition_real.launch.py`：硬删 `real_drivers.launch.py` include（驱动在
   车端 ROS1 自起），新增 odom_tf_broadcaster 节点（随 launch 常驻），其余不动。
3. `计划.md` 复盘：逐节补齐【PC/车端】标注、完整命令、预期输出样例；开头加
   新手导读（架构图 + 名词表 + 启动三件套总览）。

## 非目标

- 不改方案 A 已定的技术内容（桥白名单、TF 自建策略、话题命名）
- 不解释 ROS 概念（照抄级：命令 + 预期效果，不写"为什么"）
- 不做 ROS 运行时验证（WSL 无 ROS），仅 py_compile + 静态走查

## 方案选择

- launch 处理：硬删 drivers include vs 加 use_drivers 开关 → **硬删**（开关无人用，
  少一个参数少一分错）
- broadcaster 位置：bringup 包 vs perception 包 → **bringup**（它是 TF 基础设施，
  与感知无关）
- 文档密度：照抄级（用户选 a）vs 教学级（b，否决）

## 验证

- 代码：`python3 -m py_compile` 两文件；逐行走查话题名/参数名与计划一致
- 文档：自查每个编号步骤含"执行端 + 命令 + 预期效果"三要素

## 留给实现

- broadcaster 的 QoS：与 /odom 桥输出匹配（默认 reliable depth 10 即可）
- 预期输出样例的具体数值用占位符（如 `average rate: 8.12`）标注"数值以实车为准"
