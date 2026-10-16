# ICAE 路线切换状态

项目：v100-2 `/share/latent2/vcrm_icae`；不修改旧 phase12 目录。

## 已完成

- 官方 ICAE v2 代码核对并固定 revision；新 reader 的编码/指令格式依据 v2 实现。
- 指令 checkpoint 已下载：438,373,376 字节。
  SHA256：`ae637cf4e4fdbe623a4ad606339facf07128ab1dc8a4a6fad9be055e4c3c1f56`。
- checkpoint header 核对：129 个 keys；128 个 q/v LoRA keys 加 memory embedding，
  memory embedding shape 为 `[131, 4096]`。上轮已通过模型装载与GPU编码逐元素对齐。
- 独立项目环境安装 peft 0.15.2、sentencepiece 0.2.0；pip check 通过。
- TempLAMA filtered train 1,215 组/10,475 QA；filtered val 525 组/4,570 QA；
  test 4,057 组/34,963 QA 原标签保持不变；LongMemEval-S 500 题仅准备不训练。
- 本轮最新45项本地及远端 CPU 测试通过；涵盖版本选择、持久化、二维/多段 shape、预算、标签隔离、
  旧 codec 禁用、worker 锁、空闲 GPU 筛选和研究门槛约束。
- 新主入口、冻结 Decoder 的可读性诊断、槽位 Codec、关键帧存储、版本/时间标记、
  精确 span 写入和训练审计代码已实现；旧短句适配已移除。
- 训练审计：1,215组、10,475段落QA、1,711时序对；C/D各16,661,760参数，
  版本时间标记98,304 FP32字节。基础配置默认`allow_training=false`。
- 用户已授权三步 GPU 实验，新增 `run_paper.sh` 多卡调度、不可变特征落盘、
  C/D 独立验证与 test 评测、配对 bootstrap 以及主表/图生成。详见 `RUN_PAPER.zh.md`。

## 未完成

- 基座与全部权重下载、哈希审计已完成，不需要重新下载。
- GPU 检查发现并修复共享 adapter checkpoint 上下文与 PEFT 解冻 LoRA 问题，
  最新 parity 正在服务器执行；结果以服务器 `icae_parity.json` 为准。
- 训练审计已完成但训练默认关闭；独立验证集可读性评分与人工错误审查仍未完成。
- C/D codec 训练授权附带完整可读性门禁与错误审查，尚未启动。
- LongMemEval 的无泄漏事实更新 resolver 及 query/event 校准尚未完成，转移不可直接声称就绪。

因此当前只能称“新路线架构代码/CPU 审计完成”，不能称 ICAE baseline 或 C/D
方法已验证。`allow_training=false` 是有意的护栏。
下载任务不占用 GPU；当前先执行统一入口 parity，再多卡完整 validation 与训练特征落盘，
必要时再考虑memory-only adapt。公开文本、无记忆和错误
版本控制，不假设 latent 持久化字节优于文本。
