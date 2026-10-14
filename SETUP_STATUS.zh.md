# ICAE 路线切换状态

项目：v100-2 `/share/latent2/vcrm_icae`；不修改旧 phase12 目录。

## 已完成

- 官方 ICAE v2 代码核对并固定 revision；新 reader 的编码/指令格式依据 v2 实现。
- 指令 checkpoint 已下载：438,373,376 字节。
  SHA256：`ae637cf4e4fdbe623a4ad606339facf07128ab1dc8a4a6fad9be055e4c3c1f56`。
- checkpoint header 核对：129 个 keys；128 个 q/v LoRA keys 加 memory embedding，
  memory embedding shape 为 `[131, 4096]`。完整模型装载 key parity 仍需 GPU 实测。
- 独立项目环境安装 peft 0.15.2、sentencepiece 0.2.0；pip check 通过。
- TempLAMA filtered train 1,215 组/10,475 QA；filtered val 525 组/4,570 QA；
  test 4,057 组/34,963 QA 原标签保持不变；LongMemEval-S 500 题仅准备不训练。
- 17 项 CPU 测试通过；涵盖版本选择、持久化、二维/多段 shape、预算、标签隔离、
  旧 codec 禁用、worker 锁、空闲 GPU 筛选和研究门槛约束。
- 新主入口、Encoder QA 适配、冻结 Decoder 的可读性诊断与门槛代码已实现。

## 未完成

- Mistral-7B-Instruct-v0.2 基座权重下载、完整文件哈希审计。
- 实际 GPU 官方 `_compress` 对齐与 QA 梯度检查。
- Encoder 训练划分适配；独立验证集可读性评分；人工错误审查。
- 新条件压缩 codec 的设计/训练、正式固定预算比较与 LongMemEval 转移评估。

因此当前只能称“新路线工程切换/CPU 验证完成”，不能称 ICAE baseline 已验证。
下载任务不占用 GPU；不自动启动训练或正式对比。下一步执行统一入口的 parity，
然后 smoke，之后才做 train-only adapt 与完整 validation。公开文本、无记忆和错误
版本控制，不假设 latent 持久化字节优于文本。
