# 后训练 Infra 工程师 · 能力图谱(4 阶段)

> **取代**旧的「阶段0–6 分层路线」(README §3 仅作第一阶段的算法读物参考)。
> **视角切换**:不推导数学证明,用**系统工程师视角解剖算法**——写任何系统代码前,先知道集群里要塞几个模型、数据怎么流动。

## 主线一句话
```
为什么训 (算法)  →  怎么算得分 (Eval Infra)  →  怎么算得快 (Megatron)  →  怎么拼在一起 (Ray/Slime)
   第一阶段             第二阶段                    第三阶段                  第四阶段
```

---

## 📌 教学铁律(所有课程 · 2026-06-27 定)
**每节课必须基于大量原始素材,不靠记忆口胡。原始素材 = 论文 + repo + 书。**
1. **锚定源**:每个结论挂一个具体出处——📄 论文章节/图表号、💻 代码 `file:line`/repo 路径、📖 书章节页码。
2. **引原文/原码**:摘真实段落、真实代码片段,不转述;数字/公式照抄原文。
3. **超截止必读源**:晚于 2026-01 的(Harbor/Slime/DeepSeek 等)一律读 repo/paper 原文再讲。
4. **可回溯**:讲义/Q&A 每节标「素材锚点」,读者能跳回原始素材核对。
> 素材库:`papers/`(14 篇论文 + Sutton&Barto 书)+ `code/`(cleanrl / harbor+cookbook / 待加 slime)。

---

## 第一阶段 · 算法直觉与数据流 (LLM RL, PPO, GRPO)
**目标**:用系统视角解剖算法。集群里到底塞几个模型?数据怎么流?

### PPO (Proximal Policy Optimization)
- **核心读物**:InstructGPT 论文。
- **系统视角解剖**:**4 个模型共存** —— Actor、Reference、Reward、Critic。
- **面试考点**:*为什么 PPO 显存压力那么大?*
  → Critic 模型要跟着 Actor 一起存活并更新;Reference 模型也占巨大显存。4 份大模型同时在显存里。

### GRPO (Group Relative Policy Optimization)
- **核心读物**:DeepSeekMath、DeepSeek-R1 论文。
- **系统视角解剖**:**去掉 Critic**。对同一 Prompt 采样 G 个不同回答(G=4 或 8),用这 G 个回答的**组内得分对比**算 Advantage。
- **面试考点**:*GRPO 对 Infra 提出什么新挑战?*
  → 去 Critic 省显存;但同一 Prompt 要**极高并发生成采样** → Inference(Rollout)吞吐瓶颈急剧放大 → 对 vLLM/SGLang 这类推理引擎的并发压榨要求极高。

> **地基已完成**(corridor 微课):L0 MDP/马尔可夫 · L1 回报 G + 概率记号 · L2 π/V/Q + 桥 V=Σπ·Q。
> 剩下:在这套概念上**叠系统视角**——4 模型拓扑、Advantage 在 token 上怎么分配、GRPO 采样的显存/吞吐画面。讲义页 `lecture_L0_L1.html`、`lecture_L2.html`。

---

## 第二阶段 · 评测基建与环境反馈 (Agent Eval Infra)
**目标**:借力打力——把日常工作转成顶级面试资本。把 Agent 的复杂执行转成高质量 **Reward**(RLHF/RLAIF 的上游)。
**工业级三层架构**:`顶层考题 → 调度编排 → 底层沙盒`。

### L1 · Benchmarks(考题层)—— 是「数据」,不是「基建」
- **SWE-bench**:几千个真实 GitHub Issue + 代码库 + 验证修复的 Unit Test。
- **OSWorld**:基于 Ubuntu 虚机的 GUI 任务(开浏览器查机票等)。
- **致命坑**:官方 runner 极简陋笨重——SWE-bench 硬拉几十 GB 重型 Docker,启动慢、爆 I/O,**无法高并发**;直接拿它跑几万并发 Rollout → 服务器当场死锁崩溃。
- **正确用法**:只拿它的**数据**(jsonl),runner 扔掉。

### L2 · Evaluation Orchestrators(调度与解耦层)—— 真正要花精力的 Eval Infra
不生产考题,负责:调度沙盒、剥离模型能力、记录轨迹。
- **Inspect**(UK AI Safety Institute):最严谨、公信力最强的标准化解耦框架。强制四模块切分 `Dataset → Solver(Agent/Harness)→ Tool/Sandbox → Scorer`。**轨迹日志极优美**、高度结构化 → 适合做「模型 vs Agent 框架」的客观能力解耦。**短板**:偏单机/轻量并发,云端极致 scale-up 非其卖点。
- **Harbor**(harbor-framework/harbor,Terminal-Bench 团队 2026 开源;**别**与 goharbor/harbor 镜像仓库混淆):为后训练 RL + 云原生大规模并发而生。
  - 解「并发」:抛弃本地 Docker 堆叠,内置对接 **Daytona/Modal** 等 serverless 微虚机云;`--n-concurrent 1000` 一条命令云端拉起上千沙盒。
  - 解「猛兽」:原生测最复杂 Agent Harness(Claude Code、OpenHands、Codex CLI)。
  - **直接对齐 RL**:原生把评测过程转成 **RL Rollouts**;官方 Cookbook 提供与 **SkyRL / Prime RL** 的集成代码。
  - 结论:**Harbor = 把 Eval Infra 变成 Post-training Infra 的那座桥。**

### L3 · Execution Sandboxes(底层并发执行沙盒层)—— 毫秒级启动、安全隔离的「物理」环境
干脏活的一层(真正执行 Agent 代码)。不用 Harbor 默认的 Daytona/Modal 商业云、想自建内网时:
- **E2B**(e2b.dev):纯为 AI Agent 设计的代码沙盒,开源,基于 **Firecracker**,百毫秒级快照热启动。
- **Fly.io / Firecracker**:要自写并发引擎,直接抄 Firecracker 的**内存快照 (Snapshotting)** 架构。

### 积木式拼装(工业标准做法)
```
题目: 下载 SWE-bench 数据 (jsonl)              ← 别碰它的烂 runner
调度: Harbor / Inspect 当主控,规范 Agent 接口、统一下发   ← 你的核心工作
执行: 把 orchestrator 的 backend 指向 E2B / 内网 MicroVM 池
输出: Harbor Rewardkit / Inspect Log 解析器 → 轨迹转 PPO/GRPO 的 Reward
```

### 我补充(讨论用)
- **吞吐数学(接 ① 的 generation-bound)**:GRPO G=8 × batch 1024 prompt = 8192 并发 rollout,每个要一个沙盒 → 这就是「为什么必须 100ms 微虚机 + serverless 突发」。算给面试官听。
- **真正的杠杆 = 快照恢复 ≠ 冷启动**:SWE-bench 痛在每仓库几十 GB Docker 冷构建;解法是**预构建快照 + 暖池 + snapshot restore**,不是单纯「换微虚机」。
- **同步 vs 异步 rollout**:SWE-bench 单任务可达分钟级 → 同步等齐会被 straggler 拖死;要不要 async/partial rollout(代价:off-policy 漂移)是 RL-eval 顶级设计题。
- **Reward 粒度**:GRPO 要每条 rollout 一个标量(pass/fail 或部分分);process reward 要每步分。轨迹解析按目标产出哪种。
- **Inspect vs Harbor 是两种活**:Inspect = 测量级评测(干净日志、出报告);Harbor = RL rollout 工厂(海量并发、喂训练)。可并用。

### 面试话术
*"我构建的不只是评测系统,而是为 LLM 后训练提供高吞吐、确定性 Environment Feedback 的沙盒基建。"*

> ✅ 已核实(clone 于 2026-06-27 → `code/harbor` + `code/harbor-cookbook`):`--n-concurrent` 真有;`adapters/swebench`(+ multi/pro/gym/lancer/smith 变体)、`src/harbor/environments/{daytona,modal,cwsandbox}`、`packages/rewardkit`、cookbook `{sky_rl, prime_rl, tinker_rl, harbor_rl}` 全部存在。额外:**separate verifier sandboxes**(验证器独立沙盒)、trajectory-format RFC、adapter wizard。
> 真实 CLI:`harbor run --dataset swebench --agent claude-code --model … --n-concurrent 100 --env daytona`。

---

## 第三阶段 · 分布式底盘与显存 (GPU Training, Megatron, ZeRO)
**目标**:纯 ML Infra 基本功 —— 面试八股 + 白板推演。算法要更新权重时,怎么在 GPU 上跑起来?

- **通信原语**:NCCL —— All-Reduce、All-Gather。
- **ZeRO 系列**:精读 DeepSpeed ZeRO 论文。优化器 / 梯度 / 参数怎么切到不同卡;Forward/Backward 的通信流。
- **3D 并行 (Megatron-LM)**:Tensor Parallelism(切矩阵)+ Pipeline Parallelism(切 Transformer 层);流水线 **气泡 (Bubble)** 怎么解。

---

## 第四阶段 · 巅峰缝合:训练与推理的交响乐 (Ray Core & Slime)
**目标**:终极难题 —— 把前三阶段在物理集群上串起来。

### Ray Core 解决异构调度
- 结合算法:Actor、Reward 模型在集群里 = 不同的 **Ray Actor 进程**。
- 结合评测:Eval Infra 的高并发结果,经 Ray 内存对象存储(**Plasma Object Store**)**零拷贝**传给 GPU 训练节点。

### Slime 解决同机显存复用 (Time-sharing)
- 深拆 Slime 架构:怎么把 **SGLang**(极致生成,喂 GRPO 海量采样)和 **Megatron-LM**(极致 3D 并行训练)部署在一起。
- **核心亮点**:Rollout 时清空训练激活值(给 KV Cache 腾地方);Train 时清空 KV Cache;经分布式 Checkpoint 服务(**DCS**)实现**毫秒级内存级权重同步**。

---

## 面试考点速查
| # | 问题 | 一句话答 |
|---|---|---|
| 1 | PPO 为什么显存压力大? | Actor+Reference+Reward+Critic 四模型共存;Critic 随 Actor 更新 |
| 2 | GRPO 对 Infra 的新挑战? | 省了 Critic,但同 Prompt 高并发采样 → Rollout 吞吐瓶颈,压榨 vLLM/SGLang |
| 3 | Eval Infra 怎么变面试资本? | 高吞吐、确定性 Environment Feedback 沙盒(MicroVM + 轨迹解析) |
| 4 | ZeRO 切什么? | 优化器态 / 梯度 / 参数三级切分 |
| 5 | 3D 并行 + 气泡? | TP 切矩阵 / PP 切层;Bubble 靠调度填 |
| 6 | Slime 怎么同机复用显存? | Rollout↔Train 互清 激活值/KV Cache;DCS 毫秒级权重同步 |

## 当前进度
```
第一阶段 [~] 概念地基 L0-L3 完成(MDP/G/π·V·Q/优势·策略梯度);下一节 L4 PPO 系统视角(4模型拓扑/ratio/clip/KL/显存)
第二阶段 [ ] (日常工作已有积累,待结构化 + Inspect/MicroVM 深化)
第三阶段 [ ]
第四阶段 [ ]
```

> 教学套路不变:概念解剖 → 白板/手撕推演 → (可选)讲义页 → 面试级问答。
> 资源:`papers/` **14 篇(标题已逐一核对✓)** · `code/`:cleanrl + harbor(+cookbook)。Slime/Harbor 无 arXiv 论文 → 读 repo。
> - RL 基础:Sutton&Barto、Sutton-PG-1999、PPO、GAE
> - 阶段①:InstructGPT、DeepSeekMath(GRPO)、DeepSeek-R1、DPO
> - 阶段③:ZeRO、Megatron-LM、Megatron-Pipeline、FlashAttention
> - 阶段④:Ray、verl/HybridFlow
