# 后训练 Infra 工程师 · 面试训练教程

> 配套 `ROADMAP.md`(知识地图)。这本是**刷题册**:把知识变成能脱口而出的答案。
> 每个考点格式:**Q** → **答(30秒)** → **追问链** → **白板题** → **杀手锏(一句话)**。

## 0. 怎么练(别只读,要"产出")
1. **主动回忆**:盖住答案,先自己讲 30 秒,再对答案。讲不出 = 没掌握。
2. **追问链**:面试官一定追问。每条都往下追 2 层,练到不慌。
3. **白板题**:拿纸真画/真算一遍(显存、通信量、bubble)。
4. **杀手锏**:每个考点背一句能"收口"的话,面试时主动抛。
5. **节奏**:每阶段 2~3 天;每天先快速默写昨天的杀手锏(spaced repetition)。

## 0.5 电梯陈述(把 4 阶段串成一句话 — 开场就抛)
> "RL 后训练是 **generation-bound** 的:核心难题是在同一批 GPU 上,让**快速推理引擎**和**分片训练器**共存、并**每步同步权重**。我的能力覆盖:为什么训(算法/显存拓扑)→ 怎么造 reward(高并发 Eval 沙盒)→ 怎么训得动(Megatron/ZeRO)→ 怎么拼起来(Ray/Slime 同机复用)。"

---

# 阶段 ① 算法直觉与数据流(PPO / GRPO / DPO)

### Q1.1 系统视角讲一遍 PPO,它要几个模型?
**答(30s)**:4 个。**Actor**(策略,训练)、**Critic**(价值,训练)、**Reference**(冻结 SFT,算 KL)、**Reward**(冻结,给分)。数据流三拍:**Rollout**(Actor 生成,存 old_logprob + Reward 分 + Ref logprob)→ **算 Advantage**(Critic+GAE)→ **Update**(重算 new_logprob、ratio、clip,**同批数据 K 个 epoch**)。
**追问链**:
- *哪些训练、哪些冻结?* → Actor/Critic 训练(带优化器+梯度+激活);Ref/Reward 只推理(只占权重)。
- *为什么能同批跑 K 个 epoch?* → 因为有 ratio 做重要性修正;否则数据是 off-policy 的。
**白板**:画三拍数据流,标出每拍哪些张量在动、哪些模型在显存里。
**杀手锏**:"PPO 真正贵的是它把一个 RL step 拆成 generate→evaluate→update 三段,每段的瓶颈不同。"

### Q1.2 为什么 PPO 显存压力那么大?(高频)
**答(30s)**:因为有**两个全训练模型**。Actor 和 Critic 都要带 Adam 优化器态——混合精度下约 **16 字节/参数**(fp16 权重2 + fp16 梯度2 + fp32 主权重4 + m 4 + v 4)+ 激活。Critic 几乎复制一份 Actor 的训练开销。再加 Reference + Reward 两个冻结模型常驻显存。**= 2 个训练模型 + 2 个推理模型同框。**
**追问链**:
- *16 字节怎么来的?* → 见白板,会算给他看。
- *怎么省?* → Ref/Reward 可 offload/量化/单独服务;Critic 可与 Actor 共享 backbone;或直接换 GRPO 砍掉 Critic。
**白板**:7B 模型,Adam 混合精度,单卡放不放得下?算 7e9 × 16B ≈ 112 GB(还没算激活/Critic)→ 必须切分(引出阶段③)。
**杀手锏**:"Critic 是第二个全尺寸训练模型,它才是 PPO 显存的隐形刺客。"

### Q1.3 GRPO 和 PPO 的核心区别?为什么能去 Critic?
**答(30s)**:GRPO **去掉 Critic**,baseline 不再用学出来的 V(s),而是**对同一 prompt 采 G 个回答**(G=4/8),用**组内得分**当 baseline:`A_i = (r_i − mean(r₁..G)) / std(r₁..G)`。省掉一个全训练模型。
**追问链**:
- *还剩几个模型?* → 3 个:Actor(训)+ Reference(KL)+ Reward。
- *能到 2 个吗?* → 能。**R1-Zero 用可验证奖励**(规则/测试通过),砍掉 Reward 模型 → 只剩 **Actor + Reference**。
**白板**:写出模型数递减链 **PPO=4 → GRPO=3 → R1-Zero=2**,标每步省了谁。
**杀手锏**:"4→3→2,是 RL 后训练怎么一步步变便宜的历史。"

### Q1.4 GRPO 对 Infra 提出了什么新挑战?(高频)
**答(30s)**:省了 Critic 显存,但把压力**推给了推理**:同一 prompt 要采 G 份 → 生成量 ×G。RL 本来就 generation-bound,这一下让 **Rollout 吞吐成为绝对瓶颈** → 必须极致压榨 vLLM/SGLang 的并发(continuous batching、KV cache 复用)。
**追问链**:
- *量化一下?* → batch 1024 prompt × G=8 = 8192 条并发生成,每条还可能是多轮 agent。
- *瓶颈搬到哪了?* → 从"显存(Critic)"搬到"推理吞吐 + 沙盒并发"(接阶段②)。
**杀手锏**:"GRPO 用推理吞吐换显存——省了一个模型,买来一个吞吐难题。"

### Q1.5 ratio / clip / KL 各管什么?(别推公式,讲作用)
**答(30s)**:**ratio = π_new/π_old**:数据是 π_old 采的、却要更新 π_new 并复用 K 轮,ratio 做重要性修正。**clip**:把 ratio 夹在 [1−ε,1+ε],防止一步更新离 π_old 太远(信任域)。**KL(→Reference)**:RLHF 里防止模型漂离 SFT 太远(防 reward hacking)。
**追问链**:
- *clip 的 KL 和 reference 的 KL 一回事吗?* → 不是。clip 约束 new↔old(更新幅度);reference-KL 约束 model↔SFT(别跑偏)。
**杀手锏**:"clip 管'步子别太大',reference-KL 管'别走错方向'。"

### Q1.6 DPO 为什么便宜?和 PPO/GRPO 什么关系?
**答(30s)**:DPO **没有 RL 循环**:用离线偏好对(chosen/rejected),闭式 loss,只要 **Policy + 冻结 Reference** 两个模型,**无 rollout、无 reward model、无 critic**。Infra 画像≈SFT。代价:off-policy、无在线探索。
**杀手锏**:"DPO 把 RLHF 压成一个加权 SFT——便宜,但丢了在线探索。"

### Q1.7 InstructGPT 三阶段是什么?
**答(30s)**:SFT(监督微调)→ RM(用人类偏好训奖励模型)→ PPO(用 RM 当 reward 做 RL,带 reference-KL)。这是 PPO 四模型的来历(SFT→Actor/Ref 初始,RM→Reward)。
**杀手锏**:"PPO 的 4 个模型不是凭空来的,是 InstructGPT 三阶段每一阶段的产物各留了一个在显存里。"

---

# 阶段 ② 评测基建与环境反馈(Agent Eval Infra)

### Q2.1 大规模 Agent Eval 的工业架构?
**答(30s)**:三层。**L1 考题**(SWE-bench/OSWorld,只是数据)→ **L2 调度编排**(Inspect/Harbor,调沙盒、剥能力、录轨迹)→ **L3 执行沙盒**(E2B/Firecracker 微虚机,毫秒级隔离环境)。reward 由轨迹+验证结果成形。
**追问链**:
- *为什么不用 SWE-bench 自带 runner?* → 它硬拉几十 GB 重型 Docker,启动慢爆 I/O,无法高并发,几万并发当场崩。只拿它的 jsonl 数据。
**杀手锏**:"benchmark 是数据,不是基建;真正的 Eval Infra 是中间那层编排器。"

### Q2.2 怎么做到一分钟并发上千个干净环境?
**答(30s)**:不用冷构建 Docker,用**微虚机快照恢复**(Firecracker snapshot / 暖池),亚秒~秒级起一个;并把突发甩给 serverless(Daytona/Modal)。**杠杆是 snapshot restore,不是"换了微虚机"这个名词。**
**追问链**:
- *状态干净怎么保证?* → 每 task 从同一快照 restore,跑完即弃,无残留 → 可复现。
**白板**:画"冷构建(分钟/GB) vs 快照恢复(亚秒)"两条时间线,标 RL rollout 为什么只能选后者。
**杀手锏**:"并发的终极答案是快照恢复,把'造环境'变成'拷贝内存页'。"

### Q2.3 Eval 怎么变成 RL 的 reward?
**答(30s)**:把 agent 执行**结构化成轨迹**(每步 observation/action),终点跑**验证器**得结果,再由 reward 层成形成标量。三种 reward 来源:**可验证/规则**(测试通过、精确匹配)、**Reward Model**、**LLM-as-judge**。可验证奖励正是 R1 能砍掉 Reward 模型的原因。
**追问链**:
- *GRPO 要什么粒度?* → 每条 rollout 一个标量;process reward 要每步分。
- *怎么防 reward hacking?* → 独立 verifier 沙盒(agent 碰不到测试)+ reference-KL。
**杀手锏**:"Eval Infra 的产物不是分数,是喂给 GRPO 的确定性 reward 信号。"

### Q2.4 Inspect vs Harbor 怎么选?
**答(30s)**:**两种活**。Inspect(UK AISI)=测量级评测:抽象极干净(Dataset→Solver→Sandbox→Scorer)、轨迹日志优美、出公信力报告,但偏单机/轻并发。Harbor(Terminal-Bench 团队)=RL rollout 工厂:云原生海量并发(`--n-concurrent`,对接 Daytona/Modal),原生把评测转成 RL rollouts。可并用。
**杀手锏**:"Inspect 量能力,Harbor 造数据;一个出报告,一个喂训练。"

### Q2.5 同步还是异步 rollout?
**答(30s)**:SWE-bench 单任务可达分钟级,**同步**(等所有 rollout 齐)会被 straggler 拖死;**异步/partial rollout** 提吞吐,但引入 off-policy 漂移,要靠 ratio/KL 兜。这是 RL-eval 的顶级权衡题。
**杀手锏**:"长任务 RL 的隐形杀手是 straggler,不是平均延迟。"

### Q2.6(话术)把日常 Eval 工作包装成面试资本
**答**:"我构建的不只是评测系统,而是为 LLM 后训练提供**高吞吐、确定性 Environment Feedback** 的沙盒基建——上接 agent harness,下接 RL rollout。"

---

# 阶段 ③ 分布式底盘与显存(Megatron / ZeRO / NCCL)

### Q3.1 NCCL 几个原语,分别用在哪?
**答(30s)**:**All-Reduce**(全规约求和,DP 同步梯度用)、**All-Gather**(收集各分片,ZeRO-3 取参数用)、**Reduce-Scatter**(求和再切分,ZeRO 切梯度用)。底层走 ring/tree 拓扑。
**杀手锏**:"All-Reduce ≈ Reduce-Scatter + All-Gather,理解这个等式就懂了 ZeRO 的通信。"

### Q3.2 ZeRO 1/2/3 各切什么?
**答(30s)**:沿数据并行维切。**ZeRO-1** 切优化器态;**ZeRO-2** 再切梯度;**ZeRO-3** 再切参数(fwd/bwd 时按需 all-gather)。**ZeRO-3 ≈ PyTorch FSDP**。
**追问链**:
- *代价?* → ZeRO-3 通信变多(每层 all-gather 参数 + reduce-scatter 梯度)。
- *省多少?* → 16 字节/参数 ÷ N 卡。
**白板**:画一行参数,标 ZeRO-1/2/3 各把哪三样(优化器/梯度/参数)切到不同卡。
**杀手锏**:"ZeRO 是用通信换显存;ZeRO-3 把单卡显存压到 1/N,但每步多两次集合通信。"

### Q3.3 混合精度 Adam 每参数多少字节?(白板必考)
**答(30s)**:**16 字节**。fp16 权重 2 + fp16 梯度 2 + fp32 主权重 4 + Adam m 4 + Adam v 4。
**白板**:7B → 112 GB,单张 80G H100 放不下 → 必须 ZeRO/TP/PP 切。
**杀手锏**:"参数量 ×16 字节,是判断'放不放得下'的口算公式。"

### Q3.4 3D 并行:TP / PP / DP 各切什么?
**答(30s)**:**DP** 复制模型切数据,梯度 all-reduce。**TP**(张量并行)切单个矩阵乘(注意力头/MLP),每层 2 次 all-reduce → 通信重 → **必须留在单机内**(NVLink)。**PP**(流水线)切 Transformer 层成多段,微批流水。
**追问链**:
- *再加两维?* → SP(序列并行,切 LayerNorm/dropout 省激活)、CP(上下文并行,长序列切注意力)。
**杀手锏**:"TP 切矩阵(通信重,锁机内),PP 切层(通信轻,跨机),DP 切数据——按通信成本分层放。"

### Q3.5 流水线气泡(bubble)是什么?怎么解?
**答(30s)**:PP 分 p 段,填充/排空时部分 GPU 空转 = bubble,占比 ≈ **(p−1)/m**(m=微批数)。解法:**多切微批**(m↑→bubble↓)+ **1F1B / interleaved 调度**。
**白板**:画 4 段流水线的 fill→steady→drain,标 bubble 区。
**杀手锏**:"bubble 是 p−1 段的固定开销,用更多微批把它摊薄。"

### Q3.6 "显存刺客"还有哪些招式?
**答(30s)**:**激活重计算(checkpointing)**——fwd 丢激活、bwd 重算,省显存 ~换 30% 算力;**FlashAttention**——不落 N² 注意力矩阵;**SP/CP**——切激活/长序列;**offload**——优化器态甩 CPU/NVMe。
**杀手锏**:"训练显存=参数态+激活;ZeRO 砍前者,重计算/FlashAttn 砍后者。"

---

# 阶段 ④ 巅峰缝合(Ray Core / Slime / verl)

### Q4.1 Ray 在 RL 后训练里干什么?
**答(30s)**:**异构调度**。Actor/Critic/Ref/Reward/Rollout 各是一组 **Ray actor 进程**,用 placement group 摆到不同 GPU;高并发 Eval 结果经 **Plasma 对象存储(共享内存,零拷贝)**传给训练节点。
**杀手锏**:"Ray 把'一堆模型 + 一堆沙盒'当成一个异构进程图来调度。"

### Q4.2 训练和推理:同机复用 vs 分离部署?(核心设计题)
**答(30s)**:**Colocate(同机时分复用)**:train 和 inference 共享同一批 GPU,利用率高,但要在显存里腾挪 + 同卡同步权重(Slime 路线)。**Disaggregate(分离)**:推理集群 + 训练集群分开,显存简单,但一方跑时另一方 GPU 闲、且权重要走网络同步。
**追问链**:
- *RL 为什么倾向 colocate?* → RL 是 generation↔train 交替,分离会让一半 GPU 长期空转。
**杀手锏**:"colocate 用'显存腾挪 + 同卡权重同步'的复杂度,换 GPU 不空转。"

### Q4.3 权重同步:整个 colocated RL 的命门
**答(30s)**:每个 train step 后,更新的 **Actor 权重必须立刻推给推理引擎**(SGLang/vLLM),否则下一轮 rollout 是 off-policy 的旧策略。colocate 时走**同卡显存拷贝 / NCCL broadcast**(快);disaggregate 时要序列化走网络(慢,成瓶颈)。
**杀手锏**:"RL infra 最硬的一道题:怎么每步把训练好的权重'零拷贝'灌进推理引擎。"

### Q4.4 Slime 怎么在同机复用显存?
**答(30s)**:把 **SGLang(极致生成,喂 GRPO 海量采样)+ Megatron(极致 3D 并行训练)**部署在同一批 GPU 上时分复用:**Rollout 时清空训练激活值**给 KV cache 腾地方;**Train 时清空 KV cache**;经分布式 checkpoint 服务(DCS)做**毫秒级内存级权重同步**。
**追问链**:
- *为什么要互相清空?* → 训练激活和推理 KV cache 都是显存大户,同机放不下两份 → 按相位轮流占用。
**杀手锏**:"Slime 让 train 和 inference 在同一块卡上'轮流值班',靠 DCS 秒级交接权重。"

### Q4.5 verl vs Slime?
**答(30s)**:都解"train+inference 同机 + 权重同步"。**verl**(HybridFlow):Ray **单控制器**编排 actor/critic/ref/reward/rollout 的数据流,通用、生态广。**Slime**:SGLang+Megatron 深度绑定、极致显存时分复用。verl 更像通用框架,Slime 更像极致实现。
**杀手锏**:"verl 给你一张通用的 RLHF 数据流图,Slime 给你一台榨干单卡的复用机器。"

### Q4.6 一句话收口:为什么这套这么难?
**答(30s)**:因为它要同时满足两个矛盾的极致——**推理要 KV cache、训练要激活+优化器态**,还得在它们之间**每步同步权重**,全在同一批 GPU 上。这就是后训练 infra 的终极工程。

---

# 模拟面试题库(盖答案,限时自测)
**算法**:① 系统视角讲 PPO ② PPO 为什么显存大 ③ Critic 估什么 ④ GRPO 怎么算 advantage ⑤ GRPO 对 infra 的挑战 ⑥ ratio 为什么出现 ⑦ clip-KL 和 reference-KL 区别 ⑧ DPO 为什么便宜 ⑨ 模型数 4→3→2 怎么来。
**Eval**:⑩ 三层架构 ⑪ 为什么不用官方 runner ⑫ 一分钟并发千沙盒怎么做 ⑬ eval 怎么变 reward ⑭ Inspect vs Harbor ⑮ 同步 vs 异步 rollout。
**分布式**:⑯ NCCL 三原语 ⑰ ZeRO 1/2/3 切什么 ⑱ 每参数 16 字节怎么来 ⑲ TP/PP/DP 切什么 ⑳ bubble 及解法 ㉑ 激活重计算。
**缝合**:㉒ Ray 调度什么 ㉓ colocate vs disaggregate ㉔ 权重同步为什么是命门 ㉕ Slime 怎么复用显存 ㉖ verl vs Slime ㉗ 为什么 RL 是 generation-bound。

# 训练日程(参考)
| 天 | 内容 | 自测目标 |
|---|---|---|
| 1–2 | 阶段① 9 题,刷到脱口而出 | 默写 4→3→2 + PPO 三拍数据流 |
| 3–4 | 阶段② 6 题 + 把日常工作套话术 | 讲清三层 + 快照恢复杠杆 |
| 5–6 | 阶段③ 6 题 + 白板算显存/bubble | 口算 7B 显存、画 ZeRO-3 切分 |
| 7–8 | 阶段④ 6 题 | 讲清权重同步 + colocate 取舍 |
| 9 | 全 27 题随机抽测 + 电梯陈述 | 任意题 30 秒成答 |

> 每天开练前先默写前一天所有"杀手锏"。9 天一轮,面试前再过一轮。
