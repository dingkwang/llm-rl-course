# LLM RL 课程 (PPO → GRPO → verl/slime)

> 主线优先。目标不是当传统 RL 全栈研究员,而是**看懂 PPO/GRPO 的数学 + 代码,再读懂 verl/slime 框架的 actor/critic/ref/reward worker**。
> 老师:Claude(每节课设「读 → 核心 → 动手算 → 验收」,做完发答案,批改后放行下一节)。

---

## 0. 怎么用这个目录

```
llm-rl-course/
├── README.md            ← syllabus + 详细读物清单
├── ROADMAP.md           ← 主线路线图(4 阶段能力图谱)+ 进度
├── INTERVIEW_PREP.md    ← 面试训练教程(27 题刷题册:答+追问+白板+杀手锏)
├── lecture_L0_L1.html   ← L0+L1 讲义(MDP / G / 概率记号 / 10题自测)
├── lecture_L2.html      ← L2 讲义(π / V / Q / 桥 V=Σπ·Q / 7题自测)
├── lecture_L3.html      ← L3 讲义(优势 A=Q−V / 策略梯度 / 含素材锚点 / 7题自测)
├── references/          ← 论文分析参考(Qwen-AgentWorld 等,由 looper 会话产出)
├── papers/              ← 已下载的论文/书 PDF
├── code/cleanrl/        ← CleanRL 单文件实现(读代码锚点)
└── notes/               ← 每节课的笔记 / 你的练习答案 / 我的批改
```

学习节奏:每层做完「验收」才进下一层。卡住就停,别硬推。

---

## 1. 终极目标 & 结课自测(10 题)

学完 PPO 这一段,下面 10 题都能讲清楚 = 可以进 GRPO:

1. policy gradient 为什么是 `log π × reward/advantage`?
2. baseline 为什么不改变期望梯度,却能降方差?
3. advantage 比 raw return 好在哪?
4. critic/value model 在 PPO 里预测什么?(≠ reward model)
5. old policy 和 new policy 为什么同时出现?
6. `ratio = π_new/π_old` 的直觉是什么?
7. clipping 到底 clip 的是什么?
8. 为什么 positive / negative advantage 的 clip 方向不同?
9. PPO 里的 KL 和 RLHF 里的 reference-KL 有什么区别?
10. PPO 为什么需要 critic,而 GRPO 可以不要?

> GRPO 自然延伸:PPO 用 critic 估 baseline;GRPO 同一 prompt 采样多个 response,用**组内 reward 相对值**当 baseline,所以不需要 critic。

---

## 2. 资源清单(manifest)

### ✅ 已下载到本地 (`papers/`, `code/`)

| 文件 | 用途 | 层 |
|---|---|---|
| `papers/SuttonBarto_RL_2ndEd.pdf` | RL 经典教材,源头语言 | L0 |
| `papers/sutton_policy_gradient_1999.pdf` | policy gradient theorem 理论来源(知道即可,不啃证明) | L1 |
| `papers/gae_2015_schulman.pdf` | GAE:用 value function 降 PG 方差 | L2 |
| `papers/ppo_2017_schulman.pdf` | PPO 原论文,clipped surrogate objective | L3-L4 |
| `code/cleanrl/cleanrl/ppo.py` | PPO 单文件实现(~300 行),读代码主锚点 | 代码 |

### 🌐 在线读(网页更新频繁,留链接更好)

| 资源 | 链接 | 层 |
|---|---|---|
| **Lilian Weng — Policy Gradient Algorithms** ⭐ 必看第一遍 | https://lilianweng.github.io/posts/2018-04-08-policy-gradient/ | L0-L1 |
| **OpenAI Spinning Up — PPO** ⭐ 主资料 | https://spinningup.openai.com/en/latest/algorithms/ppo.html | L2-L5 |
| **RLHF Book — Policy Gradients (ch6)** ⭐ 贴近 LLM | https://rlhfbook.com/c/06-policy-gradients | L5 |
| **verl — PPO docs** | https://verl.readthedocs.io/en/latest/algo/ppo.html | L2/L6 |
| HuggingFace TRL — PPO Trainer | https://huggingface.co/docs/trl/en/ppo_trainer | L5 |
| Berkeley CS285 (PG / actor-critic / advanced PG) 选看 | https://rail.eecs.berkeley.edu/deeprlcourse/ | L1 |
| CleanRL docs(配合代码) | https://docs.cleanrl.dev/ | 代码 |

> 想要任何网页的离线快照(PDF)告诉我,我用 headless 渲染抓下来。

---

## 3. 学习路线图(6 层 + 日程)

### 六层概念阶梯

| 层 | 主题 | 学会回答 |
|---|---|---|
| **L0** | 最小 RL 语言 | 看到 `πθ(a\|s)`、`Vπ(s)`、`Aπ(s,a)` 不懵(详见 §5 教案) |
| **L1** | Policy Gradient | 为什么 loss 里有 `log πθ`?advantage 为什么能当 sample weight?baseline 为什么不改期望梯度但降方差? |
| **L2** | Advantage / GAE / Value Model | critic(估未来 reward)≠ reward model(评 response);GAE 为什么用 value 降方差 |
| **L3** | old/new policy ratio | `r_t(θ)=πθ/π_old`:不是概率校准,是 **old-policy rollout 数据复用时对 policy change 的度量** |
| **L4** | Clipping | `L^CLIP=E[min(r·A, clip(r,1−ε,1+ε)·A)]`;为什么正/负 advantage 的 clip 方向不同 |
| **L5** | KL Penalty | 区分两种 KL:① PPO 限制 new vs old 变太快;② RLHF `reward−β·KL(π‖π_ref)` 限制漂离 SFT 太远 |
| **L6** | Actor-Critic / Value Model | 看到 verl/slime 的 actor/critic/ref/reward worker 立刻知道各自干啥 |

### 建议日程(概念→公式→代码→LLM)

- **第1天 概念直觉**:Lilian Weng + Spinning Up PPO。只求懂 policy gradient / advantage / actor-critic / clipping。**不读代码。**
- **第2天 公式**:精读 PPO 的 `ratio`、`L^CLIP`、value loss、entropy bonus、KL。读 PPO paper(抓主线:采样后优化 surrogate,比 TRPO 简化,允许多轮 minibatch)+ Spinning Up。
- **第3天 代码**:读 `cleanrl/ppo.py`。重点定位:rollout 收集 / logprob 存储 / advantage / returns / `ratio=exp(new−old)` / `pg_loss1` / `pg_loss2` / clip / value loss / entropy / optimizer step。
- **第4天 转 LLM PPO**:TRL PPO Trainer + RLHF Book + verl docs。重点是变量对应,不是跑 benchmark。

### 普通 RL ↔ LLM PPO 对照表(背下来)

| 普通 RL | LLM PPO |
|---|---|
| state | prompt + partial response |
| action | next token |
| trajectory | 生成的整段 response |
| environment reward | reward model / rule reward |
| policy | actor LLM |
| value function | critic / value head |
| old policy | rollout 时的 actor |
| reference policy | 冻结的 SFT/base model |
| KL | 防止偏离 reference model |

### 资料优先级

- **必看**:① Lilian Weng PG → ② Spinning Up PPO → ③ PPO paper → ④ CleanRL ppo.py → ⑤ RLHF Book PG → ⑥ verl PPO docs
- **可选**:CS285 lectures、GAE paper、TRL docs
- **暂不看**:slime 源码、verl `ray_trainer.py` 深处、Megatron+SGLang weight sync、Sutton&Barto 从第1页通读

### 最短路径

```
Lilian Weng 直觉 → Spinning Up 公式 → CleanRL 代码
→ TRL/RLHF Book LLM PPO → verl PPO worker mapping → GRPO → slime
```

---

## 4. 进度追踪

- [~] **L0** 最小 RL 语言(corridor MDP 教案,§5)
  - [x] L0 热身(§3.1 + 马尔可夫性)  - [x] L1 G(回报)+ 概率记号前置  - [~] L2 π/V/Q 进行中  - [ ] L3 A=Q−V  - [ ] L4 闭卷验收
- [ ] **L1** Policy Gradient(Lilian Weng + CS285)
- [ ] **L2** Advantage / GAE / Value Model(GAE paper + Spinning Up + verl)
- [ ] **L3** old/new ratio(PPO paper + Spinning Up)
- [ ] **L4** Clipping(Spinning Up + PPO paper + CleanRL)
- [ ] **L5** KL Penalty(TRL + RLHF Book + verl)
- [ ] **L6** Actor-Critic / Value Model(CS285 + verl)
- [ ] **结课**:10 题自测全过
- [ ] **进阶**:GRPO → verl/slime

---

## 5. Layer 0 详细教案 —— corridor MDP

贯穿玩具:`A —— B —— G(终点)`,`γ=0.9`
- 在 A:`→`到B(r=−1);`←`留A(r=−1)
- 在 B:`→`到G(r=+10);`←`回A(r=−1)

依赖链:`s,a,r → τ → G → π → V → Q → A`

### L0 热身(15min)
读 §3.1。RL=`看状态→动作→奖励+新状态`循环;马尔可夫性=只看当前 s 就够。

### L1 原子+目标(1h)
读 Lilian Weng *Notation* → Sutton §3.2–3.3。
**钉死**:`r`=即时打分;`G_t=Σ_k γ^k r_{t+k}`=要最大化的目标。
**🖊️动手算1**:`r₀=−1,r₁=−1,r₂=+10,γ=0.9`,求 `G₂/G₁/G₀`(从后往前:`G_t=r_t+γ·G_{t+1}`)。

### L2 评估(1.5h)
读 §3.5(精)、§3.6(略)。
- `π(a\|s)` 概率分布;`π_θ`=参数θ的策略网络
- `V^π(s)=E_π[G\|s]`;`Q^π(s,a)=E_π[G\|s,a]`
- **桥**:`V^π(s)=Σ_a π(a\|s)Q^π(s,a)`
**🖊️动手算2**:`Q(B,→)=+8, Q(B,←)=+3, π(→\|B)=0.7, π(←\|B)=0.3`。求 `V(B)`;若改贪心 `π(→\|B)=1`,`V(B)`=? 为什么 V 必带上标 π?

### L3 优势(1h)
读 §13.1、§13.4(重点,advantage 在 baseline 处登场)。
`A^π(s,a)=Q^π(s,a)−V^π(s)`;`E_{a~π}[A]=0`。
PG 用 A 不用 Q:减基线 V → 只奖励「超出平均」→ 方差小训练稳。
**🖊️动手算3**:求 `A(B,→)`、`A(B,←)`;验证 `0.7·A(B,→)+0.3·A(B,←)≈0`;一句话说为什么「等于0」对训练好。

### L4 闭卷验收(30min)
合书默写:`G_t`、`V^π(s)`、`A^π(s,a)`、一句话「A=Q−V 为何对 PG 有用」。
四条顺手 → L0 结课,进 L1。

**下一坎预告**:`∇J=E[∇log π_θ(a\|s)·A]` 怎么从「最大化期望回报」推出——这是「认词」到「懂 PPO」之间唯一那道坎。
