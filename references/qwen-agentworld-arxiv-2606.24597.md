# Qwen-AgentWorld (arXiv:2606.24597)

**Source**: Paper "Language World Models for General Agents" + GitHub repo (cloned analysis at qwen_agentworld_analysis.md).

**Core Contribution**: First *native* Language World Model (LWM) that learns environment dynamics as primary objective (not post-hoc). Trained on >10M real trajectories across 7 domains. Accompanied by AgentWorldBench (5-dim grounded rubric eval).

## Key Concepts & Course Mapping

- **Agent loop = policy + world model**: Explicitly decomposes the loop into (1) policy (state → action) and (2) world model ((state, action) → next state / observation). This directly grounds L0 (dynamics / MDP transitions): world model learns P(s'|s,a) as first-class citizen via next-state prediction on trajectories. Unlike model-free policy gradient (L1+), here dynamics are modeled explicitly (cf. L0 state/action/reward → transition).

- **CPT → SFT → RL pipeline**: 
  - CPT: Injects dynamics knowledge (non-thinking next-state pred on massive real trajectories) — pre-trains the "environment model" core to L0.
  - SFT: Activates explicit thinking/CoT for structured next-state output (7k samples).
  - RL (92k samples, hybrid rewards): Sharpens fidelity. Uses rubric-based LLM judge (5 dims: Format/Factuality/Consistency/Realism/Quality) + rule-based verifiers. Parallels stage 4 (LLM PPO): maps "state=prompt+history, action=tool call, reward=structured judge+rules" instead of chat responses. Also stage 5 (GRPO-like post-training): optimizes simulation policy with relative/group rewards for better fidelity, enabling Sim RL (decoupled simulator > real env in some cases).

- **Reward design as structured LLM-as-judge + rules**: Hybrid rewards (not pure human prefs) provide reference-grounded, multi-dim signal. Directly relevant to LLM PPO/RLHF (stage 4-5) where reward models guide post-training; here applied to world modeling instead of alignment.

- **Transfer & applications**: RL on one domain improves others (cross-domain generalization of dynamics). Controllable simulation for adversarial testing. "Sim RL can surpass Real RL" — shows value of learned world model for agent training (L0 dynamics → practical policy improvement).

- **Evaluation**: Reference-grounded 5-dim rubric (vs free-form) + differentiated matching (exact for deterministic, plausibility for pre-existing). Informs better eval harnesses for LLM post-training (stage 4-5).

**See also**: Full analysis in `qwen_agentworld_analysis.md` (local copy). Repo: https://github.com/QwenLM/Qwen-AgentWorld
