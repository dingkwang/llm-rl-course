# Qwen-AgentWorld: Technical Analysis Report

**Source artifacts analyzed:**
- Paper: arXiv:2606.24597 (https://arxiv.org/abs/2606.24597) — "Qwen-AgentWorld: Language World Models for General Agents"
- Repo: https://github.com/QwenLM/Qwen-AgentWorld (cloned to /home/dingkwang/.tmp/Qwen-AgentWorld)

**Analysis date:** 2026-06-24

---

## (a) TL;DR — One Paragraph

Qwen-AgentWorld is the first **native language world model (LWM)** that simulates seven text-based agent environments (MCP, Search, Terminal, SWE, Android, Web, OS) via long chain-of-thought next-state prediction. Unlike prior approaches that bolt world modeling onto an existing LLM post-hoc, Qwen-AgentWorld trains environment modeling as the explicit objective from continual pre-training (CPT) onward through a three-stage pipeline (CPT→SFT→RL with hybrid rubric+rule rewards) on >10M real interaction trajectories. The model is released in two sizes (35B-A3B MoE and 397B-A17B MoE) and is accompanied by AgentWorldBench, a ground-truth-grounded 5-dimension rubric benchmark (Format/Factuality/Consistency/Realism/Quality) constructed from real frontier-model trajectories on 9 established agent benchmarks. Beyond the foundation model, the work demonstrates two applications: (i) **decoupled simulator** for scalable/controllable Sim RL that can exceed real-environment training, and (ii) **unified agent foundation model** where single-turn non-agentic LWM RL warm-up transfers to multi-turn tool-calling agentic tasks across 7 benchmarks including 3 OOD domains.

---

## (b) Key Contributions / Architecture / How the Agent-World Works

### Core Idea

In the agent–environment loop, two components are essential: the **policy** (state → action) and the **world model** ((state, action) → next state). Prior LLM-agent research focused almost exclusively on the policy side. Qwen-AgentWorld treats world modeling as a first-class training objective.

### Unified Environment Trajectory Schema (Paper §2.2, README)

All seven domains are unified under a single textual schema:

```
system_prompt := task_description ⊕ action_space ⊕ initial_state ⊕ demonstrations ⊕ simulation_instruction
turn_t := (action_t, observation_t)
trajectory := system_prompt ⊕ [turn 1, ..., turn T]
```

**Five system prompt components** (see `prompts/{domain}/system_prompt.txt` and Figure 3 in paper):
1. **Task Description** (static) — "You are a Terminal World Model — predict exact next state..."
2. **Action Space** (static) — JSON schema for commands/keystrokes/tool calls
3. **Initial State** (dynamic, per-trajectory) — container snapshot, installed packages, cwd, UI hierarchy
4. **Demonstrations** (static few-shot) — (action, observation) examples
5. **Simulation Instruction** (dynamic) — controllable perturbations, e.g., "hide answer from web_search"

### Seven Domains (Paper Table 1, README §Introduction)

| Domain | Action | Observation | Core Capability |
|--------|--------|-------------|-----------------|
| MCP | JSON Tool Call | Tool response | Factual world knowledge |
| Search | Web Search / Extractor | Conversation history | Factual world knowledge |
| Terminal | Bash / Keystrokes | stdout + prompt | Long-context causal reasoning |
| SWE | Read/Edit/Bash | Tool output + diffs | Code execution reasoning |
| Android | Touch/Swipe/Type | UI view hierarchy | Visual state reasoning |
| Web | Click/Type/Navigate | Accessibility tree | Visual state reasoning |
| OS | Mouse/Keyboard | Accessibility tree | Visual state reasoning |

GUI domains use accessibility trees / UI view hierarchies (not pixels).

### Three-Stage Training Pipeline (Paper §3, "CPT injects, SFT activates, RL sharpens")

1. **CPT (Continual Pre-Training)** — Injects general-purpose world modeling from state-transition dynamics + augmented professional corpora. Non-thinking trajectories. Data from: dedicated agent infrastructure (sandboxes, MCP servers, persistent GUI VMs), open interaction traces (cleaned via multi-agent pipeline), in-house SFT trajectories.

2. **SFT (Supervised Fine-Tuning)** — Activates next-state-prediction as explicit thinking pattern. 7,094 SFT samples across domains (Table 2).

3. **RL (Reinforcement Learning)** — Sharpens fidelity via hybrid rubric-and-rule rewards. 92,308 RL training samples, avg ~19k tokens / 13.4 turns. Rewards combine:
   - Rubric-based (LLM judge on 5 dimensions)
   - Rule-based verifiers (deterministic checks on targeted capabilities)

**Native world model**: Environment modeling is the training objective from CPT onward, not post-hoc adaptation.

### Key Files in Repo

- `README.md` — Overview, results tables, quickstart
- `prompts/{domain}/system_prompt.txt` — LWM system prompts (7 domains)
- `prompts/{domain}/judge_system_prompt.txt` — Judge rubrics per domain
- `eval/eval.py` — 3-step eval pipeline (infer → judge → score)
- `eval/lwm_eval_utils/`:
  - `task_configs.py` — Per-domain response markers, SCORE_DIMENSIONS
  - `output_parser.py` — `_remove_thinking_tags()`, `parse_model_output()`
  - `judge_parser.py` — Robust JSON extraction, `_extract_scores()`, Turing-test choice extraction

### Architecture for Controllability (Paper §6.1.2)

Control instructions in `simulation_instruction` enable:
- **Environment Adaptation**: Inject targeted perturbations (intermittent errors, paginated responses, partial results) to expose agent weaknesses
- **Fictional-World Construction**: Generate entirely invented but self-consistent worlds (e.g., 2030 smartphone market with fictional models) so agents must search rather than recall, and cannot confuse sim facts with real knowledge

---

## (c) Benchmark / Eval Methodology

### AgentWorldBench (Paper §4, README §"Evaluate on AgentWorldBench")

**Construction**: Real interaction trajectories from 5 frontier models (Claude Opus 4.6, GPT-5.4, etc.) executing on 9 established agent benchmarks (Tool Decathlon, Terminal-Bench 1.0/2.0, OSWorld-Verified, MCPMark, WideSearch, Claw-Eval, etc.). Entirely OOD from training data.

**Data format** (`*_test.jsonl` per domain):
```json
{
  "task": "terminal",
  "id": 145256090131919,
  "prompt": ["### Turn 1\n**Action:**\n```json\n{...}\n```"],
  "response": ["**Environment Observation:**\n..."],
  "current_prompt": "### Turn 1\n**Action:**\n...",
  "system_str": "# Role and Objective\n\nYou are a **Terminal World Model**...",
  "turn_idx": 1,
  "total_turns": 5
}
```

**Five evaluation dimensions** (1-5 scale, normalized to 0-100):
1. **Format** — Structure/layout fidelity (prompt patterns, command echo, line breaks)
2. **Factuality** — Correctness vs ground truth (deterministic content must match exactly)
3. **Consistency** — State coherence across turns (no contradictions with prior state)
4. **Realism** — Behavioral fidelity (appropriate errors, progress stages, value plausibility)
5. **Quality** — Completeness and appropriate conciseness

**Differentiated matching criteria** (Paper §4.2):
- Deterministic content → exact match required
- Pre-existing environment content → format + plausibility only
- Runtime metadata (timestamps, PIDs) → format + range only

**Judge protocol**:
- Reference-grounded: judge receives ground-truth observation alongside prediction
- Domain-aware rubrics in `prompts/{domain}/judge_system_prompt.txt`
- Judge selection via double-blind Turing test (Gemini 3 Flash, Claude Sonnet 4.5, GPT-5.2); GPT-5.2 selected for highest Turing-test accuracy; rankings stable across judges (Spearman ρ=0.92–0.99)

**Eval script** (`eval/eval.py`):
```
python eval.py infer --data-dir ../AgentWorldBench --model-base-url ... --model-name ...
python eval.py judge --predictions ./results/predictions.jsonl --judge-base-url ... --judge-model ...
python eval.py score --predictions ./results/judged.jsonl
```

### Main Results (Paper Table 5, README)

Qwen-AgentWorld-397B-A17B: **58.71 overall** (highest), beats GPT-5.4 (58.25).
- Text domains: leads on Terminal (57.73 vs 53.69), SWE (68.49 vs 66.29), Search (37.82 vs 37.26)
- GUI domains: competitive (5th), Claude/GPT lead due to multimodal pre-training

Effect of LWM training: +8.66 at 35B scale, +3.97 at 397B scale over same-base without LWM.

---

## (d) Concrete Relevance to Tesla Agentic Tooling

### 1. World Model as Environment Simulator → Skill / Agent Evaluation Harness

**Current Tesla pattern**: `bug-hunter` and `review-code` skills (see `.grok/marketplace-cache/.../skills/{bug-hunter,review-code}/SKILL.md`) run sub-agents against diffs or code. Evaluation is against static artifacts (diffs, checklists) with human or LLM judgment.

**Qwen-AgentWorld idea**: Train a domain-specific "world model" that predicts the *next state* of an environment (e.g., CI outcome, test result, log pattern) given an action (e.g., "run this test", "apply this patch"). Use it for:
- **Simulated rollouts** before touching real infra
- **Controllable adversarial testing**: inject targeted perturbations ("make this test flake", "return partial logs") to expose weaknesses in bug-hunter/review-code

**Why worth adopting**: Tesla's agentic tooling already has 200+ skills and a skill router (see below). Adding a "simulation" capability lets you scale evaluation environments without dedicated sandboxes, and controllability lets you target rare failure modes that real CI rarely surfaces.

### 2. Skill Router for 200+ Skills — Structured Selection + Warm-up

**Current Tesla pattern** (ai-oncall-bots):
- `select_skill(skill=...)` is the mandated first action (see `app/prompts/agent_prompt.py:55`)
- `get_skill_catalog()` returns `SKILL_METADATA` (see `app/prompts/channel_contexts.py:127`)
- `SKILL_ALIAS_MAP` + `normalize_skill_name()` for canonicalization
- ~10-15 skills defined in `SKILL_METADATA` (ap-help, nn-mlir-help, rodeo-programs, tclips-alarms, general, ...)
- Skills live at `~/.claude/skills/{name}/SKILL.md` and `~/.claude/plugins/cache/.../skills/`
- Looper scans `~/.claude/skills`, project `.claude/skills`, and plugin caches (`server.py:5835`)

**Qwen-AgentWorld idea**:
- LWM RL warm-up (single-turn, non-agentic next-state prediction) transfers to multi-turn tool-calling agentic tasks (+8.96 average across 7 benchmarks, including +9.0 on BFCL v4, +11.3 on Claw-Eval) — see Paper §6.2, Table 9.
- The mechanism: the model learns to *mentally simulate* environment responses before acting (Figure 11 case study: "prediction-driven action refinement").

**Concrete adoption**:
- **Pre-train / warm-up a lightweight router model** on (channel context, skill choice) pairs derived from historical ai-oncall-bot sessions. Treat "what skill should handle this query" as next-state prediction: given (channel, query), predict (skill, rationale).
- Use the same 5-dimension rubric discipline for router eval: Format (valid skill name), Factuality (skill actually exists and is appropriate), Consistency (same query → same skill), Realism (matches historical expert routing), Quality (sufficient justification).
- **Controllable simulation for router training**: synthesize fictional but realistic channel/skill scenarios (e.g., "new CI system with novel error taxonomy") to train robust routing without polluting real logs.

**Why worth adopting**: The current skill catalog is small and hand-curated. As the number of skills grows to 200+, manual routing and prompt stuffing become brittle. A learned router with world-model-style training (predict "what the skill will do / return") could improve first-action accuracy and reduce wasted turns.

### 3. Reference-Grounded Rubric Evaluation → Agent Eval Infrastructure

**Current Tesla pattern**: `bughunt-eval/` runs scenarios and collects `eval_report_main.json`; reviews compare agent output against expectations with free-form critique.

**Qwen-AgentWorld idea**:
- Ground eval in *actual environment observations* (real CI logs, real terminal output, real PR diffs) rather than model self-assessment.
- Use differentiated matching: deterministic fields (exit codes, pass/fail) must match; runtime metadata (timestamps, container IDs) only plausibility-checked.
- 5 explicit dimensions force the judge to separate concerns instead of giving a single "good/bad" score.

**Concrete adoption**:
- For bug-hunter: define ground-truth "observations" (e.g., the actual bug location + failure mode) and score predictions on Factuality (did it name the right file:line?) and Consistency (does the explanation match the observed symptom chain?).
- Store rubric prompts alongside each skill (cf. `prompts/{domain}/judge_system_prompt.txt`) so eval criteria evolve with the skill.

### 4. Cross-Domain Generalization Signal → Multi-Skill Transfer

**Paper finding** (§5.3, Figure 8): RL on Terminal data alone improved held-out domains (MCP +5.0, SWE +11.5, Search +11.8) within the first 10 steps, suggesting RL reinforces *generalizable world knowledge* (how state transitions compose, how errors propagate) rather than domain-specific formats.

**Relevance**: If Tesla builds per-skill or per-domain "mini world models" (e.g., a CI-failure world model, a log-world model), training on one may transfer to others. This argues for a shared representation layer and against fully siloed per-skill fine-tunes.

### 5. Deliberative Self-Correction Pattern (Paper §7.1)

Qwen-AgentWorld emits 10.4 "Wait!" interrupts per turn on average, decomposing into factual, epistemological, and perspective-taking corrections. This converts single-pass generation into constrained satisfiability search.

**Relevance**: For review-code / bug-hunter, explicitly prompting for self-correction checkpoints ("re-examine your assumption about the call graph") may improve depth without increasing context length.

---

## (e) Honest Limitations / What to Ignore

### What the Paper Does NOT Show

1. **No direct head-to-head with real-environment training cost/latency**. The claim "Sim RL can surpass Real RL" (WideSearch 50.3% vs 45.6%) is on task metric, not wall-clock or $ cost. Real RL may still be cheaper per effective sample if the simulator is slower or requires heavy judge calls.

2. **Controllability requires high-quality initial state** (Paper §6.1 takeaway: "State is the bottleneck"). Without detailed initial state (installed packages, file layout, DB contents), simulation fidelity degrades and downstream gains vanish. Tesla's environments (Stark clusters, vehicle configs, ECU farms) are far more heterogeneous than the paper's containerized sandboxes — constructing faithful initial states at scale is non-trivial.

3. **Fictional-world gains assume the downstream task is search-like** (learn procedure, not memorize facts). For Tesla domains where domain facts *are* the value (e.g., specific register maps, known-good firmware hashes, safety invariants), training in a fictional world may teach the wrong prior.

4. **Judge is still an LLM** (GPT-5.2). Even with reference-grounding and Turing-test calibration, the eval pipeline has a privileged-judge problem. If the judge model has blind spots (e.g., misunderstands a new MCP schema), all scores are biased.

5. **Repo is minimal**. The GitHub release is primarily prompts + eval harness (≈1.1k LOC Python). The actual 10M+ trajectories, CPT/SFT/RL training code, and the 397B model weights are not in the repo. Reproducing the training pipeline from scratch is not feasible from this artifact alone.

6. **"Native" is a training story, not an inference-time guarantee**. At inference you still call the model with a carefully engineered system prompt. The "native" claim is about what objective it was optimized for, not that it magically knows environments without prompting.

7. **GUI results lag text**. Qwen-AgentWorld-397B-A17B is 5th on GUI domains; Claude/GPT lead due to multimodal pre-training. If Tesla work involves significant GUI/Android/Web interaction, pure-text LWM is incomplete.

8. **No multi-agent or adversarial robustness results**. The paper does not evaluate whether agents trained in LWM-simulated environments are more or less susceptible to prompt injection, tool-schema confusion, or environment-specific exploits.

### What to De-emphasize for Tesla Context

- **Scale claims (397B-A17B)**: Tesla's agentic tooling runs on frontier hosted models (Claude/Grok) via existing inference stacks; training a 400B MoE world model is not on the roadmap.
- **Video/world-model literature survey** (Paper §8): Mostly irrelevant; Tesla agentic tooling is text/tool-call based.
- **"First language world model across 7 domains"** marketing: The engineering pattern (CPT on transitions + SFT on thinking + RL on fidelity) is the transferable artifact, not the "first" label.

---

## Summary of Actionable Ideas

| Idea | Source | Tesla Mapping | Effort |
|------|--------|---------------|--------|
| Train a lightweight "router world model" on (context, skill) → (chosen_skill, rationale) | §6.2 transfer results | ai-oncall-bot skill router, 200+ skills | Medium |
| Add reference-grounded 5-dim rubric eval to bug-hunter/review-code | §4.2 | bughunt-eval, skill eval harness | Low-Medium |
| Use controllable simulation instructions for adversarial test generation | §6.1.2 | CI failure injection, log perturbation | Medium |
| Store per-skill judge prompts alongside SKILL.md | §4.2, prompts/ layout | claudecode_plugins/skills/*/ | Low |
| Exploit cross-domain transfer: share world-model pre-training across related skills | §5.3 | Common infra for log/CI/trace skills | Medium |
| Explicit "Wait!" self-correction checkpoints in review prompts | §7.1 | review-code, thorough-pr-review | Low |

---

**Report path**: `/home/dingkwang/.tmp/qwen_agentworld_analysis.md`

**Looped back via**: `looper-cli send "Looper Manager" "<summary + report path>"` (to be executed by caller).

---

## Scoped Excerpts for robodev #1773 and review-code

**Requested via reveval-to-robodev (2026-06).** Two items only.

### (1) Differentiated-Matching Rubric (verbatim)

**Source:** `prompts/terminal/judge_system_prompt.txt` (implements Paper §4.2 criteria; core logic also appears in SWE prompt and eval patterns).

**Content Type Classification**

**Important Context:** The Terminal World Model has no access to the real environment state. It can only "know" information **explicitly shown or created during the current interaction session**. For any pre-existing state (e.g., file contents not written in this session, installed packages, system configurations, directory structures), the model must infer plausible values.

Before evaluation, apply different verification standards based on information availability:

| Content Type | Verification Standard | Examples |
|--------------|----------------------|----------|
| **Deterministic content** | Must match Ground Truth exactly. These are outputs fully determined by the command and known session state. | `echo` output, `cat` of a file written earlier in this session, computation results |
| **Pre-existing environment content** | Verify format and plausibility only. Do NOT penalize different but reasonable values. | `ls` of a pre-existing directory, `cat` of a pre-existing file, version numbers |
| **Runtime metadata** | Verify format and plausibility only. | Timestamps, PIDs, container IDs, memory addresses, download speeds, file sizes of pre-existing files |

**Key criteria from Factuality:**

- **Deterministic Content — Strict Match:** Outputs fully determined by the command and known session state must match Ground Truth exactly — including computational results, reads of session-created files, error types for deterministic failures, and success/failure status.
- **Pre-existing Content — Plausibility Check:** For content depending on unknown environment state, do NOT require exact matches. Instead verify: (1) format matches Ground Truth's pattern, (2) content is plausible for the domain, (3) no internal contradictions.
- **Runtime Metadata:** Timestamps, PIDs, and other dynamic metadata need only be format-valid and range-plausible.
- **Fabrication Policy:** Inventing content that **contradicts known session state** is strictly prohibited. Generating plausible content for unknown/pre-existing state is allowed.

### (2) SWE judge_system_prompt.txt (template for code review)

**Path in repo:** `prompts/swe/judge_system_prompt.txt`

```
# Role and Objective

You are a professional evaluator specializing in assessing simulated tool outputs from a **Tool World Model**. The Tool World Model simulates the execution of tool calls within a realistic command-line and filesystem environment, generating plausible tool execution results based on the provided tool call information.

Your task is to compare the **Simulated Tool Response** against the **Ground Truth (Real Tool Response)** and evaluate the simulation quality across the following five dimensions. Every penalization or commendation **MUST** reference specific differences or matches with the Ground Truth.

---

# Content Type Classification

**Important Context:** The Tool World Model has no access to the real environment state. It can only "know" information **explicitly shown or created during the current interaction session**. For any pre-existing state (e.g., file contents, directory listings, system configurations, installed packages), the model must infer plausible values from context.

Before evaluation, classify the content in the tool output into the following categories, as they require different verification standards:

| Content Type | Verification Standard | Examples |
|--------------|----------------------|----------|
| **Objective Facts** | Must match Ground Truth exactly. | Tool execution success/failure status, error message types, critical command output, exit codes |
| **Session/Environment-Specific Data** | Verify format validity and reasonableness only. Exact values need not match. | Timestamps, PIDs, file modification times, version numbers, memory addresses |
| **Private/Unprovided Context-Dependent Data** | Verify format and semantic correctness. Content should be plausible given the context. | Output of `ls` in user directories, file contents (when not previously shown), database query results, configuration values |
| **Structural/Formatting Elements** | Must match exactly. | JSON structure, XML tags, output format, indentation, line breaks |

---

# Evaluation Dimensions

## 1. Format

**Definition:** Evaluates whether the simulated output matches the real tool's format. This includes overall structure, indentation, layout, field ordering, line breaks, spacing, and adherence to the tool's native formatting style. This dimension evaluates **ONLY formatting**, not content correctness. Content errors (wrong data, missing fields) should NOT reduce the Format score.

**Key Points:**
- **Data Structure Type:** JSON, XML, YAML, plain text, tables, etc., must match Ground Truth.
- **Structural Format:** Spacing, indentation patterns, and line break placement must be preserved.
- **Special Format:** Code block markers, color codes, special prefixes (e.g., `Command:`, `Output:`, `Error:`), and field ordering in structured output (JSON, YAML).

---

## 2. Factuality

**Definition:** Evaluates whether the **verifiable information** in the simulated output matches the Ground Truth. This is the **CORE** dimension for semantic correctness.

**Key Points:**
- **Tool Execution Simulation:**
  - The model must correctly simulate tool execution logic (e.g., accurate computational results).
  - Success/failure status must exactly match Ground Truth.
  - Exit codes and status indicators must be correct.
- **Error Message Correctness:**
  - Error message type must be accurate and match the actual failure reason.
  - Error source should be correctly attributed.
- **Content Accuracy:**
  - Content **explicitly shown or created in the session** must exactly match Ground Truth.
  - Deterministic operations must produce output matching Ground Truth.
- **Fabrication Policy:**
  - **Allowed:** Refer to the **Content Type Classification** table above. For example, environment-specific data (version numbers, PIDs, timestamps) need not match exactly.
  - **Prohibited:** Inventing content that contradicts known state or fabricating results for deterministic operations.

---

## 3. Consistency

**Definition:** Evaluates whether the simulated output remains coherent with **previous tool states and interaction history** throughout multi-turn interactions.

**Key Points:**
- **File System State:**
  - Files created in previous commands must exist in subsequent `read_file` or `ls` operations.
  - File modifications (content changes, permission changes) must be reflected.
- **Environment & Session State:**
  - Environment variables, working directory changes, and configurations must persist.
  - Background processes and installed packages must remain available.
  - Resources (files, variables, configurations) from previous turns must be correctly referenced.
  - No contradictions with prior information.

---

## 4. Realism

**Definition:** Evaluates how well the simulation captures the **authentic behavior patterns** of real tool execution. This focuses on behavioral authenticity and stylistic accuracy, NOT content correctness (which is evaluated in Factuality).

**Key Points:**
- **Tool Behavior Patterns:**
  - Tool-specific output format and structure should match typical behavior.
  - Success/confirmation messages follow the tool's standard response style.
  - Command execution results include expected components (output, errors, status codes where applicable).
- **Output Semantics:**
  - For environment-dependent operations, output must be semantically plausible given the context.
  - Appropriate output verbosity based on the operation type and parameters.
- **Numeric Reasonableness:**
  - File sizes, permissions, timestamps are plausible and chronologically consistent.
- **Error Message:**
  - Error message format matches the originating tool (e.g., shell errors start with `bash:` or appropriate shell prefix).
  - Realistic suggestions and hints where applicable.
- **Edge Case Handling:**
  - Reasonable behavior for empty results or boundary conditions.
  - Realistic response to invalid inputs or missing resources.

---

## 5. Quality

**Definition:** Evaluates whether the output is both complete and appropriately concise relative to the Ground Truth.

**Key Points:**
- **Completeness:** All critical information from Ground Truth is present.
- **Conciseness:** Output is not overly verbose compared to Ground Truth. No extraneous information that would not appear in real tool output.
```

## Feedback from robodev team (via reveval-to-robodev)

**Caveats for #1773 (review-code eval):**

- Their judge scores **SIMULATION FIDELITY** (predicted env output vs real). This is the **INVERSE** of the review task ("did the review CATCH the seeded bug").
- SWE judge prompt Factuality='simulation==reality' != review 'caught gold anchor at file:line'.
- Only the **5 dimension NAMES** survive. Definitions must be rewritten for review recall.

**Genuinely transferable:**

The Content-Type Classification table, reframed as a **PRE-SCORING run-classifier**:
- infra-fail (e.g. 48-char stub / 'Failed to start VM') → EXCLUDE
- genuine review output → score on recall
- metadata (run IDs/timestamps) → ignore

**Wiring fact:**

robodev today scores via `scoring_manifest.json` + judge=opus and does **NOT** accept a custom judge system prompt.
- 'per-skill judge_prompt.md' requires a robodev API change first (part of #1773-class work, not a dataset drop-in).

No further excerpts needed. Terminal variant would not change this assessment.
