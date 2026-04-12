# AGENTS.md

## Project mission

This repository is an **on-device native mobile RAG research prototype**.

Current stack assumptions:

- **LLM runtime:** `llama.cpp` with QNN integration
- **Hardware targets:** mobile **CPU + GPU + NPU**
- **Storage / metadata:** SQLite
- **Dense retrieval:** Faiss
- **Platform goal:** Android-first, native-first, privacy-preserving, offline-friendly

The near-term research target is a **mobile systems paper** around:

- **budgeted retrieval-graph instantiation**
- **self-evolving multi-representation indexes**
- **resource-aware execution on heterogeneous mobile SoCs**

Working name for the upgraded idea:

- **MorphRAG**
- **Quality-Constrained Retrieval-Graph Instantiation on Heterogeneous Mobile SoCs**

## Core research position to preserve

This repo must preserve the following paper positioning:

1. **We are not building just another fixed mobile RAG pipeline.**
2. **We are not only scheduling existing stages across CPU/GPU/NPU.**
3. **We are deciding which retrieval graph should be instantiated at all, under mobile budgets.**
4. **Retrieval representation itself is a runtime decision.**
5. **SQLite is the source of truth; Faiss is an accelerator, not the whole system.**
6. **Evidence sufficiency is a control target, not only final F1.**
7. **Dense-only, static-tiered, and adaptive baselines must remain runnable.**

## Innovation boundary that must remain explicit

When implementing new code, preserve the distinctions below. Do **not** blur them.

### Compared with HeRo / HedraRAG

- Those systems mainly optimize **how to execute a given heterogeneous / agentic RAG graph**.
- This project must emphasize **which retrieval graph should be instantiated in the first place**.
- Do not reduce this work to only xPU placement, stage scheduling, or concurrency tuning.

### Compared with METIS / per-query tuning systems

- Those systems mainly tune parameters such as chunk count or synthesis settings.
- This project must expose **retrieval modes / retrieval representations / candidate depth** as first-class runtime knobs.

### Compared with plain mobile RAG systems

- This project must treat retrieval as a **multi-representation runtime**:
  - SQLite metadata / FTS
  - semantic hash prefilter
  - Faiss dense rerank
  - optional reranker
  - optional context compression
  - optional callback / second-hop

### Compared with static hash-RAG ideas

- This project must support **online-evolving data** and **hot/cold promotion-demotion**.
- The system should not assume a fixed static corpus.

## What success looks like

Every meaningful change should improve at least one of the following **without silently regressing the others**:

- end-to-end latency
- peak memory / RSS
- energy / thermal behavior
- retrieval quality
- answer quality
- source traceability / citation quality
- robustness of the mobile runtime
- maintainability and reproducibility of experiments

For this repo, **system quality and research clarity matter more than novelty in code style**.

## Upgraded system abstraction to implement

The system should evolve toward a **two-level runtime**.

### Outer controller: retrieval-graph instantiation

Given:

- query features
- retrieved-score features
- evidence coverage / evidence sufficiency features
- latency / memory / thermal / energy budget
- CPU/GPU/NPU availability
- index state (hot/cold, dense available or not)

The outer controller chooses:

- retrieval graph
- candidate depth
- whether to escalate to reranking / compression / callback
- whether to remain on a cheap graph or upgrade to a more expensive graph

Typical candidate graphs:

- `G0`: metadata / SQLite FTS only
- `G1`: FTS + semantic hash prefilter
- `G2`: G1 + Faiss dense rerank
- `G3`: G2 + neural reranker
- `G4`: G3 + context compression
- `G5`: G4 + callback / second-hop retrieval

### Inner runtime: xPU execution

Given the selected graph:

- place operators on CPU / GPU / NPU
- respect bandwidth, memory, and thermal budget
- preserve observability and reproducibility

The inner runtime may reuse existing scheduling / profiling infrastructure, but the **paper novelty must remain centered on graph instantiation and evolving indexes**.

## Preferred index evolution model

Treat the retrieval substrate as a **self-evolving multi-representation index**.

Each document / chunk may have:

- SQLite metadata
- SQLite FTS text
- semantic hash code
- optional dense vector in Faiss
- optional rerank/cache features
- provenance and evidence-span metadata

Preferred state model:

- **cold partitions:** metadata + FTS + hash only
- **warm partitions:** metadata + FTS + hash + selected dense vectors
- **hot partitions:** metadata + FTS + hash + dense + optional cached rerank features

Support:

- promotion
- demotion
- rebuild logging
- reproducible state snapshots for experiments

Do not assume every item must always keep a dense representation resident.

## Hard constraints

1. **Do not replace `llama.cpp` / QNN unless explicitly asked.**
2. **Do not remove SQLite or Faiss.** New layers should build on top of them.
3. **Do not introduce cloud dependencies** for the main runtime path.
4. **Do not assume server-class memory or storage.**
5. **Do not move hot-path logic to Python.** Hot path belongs in C++/JNI/native code.
6. **Do not break offline execution.**
7. **Do not silently change schemas, prompts, retrieval settings, or evaluation settings.**
8. **Do not optimize only average latency.** Always check tail latency, memory, and fallback quality.
9. **Do not delete baselines.**
10. **Do not hide retrieval/controller logic inside prompts when it should live in code.**

## Preferred architecture

Prefer a structure close to this, but adapt to the actual repo layout instead of inventing directories:

- `app/` or Android module: UI / orchestration / settings / experiments
- `native/` or `cpp/`: JNI bindings and native core
- `runtime/`: llama.cpp / QNN interaction
- `storage/`: SQLite schema, migrations, FTS, metadata store, state snapshots
- `retrieval/`
  - hash generation / binary code generation
  - Hamming-distance prefilter
  - Faiss dense rerank
  - optional reranker / context reducer
  - callback / second-hop support
- `controller/`
  - query feature extraction
  - evidence sufficiency features
  - budget estimation
  - graph / mode selection
  - hot/cold promotion-demotion policy
- `evaluation/`
  - benchmark harness
  - logging / traces
  - replay tools
  - CSV/JSON export
- `docs/`
  - feature notes
  - architecture notes
  - schema changes
  - experiment playbooks

If the repo already has a different layout, **follow the existing structure** and document where each responsibility lives.

## Baselines that must remain runnable

Keep these baselines easy to enable with flags:

1. **Dense-only baseline**
   - SQLite metadata filter (optional)
   - Faiss top-k
   - reader model answers directly

2. **Static tiered retrieval baseline**
   - fixed `FTS/hash -> Faiss` pipeline
   - no adaptive graph switching

3. **Adaptive retrieval-graph system**
   - graph selection per query
   - candidate-depth adaptation
   - optional rerank / compression / callback steps

4. **Optional xPU-only scheduling baseline**
   - same retrieval graph
   - different execution placement only
   - useful to show that graph instantiation matters beyond scheduling

Never remove the simplest baseline just because the new method is better.

## Data and storage principles

SQLite is the canonical store. Treat it as the **source of truth** for:

- chunk metadata
- raw text / snippet pointers
- timestamps
- app / document provenance
- evidence spans / citation spans
- experiment logs
- retrieval traces
- schema versioning
- semantic hash codes
- partition state (cold / warm / hot)
- promotion / demotion history

Faiss should be treated as an **accelerator for dense search**, not the only knowledge layer.

If semantic hashing is added, prefer storing hash codes and related metadata in SQLite, with a native in-memory cache only when measurement proves it helps.

## Quality target to preserve

Do **not** define success only by final answer F1.

The controller should be allowed to optimize for **evidence sufficiency under budget**.
Useful sufficiency features include:

- score margin / score sharpness
- evidence-slot coverage
- citation span availability
- unresolved entity / date / constraint count
- answer confidence or entropy proxy
- fallback / escalation history

When quality is uncertain, the system should escalate to a richer graph or fall back to a safer baseline.

## Performance rules

When touching the hot path:

- avoid unnecessary heap allocation
- avoid repeated string copies
- avoid blocking I/O during query execution
- avoid JNI chatter in tight loops
- batch native transitions when possible
- reuse buffers and intermediate objects when safe
- instrument before and after optimizing

When adding concurrency:

- prefer explicit ownership and shutdown semantics
- document which thread owns which resources
- do not introduce races around SQLite handles, Faiss indexes, or model contexts
- fail closed and fall back to a simpler path on runtime uncertainty

## Evaluation rules

Because small reader models can hide system improvements, always separate:

### Retrieval metrics

- Recall@k
- nDCG@k
- hit rate of gold evidence / cited span
- candidate-set size after each stage
- hash prefilter reduction ratio
- graph-selection distribution
- promotion / demotion statistics

### Generation metrics

- EM / F1 when applicable
- citation faithfulness
- answer-source consistency
- structured answer parse success
- evidence-backed answer rate

### System metrics

- P50 / P95 end-to-end latency
- TTFT if available
- retrieval latency breakdown
- prefill / decode latency breakdown
- per-operator latency on CPU/GPU/NPU when measurable
- peak RSS / memory
- energy or battery proxy if measurable
- thermal throttling events if measurable
- fallback frequency and fallback reasons

Do not report only final F1. If final answer quality is weak because of the reader model, the retriever/system analysis still matters.

## Experiment discipline

For any non-trivial change:

1. keep a flag to disable it
2. add logs for before/after comparison
3. run at least one regression benchmark
4. record the device, model, quantization, context length, and database size
5. note whether the result changes answer quality, retrieval quality, or only runtime
6. record which graph was instantiated and why
7. save enough information for trace replay when feasible

If you change prompts or answer formatting, call that out explicitly because it affects F1/EM.

## Mandatory documentation requirement

**Any code change that modifies functionality is incomplete unless the key implementation points are documented in Markdown.**

This is a hard requirement.

For every meaningful feature change, bug fix affecting behavior, schema change, controller update, retrieval-path update, or evaluation change:

- update an existing document **or**
- add a new document under `docs/`

Preferred locations:

- `docs/features/<feature-name>.md`
- `docs/changes/YYYY-MM-DD-<short-name>.md`
- `docs/architecture/<component-name>.md`
- `docs/evaluation/<benchmark-or-protocol>.md`

### Minimum documentation contents for each functional change

The doc must clearly describe:

1. **Purpose / motivation**
2. **What behavior changed**
3. **Key implementation points**
4. **Main files / modules touched**
5. **Runtime path / execution flow**
6. **Config flags / thresholds / defaults**
7. **Fallback behavior**
8. **Schema or storage changes**
9. **Metrics / logs added**
10. **How to test / reproduce**
11. **Known limitations / TODOs**

### Documentation style requirement

- Keep docs concise but explicit.
- Prefer Markdown.
- Use diagrams or flow bullets when helpful.
- Do not leave behavior changes explained only inside code comments.
- If a feature changes runtime decisions, document the decision logic.
- If a feature changes metrics, document how the metrics are computed.

### Commit/task completion rule

A task is **not complete** unless:

- code is updated,
- tests or benchmarks are updated when needed,
- and the corresponding documentation for key implementation points is updated.

## Coding rules for Codex

When working on a task:

1. **Inspect the repository first.**
   - Do not assume filenames.
   - Do not invent interfaces if an equivalent one already exists.

2. **Make the smallest change that preserves research clarity.**
   - Prefer incremental patches over rewrites.
   - Keep existing APIs stable unless the task explicitly requires refactoring.

3. **Explain the implementation boundary in code comments and docs.**
   - What is baseline logic?
   - What is adaptive logic?
   - What metrics are recorded?
   - What decisions are heuristic vs. learned?

4. **Add observability with the feature.**
   - counters
   - timing spans
   - selected graph / retrieval mode
   - candidate counts per stage
   - fallback reason
   - evidence sufficiency features when relevant

5. **Prefer deterministic behavior in evaluation mode.**
   - stable seeds
   - stable sorting / tie-breaking
   - explicit config snapshots

6. **Document schema changes and migration steps.**

7. **When adding a new feature, also add or update a doc that explains the key implementation points.**

## What not to do

- Do not replace SQLite with a server DB.
- Do not add a heavyweight orchestration framework unless requested.
- Do not hide important logic inside prompts if it belongs in retrieval/controller code.
- Do not hard-code device-specific assumptions without guards.
- Do not optimize by deleting instrumentation.
- Do not mix benchmark code with production runtime without clear compile/runtime flags.
- Do not silently degrade citation/source tracking.
- Do not turn the project into only a scheduling paper.
- Do not turn the project into only a model-comparison benchmark.

## Preferred implementation order for the upgraded idea

Unless instructed otherwise, implement in this order:

### Phase 1: measurement and plumbing

- add timing hooks across retrieval / rerank / reader
- log memory and candidate counts
- expose feature flags for retrieval graphs
- add docs describing the instrumentation and logging points

### Phase 2: semantic hash layer

- add hash-code storage to SQLite
- add native Hamming-distance prefilter
- add a static `FTS/hash -> Faiss` pipeline
- document schema and execution flow

### Phase 3: evidence sufficiency features

- add evidence coverage features
- add score-margin / confidence features
- add trace logging for graph escalation decisions
- document the feature definitions

### Phase 4: outer controller

- add graph-selection policy
- add budget-aware decision logic
- log selected graph and fallback reasons
- document the decision policy and thresholds

### Phase 5: evolving index state

- add hot/cold promotion-demotion logic
- persist index-state metadata in SQLite
- support reproducible snapshots when possible
- document lifecycle and state transitions

### Phase 6: inner xPU integration

- map selected graph operators to CPU/GPU/NPU
- preserve observability and fallback paths
- document the placement assumptions and constraints

### Phase 7: paper-grade evaluation

- automate benchmark runs
- emit CSV/JSON summaries
- support ablations and replay
- document the benchmark protocol and reporting format

## When changing prompts or model usage

Keep prompts short, explicit, and benchmark-friendly.
Prefer structured outputs for evaluation, e.g. JSON with fields like:

- `answer`
- `citations`
- `confidence`

If prompts influence graph selection, evidence sufficiency, or fallback behavior, document that explicitly.

## Final rule

Preserve the paper story:

**This project is about selecting and instantiating the cheapest sufficient retrieval graph on a heterogeneous mobile SoC, with an evolving multi-representation index and reproducible evidence-aware evaluation.**

Any implementation that weakens this story should be avoided or clearly isolated as a baseline.

## 默认设备约束

- 端侧设备固定为通过 `adb` 连接的设备：`fd8657d6`。
- 任何涉及端侧运行、部署、调试、性能测试、日志抓取或验证的操作，默认都应显式面向该设备，例如优先使用 `adb -s fd8657d6 ...`。
- 在开始任何端侧相关操作前，先检查设备状态：

```bash
adb devices
```

- 如果 `fd8657d6` 没有出现在 `device` 状态中，必须立即停止，不要继续执行，不要自行切换到其他设备，不要改用模拟器，也不要假设结果有效。
- 遇到设备不在线的情况时，代理应明确告知用户：当前必须等待用户重新连接设备，然后再继续。
