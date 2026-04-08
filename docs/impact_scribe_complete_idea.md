# IMPACT-Scribe Complete Idea

## Purpose

This document explains the full research idea behind the current `IMPACT_AS` codebase and the proposed next-stage method.

The goal is to make the project understandable to another AI or engineer without requiring them to reverse-engineer the whole repository first.

This is not just a GUI/tool description.
The intended paper framing is:

- an interactive annotation method for temporal action segmentation
- a new supervision form based on temporal boundary scribbles
- a query planning system that decides what the user should correct next
- a structured correction-learning framework that improves both annotation efficiency and downstream segmentation models

Companion implementation documents:

- `docs/impact_scribe_implementation_spec.md`
  - baseline implementation roadmap and landed prototype structure

- `docs/impact_scribe_closure_plan.md`
  - closure status and offline handoff for evaluation/export completion work

Suggested paper title:

- `IMPACT-Scribe: Scribble-Centered Query Planning and Structured Correction Learning for Interactive Action Segmentation`

---

## 1. High-Level Idea

### Core problem

Dense temporal action segmentation annotation is expensive because annotators usually need to:

- inspect long videos
- identify approximate boundaries
- refine exact boundaries
- assign labels to segments
- keep consistency across the whole video
- optionally keep consistency across multiple synchronized views and procedure states

This is inefficient because the human is forced to structure the whole video manually.

### Proposed shift

Instead of asking the annotator to explicitly segment the entire video, we reduce the task to repeated local decisions:

- the system proposes suspicious regions
- the annotator scribbles over a local time interval where a boundary may exist
- the local model refines the exact boundary and left/right action identity
- the confirmed correction is propagated to the whole sequence
- the correction history is used to improve future proposals and future training

The key intuition is:

- global dense segmentation is hard for humans
- local boundary judgment is much easier
- if we accumulate enough local confirmations, we can reconstruct high-quality global segmentation

---

## 2. What Already Exists in the Codebase

The current codebase already contains several parts of the future method.

### 2.1 Application entry and host

- `app.py`
  - bootstraps Qt
  - loads feature defaults
  - creates `MainWindow`
  - optionally enables operation logging

- `ui/main_window.py`
  - hosts the main task switcher
  - currently supports:
    - Action Segmentation
    - Assembly State (PSR/ASR/ASD)
  - wires the main `ActionWindow` and the PSR panel together

### 2.2 Main action annotation workspace

- `ui/action_window.py`
  - this is the real core of the project
  - it already contains:
    - video playback
    - multiview synchronization
    - timeline interaction
    - auto-label import
    - assisted review queue
    - segment embedding candidates
    - boundary energy estimation
    - correction sessions
    - validation logging
    - PSR/ASR/ASD state reasoning and conflict analysis

### 2.3 Timeline and visual interaction substrate

- `ui/timeline.py`
  - contains the timeline widgets
  - supports:
    - current-frame tracking
    - segment selection
    - combined row interaction
    - interaction point rendering
    - edit masks
    - boundary flashing
    - segment cut overlays

This file is the natural place to add the future scribble canvas interaction.

### 2.4 Logging and correction memory

- `utils/op_logger.py`
  - `OperationLogger`
  - stores timestamped GUI/user events

- `core/action_corrections.py`
  - `CorrectionBuffer`
  - groups multiple low-level edits into a higher-level correction session

These two files already provide the seed for correction-driven learning.

### 2.5 Procedure/state reasoning

- `core/procedure_trace.py`
  - `analyze_trace_conflicts`
  - identifies mismatches between action episodes and state transitions

- `core/psr_state.py`
  - derives procedure-state sequences from action labels and rules

This is the basis for structured constraints.

### 2.6 Feature extraction and model-facing tools

- `tools/feature_extractors.py`
  - backbone-agnostic feature extraction
  - currently supports:
    - DINOv2
    - ResNet50

- `tools/asot_full_infer_adapter.py`
  - optional external pre-label inference for current-video feature directories

- `tools/boundary_eval.py`
  - evaluates boundary quality

This means the codebase already has enough structure to support:

- a fast local model
- a global segmentation model
- boundary-centered evaluation

---

## 3. Current Code Logic That Matters for the Idea

This section explains the current implementation logic in conceptual terms.

### 3.1 Hover preview already acts like local temporal inspection

Relevant code:

- `ui/action_window.py:_flush_timeline_hover_preview`
- `ui/action_window.py:_on_timeline_hover_frame`
- `ui/timeline.py:set_current_frame`

Current behavior:

- when the user hovers over the timeline, the system asks the active video player to seek in preview mode
- if multiview synchronization is enabled, other views also update
- alignment can be:
  - absolute frame
  - relative offset from active view

Why this matters:

- the future scribble method does not need a new interaction paradigm from scratch
- the existing hover-preview mechanism already gives local temporal inspection around suspected boundaries
- the proposed scribble canvas is an extension of this behavior, not a replacement

### 3.2 The system already computes boundary evidence

Relevant code:

- `ui/action_window.py:_resolve_features_dir_for_snap`
- `ui/action_window.py:_ensure_boundary_snap_cache`
- `ui/action_window.py:_boundary_energy_at_frame`

Current behavior:

- the code locates a `features.npy` directory for the current video
- it caches either:
  - features
  - logits
- it maps feature indices back to frame indices
- it computes a boundary energy score at a frame using:
  - frame-to-frame logit change, or
  - cosine distance between adjacent feature vectors

Why this matters:

- the future query planner already has a local boundary signal
- the future scribble localizer can use this energy to propose or refine candidate boundary positions
- boundary energy can be one term inside query utility

### 3.3 The system already builds a review queue from uncertainty-like logic

Relevant code:

- `ui/action_window.py:_build_assisted_points_from_store`
- `ui/action_window.py:_boundary_query_score`
- `ui/action_window.py:_label_query_score`
- `ui/action_window.py:_label_candidates_with_source`

Current behavior:

- current annotations are converted into ordered segments
- for each adjacent pair of segments, the system may create a boundary review point
- for each segment, the system may create a label review point
- each point gets:
  - a `query_score`
  - component terms such as uncertainty/confusion/energy/training utility
- points are sorted by `query_score`

Why this matters:

- the repository already contains the seed of a query planner
- it is currently heuristic and local
- the paper method upgrades this from a heuristic review queue into a formal budget-aware query planning system

### 3.4 The system already supports model suggestions from two sources

Relevant code:

- `ui/action_window.py:_label_candidates_with_source`
- `ui/action_window.py:_segment_embedding_for_span`
- `ui/action_window.py:_embedding_topk_candidates`
- `ui/action_window.py:_apply_autolabel_predictions`

Current behavior:

- label candidates can come from:
  - imported model predictions
  - segment embedding similarity against label prototypes
- a segment embedding is computed from frame features within a segment span
- the system can import auto-label predictions and store them as a prelabel source

Why this matters:

- the future method can use a fast local model for scribble-window refinement
- the current code already distinguishes between:
  - external model suggestions
  - embedding-based similarity suggestions

### 3.5 Manual segmentation mode already exists conceptually

Relevant code:

- `ui/action_window.py:enter_extra_mode`
- `ui/action_window.py:exit_extra_mode`
- `ui/action_window.py:_align_labels_to_manual_segments`

Current behavior:

- there is an interaction mode where the user marks spans with a special interaction label
- after leaving the mode, the system aligns labels inside those spans using prelabels or current labels

Why this matters:

- this is very close to the future scribble idea
- the difference is:
  - current interaction is span-based
  - future interaction should be boundary-centered and brush-based
- this means the proposed method is an evolution of an existing concept already in the code

### 3.6 Multiview synchronization already exists

Relevant code:

- `ui/action_window.py:_multiview_sync_active`
- related sync/mask code in `ui/action_window.py`

Current behavior:

- multiple views can be selected for synchronized editing/playback
- the timeline can enforce edit masks so synchronized views remain consistent where needed

Why this matters:

- the future method can use cross-view disagreement as a query term
- a confirmed scribble in one view can constrain labels/boundaries in aligned views

### 3.7 Correction sessions already exist as structured units

Relevant code:

- `core/action_corrections.py`
- `ui/action_window.py:_begin_correction_session`
- `ui/action_window.py:_commit_correction_session`

Current behavior:

- a correction session records:
  - type
  - metadata
  - number of steps
  - committed records
- the system can summarize a local editing episode as a meaningful correction rather than as isolated clicks

Why this matters:

- the future method can learn from correction sessions
- this is much better than learning from raw GUI events only

### 3.8 Operation logs already exist

Relevant code:

- `utils/op_logger.py`
- `app.py`

Current behavior:

- the app can write structured CSV logs of user operations

Why this matters:

- these logs can supervise:
  - human-cost estimation
  - correction-efficiency estimation
  - query utility estimation

### 3.9 Structured procedure-state conflicts already exist

Relevant code:

- `ui/action_window.py:_psr_recompute_cache`
- `core/procedure_trace.py:analyze_trace_conflicts`

Current behavior:

- action segments are mapped into procedural events
- state sequences are derived
- conflicts such as:
  - early commit
  - missing commit
  are detected
- a repair patch is suggested

Why this matters:

- this is exactly the type of structured prior that a strong interactive annotation method should exploit
- future query planning can prioritize regions that violate action-state consistency

---

## 4. Proposed Next-Stage Method

The proposed full method has four main pillars:

1. temporal boundary scribble interaction
2. query planning
3. structured constraints
4. correction learning

These should be understood as one integrated system, not four separate features.

---

## 5. New Interaction Primitive: Temporal Boundary Scribble

### Goal

Replace exact full-video manual segmentation with local boundary-oriented supervision.

### Basic user behavior

The user interacts on a timeline canvas:

- hover preview lets the user inspect nearby frames
- the user sees a suspicious region
- instead of placing an exact boundary or dragging a full segment, the user scribbles over a local time interval

### Scribble types

The cleanest formulation uses three brush types:

- `left scribble`
  - marks frames that should belong to the action before the boundary

- `right scribble`
  - marks frames that should belong to the action after the boundary

- `uncertain scribble`
  - marks the region where the true boundary may lie

### Why this is useful

The annotator no longer needs to solve:

- the whole temporal structure of the video

Instead the annotator only solves:

- a local boundary disambiguation problem

This is cognitively easier and can be handled by a lighter local model.

---

## 6. Full System Formulation

### Inputs

- video frames `x_1 ... x_T`
- optional multiple synchronized views
- optional prelabel predictions
- optional procedural rules/state definitions
- current interaction history

### Outputs

- dense action segmentation over the whole video
- corrected local boundary decisions
- improved future query policy
- improved background global segmentation model

### State maintained by the system

The system should conceptually maintain:

- `Y_global`
  - current full-video segmentation hypothesis

- `Q`
  - current candidate query set

- `H`
  - correction history

- `M_local`
  - local scribble-conditioned boundary refiner

- `M_global`
  - background global action segmentation model

- `C_struct`
  - structured constraints from:
    - confirmed boundaries
    - multiview consistency
    - state/procedure consistency

---

## 7. Module A: Scribble-Conditioned Local Refiner

### Purpose

Given a small window and user scribbles, predict:

- the exact boundary location
- the left-side action label
- the right-side action label
- the confidence of the local refinement

### Input

- local temporal window `W=[a,b]`
- local features `F(W)`
- scribble masks:
  - `M_left`
  - `M_right`
  - `M_uncertain`

### Output

- `p_b(t | W, S)`
  - boundary probability over frames in the window

- `p_L(y | W, S)`
  - distribution over left action label

- `p_R(y | W, S)`
  - distribution over right action label

- `c(W,S)`
  - confidence score

### Model choice

This can be lightweight.
It does not need to be a large full-sequence model.

Reason:

- the input is only a local temporal interval
- the scribble already narrows the search space

Possible implementations:

- temporal 1D conv + scribble mask channels
- tiny transformer over local features
- MLP over pooled left/right/uncertain features plus per-frame boundary head

### Relationship to existing code

This module fits naturally on top of:

- feature loading from `tools/feature_extractors.py`
- boundary energy logic already in `ui/action_window.py`
- prelabel suggestions already imported by `_apply_autolabel_predictions`

---

## 8. Module B: Query Planning

### Purpose

Choose the next most valuable question to ask the annotator.

### Query types

- `q_b`: ask for a boundary scribble in a suspicious region
- `q_l`: ask to confirm or change a segment label
- `q_s`: ask to confirm or repair a procedure-state inconsistency

### Utility function

For each candidate query `q`, define:

`U(q) = alpha * U_uncert + beta * U_disagree + gamma * U_view + delta * U_state + eta * U_train + zeta * U_history`

Choose:

`q* = argmax_q U(q) / (C_hat(q) + tau)`

### Meaning of each term

- `U_uncert`
  - model uncertainty
  - label margin
  - boundary entropy

- `U_disagree`
  - disagreement between:
    - coarse model
    - local model
    - prototype-based suggestions
    - global segmentation

- `U_view`
  - disagreement across synchronized views

- `U_state`
  - violation of procedure-state constraints

- `U_train`
  - expected gain for downstream training if this query is resolved

- `U_history`
  - learned prior from past corrections

- `C_hat(q)`
  - estimated human effort for this query

### Relationship to existing code

Current heuristic seeds already exist in:

- `_build_assisted_points_from_store`
- `_boundary_query_score`
- `_label_query_score`
- `_current_query_policy_snapshot`

The paper version should formalize and learn these values instead of keeping them mostly heuristic.

---

## 9. Module C: Structured Constraints

### Purpose

After a local boundary is confirmed, use it to improve the whole-sequence segmentation instead of only patching one tiny region.

### Constraint sources

- confirmed scribble boundary
- confirmed left/right action labels
- multiview synchronization
- procedure-state consistency
- no-gap or transition constraints where relevant

### Constraint types

- hard constraints
  - confirmed boundary must exist
  - confirmed left/right labels cannot be violated in the local region

- soft constraints
  - boundary likely lies inside the scribble interval
  - neighboring views should agree
  - state transitions should remain valid

### Global update

The global segmentation should be recomputed using constrained decoding.

Possible formulations:

- constrained dynamic programming
- semi-Markov CRF
- boundary-constrained sequence decoding

### Relationship to existing code

This is where the current PSR/ASR/ASD machinery becomes scientifically useful:

- `_psr_recompute_cache`
- `analyze_trace_conflicts`

The current code already detects structured conflicts.
The future method should use those conflicts as:

- query priorities
- global decoding penalties

---

## 10. Module D: Correction Learning

### Purpose

Use human corrections as training signal.

### Why this matters

Without this module, the system is just an interactive tool.

With this module, the system becomes a learning method:

- the more the user corrects
- the better the system becomes

### What to learn from corrections

1. `prototype memory`
- update label prototypes from confirmed segments

2. `query gain predictor`
- estimate which query types/regions actually improve quality most

3. `human cost predictor`
- estimate which query types are cheap or expensive for annotators

4. `local calibration`
- refine local boundary/label confidence based on accepted and rejected suggestions

### Relationship to existing code

- `CorrectionBuffer` already groups edits into meaningful sessions
- `OperationLogger` already stores low-level event traces

These should become the data source for learning:

- correction efficiency
- query cost
- correction success

---

## 11. End-to-End Pipeline

This is the complete intended interaction loop.

### Step 1. Initialize coarse segmentation

Use current prelabel infrastructure or a global model to obtain:

- coarse segments
- candidate boundaries
- label candidates

### Step 2. Build candidate queries

For each suspicious area, create candidate queries:

- boundary scribble query
- label query
- state query

### Step 3. Select next query

Use query planning to pick the highest-value query under estimated human cost.

### Step 4. User scribbles locally

The user:

- hovers on the timeline
- inspects frame previews
- draws left/right/uncertain temporal scribbles

### Step 5. Local model refines the region

The scribble-conditioned local model predicts:

- exact boundary
- left label
- right label

### Step 6. User confirms or edits

The user accepts or adjusts the proposal.

### Step 7. Apply structured update

The confirmed result is propagated globally using:

- temporal consistency
- multiview consistency
- state/procedure constraints

### Step 8. Log correction

Record:

- query type
- scribble width
- number of edit steps
- accepted/refined output
- local quality gain
- estimated global quality gain

### Step 9. Learn from corrections

Update:

- prototypes
- query scoring
- human-cost model
- local calibrators

### Step 10. Train or refresh background global model

When enough confirmed local windows are accumulated:

- convert them into high-confidence dense supervision
- retrain or fine-tune the global action segmentation model

---

## 12. Training Strategy

### Stage A: Synthetic scribble pretraining

Because there is no real scribble dataset at the beginning, generate synthetic scribbles from dense annotations:

- sample a GT boundary
- generate a noisy uncertain interval around it
- optionally generate left/right brushes around the interval

Train the local refiner using this synthetic supervision.

### Stage B: Interactive correction adaptation

After real use begins:

- collect real correction sessions
- fine-tune the local model and query planner

### Stage C: Global model training from confirmed windows

Convert confirmed local corrections into supervision for the full-sequence model.

This is the main mechanism by which local scribble annotation improves downstream action segmentation.

---

## 13. Loss Design

This is one reasonable paper-ready formulation.

### 13.1 Boundary localization loss

Let `I+` be the uncertain scribble interval containing the true boundary.

`L_boundary = -log(sum_{t in I+} p_b(t | W,S))`

Optionally add penalties for confident boundary mass outside the interval.

### 13.2 Left/right label loss

`L_side = CE(p_L, y_left) + CE(p_R, y_right)`

### 13.3 Structured global consistency loss

Use confirmed windows and constrained decoding outputs to supervise the global segmentation.

Call this:

`L_struct`

### 13.4 Query learning loss

Learn a ranking/regression objective so that high-value queries receive larger scores.

Call this:

`L_query`

### 13.5 Total objective

`L = lambda1 * L_boundary + lambda2 * L_side + lambda3 * L_struct + lambda4 * L_query`

---

## 14. Why This Is Better Than a Pure GUI Paper

If the work is framed only as:

- a better timeline
- a better canvas
- a scribble UI

then it will likely be judged as engineering.

If the work is framed as:

- a new local temporal supervision signal
- a query-planning framework
- a structured correction propagation method
- a correction-driven learning system

then it becomes a computer-vision method paper.

That is the correct framing.

---

## 15. Why This Is Better Than Only “Fast Model + Scribble”

If the method is only:

- user scribbles a region
- a fast classifier or recognizer picks the local answer

then the idea is still useful, but too weak as a full method paper.

It lacks:

- a principled query mechanism
- a global propagation mechanism
- a learning-from-corrections mechanism
- a strong bridge to downstream segmentation training

The full version fixes all of these.

---

## 16. Mapping the Proposed Method to Files

This section tells another AI where to work in the repository.

### Existing files to reuse

- `app.py`
  - entry point only

- `ui/main_window.py`
  - task host

- `ui/action_window.py`
  - main place for:
    - query planner
    - local model invocation
    - correction sessions
    - state-aware global updates

- `ui/timeline.py`
  - main place for:
    - scribble canvas
    - brush interaction
    - scribble visualization
    - brush overlays on the timeline

- `tools/feature_extractors.py`
  - feature backbone access

- `core/action_corrections.py`
  - correction session abstraction

- `utils/op_logger.py`
  - event log persistence

- `core/procedure_trace.py`
  - structured constraints and conflict signals

### New code that likely needs to be added

- `core/temporal_scribble.py`
  - data structures for temporal scribbles

- `core/query_planner.py`
  - unified query scoring and cost model

- `core/local_boundary_refiner.py`
  - scribble-conditioned local model

- `core/structured_decode.py`
  - hard/soft constrained global update

- `tools/simulate_scribbles.py`
  - synthetic scribble generation from GT annotations

- `tools/train_local_refiner.py`
  - standalone training entry for the local model

---

## 17. Minimal Viable Research Prototype

If another AI needs to implement the smallest paper-valid version, the order should be:

1. add temporal scribble interaction to the timeline
2. represent scribbles as intervals plus brush type
3. use current boundary energy to rank candidate windows
4. add a simple local refiner over feature windows
5. patch global segmentation inside confirmed windows
6. log correction sessions with scribble metadata
7. train a basic query score regressor from logged sessions

This is the smallest version that already has research meaning.

---

## 18. Stronger Version for a Full Paper

The stronger full-method version adds:

- multiview disagreement terms
- procedure-state conflict prioritization
- constrained global decoding
- correction-cost prediction
- background global model fine-tuning from confirmed windows

This is the version that should be targeted for a strong paper.

---

## 19. Recommended Evaluation

### Annotation efficiency

Measure:

- Boundary-F1 vs number of interactions
- Edit / F1@10 / F1@25 / F1@50 vs time budget

### Human study

Compare:

- click boundary annotation
- drag segment annotation
- temporal scribble annotation

Measure:

- completion time
- final quality
- subjective ease

### Downstream model value

Use the generated annotations to train a global action segmentation model and compare:

- same budget, different annotation protocol

This is critical.

Without downstream gains, the method is much easier to dismiss as only an interface contribution.

---

## 20. One-Sentence Summary for Another AI

`IMPACT-Scribe` is an interactive temporal action segmentation method that converts dense full-video annotation into a sequence of local temporal boundary scribble corrections, uses query planning to decide what to ask next, uses structured constraints to propagate confirmed local edits to the whole sequence, and uses logged correction sessions to continually improve both future interaction efficiency and downstream global segmentation training.

---

## 21. Best-Practice Revision for This Project

This section records the current best-practice version of the idea for a lightweight, low-latency desktop annotation tool.

The key principle is:

- keep the interaction loop light
- keep the online model path tiny
- reserve heavier learning for asynchronous or offline updates

### 21.1 Interaction primitive

The recommended default interaction is:

- expose `uncertain scribble` by default
- expose `left scribble` and `right scribble` only in a refine mode

Rationale:

- most users should first express only: "the boundary is roughly here"
- users should not be forced to provide a unique precise answer too early
- annotator ambiguity is often informative and should be modeled as a soft supervisory region rather than discarded

The project should not treat frame-accurate boundary dragging as the primary interaction mode.

### 21.2 Scribble representation

The recommended representation is interval-mask based, not free-curve based.

Use:

- `M_uncertain`
- `M_left`
- `M_right`
- optional previous-proposal or previous-logits channel

The model should learn from masks/channels derived from scribbles rather than from the raw stroke trajectory itself.

### 21.3 Scribble simulation

Synthetic scribble generation should include:

- short uncertain intervals centered near true boundaries
- left/right side intervals
- noisy intervals with offset, width perturbation, and broken spans
- false intervals near incorrect boundary candidates
- random continuous intervals for robustness

The goal is to prevent the local refiner from overfitting to perfect GT-derived scribbles.

### 21.4 Local refiner best practice

The online local model should follow this recipe:

- frozen offline backbone features
- tiny local head only
- local window over a limited frame span
- boundary energy used as one feature, not the only decision rule

The project should avoid placing heavy models in the live interaction loop, including:

- full RGB re-encoding during interaction
- diffusion-based refinement
- large multi-stage refinement stacks

### 21.5 Query planning best practice

The recommended query planner is two-stage:

1. choose query type
2. choose query location

Supported query types:

- boundary scribble
- label confirm/change
- state repair

Supported query terms:

- uncertainty
- disagreement
- multiview conflict
- state conflict
- propagation gain
- human cost

Candidate-set label selection is preferred over exposing the full taxonomy at every label decision.

### 21.6 Uncertainty and calibration

For this tool, uncertainty must be lightweight.

Preferred approach:

- small calibration or uncertainty head
- sparse evaluation on candidate windows only

Avoid:

- deep ensembles in the live path
- MC-dropout style heavy repeated inference in the live path

The uncertainty estimate is mainly used for:

- query scoring
- proposal confidence
- abstain or escalate behavior

### 21.7 Structured global update

The recommended global update should be lightweight:

- confirmed windows become hard anchors
- state, procedure, and multiview constraints become soft penalties
- constrained DP or Viterbi-style decoding updates the whole sequence

The project should avoid pushing heavy global neural re-optimization into every interaction step.

### 21.8 Multiview usage

Multiview should be used primarily as:

- a query signal
- a soft propagation cue

For the first version, multiview should not become the main heavy training target.

### 21.9 Correction learning best practice

Correction learning should be layered by cost:

- immediate online update:
  - prototype memory
  - confidence calibration
  - human cost predictor

- periodic lightweight update:
  - tiny local head
  - adapter
  - LoRA-like parameter block

- offline update:
  - global segmentation model fine-tuning

This layered update schedule is much more appropriate for an interactive desktop tool than full online retraining.

### 21.10 Unknown / Other / Background handling

The UI should always preserve an escape route for:

- `Unknown`
- `Other`
- `Background`

Users should not be forced to assign every ambiguous region to a known action class.

### 21.11 Training best practice

The preferred training order remains:

1. synthetic scribble pretraining
2. real correction adaptation
3. confirmed-window distillation into the global model

For the first strong implementation, the loss stack should stay light:

- interval-based boundary loss
- left/right label loss
- confidence calibration loss
- lightweight structural loss

Query learning can start with a simple regressor before moving to a heavier ranking model.

### 21.12 Evaluation best practice

Interactive evaluation must not report only the final oracle metric.

The minimum recommended evaluation includes:

- budget or interaction curves
- synthetic scribble evaluation
- real human scribble evaluation
- acceptance rate
- override rate
- second-correction rate
- equal-time downstream training comparison

This is essential for presenting the method as a human-centered intelligent system rather than a pure GUI enhancement.

---

## 22. What Is Best Practice vs. What Is Project-Specific Innovation

Another AI should not confuse community-backed engineering practice with project-specific research ideas.

### 22.1 Safe best-practice choices

These are the parts that can be treated as strong default design decisions for this project:

- temporal scribble masks with `uncertain` as the default interaction
- frozen backbone plus tiny local head
- two-stage query policy
- lightweight uncertainty and calibration
- constrained DP or Viterbi global update
- prototype memory
- all-local, cached, low-latency interaction path
- both synthetic and real-human interactive evaluation

### 22.2 Project-specific but high-value research ideas

These are not generic standard parts, but they are especially suitable for this project and can increase paper distinctiveness:

- Expected Propagation Gain
- interaction-episode cost modeling
- second-correction and trust-friction style human-centered metrics
- explicit `state repair` query bucket in the query gate

These should be presented as project innovations, not as universal standard components.

### 22.3 Things to avoid in the first strong version

The project should avoid the following in the first high-quality implementation:

- heavy transformers or diffusion modules in the live interaction path
- full-model online fine-tuning after each correction
- forcing users to repeatedly search across a large label taxonomy
- making multiview joint training the main technical focus of version one

---

## 23. Final Operational Recipe

If another AI needs the most concise operational recipe for the recommended version, use this:

- precompute features
- encode temporal scribbles as interval masks
- run a tiny local boundary-and-side head
- use a two-stage cost-aware query policy
- update the whole sequence with lightweight constrained decoding
- update prototypes, calibration, and cost estimates online
- evaluate under equal human-time budgets, not only final oracle metrics

---

## 24. PSR/ASR-Integrated Mainline

The recommended way to integrate PSR/ASR into the main story is:

- keep interactive action segmentation as the primary task
- use PSR/ASR as structured procedural supervision
- let state inconsistency influence what to query, how to decode globally, and how to measure correction value

This means the project should not be described as:

- one tool for Action Segmentation
- plus another independent tool for PSR/ASR/ASD

Instead, it should be described as:

- an action-state co-consistent interactive annotation framework

In that framing:

- action boundaries remain the main annotation object
- PSR/ASR provides structured priors and repair signals
- human corrections improve both action quality and procedural consistency

Suggested integrated paper title:

- `IMPACT-Scribe: Interactive Action Segmentation with Procedural State Consistency`

Suggested one-sentence positioning:

- `IMPACT-Scribe is a human-in-the-loop action segmentation framework in which temporal boundary scribbles provide local supervision and PSR/ASR consistency provides global procedural structure.`

---

## 25. Framework Diagram Text Version

The integrated framework can be described as the following pipeline.

### Block A. Coarse action proposal

Inputs:

- video frames
- offline features
- optional imported prelabels

Outputs:

- coarse action segments
- candidate boundaries
- candidate labels

### Block B. Procedural state inference

Inputs:

- current action segmentation hypothesis
- PSR/ASR rules
- procedure components

Outputs:

- state sequence
- action-state conflicts
- repair candidates

This block already corresponds to the existing PSR cache and trace-conflict logic.

### Block C. Query-type gate

The system first decides which query family is most valuable:

- `boundary scribble`
- `label confirm/change`
- `state repair`

The gate uses:

- uncertainty
- disagreement
- multiview inconsistency
- state conflict
- expected propagation gain
- estimated human cost

### Block D. Local interaction and refinement

If the chosen query type is `boundary scribble`:

- the user draws an uncertain scribble, with optional left/right refinement
- the local refiner predicts:
  - boundary distribution
  - left label
  - right label
  - confidence

If the chosen query type is `label`:

- the user confirms or changes the current label within a candidate set

If the chosen query type is `state repair`:

- the user confirms or corrects a state inconsistency suggested by the PSR/ASR logic

### Block E. Structured global update

After local confirmation:

- confirmed boundaries act as hard anchors
- PSR/ASR consistency acts as a soft penalty
- multiview consistency acts as a soft penalty
- constrained decoding updates the whole sequence

### Block F. Correction learning

Every confirmed interaction updates:

- prototype memory
- confidence calibration
- query cost estimate
- query utility estimate

Periodic updates then improve:

- the tiny local head
- the background global segmentation model

---

## 26. Joint Objective with PSR/ASR

The integrated objective should reflect both action quality and procedural consistency.

### 26.1 Action segmentation loss

Let the current global action prediction be `Y_action`.

Use standard framewise or segmentwise action supervision:

`L_action`

This can be implemented using:

- framewise cross-entropy on confirmed or high-confidence spans
- or segment-level supervision, depending on the chosen global model

### 26.2 Boundary scribble loss

For a boundary scribble with uncertain interval `I+`:

`L_boundary = -log(sum_{t in I+} p_b(t | W,S))`

This keeps the supervision interval-based rather than forcing a single exact annotated frame at all times.

### 26.3 Side-label loss

For the local refiner:

`L_side = CE(p_L, y_left) + CE(p_R, y_right)`

### 26.4 State loss

Let `Y_state` be the procedural state sequence derived or predicted from the current action hypothesis.

Use:

`L_state`

to encourage agreement with:

- confirmed state repairs
- PSR/ASR component state labels
- state spans implied by accepted action-state mappings

### 26.5 Consistency loss

The integrated consistency term penalizes disagreement between action dynamics and procedural structure:

`L_consistency`

This term should increase when:

- an action episode implies a state transition that is missing
- a state transition appears too early
- aligned multiview evidence strongly disagrees
- confirmed local anchors are violated by the global decode

### 26.6 Total objective

The recommended joint objective is:

`L_total = lambda_a * L_action + lambda_b * L_boundary + lambda_s * L_side + lambda_p * L_state + lambda_c * L_consistency + lambda_q * L_query`

Interpretation:

- `L_action` improves the dense action sequence
- `L_boundary` improves local boundary localization from scribbles
- `L_side` improves left/right action discrimination around the boundary
- `L_state` improves PSR/ASR state supervision
- `L_consistency` couples action predictions with procedural structure
- `L_query` improves future interaction efficiency

For a first implementation, `L_state` and `L_consistency` can be kept lightweight:

- `L_state` from accepted state repairs only
- `L_consistency` from a small number of rule-based penalties

This is enough to make the integration scientifically meaningful without making the live path heavy.

---

## 27. Final Three Contributions

The final paper contribution set should be stated in a compact, disciplined way.

### Contribution 1

We propose a new interactive temporal supervision primitive, `temporal boundary scribble`, which converts dense full-video annotation into local boundary-centered corrective acts with low interaction cost.

### Contribution 2

We introduce a query-planned and procedurally structured human-in-the-loop framework that jointly selects boundary, label, and state queries and propagates confirmed local edits through lightweight constrained decoding.

### Contribution 3

We show that correction-driven learning, combining prototype updates, calibration, and procedural consistency, improves both annotation efficiency and the quality of downstream action segmentation models under equal human-time budgets.

---

## 28. Final Practical Framing

If another AI or future author needs the shortest correct framing, use this:

- the main task is interactive action segmentation
- PSR/ASR is the structured procedural layer, not a separate side feature
- the interaction primitive is temporal boundary scribble
- the algorithmic core is query planning plus constrained propagation
- the learning core is correction-driven adaptation
