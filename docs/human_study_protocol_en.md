# IMPACT-Scribe Human Study Protocol

This document defines only the three comparison conditions required by the current paper-facing study, so participants are not distracted by unrelated tool functionality.

## Goal

Compare the efficiency and final annotation quality of the following three protocols under the same videos and the same starting conditions:

1. Manual annotation with another existing tool
2. IMPACT-Scribe `Scribble Only`
3. IMPACT-Scribe `Scribble + Planner`

## Fairness Principles

- Use the same set of videos across all three conditions whenever possible.
- If an initial coarse segmentation or prelabels are provided, all three conditions must start from the same initialization.
- If the external manual tool cannot import prelabels, explicitly record that condition as `from scratch` and do not mix it with `with prelabels`.
- Each participant should complete a short practice session before the timed evaluation begins.

## Condition A: Manual Annotation with Another Tool

This condition is not performed inside IMPACT-Scribe.

Recommended fields to record:

- start time and end time
- video ID
- whether initial prelabels were used
- exported annotation file path

Recommended reported metrics:

- `Boundary-F1@10 / @25 / @50`
- `edit score`
- `frame accuracy`
- completion time

## Condition B: Scribble Only

### Participant Actions

1. Open `app.py`.
2. Load the video, features, and existing annotation or prelabel.
3. Select `Study: Scribble Only` in the top `Study:` dropdown.
4. Draw boundary scribbles directly on the timeline.
5. If the proposal is unsatisfactory, add another scribble at the same location or drag the proposal boundary directly.
6. Click `Accept Boundary`.
7. Repeat until the current video is complete.

### System Behavior in This Condition

- the study mode is fixed to `Coarse`
- `Fine`-related entry points are hidden
- the `Interaction` dropdown is hidden
- the planner is disabled
- the participant must decide where the next boundary should be

## Condition C: Scribble + Planner

### Participant Actions

1. Open `app.py`.
2. Load the video, features, and existing annotation or prelabel.
3. Select `Study: Scribble + Planner` in the top `Study:` dropdown.
4. Click `Next Boundary`.
5. The system jumps to the next suggested boundary.
6. Draw a boundary scribble on the timeline; if needed, add another scribble or drag the proposal boundary.
7. Click `Accept Boundary`.
8. Click `Next Boundary` again, or continue according to the current interface flow to the next suggested boundary.
9. Repeat until the current video is complete.

### System Behavior in This Condition

- the study mode is fixed to `Coarse`
- `Fine`-related entry points are hidden
- the `Interaction` dropdown is hidden
- the planner recommends only `boundary` queries and does not mix in label review or state repair

## Boundary Semantics in the Current Study Mode

The current paper-facing study uses a single unified `boundary scribble` gesture:

- scribbling in blank space means creating a new boundary
- a short and narrow scribble centered on an existing boundary means deleting that boundary and merging the adjacent segments
- a wider scribble crossing an existing boundary means refining that boundary

All of these cases reuse the same `Accept Boundary` flow.

## Blank-Canvas Semantics

On a blank canvas, the system works as a continuous temporal partition rather than a set of isolated patches:

- the first accepted boundary fills the immediately preceding unlabeled span
- the second accepted boundary fills the next adjacent unlabeled span
- the already visited part of the timeline should not be left with gaps

In other words, blank-canvas interaction should behave like progressive forward segmentation and filling along the timeline.

## Selection and Drawing Behavior

To reduce accidental edits, the current interaction priority is:

- clicking an existing segment or marker selects it first
- a scribble only starts after the drag distance passes the threshold
- right click preserves the delete semantics for existing annotations or markers

Therefore, participants should not sketch many locations first and accept them later in a batch. They should process one boundary at a time:

1. draw or adjust the current boundary
2. inspect the proposal
3. click `Accept Boundary`
4. move on to the next boundary

## Label Assistance

The current paper-facing study remains boundary-first, but label assistance is already integrated into the workflow:

- the left panel shows recommended labels for the current segment or gap
- for `blank fill` and `remove boundary` proposals, the recommended label can be overridden before clicking `Accept Boundary`
- candidate labels come from prototype memory, imported candidates, optional text priors, and runtime confusion memory

Notes:

- in study mode, the left label panel title is shown as `Recommended Labels` / `Labels`
- this label path is only an aid and is not the primary workflow participants must learn first

## Main IMPACT-Scribe Entry Points Used in the Study

Only the following in-tool entry points should be used during the human study:

- `Study:` dropdown  
  for switching between `Standard / Scribble Only / Scribble + Planner`

- `Next Boundary`  
  used only in `Scribble + Planner` to jump to the next suggested boundary

- `Boundary Scribble` on the timeline

- `Accept Boundary` in the footer

## Recommended Logging Fields

For each session, record:

- `participant_id`
- `condition`
- `video_id`
- `start_time`
- `end_time`
- `total_elapsed_seconds`
- `interaction_count`
- `accept_count`
- `reject_count`
- `second_correction_count`
- `output_annotation_path`
- `output_scribble_sidecar_path`

IMPACT-Scribe records `study_condition` in the correction session and scribble sidecar so later analysis can group results by condition.

## Recommended Comparisons

Two primary comparisons are recommended:

1. `manual annotation with another tool` vs `Scribble Only`  
   to evaluate whether boundary-scribble interaction itself reduces effort.

2. `Scribble Only` vs `Scribble + Planner`  
   to evaluate whether the planner reduces search cost and improves return per unit time.

Do not use `manual annotation with another tool` and `Scribble + Planner` as the only direct comparison. `Scribble Only` must remain in the middle, otherwise the planner contribution cannot be isolated.
