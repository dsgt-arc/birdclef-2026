---
name: Experiment Summary
description: Get a comprehensive status summary of a target experiment for session context.
category: Research
tags: [experiment, summary, status, context]
---

**Purpose**

Produce a structured summary of a specific experiment to understand what's been done, what artifacts exist, and what to work on next. Ideal for starting a new session or resuming work.

**Usage**

```
/experiment:summary [experiment-path-or-id]
```

If no argument is provided, list available experiments for selection.

**Arguments**

- `$ARGUMENTS` - Experiment path or ID. Accepts:
  - Full path: `user/<name>/experiments/003-model-training`
  - User + ID: `<name>/003` or `<name>/003-model-training`
  - Just ID: `003` (glob across `user/*/experiments/003-*`, ask if ambiguous)
  - If ambiguous or no user specified, list matches and ask

**Steps**

1. **Resolve the experiment path**:
   - If full path provided, use directly
   - If ID provided, ask user for their name (if not known) and find the matching directory in `user/<name>/experiments/`
   - If only a number, glob for `user/*/experiments/<number>-*`
   - Verify the directory exists

2. **Read proposal.md** and extract:
   - Status field (proposed/approved/in-progress/completed/abandoned)
   - Created and Updated dates
   - Author
   - Hypothesis (1-2 sentences)
   - Method approach (brief)
   - Evaluation metrics and success criteria

3. **Read tasks.md** and extract:
   - Count all checkboxes: `- [x]` (done), `- [~]` (in-progress), `- [ ]` (pending)
   - List task group headings (`##` sections) with per-group completion
   - Identify the first 3-5 unchecked tasks (likely next work items)

4. **Read results.md** (if exists) and extract:
   - Verdict (confirmed/rejected/inconclusive)
   - Summary line
   - Key metrics table
   - Next steps listed

5. **Read design.md** (if exists) and note:
   - Overview section summary

6. **Scan artifacts/ directory**:
   ```bash
   # List artifact subdirectories and file counts
   find user/<name>/experiments/NNN-name/artifacts -type f | head -30
   # Recent files
   find user/<name>/experiments/NNN-name/artifacts -type f -printf '%T@ %p\n' | sort -rn | head -5
   ```

7. **Check for other files** in the experiment directory:
   - `pyproject.toml`, `config.yaml`, `config/`
   - `sbatch/`, `*.sbatch`
   - `src/`, `train.py`, `*.py`
   - `.env`

8. **Produce the summary**:
   ```
   ## Experiment Summary: NNN-name

   **Path**: user/<name>/experiments/NNN-name/
   **Status**: [status] | **Author**: [author]
   **Created**: [date] | **Updated**: [date]

   ### Hypothesis
   [1-2 sentences from proposal.md]

   ### Progress
   Tasks: [done]/[total] ([pct]%)
   - [x] Group 1 (N/N)
   - [~] Group 2 (N/N)
   - [ ] Group 3 (N/N)

   ### Key Results
   [Verdict + summary + metrics, or "No results recorded yet."]

   ### Artifacts
   [Directories, file counts, recent files]

   ### What to Work on Next
   [First unchecked tasks, or next steps from results, or "completed"]

   ### Files Present
   ✓ proposal.md  ✓ tasks.md  ✗ results.md  ✓ artifacts/
   ✓ pyproject.toml  ✓ config.yaml  ✓ sbatch/  ✗ design.md
   ```

**Example Output**

```
## Experiment Summary: 003-model-training

**Path**: user/<name>/experiments/003-model-training/
**Status**: completed | **Author**: <name>
**Created**: 2026-02-08

### Hypothesis
Training ConvNeXT-Tiny and ViT-Small from scratch on HSN will produce
substantially lower cmAP than pretrained variants.

### Progress
Tasks: 12/14 (86%)
- [x] Setup (6/6)
- [x] GPU Utilization Optimization (5/5)
- [~] Training Runs (0/4 — marked as running)
- [x] Analysis (3/3)

### Key Results
**Verdict**: confirmed
ConvNeXT-Tiny pretrained (cmAP=0.180) is the best performer, surpassing
ViT-Small pretrained (cmAP=0.121) with pretraining providing a 1.7-3.0x lift.

### Artifacts
  artifacts/training/ — 4 model subdirectories

### What to Work on Next
Experiment is completed. Follow-up directions from results:
- Explore domain-specific pretraining (BirdNET, bioacoustics)
- Test larger model variants with more data

### Files Present
✓ proposal.md  ✓ tasks.md  ✓ results.md  ✓ artifacts/
✓ pyproject.toml  ✓ config/  ✓ sbatch/  ✗ design.md
```
