---
name: experiment:context
description: Load full context for an experiment at session start. Reads the experiment's proposal, tasks, results, README, and key referenced experiments to bootstrap the agent's understanding. Use this at the beginning of a conversation to resume work on an experiment.
argument-hint: "[experiment-id-or-path]"
---

# Experiment Context Loader

Load comprehensive context for an experiment so we can resume work effectively.
This goes beyond a simple summary — it builds the agent's working knowledge of the
experiment, its dependencies, and the current state of play.

## Arguments

- `$ARGUMENTS` - Experiment path or ID. Accepts:
  - Full path: `user/murilo/experiments/017-mel-overhaul`
  - ID with name: `017-mel-overhaul`
  - Just ID: `017`
  - If no argument, list experiments under `user/murilo/experiments/` for selection

## Steps

### 1. Resolve the experiment path

- Default user: `murilo` (check memory for user preferences)
- Glob for `user/murilo/experiments/$ARGUMENTS*` if partial match
- Verify the directory exists

### 2. Read core experiment files

Read these files (not all will exist — that's fine):

- **README.md** — Full experiment overview, results summary table, directory structure
- **proposal.md** — Hypothesis, motivation, approach, references
- **tasks.md** — Task checklist with completion status, config comparison table
- **results.md** — Detailed offline + LB results, interpretation, artifacts

Extract and present:
- Current status and what step we're on
- Results summary table (variants, val/auroc, LB scores)
- Next unchecked tasks (what to work on)
- Key config differences between variants
- Known issues or risks noted in the docs

### 3. Identify and load referenced experiments

Scan the experiment's README/proposal/tasks for references to other experiments.
Common patterns:
- `exp 010-submission-pipeline` — ONNX export and Kaggle submission infrastructure
- `exp 016-sed-architecture` — Previous SED experiment (baseline comparisons)
- References in format: `exp NNN`, `experiment NNN`, `../NNN-name`

For each referenced experiment, read its **README.md** (or proposal.md if no README)
to understand:
- What it does and how it relates to the current experiment
- Key commands/scripts the current experiment depends on
- Any shared infrastructure (e.g., submission pipeline, training scripts)

**Priority references to always check:**
- `010-submission-pipeline` — if the experiment involves Kaggle submissions, read its
  README to understand the export → upload → submit workflow
- The immediately preceding experiment (e.g., exp 016 for exp 017) — for baseline context

### 4. Check recent git activity

```bash
git log --oneline -10 -- user/murilo/experiments/<exp-dir>/
```

This shows what was worked on most recently.

### 5. Check running SLURM jobs

```bash
squeue -u $USER --format="%.8i %.30j %.8T %.10M" 2>/dev/null | grep -i "<exp-id>" || echo "No running jobs"
```

### 6. Present the context

Structure the output as:

```
## Experiment Context: NNN-name

**Status**: [current step/phase]
**Branch**: [git branch if noted]
**Best result**: [best LB score and variant]

### Results Summary
[Results table from README/results.md]

### Current Progress
[Task completion summary + next unchecked items]

### Key References
- exp 010: [one-line summary of what it provides]
- exp 016: [one-line summary of baseline]

### Recent Activity
[Last 5 git commits touching this experiment]

### Running Jobs
[Any active SLURM jobs]

### Ready to Work On
[Specific next tasks based on unchecked items and pending results]
```

## Guidelines

- **Read actual file contents**, don't just check existence — the value is in synthesis
- **Prioritize actionable context**: what's the current best score, what's running, what's next
- **Include the config comparison table** from tasks.md if it exists — this is essential
  for understanding what each variant changes
- **Don't read every referenced experiment in full** — just README/proposal for context
- **Note any pending LB results** or running jobs that affect next steps
- **Keep it concise but complete** — this replaces reading 5+ files manually
