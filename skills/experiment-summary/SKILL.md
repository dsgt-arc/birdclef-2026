---
name: experiment-summary
description: Use this skill when the user wants to understand the current state of a specific experiment. This includes getting a summary of what's been done, what tasks remain, what artifacts exist, and what to work on next. Activate when the user asks about experiment status, wants to resume work on an experiment, or starts a new session and needs context on an experiment.
metadata:
  author: dsgt-arc
  version: "1.0"
---

# Experiment Summary

## Overview

Produces a structured summary of a target experiment to bootstrap agent context. This is the go-to skill when starting a new session or resuming work on an experiment. It reads all experiment files and artifacts to build a complete picture of where things stand.

## When to Use

- Starting a new session and need context on an experiment
- Resuming work after a break
- Checking what's left to do on an experiment
- Understanding what artifacts have been produced
- Before deciding next steps on an experiment

## Locating Experiments

Experiments live under user workspaces:

```
user/<name>/experiments/NNN-descriptive-name/
```

**To resolve a target experiment:**

1. If the user provides a full path, use it directly
2. If the user provides an experiment ID (e.g., `003` or `003-model-training`):
   - Ask the user for their name if not known
   - Look in `user/<name>/experiments/` for a directory matching the ID prefix
3. If the user provides just a name (e.g., `model-training`):
   - Search across all user workspaces: `user/*/experiments/*model-training*`
4. If ambiguous, list matches and ask the user to clarify

## Gathering Information

Read the following files in the experiment directory. Not all files will exist — that's expected and informative.

### 1. proposal.md (required)

Extract:
- **Status**: The current lifecycle state (`proposed`, `approved`, `in-progress`, `completed`, `abandoned`)
- **Created/Updated dates**
- **Author**
- **Hypothesis**: The core claim being tested (1-2 sentences)
- **Method summary**: Brief description of the approach
- **Evaluation criteria**: What metrics define success/failure

### 2. tasks.md (required)

Extract:
- **Total tasks**: Count all checkbox items (`- [ ]`, `- [x]`, `- [~]`)
- **Completed tasks**: Count `- [x]` items
- **In-progress tasks**: Count `- [~]` items (running/partial)
- **Remaining tasks**: Count `- [ ]` items
- **Task groups**: List the `##` section headings
- **Next unchecked tasks**: The first 3-5 unchecked items (these are likely what to work on next)

### 3. results.md (optional — only exists for completed experiments)

Extract:
- **Verdict**: `confirmed`, `rejected`, or `inconclusive`
- **Summary**: The one-line outcome
- **Key metrics**: The primary quantitative results table
- **Next steps**: Any follow-up items listed

### 4. design.md (optional)

Note its existence and extract:
- **Overview**: The high-level design summary

### 5. artifacts/ directory

Scan for what outputs exist:
- **Subdirectories**: `training/`, `logs/`, `images/`, `outputs/`, `config/`, etc.
- **File types**: `.json` (metrics), `.log` (logs), `.png`/`.jpg` (figures), `.ckpt`/`.pt` (checkpoints), `.yaml` (configs)
- **File count and total size** per subdirectory
- **Recent files**: The 5 most recently modified files (these indicate recent activity)

### 6. Other files

Check for and note:
- `pyproject.toml` — experiment dependencies
- `config.yaml` or `config/` — experiment configuration
- `sbatch/` or `*.sbatch` — SLURM job scripts
- `src/` — experiment-specific source code
- `train.py` or similar scripts — entry points
- `.env` — environment configuration

## Producing the Summary

Output a structured summary with these sections:

```
## Experiment Summary: NNN-name

**Path**: user/<name>/experiments/NNN-name/
**Status**: [status from proposal.md]
**Author**: [author] | **Created**: [date] | **Updated**: [date]

### Hypothesis
[1-2 sentence hypothesis from proposal.md]

### Progress
Tasks: [completed]/[total] ([percentage]%)
- [x] Phase 1: [description] (N/N tasks)
- [~] Phase 2: [description] (N/N tasks)
- [ ] Phase 3: [description] (N/N tasks)

### Key Results
[If results.md exists: verdict + summary + key metrics table]
[If no results.md: "No results recorded yet."]

### Artifacts
[List of artifact directories/files with counts and sizes]
[Most recent 5 files with modification dates]

### What to Work on Next
[Based on the first unchecked tasks in tasks.md]
[Or based on results.md next steps if completed]
[Or "Experiment is completed — consider follow-up experiments"]

### Files Present
[Checklist of what files exist in the experiment directory]
```

## Guidelines

- **Be concise but complete**: The summary should give enough context to resume work without reading every file, but flag important details.
- **Highlight blockers**: If tasks depend on something external (e.g., waiting for SLURM jobs), call it out.
- **Show recent activity**: Recent artifact modifications indicate what was worked on last.
- **Suggest next actions**: Based on the experiment's status and remaining tasks, recommend what to do next.
- **Read file contents, not just check existence**: The value of this skill is in synthesizing the actual content, not just listing files.
