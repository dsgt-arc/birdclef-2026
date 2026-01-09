---
name: Docs: Experiment
description: Propose and scaffold a new experiment with hypothesis and method.
category: Research Docs
tags: [docs, experiment, proposal]
---

**Purpose**

Create an experiment to test a hypothesis from a concept. Experiments have two phases: proposal (before running) and results (after completion).

**Guardrails**

- Every experiment must have a falsifiable hypothesis.
- Define success/rejection criteria before running the experiment.
- Keep experiments focused; test one thing at a time when possible.
- Link back to the concept that motivated the experiment.

**Steps**

1. Determine the next experiment ID:
   - List existing experiments: `ls docs/experiments/`
   - Use the next available number: `NNN-descriptive-name/`
   - Example: if 001, 002 exist, use 003

2. Create the experiment directory:
   ```bash
   mkdir -p docs/experiments/NNN-descriptive-name
   mkdir -p docs/experiments/NNN-descriptive-name/artifacts
   ```

3. Create `proposal.md` using the template from `docs/templates/experiment-proposal.md`:
   - Set Status to `proposed`
   - Write a specific, falsifiable Hypothesis
   - Add Background linking to the motivating concept
   - Define the Method:
     - Approach (high-level description)
     - Setup (data, compute, dependencies, code)
     - Procedure (numbered steps)
     - Variables (independent, dependent, controlled)
   - Specify Evaluation:
     - Metrics with descriptions
     - Baseline for comparison
     - Success Criteria (confirm if / reject if)
   - Note Limitations and assumptions

4. Get approval before running (change Status to `approved`).

5. Update Status to `in-progress` when starting execution.

6. After completion, use `/docs:result` to add results.

**Template Location**

`docs/templates/experiment-proposal.md`

**Status Lifecycle**

```
proposed → approved → in-progress → completed
    ↓         ↓           ↓
 (abandoned) (abandoned) (abandoned)
```

**Directory Structure**

```
docs/experiments/NNN-descriptive-name/
├── proposal.md      # Method (created first)
├── results.md       # Observations (added after completion)
└── artifacts/       # Logs, configs, figures, checkpoints
```

**Example**

```bash
# Experiment on learning rate scheduling
docs/experiments/001-cosine-lr-schedule/proposal.md

# Experiment on model architecture
docs/experiments/002-ast-vs-cnn-comparison/proposal.md
```
