---
name: Status
description: Show experiment lifecycle status across all systems (docs, code, weights, kaggle, openspec).
category: Consistency
tags: [status, overview, lifecycle]
---

**Purpose**

Provide a unified view of all experiments showing their status across documentation, code, training, Kaggle, and OpenSpec. Helps identify incomplete or orphaned experiments.

**Arguments**

- `$ARGUMENTS` - Optional experiment ID (e.g., `exp005` or `005`). If omitted, shows all experiments.

**Guardrails**

- This is a read-only status check.
- Show clear visual indicators for each system.
- Highlight gaps and missing components.

**Steps**

1. **Gather all experiment IDs** from all sources:

   a. **From docs**:
      ```bash
      ls -d docs/experiments/*/ 2>/dev/null | grep -oE '[0-9]{3}'
      ```

   b. **From code**:
      ```bash
      ls -d src/experiments/exp*/ 2>/dev/null | grep -oE 'exp[0-9]{3}'
      ```

   c. **From weights** (canonical location):
      ```bash
      ls -d data/models/exp*/ 2>/dev/null | grep -oE 'exp[0-9]{3}'
      ```

   d. **From Kaggle kernels**:
      ```bash
      ls -d kaggle/experiments/exp*/ 2>/dev/null | grep -oE 'exp[0-9]{3}'
      ```

   e. **From OpenSpec** (archived changes):
      ```bash
      ls openspec/changes/archive/ 2>/dev/null
      ```

2. **Build unified experiment list**:
   Merge all experiment IDs and deduplicate.

3. **For each experiment**, check status in each system:

   a. **Documentation** (`docs/experiments/NNN-*/`):
      - `--` = not found
      - `proposal` = only proposal.md exists
      - `complete` = both proposal.md and results.md exist
      - Read Status field from proposal.md if exists

   b. **Code** (`src/experiments/expNNN_*/`):
      - `--` = not found
      - `✓` = directory exists with required files

   c. **Weights** (`data/models/expNNN_*/`):
      - `--` = not found
      - `N/5` = partial folds trained
      - `5/5` = all folds complete

   d. **Kaggle Kernel** (`kaggle/experiments/expNNN_*/`):
      - `--` = not found
      - `ready` = kernel files exist
      - Check if dataset dependencies are uploaded

   e. **OpenSpec**:
      - `--` = no related change
      - `active` = change in progress
      - `archived` = change completed

4. **Check for orphans**:
   - Docs without code
   - Code without docs
   - Weights without code
   - Kaggle without weights

5. **Display status table**:
   ```
   Experiment Status Overview
   ==========================

   ID    | Docs      | Code | Weights | Kaggle | OpenSpec | Notes
   ------|-----------|------|---------|--------|----------|-------
   exp001| complete  | ✓    | --      | --     | archived | baseline only
   exp002| complete  | ✓    | 5/5     | ready  | archived |
   exp003| complete  | ✓    | 5/5     | ready  | active   | in progress
   exp004| complete  | ✓    | 5/5     | --     | archived | no submission
   exp005| complete  | ✓    | 5/5     | ready  | archived |

   Legend:
   - Docs: proposal | complete | --
   - Code: ✓ | --
   - Weights: N/5 | -- (in data/models/)
   - Kaggle: ready | -- (kernel exists)
   - OpenSpec: active | archived | --

   Issues Found:
   [!] exp001: No weights trained
   [!] exp004: No Kaggle kernel
   ```

6. **If specific experiment requested**, show detailed view:
   ```
   Experiment: exp005 (Pyro Causal Model)
   ======================================

   Documentation:
     Location: docs/experiments/005-pyro-causal/
     proposal.md: ✓ (Status: completed)
     results.md: ✓
     LB Score: 0.61

   Code:
     Location: src/experiments/exp005_pyro_causal/
     Files: __init__.py, module.py, model.py, config.yaml
     Contract: BiomassModule exported

   Weights:
     Location: data/models/exp005_pyro_causal/
     fold0: ✓ best.ckpt (45.2 MB)
     fold1: ✓ best.ckpt (45.1 MB)
     fold2: ✓ best.ckpt (45.3 MB)
     fold3: ✓ best.ckpt (45.0 MB)
     fold4: ✓ best.ckpt (45.2 MB)

   Kaggle:
     Kernel: kaggle/experiments/exp005_pyro_causal/
     Datasets:
       ✓ acmiyaguchi/image2biomass-package
       ✓ acmiyaguchi/exp005-pyro-causal-weights
       ✓ acmiyaguchi/dinov3-vitb16-pretrain

   OpenSpec:
     Change: add-pyro-causal-model
     Status: archived (2025-12-30)

   Lifecycle: COMPLETE
   ```

**Lifecycle Stages**

An experiment progresses through these stages:

```
1. Proposed    → docs/experiments/NNN-*/proposal.md created
2. Implemented → src/experiments/expNNN_*/ created
3. Trained     → data/models/expNNN_*/ with 5/5 folds
4. Submitted   → kaggle/experiments/expNNN_*/ pushed
5. Completed   → docs results.md added, openspec archived
```

**Status Indicators**

| System | Status | Meaning |
|--------|--------|---------|
| Docs | `--` | No documentation |
| Docs | `proposal` | Only hypothesis documented |
| Docs | `complete` | Results documented |
| Code | `--` | No implementation |
| Code | `✓` | Module exists |
| Weights | `--` | Not trained |
| Weights | `N/5` | Partially trained |
| Weights | `5/5` | Fully trained |
| Kaggle | `--` | No kernel |
| Kaggle | `ready` | Kernel prepared |
| OpenSpec | `--` | No formal change |
| OpenSpec | `active` | Change in progress |
| OpenSpec | `archived` | Change completed |

**Common Workflows**

Check what needs to be done:
```bash
# See all experiments
/status

# Focus on one
/status exp005
```

After seeing status, run specific checks:
```bash
/check:docs exp005      # Validate documentation
/check:experiment exp005  # Validate code contract
/check:weights exp005   # Check training outputs
/check:kaggle exp005    # Pre-push validation
```
