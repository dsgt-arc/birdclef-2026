---
name: Status
description: Show high-level experiment/documentation status for this repository.
category: Consistency
tags: [status, overview, lifecycle]
---

**Purpose**

Provide a read-only overview of experiment documentation and repo state using the current repository layout.

**Usage**

```text
/status [optional user-or-experiment-filter]
```

**Steps**

1. Inspect `docs/STATUS.md` for the project-level summary.
2. List experiment directories under `user/*/experiments/`.
3. For each matching experiment, check whether these files exist:
   - `README.md`
   - `proposal.md`
   - `tasks.md`
   - `results.md`
4. Summarize status from the first matching status/verdict lines in those files when present.
5. Highlight gaps:
   - experiment with proposal but no results,
   - experiment with results but unchecked tasks,
   - experiment with README/proposal references to missing files.
6. Present a concise table.

**Example output**

```text
| Experiment | Proposal | Tasks | Results | Notes |
|---|---|---|---|---|
| user/murilo/experiments/017-mel-overhaul | yes | yes | yes | complete |
| user/anthony/experiments/104-wavtokenizer-retrieval | yes | yes | yes | negative result |
```

Prefer `docs/STATUS.md` for final project state; use this command for consistency checks across experiment folders.
