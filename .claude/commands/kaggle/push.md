# Kaggle Push and Submit Command

Push a notebook to Kaggle and optionally submit to the competition.

## Arguments
- `$ARGUMENTS` - Experiment ID (e.g., `exp002_baseline`)

## Instructions

1. **Validate the experiment directory** exists:
   ```bash
   ls kaggle/experiments/$ARGUMENTS/
   ```
   Should contain: `submission.py`, `kernel-metadata.json`

2. **Push the notebook** to Kaggle:
   ```bash
   kaggle kernels push -p kaggle/experiments/$ARGUMENTS/
   ```

3. **Extract the kernel ID** from `kernel-metadata.json`:
   ```bash
   jq -r '.id' kaggle/experiments/$ARGUMENTS/kernel-metadata.json
   ```

4. **Wait for completion** by polling status every 30 seconds:
   ```bash
   kaggle kernels status <kernel_id>
   ```
   Keep checking until status is COMPLETE or ERROR.

5. **If ERROR**, download and show the log:
   ```bash
   kaggle kernels output <kernel_id> -p /tmp/kaggle-output/
   cat /tmp/kaggle-output/*.log
   ```
   Report the error and stop.

6. **If COMPLETE**, ask user if they want to submit to competition.

7. **To submit**, get the competition name from kernel-metadata.json and run:
   ```bash
   COMPETITION=$(jq -r '.competition_sources[0]' kaggle/experiments/$ARGUMENTS/kernel-metadata.json)
   kaggle competitions submit $COMPETITION \
     -k <kernel_id> \
     -f submission.csv \
     -v <version> \
     -m "$ARGUMENTS submission"
   ```

8. **Check leaderboard score**:
   ```bash
   kaggle competitions leaderboard $COMPETITION -d
   ```

9. **Report results** with score and rank.

## Directory Structure

```
kaggle/experiments/{exp_id}/
├── submission.py
├── kernel-metadata.json
└── README.md
```
