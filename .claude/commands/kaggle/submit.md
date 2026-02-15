# Kaggle Submission Command

Submit a kernel to the BirdCLEF competition.

## Arguments
- `$ARGUMENTS` - Optional: kernel version number to submit (e.g., "3"). If omitted, submits latest version.

## Paths

```
EXP_DIR=user/anthony/experiments/009-birdclef-pipeline
KERNEL_DIR=$EXP_DIR/kaggle_kernel
```

## Instructions

1. **Extract the kernel ID and competition** from `kernel-metadata.json`:
   ```bash
   KERNEL_ID=$(jq -r '.id' $KERNEL_DIR/kernel-metadata.json)
   COMPETITION=$(jq -r '.competition_sources[0]' $KERNEL_DIR/kernel-metadata.json)
   echo "Kernel ID: $KERNEL_ID"
   echo "Competition: $COMPETITION"
   ```

2. **Check kernel status** before submitting:
   ```bash
   kaggle kernels status $KERNEL_ID
   ```
   The kernel must be COMPLETE before submitting.

3. If status is COMPLETE, **submit kernel to competition**:
   ```bash
   kaggle competitions submit $COMPETITION -k $KERNEL_ID -f submission.csv -v $ARGUMENTS -m "SSAMBA submission"
   ```
   Note: This is a code competition — we submit the kernel output, not a CSV file.
   The `-v` flag requires the numeric version (e.g., 3), not "Version 3".
   If `$ARGUMENTS` is empty, check the Kaggle kernel page for the latest version number.

4. **Check submission status** (may take a few minutes):
   ```bash
   kaggle competitions submissions $COMPETITION | head -10
   ```

5. **Report the results** including:
   - Kernel status and version
   - Submission success/failure
   - Public leaderboard score (once available)

## Notes
- Internet is disabled in competition notebooks
- Kernel must output `submission.csv` to `/kaggle/working/`
- The kernel runs on CPU (no GPU) with a 120-minute time limit
