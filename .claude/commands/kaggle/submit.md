# Kaggle Submission Command

Submit a kernel to a Kaggle competition.

## Arguments
- `$ARGUMENTS` - Experiment ID (e.g., `exp002_baseline`)

## Instructions

1. **Extract the kernel ID and competition** from `kernel-metadata.json`:
   ```bash
   KERNEL_ID=$(jq -r '.id' kaggle/experiments/$ARGUMENTS/kernel-metadata.json)
   COMPETITION=$(jq -r '.competition_sources[0]' kaggle/experiments/$ARGUMENTS/kernel-metadata.json)
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
   # For code competitions, specify: kernel, output file, and version number
   kaggle competitions submit $COMPETITION -k $KERNEL_ID -f submission.csv -v <version_number> -m "$ARGUMENTS submission"
   ```
   Note: This is a code competition - we submit the kernel output, not a CSV file.
   The `-v` flag requires the numeric version (e.g., 7), not "Version 7".
   Check the Kaggle kernel page to find the current version number.

4. **Check submission status** (may take a few minutes):
   ```bash
   uv run --with kaggle python3 -c "
   from kaggle.api.kaggle_api_extended import KaggleApi
   api = KaggleApi()
   api.authenticate()
   subs = api.competition_submissions('$COMPETITION')
   for s in subs:
       print(f'{s.date} | {s.status} | {s.public_score} | {s.description}')
   "
   ```

5. **Check leaderboard** for the submission score:
   ```bash
   kaggle competitions leaderboard $COMPETITION -s | head -20
   ```
   Note: Leaderboard only shows top entries. Use step 4 to see your own submissions.

6. **Report the results** including:
   - Kernel status and version
   - Submission success/failure
   - Leaderboard score and rank
   - Comparison to expected score from `kaggle/experiments/$ARGUMENTS/README.md`

## Notes
- Internet is disabled in competition notebooks
- Kernel must output `submission.csv` to `/kaggle/working/`
