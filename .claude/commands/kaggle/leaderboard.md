# Kaggle Leaderboard Check

Check the current leaderboard status for the competition.

## Arguments
- `$ARGUMENTS` - Required: competition slug (e.g., `birdclef-2026`)

## Instructions

1. **Download the latest leaderboard**:
   ```bash
   kaggle competitions leaderboard $ARGUMENTS -d
   unzip -o $ARGUMENTS.zip -d /tmp/leaderboard
   ```

2. **Show top 10 entries**:
   ```bash
   head -12 /tmp/leaderboard/*.csv
   ```

3. **Report**:
   - Top scores for comparison
   - Current standings
