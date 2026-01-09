# Kaggle Kernel Status Command

Check the status of a Kaggle kernel.

## Arguments
- `$ARGUMENTS` - Experiment ID (e.g., `exp002_linear_dino`)

## Instructions

1. **Extract the kernel ID** from `kernel-metadata.json`:
   ```bash
   jq -r '.id' kaggle/experiments/$ARGUMENTS/kernel-metadata.json
   ```

2. **Check kernel status**:
   ```bash
   kaggle kernels status <kernel_id>
   ```

3. **Report based on status**:

   - **QUEUED**: Kernel is waiting to run
   - **RUNNING**: Kernel is currently executing
   - **COMPLETE**: Download output and report runtime
     ```bash
     kaggle kernels output <kernel_id> -p /tmp/kaggle-output/
     ls -la /tmp/kaggle-output/
     ```
   - **ERROR**: Download and display logs
     ```bash
     kaggle kernels output <kernel_id> -p /tmp/kaggle-output/
     cat /tmp/kaggle-output/*.log
     ```

4. **Show summary** including:
   - Current status
   - Runtime (if complete)
   - Output files (if complete)
   - Error details (if failed)
