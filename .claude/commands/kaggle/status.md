# Kaggle Kernel Status Command

Check the status of the SSAMBA inference kernel on Kaggle.

## Arguments
- `$ARGUMENTS` - Optional: kernel slug override. Defaults to reading from kernel-metadata.json.

## Paths

```
EXP_DIR=user/anthony/experiments/009-birdclef-pipeline
KERNEL_DIR=$EXP_DIR/kaggle_kernel
```

## Instructions

1. **Extract the kernel ID** from `kernel-metadata.json`:
   ```bash
   KERNEL_ID=$(jq -r '.id' $KERNEL_DIR/kernel-metadata.json)
   echo "Kernel: $KERNEL_ID"
   ```
   If `$ARGUMENTS` is provided, use that as the kernel ID instead.

2. **Check kernel status**:
   ```bash
   kaggle kernels status $KERNEL_ID
   ```

3. **Report based on status**:

   - **QUEUED**: Kernel is waiting to run
   - **RUNNING**: Kernel is currently executing
   - **COMPLETE**: Download output and report runtime
     ```bash
     kaggle kernels output $KERNEL_ID -p /tmp/kaggle-output/
     ls -la /tmp/kaggle-output/
     ```
   - **ERROR**: Download and display logs
     ```bash
     kaggle kernels output $KERNEL_ID -p /tmp/kaggle-output/
     cat /tmp/kaggle-output/*.log | tail -50
     ```

4. **Show summary** including:
   - Current status
   - Runtime (if complete)
   - Output files (if complete)
   - Error details (if failed)
