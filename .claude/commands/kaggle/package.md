# Kaggle Package Build Command

Build the project package and/or dependencies for Kaggle.

## Arguments
- `$ARGUMENTS` - Required: "package", "deps", or "all"
  - `package` - Build just the project wheel (fast)
  - `deps` - Build all dependency wheels (slow, rarely needed)
  - `all` - Build both

## Instructions

### Building the Package (frequent)

1. **Run the package build script**:
   ```bash
   ./scripts/build-kaggle-package.sh
   ```

   This will:
   - Build the project wheel
   - Copy to `kaggle/datasets/<project>-package/`

2. **Upload to Kaggle** (run in background to avoid context exhaustion):
   ```bash
   kaggle datasets version -p kaggle/datasets/<project>-package/ -m "Update to <version>"
   ```
   **Note**: Run this in a background task or subagent—the upload generates verbose output that can overwhelm context.

### Building Dependencies (infrequent)

Only needed when `pyproject.toml` dependencies change.

1. **Run the deps build script**:
   ```bash
   ./scripts/build-kaggle-deps.sh
   ```

   This will:
   - Compile requirements (excluding torch, which Kaggle provides)
   - Download all dependency wheels
   - Copy to `kaggle/datasets/<project>-deps/`

2. **Upload to Kaggle** (run in background to avoid context exhaustion):

   First time:
   ```bash
   kaggle datasets create -p kaggle/datasets/<project>-deps/
   ```

   Or update:
   ```bash
   kaggle datasets version -p kaggle/datasets/<project>-deps/ -m "Update to <version>"
   ```

   **Note**: Run these in a background task or subagent—large uploads generate verbose output that can overwhelm context.

## Dataset Structure

```
kaggle/datasets/
├── <project>-package/        # Package only (updates frequently)
│   ├── <project>-*.whl
│   └── dataset-metadata.json
└── <project>-deps/           # Dependencies (rarely changes)
    ├── pip-packages/
    │   ├── *.whl
    │   └── ...
    ├── requirements.txt
    └── dataset-metadata.json
```

## Report

- Package wheel size
- Number of dependency wheels (if building deps)
- Upload status

## Notes
- PyTorch/CUDA excluded (Kaggle provides them)
- Targets Linux x86_64 Python 3.12
- Version is based on timestamp + git commit hash
- Experiments must list both datasets in kernel-metadata.json
