# Kaggle Vendor Command

Vendor a Kaggle notebook or dataset into `vendor/kaggle/` and track it in `.kaggle-datasets`.

## Arguments

- `$ARGUMENTS` - One of:
  - `<slug-or-url>` - Kaggle slug (`owner/name`) or full URL to vendor
  - `sync` - Pull/update all entries in `.kaggle-datasets`
  - `list` - Show all tracked entries

Optional flags (parsed from arguments string):
- `--competition <comp>` - Competition slug (e.g., `birdclef-2025`)
- `--type notebook|dataset` - Asset type (default: notebook)

## Instructions

### Vendoring a new asset (`<slug-or-url>`)

1. **Parse the argument** to extract slug and options:
   - If it's a URL like `https://www.kaggle.com/code/owner/name`, extract `owner/name` and infer type=notebook
   - If it's a URL like `https://www.kaggle.com/datasets/owner/name`, extract `owner/name` and infer type=dataset
   - If it's a URL with `/competitions/<comp>/`, extract the competition slug
   - If it's a plain slug like `owner/name`, use it directly
   - Parse `--competition` and `--type` flags if present

2. **Determine competition**:
   - Use `--competition` value if provided
   - If inferred from URL, use that
   - Otherwise, ask the user which competition this belongs to

3. **Derive the entry name** from the slug:
   - Take the kernel/dataset name part (after `/`)
   - This becomes the directory name under the competition prefix
   - Entry key format: `<competition>/<name>` (e.g., `birdclef-2025/some-kernel`)

4. **Fetch metadata** to get votes/title:
   - For notebooks: `kaggle kernels list -s <slug> --csv`
   - For datasets: `kaggle datasets list -s <slug> --csv`
   - Extract the vote count from the CSV output

5. **Pull the content**:
   - For notebooks:
     ```bash
     mkdir -p vendor/kaggle/notebooks/<competition>/<name>
     kaggle kernels pull -p vendor/kaggle/notebooks/<competition>/<name> -m <slug>
     ```
   - For datasets:
     ```bash
     mkdir -p vendor/kaggle/datasets/<competition>/<name>
     kaggle datasets download -p vendor/kaggle/datasets/<competition>/<name> --unzip <slug>
     ```

6. **Add entry to `.kaggle-datasets`**:
   Append a new section to the manifest file:
   ```ini
   [notebook "<competition>/<name>"]
   	slug = <owner>/<name>
   	competition = <competition>
   	path = vendor/kaggle/notebooks/<competition>/<name>
   	votes = <number>
   ```
   Or for datasets:
   ```ini
   [dataset "<competition>/<name>"]
   	slug = <owner>/<name>
   	competition = <competition>
   	path = vendor/kaggle/datasets/<competition>/<name>
   	votes = <number>
   ```

7. **Report** what was added: slug, type, path, votes.

### Syncing all entries (`sync`)

1. Read `.kaggle-datasets` and parse all entries
2. For each entry, re-pull:
   - Notebooks: `kaggle kernels pull -p <path> -m <slug>`
   - Datasets: `kaggle datasets download -p <path> --unzip <slug>`
3. Report which entries were updated

### Listing entries (`list`)

1. Read `.kaggle-datasets` and parse all entries
2. Display a table with columns: Type, Competition, Name, Slug, Votes, Path
3. If no entries, report that the manifest is empty

## Manifest Format

The `.kaggle-datasets` file uses INI format mirroring `.gitmodules`:

```ini
# Kaggle asset manifest (analogous to .gitmodules)
#
# Datasets:  kaggle datasets download -p <path> --unzip <slug>
# Notebooks: kaggle kernels pull -p <path> -m <slug>

[notebook "birdclef-2025/some-kernel"]
	slug = owner/some-kernel
	competition = birdclef-2025
	path = vendor/kaggle/notebooks/birdclef-2025/some-kernel
	votes = 42

[dataset "birdclef-2025/some-data"]
	slug = owner/some-data
	competition = birdclef-2025
	path = vendor/kaggle/datasets/birdclef-2025/some-data
	votes = 10
```

## Notes

- Vendor paths use `vendor/kaggle/{notebooks,datasets}/<competition>/<name>/`
- The `.kaggle-datasets` manifest is checked into git; vendored content in `vendor/kaggle/` should be gitignored or kept small
- Use tabs for indentation in the manifest (matching `.gitmodules` convention)
- If `kaggle` CLI is not available, inform the user to install it: `pip install kaggle`
