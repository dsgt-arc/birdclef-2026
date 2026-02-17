# Kaggle Summary Command

Create a clean-room summary of a vendored Kaggle notebook or dataset.

## Arguments

- `$ARGUMENTS` - The entry name to summarize, in `<competition>/<name>` format (e.g., `birdclef-2025/some-kernel`)
  - If omitted, list available vendored entries from `.kaggle-datasets` and ask which to summarize

## Instructions

1. **Find the vendored content**:
   - Read `.kaggle-datasets` to find the entry matching the argument
   - If no match, list available entries and ask the user to choose
   - Locate the vendored files at the entry's `path` (e.g., `vendor/kaggle/notebooks/birdclef-2025/some-kernel/`)

2. **Read the vendored content**:
   - For notebooks: read the `.ipynb` or `.py` file(s) in the vendor path
   - For datasets: read any README, metadata, or description files
   - Understand the approach, architecture, training, and inference pipeline

3. **Create the summary directory** if needed:
   ```bash
   mkdir -p docs/references/kaggle/<competition>/
   ```

4. **Create the summary** at `docs/references/kaggle/<competition>/<name>.md`:
   - Use the template from `docs/_templates/kaggle-summary.md`
   - Fill in the frontmatter: slug, type, competition, date_captured (today), votes
   - Write each section based on your understanding of the vendored content

5. **Clean-room principle** (CRITICAL):
   - Describe techniques, architectures, and ideas in your own words
   - NEVER copy code verbatim from the notebook
   - NEVER include code blocks that reproduce the original implementation
   - Focus on "what they did and why" not "how to reproduce their code"
   - It's fine to mention specific library names, function names, or hyperparameter values
   - Think of this as writing a paper summary, not a code review

6. **Fill in connections**:
   - Link to relevant concept docs in `docs/concepts/`
   - Link to experiments this could inform
   - Link to similar kaggle summaries if any exist

## Output Path

```
docs/references/kaggle/<competition>/<name>.md
```

## Example

For entry `birdclef-2025/birdclef-2025-ssamba`:
- Reads from: `vendor/kaggle/notebooks/birdclef-2025/birdclef-2025-ssamba/`
- Creates: `docs/references/kaggle/birdclef-2025/birdclef-2025-ssamba.md`

## Notes

- Always use `/kaggle:vendor` first to pull the content before summarizing
- The clean-room principle protects against inadvertent code copying
- Summaries are reference documents — they feed into concepts and experiments
- If the notebook is very long, focus on the most novel or impactful sections
