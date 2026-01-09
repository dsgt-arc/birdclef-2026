---
name: Docs: Vendor
description: Document an external tool, library, or dataset.
category: Research Docs
tags: [docs, vendor]
---

**Purpose**

Create documentation for external tools, libraries, datasets, or models that are relevant to the project. Vendor docs capture how external resources fit into our pipeline.

**Guardrails**

- Only document tools/datasets actually relevant to the project.
- Include source URL and license.
- Focus on relevance to our pipeline, not comprehensive documentation.
- Keep usage examples minimal and practical.

**Steps**

1. Check if vendor doc already exists:
   ```bash
   ls docs/vendor/
   ```

2. Choose a filename using kebab-case: `<tool-name>.md`
   - Examples: `birdnet-analyzer.md`, `esc50.md`, `beats.md`

3. If the tool has a git repo we need locally, add it as a submodule:
   ```bash
   git submodule add <repo-url> vendor/<tool-name>
   ```
   - Add submodule if: we need the code/data locally (models, datasets, libraries)
   - Skip submodule if: it's just a reference, or available via pip/huggingface

4. Create the vendor doc using `docs/vendor/_TEMPLATE.md`:
   - Fill in frontmatter (name, source, license, category, pipeline_stage, relevance, status)
   - Write Overview (2-3 sentences)
   - Document Key Components if applicable
   - Explain Relevance to Pipeline
   - Add minimal Usage example
   - Note any Dependencies or gotchas
   - If submodule added, note the local path in Key Components

5. Categories:
   - `tokenization` - Audio tokenizers, codecs
   - `self-supervised` - SSL methods, pretraining
   - `pretrained-model` - Ready-to-use models
   - `dataset` - Training/evaluation datasets
   - `utility` - Tools, utilities

6. Pipeline stages:
   - `tokenization` - Converts audio to discrete tokens
   - `representation-learning` - Learns embeddings
   - `embedding` - Produces fixed vectors
   - `inference` - Final prediction
   - `data-processing` - Data preparation

**Template Location**

`docs/vendor/_TEMPLATE.md`

**Example**

```bash
# With submodule (need code/data locally)
git submodule add https://github.com/karolpiczak/ESC-50 vendor/ESC-50
# Then create docs/vendor/esc50.md

# Without submodule (pip install or just reference)
# Just create docs/vendor/perch.md
```

**Frontmatter Example**

```yaml
---
name: "ESC-50"
source: "https://github.com/karolpiczak/ESC-50"
license: "CC BY-NC 3.0"
category: "dataset"
pipeline_stage:
  - "evaluation"
relevance: "Standard audio classification benchmark"
status: "active"
---
```
