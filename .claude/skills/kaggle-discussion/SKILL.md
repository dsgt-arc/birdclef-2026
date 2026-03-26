---
name: kaggle:discussion
description: Fetch a Kaggle competition discussion post and save it as a reference doc. Uses the kagglesdk search API to retrieve the discussion content (markdown + metadata). Use when the user shares a Kaggle discussion URL or asks to document a competition discussion.
argument-hint: "[discussion-url-or-search-query]"
disable-model-invocation: true
---

# Kaggle Discussion Fetcher

Fetch a Kaggle competition discussion post via the kagglesdk search API and save
it as a structured reference document.

## Arguments

- `$ARGUMENTS` - One of:
  - Kaggle discussion URL: `https://www.kaggle.com/competitions/birdclef-2026/discussion/683822`
  - Search query: `HGNetV2 baseline birdclef 2026`

## Prerequisites

- Python venv with `kagglesdk` installed: `~/clef/birdclef-2026/.venv`
- Kaggle credentials at `~/.config/kaggle/kaggle.json`

## Steps

### 1. Parse the argument

- If URL: extract the discussion/topic ID (the number at the end)
- If search query: use it directly for the search API

### 2. Fetch the discussion via kagglesdk

Use the search API to find and retrieve the discussion content:

```python
source ~/clef/birdclef-2026/.venv/bin/activate
python3 -c "
from kagglesdk import KaggleClient
from kagglesdk.search.types.search_api_service import (
    ListEntitiesRequest, ListEntitiesFilters,
    ApiSearchDiscussionsFilters,
    SearchDiscussionsSourceType, DocumentType,
)
import json

client = KaggleClient()
req = ListEntitiesRequest()
req.page_size = 5

filters = ListEntitiesFilters()
filters.query = '<SEARCH_QUERY_OR_TITLE>'
filters.document_types = [DocumentType.TOPIC]
disc_filters = ApiSearchDiscussionsFilters()
disc_filters.source_type = SearchDiscussionsSourceType.SEARCH_DISCUSSIONS_SOURCE_TYPE_COMPETITION
filters.discussion_filters = disc_filters
req.filters = filters

resp = client.search.search_api_client.list_entities(req)

for doc in resp.documents:
    full = json.loads(doc.to_json())
    disc = full.get('discussionDocument', {})
    print('=== TITLE ===')
    print(full.get('title', ''))
    print('=== AUTHOR ===')
    print(json.dumps(full.get('ownerUser', {}), indent=2))
    print('=== ID ===')
    print(full.get('id'))
    print('=== VOTES ===')
    print(full.get('votes'))
    print('=== DATE ===')
    print(full.get('createTime'))
    print('=== MARKDOWN ===')
    print(disc.get('messageMarkdown', ''))
    print('=== END ===')
"
```

### 3. If the discussion references notebooks, fetch them too

If the discussion links to Kaggle notebooks (e.g., training/inference code):

```bash
mkdir -p /tmp/kaggle-notebooks
kaggle kernels pull <owner>/<notebook-slug> -p /tmp/kaggle-notebooks/<slug>/
```

Read the downloaded .ipynb to extract technical details.

### 4. Create the reference document

Save to: `docs/references/kaggle/<competition>/discussions/<slug>.md`

Determine the slug from the discussion title (kebab-case).

Use this frontmatter format:

```markdown
---
title: "<discussion title>"
source: "<discussion URL>"
author: "<display name> (<username>, <tier>)"
date_posted: "<YYYY-MM-DD>"
date_captured: "<today's date>"
votes: <number>
related: ["other-doc.md"]
---

# <title>

[Document the key technical content, organized by topic]
```

### 5. Organize the content

Structure the document by extracting and organizing:
- Architecture details (backbone, head, pooling)
- Training configuration (loss, optimizer, scheduler, epochs, augmentation)
- Mel/spectrogram parameters
- Data pipeline (how data is loaded, splits, soundscape handling)
- Inference pipeline (how predictions are made)
- Reported scores (LB, CV, ablation results)
- Key insights and tips
- Relevance to our experiments

Include exact code snippets for key components.

## Notes

- The kagglesdk search API returns the full markdown content of the original post
- It does NOT return comments/replies — only the original post body
- Very long posts may be truncated in the search results
- Always attribute the author and link to the source
- If the discussion references notebooks, consider using `/kaggle:notebook` to analyze them
