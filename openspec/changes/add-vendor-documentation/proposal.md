# Change: Add Vendor Repository Documentation

## Why

The project vendors 14 external repositories covering audio codecs, self-supervised learning, bioacoustics models, and utilities. Without documentation, it's difficult to understand:

1. What each repository provides
2. How it's relevant to the BirdCLEF audio classification task
3. Which components to use for tokenization, representation learning, or inference

## What Changes

- Create `docs/vendor/` directory with structured documentation for each vendored repository
- Establish a YAML frontmatter template capturing key metadata
- Document each repository with:
  - Purpose and relevance to our pipeline
  - Key components and entry points
  - Dependencies and compatibility notes
  - Usage examples where applicable

## Impact

- Affected specs: New documentation capability
- Affected code: None (documentation only)
- New files: `docs/vendor/README.md`, `docs/vendor/<repo>.md` for each vendor
