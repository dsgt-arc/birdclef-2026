# Results: Data Analysis Scratch Space

## PaCMAP Perch Soundscape Embeddings

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| Status | initial analysis complete |
| Analysis code | `src/data_analysis/soundscapes/pacmap_embeddings.py` |
| Input embeddings | `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_soundscapes/embeddings.parquet` |
| Label filter | `/mnt/data/scratch/birdclef-2026/train_soundscapes_labels.csv` |
| Scope | 66 labeled soundscape files |
| Embedding model | Perch v2 TensorFlow inference outputs |
| Projection | standardized embeddings → PCA up to 50 dims → PaCMAP 2D |
| Output directory | `artifacts/soundscapes/pacmap/` |

### At-a-glance takeaway

Perch soundscape embeddings separate the labeled data into a few strong **acoustic-context regimes**. The plots are not pure species-identity maps: because each labeled point comes from whole 5-second soundscape windows, species prototypes are strongly shaped by site/background and co-occurring taxa.

Main regimes visible in the plots:

1. **Frog chorus / amphibian regime** — mostly S03, S13, S18, and much of S22.
2. **Insect-heavy S08 regime** — tight bottom/right island in species×site space.
3. **Bird/insect mixed S15 regime** — separate right/upper island.
4. **Isolated S19 regime** — frogs and insects from S19 move together into a distinct top cluster.
5. **Mixed S09/S23 regime** — non-amphibian, heterogeneous bird/insect/mammal/reptile context.

### Artifacts

| Plot | Description |
|---|---|
| `windows_pacmap_by_site.png` | every available 5-second Perch embedding window for labeled clips |
| `clip_pooled_pacmap_by_site.png` | one average-pooled embedding per soundscape clip/file |
| `site_pooled_pacmap_by_site.png` | one average-pooled embedding per site |
| `species_pooled_pacmap_by_class.png` | one average-pooled embedding per labeled species/taxon |
| `species_site_pooled_pacmap_by_class.png` | one average-pooled embedding per species×site, colored by class |
| `species_site_pooled_pacmap_by_site.png` | one average-pooled embedding per species×site, colored by site |

Coordinate CSVs with the same prefixes are saved alongside the plots.

### Data volume

| Level | Points |
|---|---:|
| Window-level embeddings | 792 |
| Clip-pooled embeddings | 66 |
| Site-pooled embeddings | 9 |
| Species-pooled embeddings | 75 |
| Species×site-pooled embeddings | 112 |

### Interpretation by plot

#### Window-level plot

The raw 5-second windows show site-colored chains and islands rather than compact blobs. This suggests consecutive windows within a clip trace smooth acoustic trajectories as calls/background change over time. Window-level clustering is informative, but noisy for species-level interpretation.

#### Clip-pooled plot

Average pooling by clip stabilizes the view. S22 remains broad and multi-modal, while smaller sites form more compact groups. This suggests S22 should not be treated as one homogeneous acoustic domain.

#### Site-pooled plot

Site pooling shows broad site/domain relationships:

- **S03 and S18** are nearly overlapping: small amphibian-only sites with similar average acoustic context.
- **S13 and S22** are close: both frog-heavy, though S22 is much more diverse internally.
- **S08 and S23** move away from the frog-heavy sites: insect-heavy / mixed contexts.
- **S09 and S15** occupy another non-amphibian-ish region: bird/mammal/insect mixtures.

#### Species-pooled plot

Pooling all windows for each species/taxon creates three coarse clusters:

1. amphibian/frog cluster,
2. insect/mixed nocturnal cluster,
3. bird-ish cluster.

Some birds/owls/nightjars appear near frogs. This likely means their labeled 5-second windows are dominated by frog/background context, not that their vocalizations are intrinsically frog-like.

#### Species×site-pooled plot

This is the most diagnostic plot. Splitting each species by site shows that **site context can move the same species into different clusters**.

Examples:

- **Chaco Chachalaca** appears in several distinct regions depending on site:
  - `@ S08` sits in the S08 insect island,
  - `@ S15` sits in the S15 bird/insect region,
  - `@ S22` moves into the frog-heavy region,
  - `@ S09` / `@ S23` sit in a mixed middle region.
- **Whistling Grass Frog**:
  - `@ S03` / `@ S22` are in the frog cluster,
  - `@ S19` moves to the isolated S19 cluster.
- **Undulated Tinamou**:
  - `@ S15` is in the bird/insect S15 region,
  - `@ S22` is pulled into the frog-heavy S22 context.

This supports the conclusion that label-conditioned soundscape embeddings are **species-in-context prototypes**, not clean isolated vocalization prototypes.

### Modeling implications

- Treat site/background as a major source of representation shift.
- Avoid assuming one global prototype per species from labeled soundscapes.
- For species prototype methods, prefer species×site or species×clip prototypes when enough data exists.
- S22 likely needs multi-prototype handling; a single site mean loses important structure.
- Bird labels embedded in frog/insect clusters may be context-confounded and should be audited before using as clean positives.
- Frogs, insects, and bird/insect mixtures may benefit from different postprocessing or calibration regimes.

### Next useful checks

- Cluster the species×site pooled coordinates or original embeddings with HDBSCAN/k-means and write a cluster membership table.
- For multi-site species, quantify within-species cross-site distance vs between-species same-site distance.
- Make species×clip pooled plots to see whether site clusters are driven by a few clips or consistent within site.
- Overlay time-of-night / recording timestamp to separate site effects from diel effects.
