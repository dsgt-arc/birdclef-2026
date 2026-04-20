"""Row-ID parsing for BirdCLEF 2026 test soundscapes.

Row ID format (docs/references/birdclef-2026-competition.md line 175):
    BC2026_Test_<file_ID>_<site>_<date>_<time>_<end_seconds>

Each 60s test soundscape produces 12 rows at end_seconds in {5, 10, ..., 60}.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Default set of sites present in the labeled soundscape training set.
# Matches exp 107's site index (UNION of labeled + unlabeled SC sites).
KNOWN_SITES: frozenset[str] = frozenset(f"S{i:02d}" for i in range(1, 24))

_ROW_ID_RE = re.compile(
    r"^BC2026_Test_(?P<file_id>[^_]+)_(?P<site>S\d+)_(?P<date>\d+)_(?P<time>\d+)_(?P<end>\d+)$"
)


@dataclass(frozen=True)
class ParsedRowId:
    row_id: str
    soundscape_id: str  # everything before the trailing _<end_seconds>
    file_id: str
    site: str
    date: str
    time: str
    end_seconds: int
    window_idx: int  # 1..12
    is_known_site: bool


def parse(row_id: str, known_sites: frozenset[str] | set[str] = KNOWN_SITES) -> ParsedRowId:
    """Parse a BC2026 test row_id.

    Raises `ValueError` if the row_id does not match the documented format.
    """
    m = _ROW_ID_RE.match(row_id)
    if m is None:
        raise ValueError(f"Malformed BC2026 row_id: {row_id!r}")

    file_id = m.group("file_id")
    site = m.group("site")
    date = m.group("date")
    time = m.group("time")
    end_seconds = int(m.group("end"))

    if end_seconds <= 0 or end_seconds % 5 != 0:
        raise ValueError(
            f"end_seconds must be a positive multiple of 5, got {end_seconds} in {row_id!r}"
        )

    soundscape_id = f"BC2026_Test_{file_id}_{site}_{date}_{time}"
    return ParsedRowId(
        row_id=row_id,
        soundscape_id=soundscape_id,
        file_id=file_id,
        site=site,
        date=date,
        time=time,
        end_seconds=end_seconds,
        window_idx=end_seconds // 5,
        is_known_site=site in known_sites,
    )


def site_to_index(site: str, known_sites: frozenset[str] | set[str] = KNOWN_SITES) -> int | None:
    """Map a site string to its 0-based index in the sorted KNOWN_SITES.

    Returns None if the site is out-of-distribution (triggers OOD-site fallback
    at inference time). The sorted order matches exp 107's `site_to_idx` built
    from the labeled+unlabeled union.
    """
    if site not in known_sites:
        return None
    return sorted(known_sites).index(site)
