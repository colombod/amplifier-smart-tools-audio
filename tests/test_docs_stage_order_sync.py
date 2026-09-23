"""Guard against docs/ARCHITECTURE.md drifting from aud.plan.STAGE_ORDER.

AGENTS.md #7 is explicit: "`src/aud/plan.py::STAGE_ORDER` is the executable
form of the canonical order. If it and the contract disagree, one of them is
lying to a caller." The same argument holds for docs/ARCHITECTURE.md's own
canonical-order listing -- a reader trusting the doc gets told something
false the moment `plan.py` gains, loses, or reorders a stage and the doc
does not move with it.

This has already happened once: the doc's listing was missing `gate` and
`expand`, and drifted further out of sync when `resample` and `downmix`
were added to STAGE_ORDER. A one-time correction rots again the next time a
stage is added, so this test parses the doc's own listing -- rather than
hardcoding a second copy of the order here, which would just be a third
thing that can drift -- and fails loudly the moment the two disagree.
"""

from __future__ import annotations

import re
from pathlib import Path

from aud.plan import STAGE_ORDER

ARCHITECTURE_MD = Path(__file__).parents[1] / "docs" / "ARCHITECTURE.md"
HEADING = "## 2. Canonical stage ordering"


def _parse_documented_order() -> list[str]:
    """Extract the canonical-order array documented under `HEADING`.

    The doc states the order as a fenced code block holding a JSON-style
    array of quoted stage names, directly below `HEADING`. Pull the quoted
    names out in the order they appear -- this is parsing the document's
    own words, not a second, independently-maintained list.
    """
    text = ARCHITECTURE_MD.read_text(encoding="utf-8")
    heading_at = text.index(HEADING)
    fence_start = text.index("```", heading_at) + 3
    fence_end = text.index("```", fence_start)
    block = text[fence_start:fence_end]
    return re.findall(r'"([a-z_]+)"', block)


def test_architecture_doc_canonical_order_matches_stage_order() -> None:
    documented = _parse_documented_order()
    assert documented == STAGE_ORDER, (
        "docs/ARCHITECTURE.md's canonical stage-order listing (under "
        f"{HEADING!r}) has drifted from aud.plan.STAGE_ORDER.\n"
        f"Documented: {documented}\n"
        f"Actual:     {STAGE_ORDER}"
    )


def test_parser_actually_found_a_listing() -> None:
    """A parser that silently returns [] on a heading rename would make the
    test above pass for the wrong reason -- vacuously. Guard the guard.
    """
    documented = _parse_documented_order()
    assert len(documented) == len(STAGE_ORDER) > 0
