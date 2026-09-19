"""aud.intelligence -- the provider-agnostic seam where a model turns
measurements into a mastering plan.

Nothing here is imported at module load time by `aud.lib`. `advise` and
`master` import it lazily, inside their own function bodies, so every
deterministic verb in `aud` keeps working with no provider installed or
configured -- see docs/ARCHITECTURE.md #7 and AGENTS.md #3/#5.

Modules:
    interface  The `IntelligenceBackend` Protocol, one HTTPS implementation
               per provider, and credential-based provider selection.
    advisor    Measurements in, a validated `aud.plan.Plan` out. Untrusted
               model output is parsed and validated through the same stage
               builders every other verb uses.
    prompts    System/user prompt construction for the advisor.
"""

from __future__ import annotations
