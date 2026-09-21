## Executive summary (read this first)

`qfbench2_track_forecasting.resolution.score_forecast_resolution` is a separate opt-in API
for private rehearsal of a fully verified `candidate-2` forecast chain. It uses the existing
canonical scorer, normalization, gates and aggregate. It does not change `score_roster`, the
ordinary C1/C2 verifier, CLI, worker, factory or live scoring path. Every returned result is
explicitly non-rankable and lists the remaining adoption/evidence gaps.

This module requires the coupled Hub `candidate-2` implementation. Existing public toolkit
tags do not provide it. Install and test the reviewed pair together; do not skip the new tests
or claim a public-tag CI run exercised this candidate. The normal scorer has no new import of
this module, so its existing entrypoints do not select this path implicitly.

The organizer supplies three signed document byte strings and immutable retained byte maps:

```python
from qfbench2_track_forecasting.resolution import score_forecast_resolution

candidate = score_forecast_resolution(
    protocol_bytes, receipt_bytes, resolution_bytes,
    organizer_trust=organizer_keys, runner_trust=runner_keys,
    receipt_trust=archive_keys, now=trusted_verifier_time,
    descriptor=descriptor_bytes, config=config_bytes,
    model_dependencies=model_bytes_by_declared_name,
    records=c2_bytes_by_unit, trees=c3_bytes_by_unit,
    forecasts=retained_forecast_member_bytes_by_unit,
    outcomes=outcome_bytes_by_unit, scales=scale_bytes_by_unit,
    input_snapshots=retained_input_member_bytes_by_unit,
    source_snapshots=source_bytes_by_digest,
)
assert candidate.rankable is False
```

Production trust is the default. Only an explicit `require_production_trust=False` permits
development-key rehearsal. No input path, caller-created verified object, or claimed digest
can substitute for actual evidence bytes. The adapter copies nested maps before verification
and consumes those same bytes. Missing, changed or inconsistent organizer evidence aborts as
an organizer fault. A validly retained but malformed participant forecast still receives the
canonical participant refusal and keeps its roster slot.

Before scoring, the full protocol/receipt/resolution verifier checks chronology, C5/image,
configuration, model dependencies, exact C2/C3 evidence, forecasts, outcomes, scales, cards,
full input snapshots and source content. The adapter additionally measures both complete
actual package source trees with the canonical Hub algorithm. Every imported Common/Track 2
module must belong to its package's file set and origin, with no linked or ambiguous tree.
Both measured digests must equal the pre-outcome protocol; they are checked again afterward.
These are source-tree identities, not wheel or production-image certification.

Fresh private temporary trees contain only the authenticated input/forecast/card/scale bytes
and a parquet encoding of the verified outcome cells. Cards use the shared parser and full
schema validator. Grids and recipe weights/joint statistics must agree. The canonical panel
and text cutoff scanners run on the committed inputs. This first adapter profile requires the
existing `panels/` and `text/` topology. All staged bytes are compared with the original verified
values before and after the canonical `score_roster` call; no retained file is rewritten.

The later C1 supplies the ordered grid and numeric scale commitment; original C2 records keep
their protocol digest. The result retains each original signed C2 digest and C3 root, plus the
verified chain. It is a `CandidateResolutionScore`, whose `diagnostic` is for organizer review;
it is not an ordinary live evaluation authorization. No public artifact is written by this API.

The candidate still does not establish live archive-clock custody, actual mounts of committed
input/config/model bytes, full worker/C7 evidence, independent first-vintage attribution, or
reproduction of the scale-generation recipe. Numeric scales are authenticated against the
signed commitment and validated by the canonical normalization loader, not claimed regenerated.
Those gaps and lack of policy adoption are always present in `acceptance_blockers`.

The synthetic tests use only the Hub's published development seed, fixture documents and
invented observations. Positive tests compute source identities from the actual imports and
execute the real gates/math. Negative controls cover immutable evidence, missing members,
signed inconsistent cards, late inputs/execution, image/source substitution, runtime origins,
and mutation during staging/scoring. No production signer, network model or paid inference is
used, and the old C1/C2 equality test continues to reject rebinding.
