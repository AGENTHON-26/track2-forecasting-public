# Monthly target periods

## Executive summary (read this first)

A monthly forecast needs the observation month being forecast, the value's vintage,
and the horizon key used in the submission. These are separate pieces of information.
The helper in `qfbench2_track_forecasting.horizons` resolves monthly step counts from
explicit observation-period metadata. The reference CLI and reasoning example use this mapping for declared monthly
level targets. The monthly practice inputs in this update include explicit period mappings.

## A synthetic example

Suppose a monthly input panel ends at December 2030, and the as-of date is February
14, 2031. The task specifies March and April observations at horizon keys 42 and 65.
The sampling steps from the panel's final level are three and four months. Counting
from February would miss the unpublished observations between the panel and cutoff.
Keep 42 and 65 in `forecast.parquet`; do not replace them with three and four.

A spec that already supplies explicit monthly `targets.target_dates` or
`questions[].target_date` needs no second mapping. The date labels identify observation
months in this contract; they are not publication dates. Where dates are absent,
the minimal optional addition to a spec with an existing target grid is:

```json
{
  "targets": {
    "asset_ids": ["INDEX"],
    "horizons": [42, 65],
    "observation_periods": ["2031-03", "2031-04"]
  }
}
```

For a spec expressed as individual `questions`, add `observation_period: "2031-03"`
to the existing row for its asset and horizon. Every supplied mapping must agree.
The helper rejects conflicting, incomplete, duplicated, or malformed mappings.

## Sampling and value interpretation

For a monthly level series, use the final available observation at or before the
as-of as the anchor. Count calendar-month transitions from that observation period
to the explicitly named target period. For a random walk with monthly innovation
standard deviation `sigma`, the marginal standard deviation after `s` transitions
is `sigma * sqrt(s)`. A fitted drift per monthly transition is multiplied by `s`.
The horizon integer alone does not supply `s`; do not divide it by an assumed number
of business days in a month or substitute the position in a sorted horizon list.

The observation month does not determine which vintage is scored. Retain the task's
stated first-release or fixed-snapshot convention. A monthly date label at the start
or end of a month can identify the same observation period; neither gives the release
day. Do not infer a new publication date or replace an authored target value.

Pass the selected monthly series' last observation dates, the as-of, and the supplied
card/spec to `monthly_horizon_steps`. It returns an array ordered by the supplied
asset and horizon lists. The caller is responsible for establishing that the selected
series are monthly; context panels and generic frequency labels are insufficient.
An unresolved mapping raises an actionable error. No silent horizon conversion is
available. The scorer continues to join on `(asset, horizon)`. The reference sampler draws
correlated monthly increments once per calendar month and reuses them at later
target periods. Its reasoning example uses the resulting monthly forecast spread
in its prompt, adjustment bounds and fallback. Daily sampling is unchanged.

The CLI reads `forecast_spec.json` beside `card.toml` when the card declares a
monthly target frequency. It verifies the selected target series have monthly
observations. Older declarations that call dense daily target observations monthly
produce an explicit warning and retain daily sampling when there is no explicit
monthly observation-period mapping. A conflicting mapping, mixed cadence, or
duplicated monthly observations is refused; context panels do not select cadence. Monthly inputs without an explicit period mapping refuse until the task input is
updated; code and metadata must be released together. The public practice inputs
include mappings established from their existing authored instructions and public
release metadata. This does not install metadata for organizer-controlled inputs.
The five scaffold adapters have not been connected to this monthly mapping.
