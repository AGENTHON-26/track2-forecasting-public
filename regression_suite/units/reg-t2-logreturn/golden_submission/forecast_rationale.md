# Forecast rationale — reg-t2-logreturn

As of **2026-05-29**, joint distribution over SYN_MOM, SYN_HML at horizon(s)
21 business days. 500 draws.

## Anchor

Zero for every asset: the target sums log(1 + daily simple return) over the horizon. The last observed daily return belongs to the history, not to that future total.
(260 rows of overlapping daily history used for the covariance).

## Adjustments

The historical mean daily log return, multiplied by the horizon. This statistical drift uses only the supplied history at or before the as-of. No text adjustment is made.

## Scale and shape

Per-asset daily standard deviation of daily log returns, log(1 + panel value), scaled by sqrt(horizon). Gaussian
shape — deliberately not fat-tailed, since nothing here justifies a tail view.

The draws are **joint**: a single innovation vector is drawn per draw from the empirical
correlation of daily log returns across assets, so cross-asset structure is preserved
rather than independent marginals. The composite's variogram term scores that structure.

## Adjustment ledger

| asset | anchor | daily drift | centre at horizon | daily sd | sd at horizon | horizon (BD) |
|---|---|---|---|---|---|---|
| SYN_MOM | 0.0000 | 0.0007 | 0.0157 | 0.0111 | 0.0509 | 21 |
| SYN_HML | 0.0000 | 0.0007 | 0.0147 | 0.0072 | 0.0328 | 21 |

Centre = 0 + historical mean daily log return × horizon.

## What the text corpus contributed

**Nothing.** 2 document(s) were present at the text path and none was read. This is the
statistical floor a reasoning agent has to beat, not an example of using text — the whole point
of Track 2 is the gap between this and an agent that reads the corpus. A real submission would
use the documents to move the centre, skew the distribution, or widen the tails, and would say
here which document drove which adjustment and by how much.

## What would change this forecast

Any evidence at all. It currently uses none beyond the panel's own volatility.
