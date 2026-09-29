# QFBench 2.0 Track-2 reference submission image.
#
# Follows the shape the other tracks already use on the shared dev box: the verb is an
# executable on PATH, CMD is the verb plus --help so `docker run <img>` is self-describing,
# and the harness overrides argv with the real invocation.
#
#   docker buildx build --platform linux/amd64 -t t2-forecast-agent:dev --load .
#   docker run --rm --network=none \
#     -v <unit>:/input:ro -v <out>:/output \
#     t2-forecast-agent:dev \
#     forecast --panels /input/panels/ --text /input/text/ --asof 2024-06-28 \
#              --out /output/forecast.parquet
#
# SUBMIT linux/amd64 ONLY. The image runs on linux/arm64 too (the reference build was verified on
# GH200), but the organizer's harness pulls linux/amd64, so a plain `docker build` on an Apple
# Silicon machine produces an image that fails every unit. Always pass --platform linux/amd64.

# Python 3.13. `pyproject.toml` declares `requires-python = ">=3.13"` and CI runs 3.13; a 3.12 base
# here meant the image could not install the package it exists to run, and that contradiction sat
# unnoticed because nothing ever installed the package into the image.
FROM python:3.13-slim-bookworm

LABEL qfbench2.interface_version=2.0
LABEL qfbench2.track=forecasting
LABEL qfbench2.verb=forecast

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Pinned. The scorer's own CI was red for a week because a pinned type checker met an unpinned
# numpy; a submission image that floats its deps has the same failure mode with worse timing.
#
# `pandas` is required by `forecast_models.py`, which imports it at module scope for the
# date-aligned step frame every card's forecast is fitted from. (It used to be here for
# `f1_pipeline/m2_unit.py`, when F1 level cards went through M2; M2 left production on
# 2026-09-27 -- see the COPY block below.) It is NOT needed by
# `qfbench2_track_forecasting/cli.py` -- that package is not in this image; the reference CLI
# ignores `--text` entirely and would score the text-blind floor.
#
# `qfbench2-common` is no longer imported at run time either (the agent path is numpy + pandas +
# pyarrow + stdlib), but the line stays: `.github/workflows/ci.yml` greps THIS FILE for
# `refs/tags/vX.Y.Z.tar.gz` and fails the build with "could not find a pinned toolkit install"
# when it finds none. It also keeps the image able to self-check against the shared schemas.
#
# Same repository and tag the README installs, in the TARBALL form rather than `git+https://`.
# That is deliberate: the base image is `python:3.13-slim-bookworm`, which ships no `git`, so the
# `git+` form fails at build time with "Cannot find command 'git'". The tarball needs no VCS
# client and no extra apt layer.
RUN pip install --no-cache-dir \
        "numpy==2.1.3" \
        "pandas==2.2.3" \
        "pyarrow==18.1.0" \
        "jsonschema==4.23.0" \
        "qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.4.4.tar.gz#subdirectory=common"

WORKDIR /work
# The TEAM agent, not the text-blind reference CLI. Three modules, all resolved by bare name, so
# all three must sit on the same path:
#   forecast_agent.py   the contract and the CLI; `from forecast_models import build_draws`
#   forecast_models.py  the model -- the cumulative walk and its per-family WALK_SETTINGS. Split
#                       out of forecast_agent.py on 2026-09-26. MISSING THIS IS THE WHOLE IMAGE:
#                       forecast_agent imports it at module scope, so the agent cannot start.
#   text_signal.py      the LLM half; `from text_signal import read_text_signal`
#
# `WALK_SETTINGS` is a literal dict in forecast_models.py, so nothing under `model_baseline/` is
# needed at run time -- that directory is notebook output that chose those numbers, not input to
# them. Same for `f1_pipeline/data/`.
#
# `f1_pipeline/m2_unit.py` is NOT on the forecast path any more (M2 left production 2026-09-27;
# every card now takes the cumulative walk). It is still copied because `text_signal._model_context`
# imports it when `_MODEL_CONTEXT_ON` is enabled, and that flag is a one-line change -- leaving the
# module out would turn flipping it into a silent degradation rather than a working feature. No
# `__init__.py` is required: `f1_pipeline` resolves as an implicit namespace package under
# `PYTHONPATH=/opt`, and it pulls in no dependency that is not already installed above.
COPY forecast_agent.py forecast_models.py text_signal.py /opt/
COPY f1_pipeline/m2_unit.py /opt/f1_pipeline/m2_unit.py
ENV PYTHONPATH=/opt

# Fail the BUILD if the agent cannot import, rather than the scoring run.
#
# Each of the three is here for a different failure mode:
#
#   text_signal      `forecast_agent.read_text_signal` CATCHES any exception from this import and
#                    returns exact neutral, so an image missing it builds, runs, exits 0, passes
#                    g0-g3 and silently scores the text-blind baseline -- a wasted submission with
#                    no error visible anywhere.
#   forecast_models  the opposite: imported at module scope with no fallback, so a missing copy is
#                    an immediate crash on every unit. Loud, but only once the scoring run starts.
#   f1_pipeline      only reached behind `_MODEL_CONTEXT_ON`, but checked here so the copy above
#                    cannot rot unnoticed.
#
# The second line checks the CONTRACT rather than the import: `build_draws` is documented as
# living at `forecast_agent.build_draws` and is re-exported there from forecast_models. An import
# that succeeds while that name is gone would still fail every unit.
RUN python3 -c "import forecast_agent, forecast_models, text_signal; from f1_pipeline import m2_unit" \
 && python3 -c "import forecast_agent; assert forecast_agent.build_draws is not None"

# The verb, as an executable on PATH.
RUN printf '#!/bin/sh\nexec python3 /opt/forecast_agent.py "$@"\n' \
      > /usr/local/bin/forecast \
 && chmod +x /usr/local/bin/forecast

# Runs as a non-root user: the harness mounts /input read-only and /output writable, and nothing
# in this image needs to write anywhere else.
RUN useradd --create-home --uid 1000 runner
USER runner

CMD ["forecast", "--help"]
