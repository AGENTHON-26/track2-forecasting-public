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
# `pandas` is deliberately absent: it was here for `qfbench2_track_forecasting/cli.py`, the only
# module in the tree that imports it, and that package is no longer in the image. Dropping the
# COPY and the dependency go together -- keeping the package while dropping pandas would leave a
# verb-adjacent code path that dies on `import pandas` instead of simply not being there.
#
# `qfbench2-common` is no longer imported at run time either (the agent path is numpy + pyarrow +
# stdlib), but the line stays: `.github/workflows/ci.yml` greps THIS FILE for
# `refs/tags/vX.Y.Z.tar.gz` and fails the build with "could not find a pinned toolkit install"
# when it finds none. It also keeps the image able to self-check against the shared schemas.
#
# Same repository and tag the README installs, in the TARBALL form rather than `git+https://`.
# That is deliberate: the base image is `python:3.13-slim-bookworm`, which ships no `git`, so the
# `git+` form fails at build time with "Cannot find command 'git'". The tarball needs no VCS
# client and no extra apt layer.
RUN pip install --no-cache-dir \
        "numpy==2.1.3" \
        "pyarrow==18.1.0" \
        "jsonschema==4.23.0" \
        "qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.4.4.tar.gz#subdirectory=common"

WORKDIR /work
# The TEAM agent, not the text-blind reference CLI. `forecast_agent.py` does a bare
# `from text_signal import read_text_signal`, so both files must sit on the same path.
COPY forecast_agent.py text_signal.py /opt/
ENV PYTHONPATH=/opt

# Fail the BUILD if the agent cannot import, rather than the scoring run.
#
# `forecast_agent.read_text_signal` catches any exception from that import and returns exact
# neutral adjustments. So an image missing `text_signal.py` builds, runs, exits 0, passes g0-g3
# and silently scores as the text-blind baseline -- the one failure mode that wastes a submission
# with no error visible anywhere. This line turns it into a build failure.
RUN python3 -c "import forecast_agent, text_signal"

# The verb, as an executable on PATH.
RUN printf '#!/bin/sh\nexec python3 /opt/forecast_agent.py "$@"\n' \
      > /usr/local/bin/forecast \
 && chmod +x /usr/local/bin/forecast

# Runs as a non-root user: the harness mounts /input read-only and /output writable, and nothing
# in this image needs to write anywhere else.
RUN useradd --create-home --uid 1000 runner
USER runner

CMD ["forecast", "--help"]
