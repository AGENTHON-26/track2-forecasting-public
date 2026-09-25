#!/usr/bin/env python3
"""Build, seal and validate `submission.json` for this track.

The descriptor is hand-writable, but three of its fields are derived and one is
self-referential, so hand-writing it is how uploads get held:

  * `descriptor_digest` is a JCS-canonical sha256 over the descriptor MINUS that field, so it
    has to be recomputed after every edit -- including a one-character digest fix;
  * `image.digest` must be the pushed amd64 manifest digest, not a local image id;
  * `team_id` comes from `qfbench2 submission alias` and never from a keyboard;
  * the schema sets `additionalProperties: false`, so one stray key is a rejection.

This script takes the two values you cannot know ahead of time, fills in everything that is
fixed for Track 2, seals the digest with the toolkit's own function, and then validates the
result twice -- once against the JSON Schema, once through `SubmissionDescriptor.from_mapping`,
which is the authority and catches cross-field problems the schema cannot see.

Neither validation consumes an upload attempt. Run it as often as you like.

    python3 tools/make_descriptor.py \
        --team-id "$(qfbench2 submission alias --team-number 42)" \
        --repository your-gh-user/t2-forecast-agent \
        --digest sha256:<64 hex from the push> \
        --out submission.json

Then: qfbench2 submission pack --descriptor submission.json --team-number 42 --out submission.zip
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# The House model identity row, copied verbatim from the shared House model guide. `version` and
# `revision` are the reported model/tokenizer snapshot (docs/NVIDIA-STACK.md). The training cutoff
# is genuinely unpublished, and the guide says to declare the literal string rather than guess a
# date -- "do not invent a training cutoff".
HOUSE_MODEL = {
    "name": "nvidia/nemotron-3-super-120b-a12b",
    "version": "rl-030326-fp8",
    "training_cutoff": "unpublished",
    "access": "api",
    "revision": "rl-030326-fp8",
}

# `api` is the only category on this track since toolkit 2.4.3 withdrew the byo-* pair; the 2.4.4
# contracts module exposes CATEGORIES == ("api", "simulator") and `pack` refuses anything else.
CATEGORY = "api"
TRACK = "forecasting"


def build(team_id: str, registry: str, repository: str, digest: str,
          phase: str, license_id: str, schema_version: str, models: list) -> dict:
    return {
        "schema_version": schema_version,
        "interface_version": "2.0",
        "competition_id": f"agenthon2026-{TRACK}-{phase}",
        "team_id": team_id,
        "track": TRACK,
        "phase": phase,
        "category": CATEGORY,
        "image": {"registry": registry, "repository": repository, "digest": digest},
        "image_access": "public",
        "models": models,
        "license": license_id,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="make_descriptor")
    p.add_argument("--team-id", required=True,
                   help="from `qfbench2 submission alias --team-number <N>`; never hand-written")
    p.add_argument("--repository", required=True,
                   help="lowercase, e.g. your-gh-user/t2-forecast-agent")
    p.add_argument("--digest", required=True,
                   help="sha256:<64 hex> of the PUSHED linux/amd64 manifest")
    p.add_argument("--registry", default="ghcr.io")
    p.add_argument("--phase", default="dev", choices=("dev", "final", "verification"))
    p.add_argument("--license", dest="license_id", default="MIT",
                   help="SPDX id for the prize-eligible source release (this repo is MIT)")
    p.add_argument("--schema-version", default="1.1.0", choices=("1.0.0", "1.1.0"))
    p.add_argument("--no-model", action="store_true",
                   help="declare models: [] -- ONLY correct if the image makes no House call")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("submission.json"))
    a = p.parse_args(argv)

    try:
        from qfbench2_common.contracts.descriptor import (
            SubmissionDescriptor, seal_descriptor_digest,
        )
    except ImportError:
        print("qfbench2-common is not installed. Install the pinned tag:\n"
              '  pip install "qfbench2-common[data] @ git+https://github.com/Agenthon-2026/'
              'Agenthon2026-public.git@v2.4.4#subdirectory=common"', file=sys.stderr)
        return 1

    models = [] if a.no_model else [HOUSE_MODEL]
    body = build(a.team_id, a.registry, a.repository, a.digest,
                 a.phase, a.license_id, a.schema_version, models)

    # Seal: JCS-canonical sha256 over the body with descriptor_digest removed. Always the
    # toolkit's own function -- a hand-rolled canonicalisation will disagree on number formats.
    sealed = seal_descriptor_digest(body)

    # Authority check. Catches what JSON Schema structurally cannot, e.g. digest self-consistency.
    try:
        SubmissionDescriptor.from_mapping(sealed)
    except Exception as exc:
        print(f"descriptor REJECTED by the contracts layer: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    # Belt and braces: the shipped JSON Schema, read from the INSTALLED toolkit so this tracks
    # whatever version is actually going to pack the zip.
    try:
        import jsonschema
        import qfbench2_common
        schema_path = (pathlib.Path(qfbench2_common.__file__).parent
                       / "schemas" / "submission.schema.json")
        jsonschema.validate(sealed, json.loads(schema_path.read_text()))
    except ImportError:
        print("note: jsonschema not installed; skipped the schema pass", file=sys.stderr)
    except Exception as exc:
        print(f"descriptor REJECTED by the JSON Schema: {exc}", file=sys.stderr)
        return 1

    a.out.write_text(json.dumps(sealed, indent=2) + "\n")
    print(f"wrote {a.out}")
    print(f"  team_id      {sealed['team_id']}")
    print(f"  image        {a.registry}/{a.repository}@{a.digest}")
    print(f"  category     {sealed['category']}   phase {sealed['phase']}   "
          f"models {len(sealed['models'])}")
    print(f"  sealed       {sealed['descriptor_digest']}")
    print(f"\nnext: qfbench2 submission pack --descriptor {a.out} "
          f"--team-number <N> --out submission.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
