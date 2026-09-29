#!/usr/bin/env bash
# Build the linux/amd64 submission image, push it, and print the digest to put in the descriptor.
#
#   tools/build_and_push.sh <gh-user> [tag]
#
# Requires `docker login ghcr.io` first, with a classic PAT scoped write:packages.
#
# Three things here are not optional:
#
#   --platform linux/amd64   the harness pulls amd64. A plain `docker build` on Apple Silicon
#                            produces arm64, which fails every unit and wastes an upload.
#   --builder                the default buildx builders use the `docker` driver, which cannot
#                            push. This creates a docker-container builder once.
#   --provenance/--sbom off  with attestations on, buildx pushes an OCI *index* (amd64 plus an
#                            unknown/unknown attestation entry) and the tag digest is the index
#                            digest, not the plain amd64 manifest digest the descriptor wants.
#
# The digest is read from --metadata-file rather than scraped off the terminal, then cross-checked
# against `imagetools inspect`. Copying a digest by eye is how the wrong image gets submitted.
set -euo pipefail

GH_USER="${1:?usage: build_and_push.sh <gh-user> [tag]}"
IMAGE_NAME="t2-forecast-agent"
TAG="${2:-dev-$(date +%Y%m%d)-$(git rev-parse --short HEAD)}"
REPOSITORY="${GH_USER}/${IMAGE_NAME}"   # lowercase: the descriptor schema's repository pattern
REPO="ghcr.io/${REPOSITORY}"
BUILDER="qfb2"
META="$(mktemp -t qfb2-build-meta).json"

if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  echo "== creating the $BUILDER builder (docker-container driver) =="
  docker buildx create --name "$BUILDER" --driver docker-container --bootstrap
fi

echo "== building and pushing ${REPO}:${TAG} (linux/amd64) =="
docker buildx build \
  --builder "$BUILDER" \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  -t "${REPO}:${TAG}" \
  --metadata-file "$META" \
  --push .

DIGEST="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["containerimage.digest"])' "$META")"

echo
echo "== cross-checking the digest and the platform =="
docker buildx imagetools inspect "${REPO}:${TAG}"
INSPECTED="$(docker buildx imagetools inspect "${REPO}:${TAG}" --format '{{.Manifest.Digest}}')"
if [ "$DIGEST" != "$INSPECTED" ]; then
  echo "DIGEST MISMATCH: build reported $DIGEST, registry reports $INSPECTED" >&2
  exit 1
fi

echo
echo "== anonymous pullability (no credentials -- a logged-in pull proves nothing) =="
TOKEN="$(curl -fsS "https://ghcr.io/token?scope=repository:${REPOSITORY}:pull&service=ghcr.io" \
         | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')"
CODE="$(curl -o /dev/null -s -w '%{http_code}' -H "Authorization: Bearer ${TOKEN}" \
        -H 'Accept: application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json' \
        "https://ghcr.io/v2/${REPOSITORY}/manifests/${DIGEST}")"
if [ "$CODE" = "200" ]; then
  echo "anonymous pull: OK (HTTP $CODE)"
else
  echo "anonymous pull: FAILED (HTTP $CODE)" >&2
  echo "The package is still private. Make it public, then re-run:" >&2
  echo "  https://github.com/users/${GH_USER}/packages/container/${IMAGE_NAME}/settings" >&2
  exit 1
fi

cat <<EOF

================================================================
image    ${REPO}:${TAG}
digest   ${DIGEST}

next:
  python3 tools/make_descriptor.py \\
      --team-id "\$(qfbench2 submission alias --team-number <N>)" \\
      --repository ${REPOSITORY} \\
      --digest ${DIGEST} \\
      --out submission.json
  qfbench2 submission pack --descriptor submission.json --team-number <N> --out submission.zip
================================================================
EOF
