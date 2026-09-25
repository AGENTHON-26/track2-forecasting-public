#!/usr/bin/env bash
# Verify this staging repo has not been bypassed.
#
# Staging is EXPECTED to be ahead of public -- that is where work happens.
# The failure this catches is the opposite: a commit on public that is NOT in
# staging, which means someone edited the public repo directly instead of
# publishing from here. That silently forks the two and is how a release
# quietly loses a fix.
#
# Two tests, both against public main:
#   history  every public commit is in this repo's history.
#   content  public main's files are exactly the files of some commit on this
#            repo's main line (the first-parent chain of the commit checked).
# The history test alone cannot see a lost change: a public commit merged here
# WITHOUT its content (for example `git merge -s ours`) counts as present. The
# content test catches that, because public main then matches no commit here.
# It looks only at public main as it is now, never at past merges.
#
# Exit 0  public is in this history AND a snapshot of this main line  -- healthy
# Exit 1  public has commits missing here, or is not such a snapshot   -- bypassed
# Exit 2  could not determine                                          -- never a pass
#
# Optional overrides, for running against refs that are already fetched:
#   STAGING_REF  the commit to check                     (default: HEAD)
#   PUBLIC_REF   public main as a local ref or commit; when set, nothing is
#                fetched and no git remote is added or removed
set -uo pipefail

STAGING_REF="${STAGING_REF:-HEAD}"
PUBLIC_REF="${PUBLIC_REF:-}"

added_remote=0
# shellcheck disable=SC2329  # called by the EXIT trap below
cleanup() {
  if [[ "$added_remote" -eq 1 ]]; then
    git remote remove _public 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -z "$PUBLIC_REF" ]]; then
  PUBLIC_REPO="${PUBLIC_REPO:-}"
  if [[ -z "$PUBLIC_REPO" ]]; then
    # staging repo is <public-name>-staging; strip the suffix.
    origin="$(git config --get remote.origin.url || true)"
    name="$(basename "${origin%.git}")"
    # Without the suffix there is nothing to strip, and the line below would set
    # PUBLIC_REPO to this very repository -- which then compares equal to itself and
    # reports OK forever. A check that cannot run is not a pass (exit 2), so refuse.
    if [[ "$name" != *-staging ]]; then
      echo "RESULT: UNDETERMINED -- '$name' has no '-staging' counterpart to compare against."
      echo "Set PUBLIC_REPO explicitly to run this check here."
      exit 2
    fi
    PUBLIC_REPO="Agenthon-2026/${name%-staging}"
  fi
fi

if ! staging_sha="$(git rev-parse --verify -q "$STAGING_REF^{commit}")"; then
  echo "RESULT: UNDETERMINED -- '$STAGING_REF' is not a commit in this repository."
  exit 2
fi
staging_name="$(git rev-parse --abbrev-ref "$STAGING_REF" 2>/dev/null || true)"
echo "staging : ${staging_name:-$STAGING_REF} @ $(git rev-parse --short "$staging_sha")"

if [[ -n "$PUBLIC_REF" ]]; then
  if ! PUB="$(git rev-parse --verify -q "$PUBLIC_REF^{commit}")"; then
    echo "RESULT: UNDETERMINED -- PUBLIC_REF '$PUBLIC_REF' is not a commit in this repository."
    echo "A check that cannot run is not a pass."
    exit 2
  fi
  echo "public  : $PUBLIC_REF @ $(git rev-parse --short "$PUB") (local ref, not fetched)"
else
  echo "public  : $PUBLIC_REPO"
  # `_public` is this script's own remote name; one that already exists was left
  # behind by an interrupted run.
  git remote remove _public 2>/dev/null || true
  if ! git remote add _public "https://github.com/${PUBLIC_REPO}.git"; then
    echo "RESULT: UNDETERMINED -- could not add a remote for $PUBLIC_REPO"
    exit 2
  fi
  added_remote=1
  if ! git fetch -q _public main --depth=200 2>/dev/null \
      || ! PUB="$(git rev-parse --verify -q "refs/remotes/_public/main^{commit}")"; then
    echo "RESULT: UNDETERMINED -- could not fetch $PUBLIC_REPO"
    echo "A check that cannot run is not a pass."
    exit 2
  fi
fi

missing="$(git rev-list --count "$staging_sha..$PUB" 2>/dev/null || echo ERR)"
ahead="$(git rev-list --count "$PUB..$staging_sha" 2>/dev/null || echo ERR)"
if [[ "$missing" == "ERR" || "$ahead" == "ERR" ]]; then
  echo "RESULT: UNDETERMINED -- histories unrelated"; exit 2
fi

echo "staging is ahead of public by : $ahead commit(s)"
echo "public has commits missing here: $missing"

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "RESULT: FAIL -- public was edited directly."
  echo "These commits exist on public and not here:"
  git log --oneline "$staging_sha..$PUB" | sed 's/^/  /'
  echo
  echo "Fix: merge them into staging before publishing again, or the next"
  echo "release will silently revert them."
  exit 1
fi

# Content. Every public commit is in this history, but a merge can record a
# commit and still drop its change, so compare the files themselves. The search
# follows first parents only: after a merge of public into this repo, public main
# is itself an ancestor here, and a search of the whole history would always find it.
pub_tree="$(git rev-parse "$PUB^{tree}")"
mainline="$(git log --first-parent --format='%T %H' "$staging_sha")"
match="$(awk -v t="$pub_tree" '$1 == t { print $2; exit }' <<<"$mainline")"

if [[ -z "$match" ]]; then
  echo
  echo "RESULT: FAIL -- public main is not a snapshot of any commit on this repo's main line."
  echo "Its commits are merged here, but its files match no commit on this main line: a"
  echo "change made on public did not come back with the merge, or came back and has not"
  echo "been published from here since."
  # The newest public commit whose files ARE a snapshot of this main line is the
  # last good publish; everything public changed after it is what to look at.
  publine="$(git log --first-parent --format='%T %H' "$PUB")"
  base_pair="$( { printf '%s\n' "$mainline"; echo '--'; printf '%s\n' "$publine"; } \
    | awk '$0 == "--" { pub = 1; next }
           !pub { if (!($1 in snap)) snap[$1] = $2; next }
           ($1 in snap) { print $2, snap[$1]; exit }')"
  base="${base_pair%% *}"
  echo
  if [[ -n "$base" ]]; then
    echo "Changed on public since its last snapshot of this main line,"
    echo "public $(git rev-parse --short "$base") (= $(git rev-parse --short "${base_pair##* }") here):"
    diff_from="$base"
  else
    echo "No public commit is a snapshot of this main line; every path that differs:"
    diff_from="$staging_sha"
  fi
  git -c core.quotePath=false diff --no-renames --name-status "$diff_from" "$PUB" \
    | while IFS=$'\t' read -r status path; do
        here="$(git rev-parse -q --verify "$staging_sha:$path" 2>/dev/null || true)"
        there="$(git rev-parse -q --verify "$PUB:$path" 2>/dev/null || true)"
        if [[ "$here" == "$there" ]]; then mark="same in staging"; else mark="differs in staging"; fi
        printf '  %s %s  [%s]\n' "$status" "$path" "$mark"
      done
  echo "  (same in staging: this commit has public's version of that path."
  echo "   differs in staging: it has another version; check that public's change is in it.)"
  echo
  echo "Fix: bring those changes into staging with a real merge (not 'git merge -s ours'),"
  echo "then publish from staging again so public main is once more a snapshot of it."
  exit 1
fi

echo "public main's files are those of staging $(git log -1 --format='%h (%s)' "$match")"
echo
echo "RESULT: OK -- nothing on public is missing here."
if git diff --quiet "$match" "$staging_sha"; then
  echo "Nothing pending: this commit's files are the published files."
else
  echo "Pending release -- merged on this main line since that snapshot:"
  git log --first-parent --oneline "$match..$staging_sha" | sed 's/^/  /'
  echo "Files that differ from public main:"
  git diff --stat "$match" "$staging_sha" | sed 's/^/  /'
fi
exit 0
