#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/deploy-pages.sh [options]

Deploys a GitHub Pages branch generated from a subdirectory using git subtree.

Options:
  -p, --prefix DIR            Source directory to publish (default: web)
  -s, --source-branch BRANCH  Branch to build from (default: main)
  -t, --target-branch BRANCH  Branch to push for Pages (default: gh-pages)
  -r, --remote NAME           Git remote to push to (default: origin)
      --no-push               Build target branch locally, skip push
  -h, --help                  Show this help
EOF
}

prefix="web"
source_branch="main"
target_branch="gh-pages"
remote_name="origin"
no_push=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prefix)
      prefix="${2:?missing value for $1}"
      shift 2
      ;;
    -s|--source-branch)
      source_branch="${2:?missing value for $1}"
      shift 2
      ;;
    -t|--target-branch)
      target_branch="${2:?missing value for $1}"
      shift 2
      ;;
    -r|--remote)
      remote_name="${2:?missing value for $1}"
      shift 2
      ;;
    --no-push)
      no_push=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository." >&2
  exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/$source_branch"; then
  echo "Error: source branch '$source_branch' does not exist locally." >&2
  exit 1
fi

if ! git rev-parse --verify "HEAD:$prefix" >/dev/null 2>&1; then
  echo "Error: directory '$prefix' does not exist in current repository state." >&2
  exit 1
fi

if [[ $no_push -eq 0 ]] && ! git remote get-url "$remote_name" >/dev/null 2>&1; then
  echo "Error: remote '$remote_name' does not exist." >&2
  exit 1
fi

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "$source_branch" ]]; then
  git checkout "$source_branch"
fi

git branch -D "$target_branch" >/dev/null 2>&1 || true
git subtree split --prefix "$prefix" -b "$target_branch"

if [[ $no_push -eq 1 ]]; then
  echo "Created local branch '$target_branch' from '$prefix' (push skipped)."
  exit 0
fi

git push --force -u "$remote_name" "$target_branch:$target_branch"
echo "Published '$prefix' to '$remote_name/$target_branch'."
