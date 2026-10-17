#!/usr/bin/env python3
"""Open, update or note a GitHub issue for a notebook health-check result.

Used by .github/workflows/notebooks.yml. Issues follow the format of the website's
tools/task_issue.mjs (see ScientistWebsite docs/MAINTENANCE.md): what happened, why it matters,
and the command to fix it in a local Claude Code session. Label `claude-task`.

    python tools/task_issue.py fail <name> --error-file err.txt [--run-url URL] [--dry-run]
    python tools/task_issue.py pass <name> [--run-url URL] [--dry-run]

fail: opens "Notebook failing: <name>", or comments on it with the new error if already open.
pass: if that issue is open, comments that the notebook passes again (the issue stays open
      until the fix is committed; /fix-notebook closes it).
Needs the GitHub CLI (`gh`) with GH_TOKEN set (issues: write).
"""
import argparse
import json
import os
import subprocess

LABEL, COLOR = "claude-task", "5319e7"
LABEL_DESC = "A task to run in a local Claude Code session; the issue gives the command"

ap = argparse.ArgumentParser()
ap.add_argument("result", choices=["fail", "pass"])
ap.add_argument("name")
ap.add_argument("--error-file")
ap.add_argument("--run-url", default="")
ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "matthewjohnpayne/science-notebooks"))
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

title = f"Notebook failing: {args.name}"
run = f" ([workflow run]({args.run_url}))" if args.run_url else ""


def gh(*a):
    return subprocess.run(["gh", *a], check=True, capture_output=True, text=True).stdout.strip()


def error_block():
    text = open(args.error_file).read() if args.error_file and os.path.exists(args.error_file) else "(no error text captured)"
    return "```\n" + text.strip()[-6000:] + "\n```"


if args.dry_run:
    existing = None
else:
    found = json.loads(gh("issue", "list", "--repo", args.repo, "--state", "open", "--search", f'"{title}" in:title',
                          "--json", "number,title"))
    existing = next((i["number"] for i in found if i["title"] == title), None)

if args.result == "pass":
    if existing:
        gh("issue", "comment", str(existing), "--repo", args.repo,
           "--body", f"The weekly check passes again{run}. Close this issue once the fix (if any) is committed.")
        print(f"Commented on #{existing}: passing again")
    else:
        print(f"{args.name}: passing, no open issue")
    raise SystemExit(0)

if existing:
    gh("issue", "comment", str(existing), "--repo", args.repo, "--body", f"Still failing{run}:\n\n{error_block()}")
    print(f"Commented on #{existing}")
    raise SystemExit(0)

body = "\n".join([
    f"<!-- {LABEL} -->",
    f"**What happened:** the weekly notebook check could not execute `notebooks/{args.name}/{args.name}.ipynb`{run}.",
    "",
    "**Why it matters:** the committed notebook may no longer run with current versions of its dependencies "
    "(the check installs the latest versions allowed by `requirements.txt`).",
    "",
    "**Error:**",
    error_block(),
    "",
    "**To fix it locally:**",
    "```bash",
    f'cd ~/git_repos/science-notebooks && claude "/fix-notebook {args.name}"',
    "```",
    "The command commits locally, closes this issue, and asks before pushing.",
    "",
    "<sub>Opened automatically by .github/workflows/notebooks.yml.</sub>",
])
if args.dry_run:
    print(f"[dry run] {args.repo}: \"{title}\" [{LABEL}]\n\n{body}")
    raise SystemExit(0)
gh("label", "create", LABEL, "--repo", args.repo, "--color", COLOR, "--description", LABEL_DESC, "--force")
print(gh("issue", "create", "--repo", args.repo, "--title", title, "--label", LABEL, "--body", body))
