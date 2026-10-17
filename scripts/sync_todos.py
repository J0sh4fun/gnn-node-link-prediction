"""Synchronize verified tasks with GitHub; never infer acceptance from filenames."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TASK = re.compile(
    r"^- \[([ x])\] \*\*(A\d{2}|B\d{2}-\d{2})( · Tuần \d+)? · "
    r"(To do|Doing|Review|Done)\*\*", re.MULTILINE
)
STATES = ("To do", "Doing", "Review", "Done")


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    return path


class GitHub:
    def __init__(self, repository: str, token: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Invalid repository")
        self.prefix = f"https://api.github.com/repos/{repository}/"
        self.token = token
        self.cache = {}

    def get(self, endpoint: str):
        if endpoint not in self.cache:
            request = urllib.request.Request(self.prefix + endpoint, headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "gnn-todo-sync",
                "X-GitHub-Api-Version": "2022-11-28",
            })
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    self.cache[endpoint] = json.load(response)
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"GitHub HTTP {error.code}: {endpoint}") from None
        return self.cache[endpoint]

    def tree(self, revision: str) -> dict[str, str]:
        data = self.get(f"git/trees/{urllib.parse.quote(revision, safe='')}?recursive=1")
        if data.get("truncated"):
            raise ValueError("Truncated GitHub tree; cannot verify source")
        return {item["path"]: item["sha"] for item in data["tree"] if item["type"] == "blob"}


def snapshot(client: GitHub, config: dict) -> dict:
    base = config["base"]
    main = client.get(f"commits/{urllib.parse.quote(base, safe='')}")["sha"]
    prs = {str(spec["pr"]): client.get(f"pulls/{spec['pr']}")
           for spec in config["verified_tasks"].values()}
    revisions = {main}
    revisions.update(proof["revision"] for proof in config["evidence"].values())
    revisions.update(pr["head"]["sha"] for pr in prs.values())
    return {"main": main, "prs": prs,
            "trees": {sha: client.tree(sha) for sha in sorted(revisions)}}


def accepted(root: Path, proof: dict) -> bool:
    """Require recorded passing tests/report; optional sanity trace must match revision."""
    try:
        suites = ET.parse(local_path(root, proof["junit"])).getroot()
        cases = list(suites.iter("testcase"))
        if len(cases) != proof["expected_tests"] or not cases:
            return False
        if any(case.find(tag) is not None for case in cases
               for tag in ("failure", "error", "skipped")):
            return False
        for suite in suites.iter("testsuite"):
            if any(int(suite.get(key, "0")) for key in ("failures", "errors", "skipped")):
                return False
        if not local_path(root, proof["report"]).read_text(encoding="utf-8").strip():
            return False
        if "sanity" in proof:
            trace = json.loads(local_path(root, proof["sanity"]).read_text(encoding="utf-8"))
            if not (trace["implementation_commit"] == proof["revision"]
                    and trace["finite_outputs_losses_all_epochs"] is True
                    and trace["finite_gradients_all_epochs"] is True
                    and trace["checkpoint_losses_strictly_decrease"] is True):
                return False
        return True
    except (OSError, ValueError, KeyError, ET.ParseError):
        return False


def task_state(root: Path, config: dict, data: dict, spec: dict) -> tuple[str, str]:
    pr = data["prs"][str(spec["pr"])]
    proof = config["evidence"][spec["evidence"]]
    if pr["base"]["ref"] != config["base"] or pr["base"]["repo"]["full_name"] != config["repository"]:
        raise ValueError("Unexpected PR destination")
    integrated = bool(pr.get("merged_at"))
    revision = data["main"] if integrated else pr["head"]["sha"]
    reference = data["trees"][proof["revision"]]
    current = data["trees"][revision]
    same_sources = all(reference.get(path) and reference.get(path) == current.get(path)
                       for path in proof["sources"])
    if not same_sources:
        return "Doing", "Code/test đã thay đổi hoặc không còn trên main; cần kiểm chứng lại."
    if not accepted(root, proof):
        return "Doing", "Chưa đủ bằng chứng nghiệm thu hợp lệ; không tự đánh dấu Done."
    if integrated:
        return "Done", "Đã merge vào main; code/test khớp phiên bản đã nghiệm thu."
    if pr["state"] == "open" and not pr.get("draft", False):
        return "Review", "Đã nghiệm thu; PR đang mở, chưa merge."
    return "Doing", "Đã có bằng chứng; PR draft hoặc đóng mà chưa merge."


def block(text: str, name: str, body: str) -> str:
    start, end = f"<!-- todo-sync:{name}:start -->", f"<!-- todo-sync:{name}:end -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if len(pattern.findall(text)) != 1:
        raise ValueError(f"Expected one todo-sync block: {name}")
    return pattern.sub(lambda _: f"{start}\n{body}\n{end}", text)


def render(root: Path, config: dict, data: dict, relative: str, date: str) -> str:
    original = local_path(root, relative).read_text(encoding="utf-8")
    matches = list(TASK.finditer(original))
    ids = [match[2] for match in matches]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate task IDs in {relative}")
    notes = []

    def replace(match):
        task, week, status = match[2], match[3] or "", match[4]
        if task in config["verified_tasks"]:
            spec = config["verified_tasks"][task]
            status, reason = task_state(root, config, data, spec)
            url = f"https://github.com/{config['repository']}/pull/{spec['pr']}"
            notes.append(f"- **{task} · {status}** — [PR #{spec['pr']}]({url}). {reason}")
        return f"- [{'x' if status == 'Done' else ' '}] **{task}{week} · {status}**"

    updated = TASK.sub(replace, original)
    counts = Counter(match[4] for match in TASK.finditer(updated))
    total = sum(counts.values())
    summary = "| To do | Doing | Review | Done | Tổng |\n|---:|---:|---:|---:|---:|\n"
    summary += "| " + " | ".join(str(counts[state]) for state in STATES) + f" | {total} |\n\n"
    summary += (f"**{counts['Done']}/{total} task đã nghiệm thu và tích hợp main.** "
                "Số task không biểu thị phần trăm khối lượng.\n\n")
    summary += "\n".join(notes)
    summary += "\n\nTask chưa có quy tắc bằng chứng giữ trạng thái thủ công; không suy diễn từ tên file/PR."
    updated = block(updated, "summary", summary)
    # Refresh the date only for substantive changes; repeated job completion is idempotent.
    if updated != original:
        updated = re.sub(r"(?m)^Đồng bộ todo: \*\*\d{4}-\d{2}-\d{2}\*\*\.",
                         f"Đồng bộ todo: **{date}**.", updated)
    return updated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--github", action="store_true")
    source.add_argument("--snapshot", type=Path, help="Replay a saved API snapshot offline")
    parser.add_argument("--save-snapshot", type=Path)
    parser.add_argument("--write", action="store_true", help="Write trackers; otherwise check only")
    args = parser.parse_args(argv)
    config = json.loads((ROOT / "configs/todo_sync.json").read_text(encoding="utf-8"))
    if args.github:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            parser.error("GITHUB_TOKEN is required for --github")
        data = snapshot(GitHub(config["repository"], token), config)
    else:
        data = json.loads(args.snapshot.read_text(encoding="utf-8"))
    date = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
    # Render/validate every tracker before writing any file or snapshot.
    rendered = {path: render(ROOT, config, data, path, date) for path in config["trackers"]}
    changed = [path for path, value in rendered.items()
               if value != local_path(ROOT, path).read_text(encoding="utf-8")]
    if args.save_snapshot:
        args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.save_snapshot.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.write:
        for path in changed:
            local_path(ROOT, path).write_text(rendered[path], encoding="utf-8", newline="\n")
    print(json.dumps({"changed": changed, "written": args.write}, ensure_ascii=False))
    return 0 if args.write or not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
