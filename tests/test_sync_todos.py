"""A merged PR alone must never imply verified acceptance."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import sync_todos as sync


class TodoSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "proof.xml").write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase name="acceptance"/></testsuite></testsuites>', encoding="utf-8")
        (self.root / "report.md").write_text("Recorded acceptance", encoding="utf-8")
        self.proof = {"revision": "verified", "sources": ["model.py", "test.py"],
                      "junit": "proof.xml", "expected_tests": 1, "report": "report.md"}
        self.spec = {"pr": 4, "evidence": "layer"}
        self.config = {"repository": "owner/repo", "base": "main",
                       "verified_tasks": {"A06": self.spec}, "evidence": {"layer": self.proof},
                       "trackers": ["todo.md"]}
        blobs = {"model.py": "model-sha", "test.py": "test-sha"}
        self.data = {"main": "main-sha", "trees": {key: dict(blobs)
                     for key in ("main-sha", "head-sha", "verified")}, "prs": {"4": {
                         "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
                         "head": {"sha": "head-sha"}, "merged_at": None,
                         "state": "open", "draft": False}}}
        self.todo = (
            "Đồng bộ todo: **2026-10-15**.\n"
            "- [ ] **A06 · Tuần 5 · Review** — Accepted layer\n"
            "- [ ] **B06-03 · To do** — Missing benchmark\n"
            "<!-- todo-sync:summary:start -->\nold\n<!-- todo-sync:summary:end -->\n"
            "History: previously Review.\n")
        (self.root / "todo.md").write_text(self.todo, encoding="utf-8")

    def state(self):
        return sync.task_state(self.root, self.config, self.data, self.spec)[0]

    def merged(self):
        self.data["prs"]["4"].update(merged_at="2026-10-15T00:00:00Z", state="closed")

    def test_open_accepted_pr_is_review_not_done(self):
        self.assertEqual(self.state(), "Review")

    def test_merged_accepted_matching_code_is_done(self):
        self.merged()
        self.assertEqual(self.state(), "Done")

    def test_closed_unmerged_and_draft_are_not_done(self):
        for update in ({"state": "closed"}, {"draft": True}):
            with self.subTest(update=update):
                self.data["prs"]["4"].update(state="open", draft=False)
                self.data["prs"]["4"].update(update)
                self.assertEqual(self.state(), "Doing")

    def test_changed_model_or_test_invalidates_evidence(self):
        for path in ("model.py", "test.py"):
            with self.subTest(path=path):
                old = self.data["trees"]["head-sha"][path]
                self.data["trees"]["head-sha"][path] = "changed"
                self.assertEqual(self.state(), "Doing")
                self.data["trees"]["head-sha"][path] = old

    def test_reverted_code_on_main_is_not_done(self):
        self.merged()
        del self.data["trees"]["main-sha"]["model.py"]
        self.assertEqual(self.state(), "Doing")

    def test_unrelated_doc_change_preserves_acceptance(self):
        self.data["trees"]["head-sha"]["README.md"] = "new"
        self.assertEqual(self.state(), "Review")

    def test_failures_errors_skips_counts_and_invalid_xml_block_done(self):
        self.merged()
        for content in (
            '<testsuite><testcase><failure/></testcase></testsuite>',
            '<testsuite><testcase><error/></testcase></testsuite>',
            '<testsuite><testcase><skipped/></testcase></testsuite>',
            '<testsuite errors="1"><testcase/></testsuite>',
            '<testsuite/>', '<broken',
        ):
            with self.subTest(content=content):
                (self.root / "proof.xml").write_text(content, encoding="utf-8")
                self.assertEqual(self.state(), "Doing")

    def test_missing_report_blocks_done(self):
        self.merged()
        (self.root / "report.md").unlink()
        self.assertEqual(self.state(), "Doing")

    def test_sanity_requires_matching_revision_and_finite_gradients(self):
        self.proof["sanity"] = "sanity.json"
        trace = {"implementation_commit": "verified", "finite_outputs_losses_all_epochs": True,
                 "finite_gradients_all_epochs": True, "checkpoint_losses_strictly_decrease": True}
        for key, bad in (("implementation_commit", "stale"), ("finite_gradients_all_epochs", False)):
            with self.subTest(key=key):
                (self.root / "sanity.json").write_text(json.dumps(dict(trace, **{key: bad})), encoding="utf-8")
                self.assertEqual(self.state(), "Doing")
        (self.root / "sanity.json").write_text(json.dumps(trace), encoding="utf-8")
        self.assertEqual(self.state(), "Review")

    def test_wrong_destination_is_rejected(self):
        self.data["prs"]["4"]["base"]["repo"]["full_name"] = "other/repo"
        with self.assertRaisesRegex(ValueError, "destination"):
            self.state()

    def test_render_counts_preserves_manual_tasks_and_history(self):
        self.merged()
        result = sync.render(self.root, self.config, self.data, "todo.md", "2026-10-16")
        self.assertIn("[x] **A06 · Tuần 5 · Done**", result)
        self.assertIn("[ ] **B06-03 · To do**", result)
        self.assertIn("| 1 | 0 | 0 | 1 | 2 |", result)
        self.assertIn("History: previously Review.", result)
        self.assertIn("**2026-10-16**", result)

    def test_next_day_without_progress_is_idempotent(self):
        first = sync.render(self.root, self.config, self.data, "todo.md", "2026-10-16")
        (self.root / "todo.md").write_text(first, encoding="utf-8")
        self.assertEqual(first, sync.render(self.root, self.config, self.data, "todo.md", "2026-10-17"))

    def test_duplicate_ids_and_missing_marker_are_rejected(self):
        for value in (self.todo + "- [ ] **A06 · Tuần 5 · Review**\n",
                      self.todo.replace("<!-- todo-sync:summary:end -->", "")):
            with self.subTest(value=value):
                (self.root / "todo.md").write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    sync.render(self.root, self.config, self.data, "todo.md", "2026-10-16")

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            sync.local_path(self.root, "../outside.md")

    def test_bad_second_tracker_does_not_partially_write(self):
        (self.root / "configs").mkdir()
        broken = copy.deepcopy(self.config)
        broken["trackers"].append("missing.md")
        (self.root / "configs/todo_sync.json").write_text(json.dumps(broken), encoding="utf-8")
        (self.root / "snapshot.json").write_text(json.dumps(self.data), encoding="utf-8")
        with patch.object(sync, "ROOT", self.root), self.assertRaises(FileNotFoundError):
            sync.main(["--snapshot", str(self.root / "snapshot.json"), "--write"])
        self.assertEqual(self.todo, (self.root / "todo.md").read_text(encoding="utf-8"))

    def test_truncated_remote_tree_fails_closed(self):
        client = sync.GitHub("owner/repo", "unused")
        with patch.object(client, "get", return_value={"truncated": True, "tree": []}):
            with self.assertRaisesRegex(ValueError, "Truncated"):
                client.tree("main")


if __name__ == "__main__":
    unittest.main()
