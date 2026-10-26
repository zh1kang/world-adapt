"""Notebook judge: folder parsing, predictor truth, quote checks, caching, aggregation and agreement. No network."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from world_adapt.runner import judge as J

ATOMS = [{"name": "single_cargo", "family": "prerequisite", "condition": "(load & truck_already_loaded)",
          "description": "Predictor incorrectly blocks loading a second crate into a truck."}]
RULES = "Standard Depot PDDL."


def make_run(root: Path, name: str, notebooks: list[str], atoms=ATOMS, completed: int | None = None,
             heldout: list[str] = ()) -> Path:
    d = root / name
    d.mkdir(parents=True)
    manifest = {"type": "evaluator_manifest", "profile": {"name": "p", "atoms": atoms}, "rules": RULES}
    (d / "evaluator.jsonl").write_text(json.dumps(manifest) + "\n")
    entries = [{"task_id": f"adapt-{i}", "tier": "easy", "start": "", "written": t} for i, t in enumerate(notebooks)]
    held = [{"task_id": f"heldout-easy-{i}", "tier": "easy", "start": "", "written": t} for i, t in enumerate(heldout)]
    (d / "notebooks.json").write_text(json.dumps({"adaptation": entries, "heldout": held,
                                                  "saved_after_adaptation": ""}))
    done = len(notebooks) if completed is None else completed
    events = [{"type": "task_complete", "task_id": f"adapt-{i}"} for i in range(done)]
    events += [{"type": "task_complete", "task_id": f"heldout-easy-{i}"} for i in range(len(heldout))]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n{\"type\": \"trunc")
    return d


def good(**changes):
    out = {"rationale": "r", "diagnosis": "no_diagnosis", "false_alarm": False, "predictor_policy": "not_mentioned",
           "false_beliefs": [], "evidence": []}
    return {**out, **changes}


def notebook(text="Trucks hold many crates.\n\nPredict blocks loading a second crate.", predictor="faulty"):
    return J.Notebook("20261021-182551-x-model1-depot-PM-s3000", "/p", "model1", "depot",
                      "faulty-predictor-saved-notebook", predictor, "saved", "adaptation", "adapt-0", 1, "easy",
                      text, RULES, "truth")


def test_parse_old_and_new_folder_names():
    old = J.parse_folder("20261021-182551-final-eight-model-2026-10-21-deepseek41-depot-UM-s3000")
    assert old == {"model": "deepseek41", "env": "depot", "condition": "accurate-predictor-saved-notebook",
                   "memory": "saved"}
    assert J.parse_folder("20261022-205711-opus-high-full-2026-10-22-opus55high-pusht-PR-s3000")["condition"] == \
        "faulty-predictor-reset-notebook"
    new = J.parse_folder("20261105-101010-study-portalmaze-faulty-predictor-reset-notebook-s2")
    assert (new["env"], new["condition"], new["memory"]) == ("portalmaze", "faulty-predictor-reset-notebook", "reset")
    derived = J.parse_folder("derived-c778adb49142-kimi-cloud-fresh-8f1408f9d657-kimi26-iceslide-"
                             "faulty-predictor-saved-notebook-s3000-derived-c778adb49142")
    assert (derived["model"], derived["env"], derived["memory"]) == ("kimi26", "iceslide", "saved")
    with pytest.raises(ValueError):
        J.parse_folder("not-a-run")


def test_predictor_truth_from_manifest_profile():
    faulty = J.predictor_truth({"name": "p", "atoms": ATOMS})
    assert "single_cargo" in faulty and "blocks loading a second crate" in faulty and "(load & truck_already_loaded)" in faulty
    assert "exact copy" in J.predictor_truth({"name": "identity", "atoms": []})


def test_load_history_skips_incomplete_and_empty_tasks(tmp_path):
    d = make_run(tmp_path, "20261021-182551-x-model1-depot-PM-s3000", ["a", "b", " ", "d"], completed=2,
                 heldout=["h"])
    assert [n.task_index for n in J.load_history(d)] == [1, 2]
    both = J.load_history(d, include_heldout=True)
    assert [(n.phase, n.task_index) for n in both] == [("adaptation", 1), ("adaptation", 2), ("heldout", 9)]
    assert both[0].predictor == "faulty" and "single_cargo" in both[0].truth and both[0].rules == RULES
    assert [r for r in J.find_runs([tmp_path])] == [d]


def test_load_history_rejects_profile_that_contradicts_the_folder(tmp_path):
    d = make_run(tmp_path, "20261021-182551-x-model1-depot-UM-s3000", ["a"])
    with pytest.raises(ValueError, match="contradicts"):
        J.load_history(d)


def test_quote_verification_normalises_whitespace_and_drops_the_rest():
    text = "Trucks hold\n  many crates.\nPredict blocks loading."
    j = good(evidence=["Trucks hold many crates.", "invented quote"],
             false_beliefs=[{"quote": "Predict   blocks loading.", "explanation": "e"},
                            {"quote": "never in notebook", "explanation": "e"}])
    checked = J.verify_quotes(j, text)
    assert checked["evidence"] == ["Trucks hold many crates."]
    assert [b["quote"] for b in checked["false_beliefs"]] == ["Predict   blocks loading."]
    assert checked["unverified_quotes"] == ["invented quote", "never in notebook"]
    assert checked["quotes_verified"] is False
    assert J.verify_quotes(good(evidence=["Predict blocks loading."]), text)["quotes_verified"] is True


@pytest.mark.parametrize("bad", [
    {"diagnosis": "somewhat"}, {"predictor_policy": "uses"}, {"false_alarm": "no"},
    {"evidence": ["a", "b", "c", "d"]}, {"evidence": "a"}, {"false_beliefs": [{"quote": "q"}]},
    {"rationale": 3},
])
def test_schema_validation_rejects_bad_output(bad):
    with pytest.raises(J.JudgeFormatError):
        J.validate_judgment(good(**bad), "faulty")


def test_validation_rejects_missing_extra_keys_and_exact_predictor_diagnosis():
    with pytest.raises(J.JudgeFormatError):
        J.validate_judgment({k: v for k, v in good().items() if k != "evidence"}, "faulty")
    with pytest.raises(J.JudgeFormatError):
        J.validate_judgment({**good(), "extra": 1}, "faulty")
    with pytest.raises(J.JudgeFormatError, match="exact"):
        J.validate_judgment(good(diagnosis="names_rule"), "accurate")
    assert J.validate_judgment(good(diagnosis="vague_warning", false_alarm=True), "accurate")
    assert J.parse_reply("```json\n" + json.dumps(good()) + "\n```", "faulty")["diagnosis"] == "no_diagnosis"
    with pytest.raises(J.JudgeFormatError):
        J.parse_reply("no json here", "faulty")


class FakeSail:
    """Counts calls and returns scripted replies as an OpenAI-style client."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        self.last = kwargs
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        return NS(choices=[NS(message=NS(content=text), finish_reason=finish)],
                  usage=NS(prompt_tokens=1000, completion_tokens=100, prompt_tokens_details=None))


def test_cache_hit_skips_the_call_and_errors_are_not_cached(tmp_path):
    nb = notebook()
    reply = json.dumps(good(diagnosis="names_rule", evidence=["Predict blocks loading a second crate."]))
    client = FakeSail([reply])
    judge = J.SailJudge(client, "m", sleep=lambda s: None)
    verdicts, fresh = J.judge_notebooks([nb, nb], "sail:m", judge, tmp_path)
    assert client.calls == 1 and len(fresh) == 1
    again, fresh2 = J.judge_notebooks([nb], "sail:m", J.SailJudge(FakeSail([]), "m"), tmp_path)
    assert fresh2 == set() and again[J.cache_key("sail:m", nb)].output["diagnosis"] == "names_rule"
    other = notebook(text="different notebook")
    flaky = FakeSail(["garbage"] * 4)
    v, f = J.judge_notebooks([other], "sail:m", J.SailJudge(flaky, "m", sleep=lambda s: None), tmp_path)
    assert flaky.calls == 4 and next(iter(v.values())).error.startswith("invalid judgment")
    assert not list(tmp_path.rglob(f"{J.cache_key('sail:m', other)}.json"))
    assert J.cache_key("sail:m", nb) != J.cache_key("sail:other", nb) != J.cache_key("sail:m", other)


def test_sail_judge_retries_transport_errors_and_counts_usage():
    reply = json.dumps(good())
    client = FakeSail([RuntimeError("boom"), reply])
    verdict = J.SailJudge(client, "m", sleep=lambda s: None)._call(J.Request("k", "s", "u", "faulty"))
    assert verdict.error is None and verdict.usage["input_tokens"] == 1000 and verdict.usage["output_tokens"] == 100


def test_sail_judge_reports_a_cut_off_reply_and_asks_again():
    client = FakeSail([('{"diagnosis": "names_rule", "evid', "length"), json.dumps(good())])
    verdict = J.SailJudge(client, "m", sleep=lambda s: None)._call(J.Request("k", "s", "u", "faulty"))
    assert verdict.error is None and client.calls == 2 and client.last["max_tokens"] == 32_768
    cut = J.SailJudge(FakeSail([("thinking...", "length")] * 4), "m")._call(J.Request("k", "s", "u", "faulty"))
    assert cut.error == "stopped at max_tokens (32768)" and cut.usage["output_tokens"] == 400


def test_cost_uses_cache_and_batch_prices():
    usage = {"input_tokens": 1_000_000, "cache_write_tokens": 0, "cache_read_tokens": 1_000_000,
             "output_tokens": 100_000}
    price = J.PRICES["anthropic:claude-opus-5-5"]
    assert J.cost_usd(usage, price, batch=False) == pytest.approx(4 + 0.2 + 2)
    assert J.cost_usd(usage, price, batch=True) == pytest.approx((4 + 0.2 + 2) / 2)
    assert J.cost_usd(usage, None, batch=False) is None


def collect(judge, requests):
    out = {}
    judge.judge(requests, lambda key, verdict: out.__setitem__(key, verdict))
    return out


def anthropic_message(obj, stop="end_turn"):
    return NS(stop_reason=stop, content=[NS(type="thinking"), NS(type="text", text=json.dumps(obj))],
              usage=NS(input_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=5, output_tokens=7))


def test_anthropic_batch_submits_once_and_resumes_from_saved_state(tmp_path):
    nbs = [notebook(text=f"notebook {i}") for i in range(2)]
    reqs = [J.Request(J.cache_key("anthropic:m", nb), "sys", J.user_prompt(nb), "faulty") for nb in nbs]
    created = []
    state = {"status": "in_progress"}

    class Batches:
        def create(self, requests):
            created.append([r["custom_id"] for r in requests])
            return NS(id="batch1")

        def retrieve(self, batch_id):
            return NS(processing_status=state["status"], request_counts=NS(processing=2, succeeded=0))

        def results(self, batch_id):
            for r in reqs:
                yield NS(custom_id=r.key, result=NS(type="succeeded", message=anthropic_message(good())))

    client = NS(messages=NS(batches=Batches()))
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        state["status"] = "ended"

    judge = J.AnthropicJudge(client, "m", state_dir=tmp_path, sleep=sleep)
    out = collect(judge, reqs)
    assert len(created) == 1 and len(created[0]) == 2 and sleeps and all(v.batch for v in out.values())
    assert json.loads((tmp_path / "batches-m.json").read_text())["batches"][0]["collected"] is True
    # A restarted process with an uncollected batch polls it instead of submitting again.
    path = tmp_path / "batches-m.json"
    saved = json.loads(path.read_text())
    saved["batches"][0]["collected"] = False
    path.write_text(json.dumps(saved))
    resumed = collect(J.AnthropicJudge(client, "m", state_dir=tmp_path, sleep=sleep), reqs)
    assert len(created) == 1 and set(resumed) == {r.key for r in reqs}


def test_anthropic_refusal_and_truncation_are_recorded():
    refusal = J.AnthropicJudge.verdict(anthropic_message(good(), stop="refusal"), "faulty", batch=False)
    assert refusal.refusal and refusal.output is None and refusal.error is None
    cut = J.AnthropicJudge.verdict(anthropic_message(good(), stop="max_tokens"), "faulty", batch=False)
    assert "max_tokens" in cut.error
    bad = J.AnthropicJudge.verdict(anthropic_message({"diagnosis": "x"}), "faulty", batch=False)
    assert bad.error.startswith("invalid judgment")


def test_request_params_use_structured_outputs_with_a_cacheable_system_prompt():
    params = J.AnthropicJudge(None, "claude-opus-5-5").params(J.Request("k", "rubric", "user", "faulty"))
    assert params["output_config"]["format"]["schema"] is J.SCHEMA and params["output_config"]["effort"] == "medium"
    assert params["thinking"] == {"type": "adaptive"} and "temperature" not in params
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_kappa_computation():
    labels = ("a", "b")
    perfect = J.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"], labels)
    assert perfect["kappa"] == 1.0 and perfect["confusion"]["a"]["a"] == 2
    chance = J.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"], labels)
    assert chance["kappa"] == 0.0 and chance["n"] == 4 and chance["observed_agreement"] == 0.5
    # Textbook example: 2x2 table 20/5/10/15 gives kappa 0.4.
    a = ["a"] * 25 + ["b"] * 25
    b = ["a"] * 20 + ["b"] * 5 + ["a"] * 10 + ["b"] * 15
    assert J.cohen_kappa(a, b, labels)["kappa"] == pytest.approx(0.4)
    assert J.cohen_kappa(["a", "a"], ["a", "a"], labels)["kappa"] is None
    assert J.fleiss_kappa([["a", "a", "a"], ["b", "b", "b"], ["a", "b", "a"], ["b", "b", "a"]], labels)["kappa"] == \
        pytest.approx(0.333333, abs=1e-5)
    assert J.fleiss_kappa([["a", "b"]] * 3, labels)["raters"] == 2


def test_majority_needs_more_than_half_of_the_configured_judges():
    assert J.majority(["a", "a", "b"], 3) == "a"
    assert J.majority(["a", "b", "c"], 3) == J.NO_MAJORITY
    assert J.majority(["a", None, None], 3) == J.NO_MAJORITY
    assert J.majority(["a", "b"], 2) == J.NO_MAJORITY
    assert J.majority([True, True, False], 3) is True


def row(history, index, judge, diagnosis, policy="not_mentioned", beliefs=(), alarm=False, phase="adaptation"):
    return {"model": "m1", "env": "depot", "condition": "faulty-predictor-saved-notebook", "predictor": "faulty",
            "notebook_mode": "saved", "history": history, "run_path": f"/runs/{history}", "phase": phase,
            "task_id": f"adapt-{index - 1}", "task_index": index, "judge": judge, "diagnosis": diagnosis,
            "false_alarm": alarm, "predictor_policy": policy, "false_beliefs": [{"quote": "q", "explanation": "e"}
                                                                                for _ in beliefs]}


A, B, C = "anthropic:claude-opus-5-5", "sail:org/model-b", "sail:org/model-c"


def test_history_aggregation_uses_the_majority_and_keeps_per_judge_columns():
    rows = []
    # Diagnosis by task: judges A and B agree from task 2; C only from task 3. Task 3 carries a false belief for A, B.
    plan = {1: ("no_diagnosis",) * 3, 2: ("locates_error", "locates_error", "vague_warning"),
            3: ("names_rule", "names_rule", "names_rule"), 4: ("names_rule", "vague_warning", "locates_error")}
    for index, labels in plan.items():
        for judge, label in zip((A, B, C), labels):
            beliefs = ("x",) if index == 3 and judge in (A, B) else ()
            policy = "restricts" if index == 4 else "not_mentioned"
            rows.append(row("h1", index, judge, label, policy, beliefs))
    table = J.notebook_table(rows, [A, B, C])
    assert [r["diagnosis"] for r in table] == ["no_diagnosis", "locates_error", "names_rule", J.NO_MAJORITY]
    [h] = J.history_rows(table, [A, B, C])
    assert h["first_locates_error"] == 2 and h["first_names_rule"] == 3
    assert h["final_task_index"] == 4 and h["final_diagnosis"] == J.NO_MAJORITY and h["final_policy"] == "restricts"
    assert h["false_belief_notebooks"] == 1 and h["false_belief_in_final"] is False
    assert h["model-b__first_names_rule"] == 3 and h["model-c__first_names_rule"] == 3
    assert h["model-c__first_locates_error"] == 4 and h["model-c__false_belief_statements"] == 0
    assert h["claude-opus-5-5__false_belief_statements"] == 1


def test_false_belief_persists_when_present_in_the_final_notebook():
    rows = [row("h", i, j, "no_diagnosis", beliefs=("x",) if i >= 2 else ()) for i in (1, 2, 3) for j in (A, B)]
    [h] = J.history_rows(J.notebook_table(rows, [A, B]), [A, B])
    assert h["false_belief_in_final"] is True and h["false_belief_notebooks"] == 2
    assert h["first_names_rule"] == ""


def test_heldout_notebooks_do_not_enter_history_rows():
    rows = [row("h", 1, A, "no_diagnosis"), row("h", 9, A, "names_rule", phase="heldout")]
    [h] = J.history_rows(J.notebook_table(rows, [A]), [A])
    assert h["notebooks"] == 1 and h["first_names_rule"] == ""


def test_agreement_and_disagreements_across_three_judges():
    rows = []
    labels = {1: ("names_rule",) * 3, 2: ("names_rule", "names_rule", "vague_warning"),
              3: ("names_rule", "vague_warning", "no_diagnosis")}
    for index, trio in labels.items():
        for judge, label in zip((A, B, C), trio):
            rows.append(row("h", index, judge, label))
    specs = [A, B, C]
    table = J.notebook_table(rows, specs)
    dis = J.disagreement_rows(table, specs)
    assert [d["task_index"] for d in dis] == [2, 3]
    assert [d["no_majority"] for d in dis] == [False, True]
    assert dis[0]["notebooks_json"] == "/runs/h/notebooks.json"
    by_notebook = {(r["history"], r["task_id"]): {} for r in rows}
    for r in rows:
        by_notebook[(r["history"], r["task_id"])][r["judge"]] = r
    out = J.agreement(by_notebook, specs)
    assert set(out["diagnosis"]["pairwise_cohen"]) == {f"{A} | {B}", f"{A} | {C}", f"{B} | {C}"}
    assert out["diagnosis"]["pairwise_cohen"][f"{A} | {B}"]["n"] == 3
    assert out["diagnosis"]["fleiss"]["n"] == 3 and out["diagnosis"]["fleiss"]["raters"] == 3
    assert "predictor_policy" in out


def test_end_to_end_outputs_with_a_fake_judge(tmp_path):
    run = make_run(tmp_path / "runs", "20261021-182551-x-model1-depot-PM-s3000",
                   ["Predict blocks loading a second crate.", "Predict may be imperfect."])
    nbs = J.load_history(run)
    replies = [json.dumps(good(diagnosis="names_rule", predictor_policy="restricts",
                               evidence=["Predict blocks loading a second crate."])),
               json.dumps(good(diagnosis="vague_warning", evidence=["not in the notebook"]))]
    client = FakeSail(replies)
    spec = "sail:org/model-b"
    verdicts, fresh = J.judge_notebooks(nbs, spec, J.SailJudge(client, "org/model-b", concurrency=1), tmp_path / "cache")
    rows = J.judgment_rows(nbs, spec, verdicts, fresh, J.Price(1.0, 4.0))
    out = tmp_path / "out"
    J.write_outputs(out, rows, [spec], {spec: J.Price(1.0, 4.0)})
    saved = [json.loads(line) for line in (out / "judgments.jsonl").read_text().splitlines()]
    assert [r["diagnosis"] for r in saved] == ["names_rule", "vague_warning"]
    assert [r["quotes_verified"] for r in saved] == [True, False] and saved[1]["unverified_quotes"]
    assert saved[0]["condition"] == "faulty-predictor-saved-notebook" and saved[0]["notebook_mode"] == "saved"
    cost = json.loads((out / "cost.json").read_text())
    assert cost[spec]["calls_this_run"] == 2 and cost["total_usd"] == pytest.approx(0.0028)
    histories = (out / "histories.csv").read_text().splitlines()
    assert len(histories) == 2 and "first_names_rule" in histories[0]
    for name in ("notebooks.csv", "disagreements.csv", "agreement.json"):
        assert (out / name).exists()


def test_sample_spreads_over_histories():
    nbs = [J.Notebook(f"h{h}", "/p", "m", "depot", "faulty-predictor-saved-notebook", "faulty", "saved", "adaptation",
                      f"adapt-{i}", i + 1, "easy", "t", RULES, "x") for h in range(10) for i in range(8)]
    sample = J.sample_notebooks(nbs, 6, seed=1)
    assert len(sample) == 6 and len({n.history for n in sample}) == 6
    assert sample == J.sample_notebooks(nbs, 6, seed=1)
