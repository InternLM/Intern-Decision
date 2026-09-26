"""Independent oracle checks and leakage/metric contract tests for the pilot."""

import copy
import json
from collections import Counter, defaultdict
from fractions import Fraction as F
from itertools import combinations, product

import pytest

from src.eval.known_distribution import base_cases, build_records, coin_filter, distinct_distribution, write_benchmark
from src.eval.score_known_distribution import aggregate, score_answer, score_benchmark


def family(name):
    return [r for r in base_cases() if r["family"] == name]


def test_size_balance_and_evidence_isolation():
    inputs, refs = build_records()
    assert len(inputs) == len(refs) == 96
    assert len({r["group"] for r in refs}) == 48
    assert len({r["family"] for r in refs}) == 24
    assert set(Counter(r["category"] for r in refs).values()) == {16}
    for row, ref in zip(inputs, refs):
        assert set(row) == {"id", "state", "questions"}
        assert not {"targets", "expected", "gold_probs", "derivation", "provenance"}.intersection(row)
        assert set(row["questions"]["decision"]) == {"type", "instructions", "criteria"}
        assert sum(F(v) for v in ref["gold_fractions"].values()) == 1
    assert build_records() == (inputs, refs)


def test_direct_and_composed_oracles():
    die = family("uniform_support")[0]["probabilities"]
    assert die == {**{str(i): F(1, 6) for i in range(1, 7)}, "7": F(0), "8": F(0)}
    assert family("weighted_inventory")[0]["probabilities"] == {"red": F(1, 2), "blue": F(3, 10), "green": F(1, 5)}
    for row in family("deterministic_support"):
        assert sorted(row["probabilities"].values()) == [0, 0, 1]
    assert [c["probabilities"]["yes"] for c in family("rare_event")] == [F(1, 100), F(1, 1000)]
    assert family("backup_reliability")[0]["probabilities"]["yes"] == F(49, 50)
    assert family("delivery_mixture")[0]["probabilities"]["yes"] == F(41, 100)
    for case in family("sum_of_dice"):
        n = case["parameters"]["sides"]
        counts = Counter(sum(pair) for pair in product(range(1, n + 1), repeat=2))
        assert all(p == F(counts[int(k)], n * n) for k, p in case["probabilities"].items())
    for case in family("independent_hits"):
        n, p = case["parameters"]["shots"], F(case["parameters"]["hit_probability"])
        oracle = defaultdict(F)
        for outcomes in product((0, 1), repeat=n):
            prob = F(1)
            for outcome in outcomes:
                prob *= p if outcome else 1 - p
            oracle[str(sum(outcomes))] += prob
        assert case["probabilities"] == dict(oracle)


def test_history_and_bayes_golden_cases():
    assert all(c["probabilities"] == {"heads": F(1, 2), "tails": F(1, 2)} for c in family("independent_streak"))
    assert [c["probabilities"] for c in family("without_replacement")] == [
        {"red": F(1, 2), "blue": F(1, 2)},
        {"red": F(3, 7), "blue": F(4, 7)},
    ]
    for c in family("restricted_observation"):
        allowed = c["parameters"]["allowed"]
        assert all(p == (F(1, len(allowed)) if int(k) in allowed else 0) for k, p in c["probabilities"].items())
    assert [c["probabilities"]["yes"] for c in family("defect_screening")] == [F(2, 13), F(8, 17)]
    assert [c["probabilities"]["yes"] for c in family("independent_repeated_tests")] == [F(16, 25), F(64, 73)]
    assert all(c["probabilities"]["yes"] == F(4, 13) for c in family("copied_correlated_tests"))
    assert [c["probabilities"]["long_gap"] for c in family("random_time_waiting")] == [F(3, 4), F(2, 3)]


def test_sensor_oracle_matches_full_latent_path_enumeration():
    for case in family("hidden_state_sensor"):
        params = case["parameters"]
        prior, stay, acc = map(F, (params["initial_busy"], params["persistence"], params["sensor_accuracy"]))
        observations = params["observations"]
        weights = defaultdict(F)
        for states in product((False, True), repeat=len(observations)):
            prob = prior if states[0] else 1 - prior
            for i in range(1, len(states)):
                prob *= stay if states[i] == states[i - 1] else 1 - stay
            for state, obs in zip(states, observations):
                prob *= acc if state == obs else 1 - acc
            weights[states[-1]] += prob
        expected = weights[True] / sum(weights.values())
        assert coin_filter(prior, stay, acc, observations) == expected == case["probabilities"]["yes"]


@pytest.mark.parametrize("informed", [True, False])
@pytest.mark.parametrize("n", [3, 5])
def test_monty_by_enumerating_prize_and_host_reveal(informed, n):
    masses = defaultdict(F)
    for prize in range(n):
        legal = [r for r in combinations(range(1, n), n - 2) if not informed or prize not in r]
        for opened in legal:
            if prize in opened:
                continue
            label = "initial_door" if prize == 0 else "other_unopened_door"
            masses[label] += F(1, n * len(legal))
    total = sum(masses.values())
    expected = {k: p / total for k, p in masses.items()}
    name = "monty_informed_host" if informed else "monty_uninformed_host"
    case = next(c for c in family(name) if c["parameters"]["doors"] == n)
    assert case["probabilities"] == expected


def test_selective_offer_and_two_child_protocols():
    for case in family("monty_selective_offer"):
        params = case["parameters"]
        masses = defaultdict(F)
        for prize in range(3):
            empty = [door for door in (1, 2) if door != prize]
            for _opened in empty:
                offer = F(params["offer_if_initial_correct"] if prize == 0 else params["offer_if_initial_wrong"])
                masses["initial_door" if prize == 0 else "other_unopened_door"] += F(1, 3 * len(empty)) * offer
        assert case["probabilities"] == {k: p / sum(masses.values()) for k, p in masses.items()}
    cases = family("two_child_selection")
    for case in cases:
        numerator = denominator = F(0)
        for children in product(("B", "G"), repeat=2):
            likelihood = (
                F(int("B" in children))
                if case["parameters"]["observation_protocol"] == "at_least_one"
                else F(children.count("B"), 2)
            )
            denominator += F(1, 4) * likelihood
            if children == ("B", "B"):
                numerator += F(1, 4) * likelihood
        assert case["probabilities"]["yes"] == numerator / denominator


def test_coupon_dp_matches_exhaustive_draws():
    for types, draws in ((3, 3), (4, 5)):
        counts = Counter(len(set(draw)) for draw in product(range(types), repeat=draws))
        oracle = distinct_distribution(types, draws)
        assert all(p == F(counts[int(k)], types**draws) for k, p in oracle.items())


def test_birthday_and_guaranteed_reward():
    counts = Counter(len(set(days)) < 4 for days in product(range(10), repeat=4))
    assert family("birthday_collision")[0]["probabilities"]["yes"] == F(counts[True], 10**4)
    assert family("guaranteed_reward")[0]["probabilities"] == {"1": F(1, 5), "2": F(4, 25), "3": F(16, 25)}
    assert family("guaranteed_reward")[1]["probabilities"] == {
        "1": F(1, 4),
        "2": F(3, 16),
        "3": F(9, 64),
        "4": F(27, 64),
    }


def test_ruin_formula_matches_rational_linear_system():
    for case in family("gambler_ruin"):
        params = case["parameters"]
        n, p = params["goal"], F(params["win"])
        # Independently solve the transient-state linear system by Gaussian elimination.
        a = [[F(int(i == j)) for j in range(n - 1)] + [F(0)] for i in range(n - 1)]
        for i in range(n - 1):
            if i:
                a[i][i - 1] -= 1 - p
            if i < n - 2:
                a[i][i + 1] -= p
            else:
                a[i][-1] = p
        for k in range(n - 1):
            pivot = a[k][k]
            a[k] = [v / pivot for v in a[k]]
            for i in range(n - 1):
                if i != k:
                    factor = a[i][k]
                    a[i] = [v - factor * w for v, w in zip(a[i], a[k])]
        assert a[params["start"] - 1][-1] == case["probabilities"]["yes"]


def test_proper_scores_against_enumeration_of_possible_realized_labels():
    ref = {"gold_fractions": {"a": "1/4", "b": "3/4"}}
    answer = {"type": "choice", "probabilities": {"a": 0.6, "b": 0.4}, "confidence": 0.0}
    s = score_answer(answer, ref)
    expected = sum(
        q * ((0.6 - int(label == "a")) ** 2 + (0.4 - int(label == "b")) ** 2) for label, q in (("a", 0.25), ("b", 0.75))
    )
    assert s["expected_brier"] == pytest.approx(expected)
    assert s["excess_expected_brier"] == pytest.approx(2 * 0.35**2)
    assert s["expected_correctness"] == 0.25
    assert s["confidence_used"] == 0.0 and s["confidence_source"] == "api"
    assert aggregate([s])["expected_ece"]["ece"] == 0.25
    del answer["confidence"]
    assert score_answer(answer, ref)["confidence_source"] == "max_p"


@pytest.mark.parametrize(
    "probs", [{"a": 1.0}, {"a": 0.8, "b": 0.4}, {"a": True, "b": 0.0}, {"a": float("nan"), "b": 0.0}]
)
def test_invalid_probabilities_rejected(probs):
    with pytest.raises(ValueError):
        score_answer({"type": "choice", "probabilities": probs}, {"gold_fractions": {"a": "1/2", "b": "1/2"}})


def test_oracle_sanity_end_to_end_and_missing_coverage(tmp_path):
    dataset = tmp_path / "dataset"
    write_benchmark(dataset)
    _, refs = build_records()
    predictions = [
        {
            "id": ref["id"],
            "model": "oracle-sanity-only",
            "answers": {"decision": {"type": "choice", "probabilities": ref["gold_probs"]}},
        }
        for ref in refs
    ]
    file = tmp_path / "predictions.jsonl"
    file.write_text("".join(json.dumps(r) + "\n" for r in predictions))
    m = score_benchmark(dataset, file, tmp_path / "scored")
    assert m["complete"] and m["valid"] == 96
    assert m["tvd"] == pytest.approx(0, abs=1e-14)
    assert m["excess_expected_brier"] == pytest.approx(0, abs=1e-14)
    assert m["expected_ece"]["ece"] == pytest.approx(0, abs=1e-14)
    assert m["expected_brier"] > 0  # Irreducible randomness is not model error.
    assert m["order_robustness"]["mean_tvd_between_variants"] == pytest.approx(0, abs=1e-14)
    file.write_text("".join(json.dumps(r) + "\n" for r in predictions[:-1]))
    missing = score_benchmark(dataset, file, tmp_path / "missing")
    assert not missing["complete"] and missing["invalid"] == 1
    assert missing["order_robustness"]["valid_pairs"] == 47
    file.write_text(json.dumps(predictions[0]) + "\n" + json.dumps(predictions[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        score_benchmark(dataset, file, tmp_path / "duplicate")


def test_reference_hash_tampering_rejected(tmp_path):
    write_benchmark(tmp_path / "dataset")
    inputs_path = tmp_path / "dataset/inputs.jsonl"
    inputs_path.write_text(inputs_path.read_text().replace("fair die", "loaded die", 1))
    predictions = tmp_path / "empty.jsonl"
    predictions.write_text("")
    with pytest.raises(ValueError, match="hash"):
        score_benchmark(tmp_path / "dataset", predictions, tmp_path / "results")


def test_option_reversal_keeps_the_same_question_and_truth():
    inputs, refs = build_records()
    for i in range(0, len(inputs), 2):
        a, b = copy.deepcopy(inputs[i]), copy.deepcopy(inputs[i + 1])
        a.pop("id")
        b.pop("id")
        assert a == b  # Python map equality ignores order; explicit ordering is reversed.
        assert list(a["questions"]["decision"]["criteria"]) == list(reversed(b["questions"]["decision"]["criteria"]))
        assert refs[i]["gold_fractions"] == refs[i + 1]["gold_fractions"]
