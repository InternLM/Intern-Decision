"""Build a small evaluation-only benchmark of explicit stochastic mechanisms.

Standard library only. Every oracle uses exact rational arithmetic; no sampled
labels, LLM-generated answers, training dependencies, or API calls are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction as F
from itertools import product
from math import comb
from pathlib import Path

REVISION = "known-distribution-pilot-v1"
CATEGORIES = {
    "direct": "Direct randomness and support",
    "composition": "Composed events and mixtures",
    "history": "History, conditioning, and hidden state",
    "daily_evidence": "Daily evidence and observation bias",
    "disclosure": "Selective disclosure and probability puzzles",
    "sequential": "Sequential and combinatorial processes",
}
INSTRUCTION = (
    "Predict the outcome of the specified random experiment using only its stated rules. "
    "The listed outcomes are mutually exclusive and exhaustive. Treat every stated "
    "probability, independence assumption, and observation protocol as exact."
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def evidence_hash(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def binary(p):
    return {"yes": p, "no": 1 - p}


def coin_filter(prior, persistence, sensor_accuracy, observations):
    """Two-state HMM: observation at t=0, then transition before each later one."""
    weights = {True: prior, False: 1 - prior}
    for index, observed in enumerate(observations):
        if index:
            weights = {
                state: sum(weights[old] * (persistence if state == old else 1 - persistence) for old in weights)
                for state in weights
            }
        weights = {
            state: weight * (sensor_accuracy if state == observed else 1 - sensor_accuracy)
            for state, weight in weights.items()
        }
    return weights[True] / sum(weights.values())


def distinct_distribution(types, draws):
    """Exact coupon occupancy DP; state is the number of distinct types seen."""
    state = {0: F(1)}
    for _ in range(draws):
        next_state = defaultdict(F)
        for seen, probability in state.items():
            next_state[seen] += probability * F(seen, types)
            if seen < types:
                next_state[seen + 1] += probability * F(types - seen, types)
        state = dict(next_state)
    return {str(k): state.get(k, F(0)) for k in range(min(types, draws) + 1)}


def base_cases():
    cases = []

    def add(category, family, difficulty, parameters, state, question, probs, derivation, descriptions=None):
        if not all(isinstance(p, F) and 0 <= p <= 1 for p in probs.values()) or sum(probs.values()) != 1:
            raise ValueError(f"Invalid exact oracle: {family}")
        case_index = 1 + sum(c["family"] == family for c in cases)
        criteria = descriptions or {label: label for label in probs}
        if set(criteria) != set(probs):
            raise ValueError("Oracle/option mismatch")
        cases.append(
            {
                "group": f"{REVISION}/{family}/{case_index:02d}",
                "category": category,
                "family": family,
                "difficulty": difficulty,
                "parameters": parameters,
                "state": state,
                "question": question,
                "criteria": criteria,
                "probabilities": probs,
                "derivation": derivation,
            }
        )

    # 1. Direct randomness: impossible labels, unequal weights, certainty, rarity.
    for n in (6, 4):
        probs = {str(k): F(1, n) if k <= n else F(0) for k in range(1, n + 3)}
        add(
            "direct",
            "uniform_support",
            "direct",
            {"faces": n},
            f"A fair die has exactly {n} faces labeled 1 through {n}. It will be rolled once. "
            "Every face is equally likely; it cannot land on an edge or leave the table.",
            "What number will the die show?",
            probs,
            f"Each real face has probability 1/{n}; labels {n + 1} and {n + 2} are impossible.",
        )
    for counts in ((5, 3, 2), (1, 7, 2)):
        labels = ("red", "blue", "green")
        add(
            "direct",
            "weighted_inventory",
            "direct",
            dict(zip(labels, counts)),
            f"A shop jar contains {counts[0]} red, {counts[1]} blue, and {counts[2]} green candies. "
            "One physical candy is selected uniformly at random. The jar has no other candies.",
            "What color is the selected candy?",
            {k: F(v, sum(counts)) for k, v in zip(labels, counts)},
            "A color's probability equals its count divided by the total physical candy count.",
        )
    for color in ("red", "blue"):
        add(
            "direct",
            "deterministic_support",
            "direct",
            {"only_color": color},
            f"A sealed box contains 20 {color} pens and no other objects. A robot selects one pen uniformly.",
            "What color is the selected pen?",
            {k: F(int(k == color)) for k in ("red", "blue", "green")},
            f"All physical pens are {color}; every other color is impossible.",
        )
    for n in (100, 1000):
        add(
            "direct",
            "rare_event",
            "direct",
            {"tickets": n},
            f"A raffle draws exactly one winning ticket uniformly from tickets numbered 1 through {n}. "
            "You own only ticket 1.",
            "Does your ticket win?",
            binary(F(1, n)),
            f"There is one favorable ticket among {n} equally likely tickets.",
        )

    # 2. Composition and mixtures.
    for sides in (4, 6):
        counts = Counter(a + b for a, b in product(range(1, sides + 1), repeat=2))
        add(
            "composition",
            "sum_of_dice",
            "derived",
            {"sides": sides},
            f"A board game rolls two independent fair {sides}-sided dice. Each die has labels 1 through {sides}.",
            "What is the sum of the two face values?",
            {str(k): F(counts[k], sides**2) for k in range(1, 2 * sides + 2)},
            "Enumerate every ordered pair of faces, each with mass 1/sides^2. Sums outside [2,2*sides] have zero mass.",
        )
    for shots, p in ((2, F(7, 10)), (3, F(3, 5))):
        add(
            "composition",
            "independent_hits",
            "derived",
            {"shots": shots, "hit_probability": str(p)},
            f"In a game simulator, a player fires {shots} shots. Each shot independently hits with probability {p}. "
            "Hits do not change the probabilities of later shots.",
            "How many shots hit?",
            {str(k): F(comb(shots, k)) * p**k * (1 - p) ** (shots - k) for k in range(shots + 1)},
            "Binomial law: P(k hits)=C(n,k)*p^k*(1-p)^(n-k).",
        )
    for p, q in ((F(9, 10), F(4, 5)), (F(3, 4), F(3, 4))):
        add(
            "composition",
            "backup_reliability",
            "derived",
            {"a": str(p), "b": str(q)},
            f"A building has two backup generators. During the next outage, A starts with probability {p}, "
            f"and B starts with probability {q}. Their starts are independent. The building has power if at least one starts.",
            "Will the building have power?",
            binary(1 - (1 - p) * (1 - q)),
            "Power fails only if both generators fail: P(power)=1-(1-pA)*(1-pB).",
        )
    for w, a, b in ((F(3, 10), F(9, 10), F(1, 5)), (F(2, 5), F(3, 4), F(1, 4))):
        add(
            "composition",
            "delivery_mixture",
            "derived",
            {"route_a": str(w), "on_time_a": str(a), "on_time_b": str(b)},
            f"A package uses route A with probability {w} and route B otherwise. Conditional on route A, "
            f"it arrives on time with probability {a}; conditional on route B, that probability is {b}. "
            "You have not observed which route was selected.",
            "Does the package arrive on time?",
            binary(w * a + (1 - w) * b),
            "Marginalize the unobserved route: P(on time)=w*pA+(1-w)*pB.",
        )

    # 3. History and conditioning.
    for streak in (5, 10):
        add(
            "history",
            "independent_streak",
            "direct",
            {"previous_heads": streak},
            f"A verified fair coin is tossed repeatedly. Tosses are independent. The last {streak} tosses were all heads. "
            "One more toss will now be made; there is no stopping or selection rule affecting that toss.",
            "What is the next toss?",
            {"heads": F(1, 2), "tails": F(1, 2)},
            "Independence means the observed streak cannot change the next-toss distribution.",
        )
    for red, blue, removed in ((3, 2, 1), (6, 4, 3)):
        add(
            "history",
            "without_replacement",
            "derived",
            {"red": red, "blue": blue, "removed_red": removed},
            f"A drawer initially contained {red} red and {blue} blue socks. Exactly {removed} red socks were "
            "removed and not replaced; no other socks were removed or added. One of the remaining socks is selected uniformly.",
            "What color is the selected sock?",
            {"red": F(red - removed, red + blue - removed), "blue": F(blue, red + blue - removed)},
            "Update remaining counts, then divide each color's count by the remaining total.",
        )
    for n, allowed in ((6, [2, 4, 6]), (8, [5, 6, 7, 8])):
        add(
            "history",
            "restricted_observation",
            "derived",
            {"faces": n, "allowed": allowed},
            f"A fair {n}-sided die labeled 1 through {n} has already been rolled. A sensor reports only whether "
            f"the value belongs to {allowed}. It reports yes, truthfully and without any additional selection process. "
            "No other information is revealed.",
            "What value did the die show?",
            {str(k): F(1, len(allowed)) if k in allowed else F(0) for k in range(1, n + 1)},
            "Condition the uniform prior on membership in the reported subset.",
        )
    for prior, stay, acc, observations in (
        (F(1, 2), F(4, 5), F(3, 4), [True, True]),
        (F(1, 3), F(3, 4), F(4, 5), [True, False, True]),
    ):
        add(
            "history",
            "hidden_state_sensor",
            "compositional",
            {
                "initial_busy": str(prior),
                "persistence": str(stay),
                "sensor_accuracy": str(acc),
                "observations": observations,
            },
            f"A road is either busy or clear. Initially it is busy with probability {prior}. At each later time step, "
            f"it stays in its current state with probability {stay} and flips otherwise, independently of earlier history given "
            f"the current state. At every time step a sensor reports the true state with probability {acc}, otherwise the opposite. "
            "Sensor errors are independent given the state sequence. The first reading is taken before any transition; "
            "one transition occurs before each later reading. The readings in order are "
            f"{['busy' if x else 'clear' for x in observations]}. There is no transition after the final reading.",
            "Is the road busy at the time of the final reading?",
            binary(coin_filter(prior, stay, acc, observations)),
            "Exact hidden Markov forward algorithm: transition, multiply observation likelihoods, then normalize.",
        )

    # 4. Daily evidence: base rates, dependence, and length-biased sampling.
    for prevalence, sensitivity, false_alarm in ((F(1, 100), F(9, 10), F(1, 20)), (F(1, 10), F(4, 5), F(1, 10))):
        posterior = prevalence * sensitivity / (prevalence * sensitivity + (1 - prevalence) * false_alarm)
        add(
            "daily_evidence",
            "defect_screening",
            "derived",
            {"defect_prior": str(prevalence), "sensitivity": str(sensitivity), "false_positive": str(false_alarm)},
            f"An item is selected uniformly from a production population where the defect probability is {prevalence}. "
            f"A scanner flags a defective item with probability {sensitivity} and flags a good item with probability {false_alarm}. "
            "The selected item was scanned exactly once and was flagged.",
            "Is the selected item defective?",
            binary(posterior),
            "Bayes: P(defect|flag)=prior*sensitivity / [prior*sensitivity+(1-prior)*false-positive].",
        )
    for count in (2, 3):
        prior, tpr, fpr = F(1, 10), F(4, 5), F(1, 5)
        add(
            "daily_evidence",
            "independent_repeated_tests",
            "compositional",
            {"tests": count},
            "A randomly selected transaction is fraudulent with probability 1/10. Each detector flags fraud with probability "
            "4/5 and flags a legitimate transaction with probability 1/5. The detectors' outputs are independent "
            f"conditional on the transaction's true status. Exactly {count} detectors were run, and all flagged the transaction.",
            "Is the transaction fraudulent?",
            binary(prior * tpr**count / (prior * tpr**count + (1 - prior) * fpr**count)),
            "Multiply conditionally independent likelihoods before applying Bayes; never multiply posterior probabilities.",
        )
    for count in (2, 3):
        add(
            "daily_evidence",
            "copied_correlated_tests",
            "compositional",
            {"displayed_flags": count},
            "A randomly selected transaction is fraudulent with probability 1/10. One detector flags fraud with probability "
            "4/5 and flags a legitimate transaction with probability 1/5. Its single output is copied unchanged "
            f"into {count} displays. All displays show a flag. No other detector ran and the displays add no new information.",
            "Is the transaction fraudulent?",
            binary(F(4, 13)),
            "Copied flags are one observation: (1/10*4/5)/(1/10*4/5+9/10*1/5)=4/13.",
        )
    for short, long in ((5, 15), (10, 20)):
        add(
            "daily_evidence",
            "random_time_waiting",
            "compositional",
            {"short_minutes": short, "long_minutes": long},
            f"A shuttle timetable repeats forever: a {short}-minute gap, then a {long}-minute gap, then the same cycle. "
            f"You arrive at a time uniformly distributed over one complete {short + long}-minute cycle, independently of the buses. "
            "Arrival exactly at a departure has probability zero.",
            "Which kind of gap contains your arrival time?",
            {"short_gap": F(short, short + long), "long_gap": F(long, short + long)},
            "Uniform time sampling weights intervals by duration, not by the equal number of gaps.",
            {"short_gap": f"The {short}-minute interval.", "long_gap": f"The {long}-minute interval."},
        )

    # 5. Host policies and selection protocols are explicit and distinguish puzzles.
    for n in (3, 5):
        add(
            "disclosure",
            "monty_informed_host",
            "compositional",
            {"doors": n},
            f"A game has {n} doors and one prize, placed uniformly. You select door 1 before receiving any information. "
            f"A host who knows the prize location always opens exactly {n - 2} unchosen doors that contain no prize, "
            "then always offers a switch to the only other unopened door. If more than one legal reveal set exists, "
            "the host chooses uniformly among them. You observe this protocol being completed, but are not told which door numbers were opened.",
            "Where is the prize relative to the final two unopened doors?",
            {"initial_door": F(1, n), "other_unopened_door": F(n - 1, n)},
            "The host never reveals the prize: initial correctness stays 1/n; all other prior mass belongs to the switching door.",
        )
    for n in (3, 5):
        # In this policy, conditioning on a safe random reveal makes the remaining doors equiprobable.
        add(
            "disclosure",
            "monty_uninformed_host",
            "compositional",
            {"doors": n},
            f"A game has {n} doors and one uniformly placed prize. You choose door 1. A host who does not know "
            f"the prize location selects exactly {n - 2} unchosen doors uniformly at random, independently of the prize, "
            "and opens them even if that would reveal the prize. In this game none of the opened doors contains the prize. "
            "There is no additional selection or offer rule. Only your door and one other door remain closed.",
            "Conditional on that safe reveal, where is the prize?",
            {"initial_door": F(1, 2), "other_unopened_door": F(1, 2)},
            "Surviving mass: initial door 1/n; unchosen prize survives with mass ((n-1)/n)*(1/(n-1))=1/n. Normalize.",
        )
    for offer_correct, offer_wrong in ((F(1), F(1, 4)), (F(1, 4), F(1))):
        w_correct, w_wrong = F(1, 3) * offer_correct, F(2, 3) * offer_wrong
        add(
            "disclosure",
            "monty_selective_offer",
            "compositional",
            {"offer_if_initial_correct": str(offer_correct), "offer_if_initial_wrong": str(offer_wrong)},
            "Three doors hide one uniformly placed prize. You choose door 1. A knowledgeable host always opens one "
            "unchosen empty door, choosing uniformly if two are available. After opening it, the host offers a switch "
            f"with probability {offer_correct} if your original door has the prize, and probability {offer_wrong} if it does not. "
            "This offer decision depends on nothing else. You are offered a switch. The identity of the opened door is not provided.",
            "Given the offer, where is the prize?",
            {"initial_door": w_correct / (w_correct + w_wrong), "other_unopened_door": w_wrong / (w_correct + w_wrong)},
            "Weight initial-correct and initial-wrong prior masses by their offer likelihoods, then normalize.",
        )
    for protocol in ("at_least_one", "random_child"):
        state = "In a toy population every family has exactly two children, distinguished as older and younger. "
        state += "Each child's recorded category is independently B or G with probability 1/2 each. "
        if protocol == "at_least_one":
            state += "A family is sampled uniformly. A truthful device reports only whether at least one child is B. It reports yes."
            p = F(1, 3)
            derivation = "Of BB, BG, GB, GG, the observation retains BB, BG, GB equally. Only BB has two B children."
        else:
            state += "A family is sampled uniformly, then one of its two children is selected uniformly and independently. The selected child is B."
            p = F(1, 2)
            derivation = "BB produces the observation with likelihood 1; BG and GB with 1/2 each. Normalize BB mass 1/4 by evidence mass 1/2."
        add(
            "disclosure",
            "two_child_selection",
            "compositional",
            {"observation_protocol": protocol},
            state,
            "Are both children in this family category B?",
            binary(p),
            derivation,
        )

    # 6. Sequential/finite combinatorics, all exact (no Monte Carlo targets).
    for types, purchases in ((3, 3), (4, 5)):
        add(
            "sequential",
            "coupon_collection",
            "compositional",
            {"types": types, "purchases": purchases},
            f"Each cereal box independently contains one of {types} sticker types, with equal probability for every type. "
            f"You initially have no stickers and buy exactly {purchases} boxes. Duplicates are allowed and you do not trade.",
            "How many distinct sticker types do you have after opening all boxes?",
            distinct_distribution(types, purchases),
            "Occupancy DP: with k types seen, the next draw repeats with probability k/m and adds a new type with (m-k)/m.",
        )
    for days, people in ((10, 4), (30, 8)):
        distinct = F(1)
        for i in range(people):
            distinct *= F(days - i, days)
        add(
            "sequential",
            "birthday_collision",
            "derived",
            {"possible_dates": days, "people": people},
            f"A toy calendar has exactly {days} possible birthday dates. Each of {people} people's birthdays is "
            "independently uniform across those dates. Ignore all real-world calendar effects.",
            "Do at least two people share a birthday?",
            binary(1 - distinct),
            "Use the complement: P(no collision)=product over i=0..n-1 of (d-i)/d.",
        )
    for start, goal, win in ((2, 5, F(1, 2)), (2, 5, F(2, 5))):
        if win == F(1, 2):
            success = F(start, goal)
        else:
            ratio = (1 - win) / win
            success = (1 - ratio**start) / (1 - ratio**goal)
        add(
            "sequential",
            "gambler_ruin",
            "compositional",
            {"start": start, "goal": goal, "win": str(win)},
            f"A simulated game starts with {start} tokens. Each round independently adds one token with probability "
            f"{win} or removes one otherwise. Play stops immediately at 0 or {goal} tokens. There is no time limit or other stopping rule.",
            f"Does the game reach {goal} tokens before 0?",
            binary(success),
            "Solve u(i)=p*u(i+1)+(1-p)*u(i-1), u(0)=0,u(N)=1. For p=1/2 use i/N; otherwise (1-r^i)/(1-r^N), r=(1-p)/p.",
        )
    for threshold, p in ((3, F(1, 5)), (4, F(1, 4))):
        add(
            "sequential",
            "guaranteed_reward",
            "derived",
            {"guarantee_attempt": threshold, "natural_success": str(p)},
            f"A game gives a reward on each ordinary attempt independently with probability {p}. You stop at your "
            f"first reward. If all first {threshold - 1} attempts fail, attempt {threshold} gives the reward with certainty. "
            "You begin with no prior attempts and will continue until the first reward.",
            "On which attempt do you first receive the reward?",
            {
                str(k): (1 - p) ** (k - 1) * p if k < threshold else (1 - p) ** (threshold - 1)
                for k in range(1, threshold + 1)
            },
            "For k below the guarantee use geometric mass (1-p)^(k-1)*p; all remaining mass is on the guaranteed attempt.",
        )
    return cases


def build_records():
    inputs, references = [], []
    for case in base_cases():
        for variant in ("canonical", "reversed_options"):
            labels = list(case["criteria"])
            if variant == "reversed_options":
                labels.reverse()
            identity = case["group"] + "/" + variant
            row = {
                "id": identity,
                "state": case["state"],
                "questions": {
                    "decision": {
                        "type": "choice",
                        "instructions": case["question"] + " " + INSTRUCTION,
                        "criteria": {k: case["criteria"][k] for k in labels},
                    }
                },
            }
            inputs.append(row)
            references.append(
                {
                    "id": identity,
                    "group": case["group"],
                    "variant": variant,
                    "category": case["category"],
                    "family": case["family"],
                    "difficulty": case["difficulty"],
                    "parameters": case["parameters"],
                    "input_sha256": evidence_hash(row),
                    "field": "decision",
                    "gold_probs": {k: float(case["probabilities"][k]) for k in labels},
                    "gold_fractions": {k: str(case["probabilities"][k]) for k in labels},
                    "derivation": case["derivation"],
                    "split": "evaluation_only",
                }
            )
    validate(inputs, references)
    return inputs, references


def validate(inputs, references):
    if len(inputs) != len(references) or len({r["id"] for r in inputs}) != len(inputs):
        raise ValueError("Identity/count mismatch")
    groups = defaultdict(list)
    for row, ref in zip(inputs, references):
        if set(row) != {"id", "state", "questions"} or row["id"] != ref["id"]:
            raise ValueError("Input envelope contains supervision or identity mismatch")
        if evidence_hash(row) != ref["input_sha256"]:
            raise ValueError("Input hash mismatch")
        q = row["questions"][ref["field"]]
        if set(q) != {"type", "instructions", "criteria"} or q["type"] != "choice":
            raise ValueError("Unexpected input schema")
        probs = {k: F(v) for k, v in ref["gold_fractions"].items()}
        if set(probs) != set(q["criteria"]) or sum(probs.values()) != 1 or any(p < 0 or p > 1 for p in probs.values()):
            raise ValueError("Invalid gold distribution")
        if any(float(p) != ref["gold_probs"][k] for k, p in probs.items()):
            raise ValueError("Float/fraction mismatch")
        groups[ref["group"]].append((row, ref))
    for pairs in groups.values():
        if len(pairs) != 2:
            raise ValueError("Expected a pair of answer-order variants")
        (a, ra), (b, rb) = pairs
        qa, qb = a["questions"]["decision"], b["questions"]["decision"]
        if a["state"] != b["state"] or qa["instructions"] != qb["instructions"] or ra["gold_probs"] != rb["gold_probs"]:
            raise ValueError("Variants are not equivalent")
        if list(qa["criteria"]) != list(reversed(qb["criteria"])):
            raise ValueError("Variant option order mismatch")


def write_benchmark(output):
    inputs, refs = build_records()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("inputs.jsonl", inputs), ("references.jsonl", refs)):
        (output / name).write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
    write_json(
        output / "manifest.json",
        {
            "revision": REVISION,
            "purpose": "evaluation_only",
            "training_ready": False,
            "rows": len(inputs),
            "independent_parameter_groups": len({r["group"] for r in refs}),
            "families": len({r["family"] for r in refs}),
            "categories": CATEGORIES,
            "counts_by_category": dict(Counter(r["category"] for r in refs)),
            "counts_by_difficulty": dict(Counter(r["difficulty"] for r in refs)),
            "oracle": "Exact Fraction arithmetic, enumeration, analytic formulas and dynamic programming; no Monte Carlo",
            "files": {name: sha256(output / name) for name in ("inputs.jsonl", "references.jsonl")},
            "generator_sha256": sha256(__file__),
            "seed": None,
            "confidence_policy": "valid API confidence when present; otherwise max(P)",
            "ece": "10-bin population ECE: replace sampled correctness by gold probability of the model argmax",
            "fit_split": None,
            "note": "Pilot test set; no fitting on these references. Variants share a group. All questions use choice.",
        },
    )
    lines = [
        "# Known-distribution calibration pilot",
        "",
        "96 questions; 48 parameter groups; 24 families; 6 categories.",
        "",
        "This is an evaluation-only pilot. Each base problem has canonical and reversed answer-order variants. "
        "The variants are paired robustness measurements, not independent observations. Model results must be obtained with the separate scorer.",
        "",
        "## Categories",
        "",
        "| Category | Families | Questions |",
        "|---|---|---:|",
    ]
    for category, title in CATEGORIES.items():
        names = sorted({r["family"] for r in refs if r["category"] == category})
        lines.append(f"| {title} | {', '.join(names)} | {sum(r['category'] == category for r in refs)} |")
    lines += [
        "",
        "## Evaluation contract",
        "",
        "Send only state and questions from inputs.jsonl to Jev; add the chosen model name at the transport layer. "
        "Pass the same evidence to the local model compiler. IDs are for result joins. Do not send references, "
        "derivations, or manifest metadata as evidence. Preserve question/option order and do not truncate inputs.",
        "",
        "These tasks predict random outcomes, not numerical probability answers. Do not turn the oracle argmax "
        "into a hard label. There is no sampled outcome and thus no empirical classification accuracy. "
        "Ordinary hard-label evaluators are not appropriate for this dataset.",
        "",
        "Score full-distribution TVD, excess expected Brier, expected multiclass Brier and impossible-outcome mass. "
        "Expected ECE uses API confidence when available, max(P) otherwise, and gold probability of the predicted label "
        "as expected correctness. This is a population calculation, not empirical ECE over observed outcomes. "
        "A model can match the full distribution yet have nonzero ECE if its separate confidence statistic differs from max(P).",
        "",
        "All mechanisms are synthetic and explicit. Medical/financial/game-like wording conveys no real-world rates. "
        "The pilot includes selected Monty Hall variants; biased-prior/partial-reveal versions, queues, payoff optimization, "
        "Simpson's paradox, and other extensions remain outside this first set.",
        "",
        "## Complete base-case review",
        "",
        "Fractions below are gold references; this document is not a model input.",
    ]
    for case in base_cases():
        lines += [
            "",
            f"### {case['group']}",
            "",
            f"Category: {CATEGORIES[case['category']]}; difficulty: {case['difficulty']}.",
            "",
            case["state"],
            "",
            case["question"],
            "",
            "| Outcome | Exact probability |",
            "|---|---:|",
        ]
        lines += [f"| {label} | {p} |" for label, p in case["probabilities"].items()]
        lines += ["", case["derivation"]]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), "rows": len(inputs), "groups": len(refs) // 2, "families": 24}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_benchmark(args.output), indent=2))


if __name__ == "__main__":
    main()
