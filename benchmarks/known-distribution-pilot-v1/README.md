# Known-distribution calibration pilot

96 questions; 48 parameter groups; 24 families; 6 categories.

This is an evaluation-only pilot. Each base problem has canonical and reversed answer-order variants. The variants are paired robustness measurements, not independent observations. Model results must be obtained with the separate scorer.

## Categories

| Category | Families | Questions |
|---|---|---:|
| Direct randomness and support | deterministic_support, rare_event, uniform_support, weighted_inventory | 16 |
| Composed events and mixtures | backup_reliability, delivery_mixture, independent_hits, sum_of_dice | 16 |
| History, conditioning, and hidden state | hidden_state_sensor, independent_streak, restricted_observation, without_replacement | 16 |
| Daily evidence and observation bias | copied_correlated_tests, defect_screening, independent_repeated_tests, random_time_waiting | 16 |
| Selective disclosure and probability puzzles | monty_informed_host, monty_selective_offer, monty_uninformed_host, two_child_selection | 16 |
| Sequential and combinatorial processes | birthday_collision, coupon_collection, gambler_ruin, guaranteed_reward | 16 |

## Evaluation contract

Send only state and questions from inputs.jsonl to Jev; add the chosen model name at the transport layer. Pass the same evidence to the local model compiler. IDs are for result joins. Do not send references, derivations, or manifest metadata as evidence. Preserve question/option order and do not truncate inputs.

These tasks predict random outcomes, not numerical probability answers. Do not turn the oracle argmax into a hard label. There is no sampled outcome and thus no empirical classification accuracy. Ordinary hard-label evaluators are not appropriate for this dataset.

Score full-distribution TVD, excess expected Brier, expected multiclass Brier and impossible-outcome mass. Expected ECE uses API confidence when available, max(P) otherwise, and gold probability of the predicted label as expected correctness. This is a population calculation, not empirical ECE over observed outcomes. A model can match the full distribution yet have nonzero ECE if its separate confidence statistic differs from max(P).

All mechanisms are synthetic and explicit. Medical/financial/game-like wording conveys no real-world rates. The pilot includes selected Monty Hall variants; biased-prior/partial-reveal versions, queues, payoff optimization, Simpson's paradox, and other extensions remain outside this first set.

## Complete base-case review

Fractions below are gold references; this document is not a model input.

### known-distribution-pilot-v1/uniform_support/01

Category: Direct randomness and support; difficulty: direct.

A fair die has exactly 6 faces labeled 1 through 6. It will be rolled once. Every face is equally likely; it cannot land on an edge or leave the table.

What number will the die show?

| Outcome | Exact probability |
|---|---:|
| 1 | 1/6 |
| 2 | 1/6 |
| 3 | 1/6 |
| 4 | 1/6 |
| 5 | 1/6 |
| 6 | 1/6 |
| 7 | 0 |
| 8 | 0 |

Each real face has probability 1/6; labels 7 and 8 are impossible.

### known-distribution-pilot-v1/uniform_support/02

Category: Direct randomness and support; difficulty: direct.

A fair die has exactly 4 faces labeled 1 through 4. It will be rolled once. Every face is equally likely; it cannot land on an edge or leave the table.

What number will the die show?

| Outcome | Exact probability |
|---|---:|
| 1 | 1/4 |
| 2 | 1/4 |
| 3 | 1/4 |
| 4 | 1/4 |
| 5 | 0 |
| 6 | 0 |

Each real face has probability 1/4; labels 5 and 6 are impossible.

### known-distribution-pilot-v1/weighted_inventory/01

Category: Direct randomness and support; difficulty: direct.

A shop jar contains 5 red, 3 blue, and 2 green candies. One physical candy is selected uniformly at random. The jar has no other candies.

What color is the selected candy?

| Outcome | Exact probability |
|---|---:|
| red | 1/2 |
| blue | 3/10 |
| green | 1/5 |

A color's probability equals its count divided by the total physical candy count.

### known-distribution-pilot-v1/weighted_inventory/02

Category: Direct randomness and support; difficulty: direct.

A shop jar contains 1 red, 7 blue, and 2 green candies. One physical candy is selected uniformly at random. The jar has no other candies.

What color is the selected candy?

| Outcome | Exact probability |
|---|---:|
| red | 1/10 |
| blue | 7/10 |
| green | 1/5 |

A color's probability equals its count divided by the total physical candy count.

### known-distribution-pilot-v1/deterministic_support/01

Category: Direct randomness and support; difficulty: direct.

A sealed box contains 20 red pens and no other objects. A robot selects one pen uniformly.

What color is the selected pen?

| Outcome | Exact probability |
|---|---:|
| red | 1 |
| blue | 0 |
| green | 0 |

All physical pens are red; every other color is impossible.

### known-distribution-pilot-v1/deterministic_support/02

Category: Direct randomness and support; difficulty: direct.

A sealed box contains 20 blue pens and no other objects. A robot selects one pen uniformly.

What color is the selected pen?

| Outcome | Exact probability |
|---|---:|
| red | 0 |
| blue | 1 |
| green | 0 |

All physical pens are blue; every other color is impossible.

### known-distribution-pilot-v1/rare_event/01

Category: Direct randomness and support; difficulty: direct.

A raffle draws exactly one winning ticket uniformly from tickets numbered 1 through 100. You own only ticket 1.

Does your ticket win?

| Outcome | Exact probability |
|---|---:|
| yes | 1/100 |
| no | 99/100 |

There is one favorable ticket among 100 equally likely tickets.

### known-distribution-pilot-v1/rare_event/02

Category: Direct randomness and support; difficulty: direct.

A raffle draws exactly one winning ticket uniformly from tickets numbered 1 through 1000. You own only ticket 1.

Does your ticket win?

| Outcome | Exact probability |
|---|---:|
| yes | 1/1000 |
| no | 999/1000 |

There is one favorable ticket among 1000 equally likely tickets.

### known-distribution-pilot-v1/sum_of_dice/01

Category: Composed events and mixtures; difficulty: derived.

A board game rolls two independent fair 4-sided dice. Each die has labels 1 through 4.

What is the sum of the two face values?

| Outcome | Exact probability |
|---|---:|
| 1 | 0 |
| 2 | 1/16 |
| 3 | 1/8 |
| 4 | 3/16 |
| 5 | 1/4 |
| 6 | 3/16 |
| 7 | 1/8 |
| 8 | 1/16 |
| 9 | 0 |

Enumerate every ordered pair of faces, each with mass 1/sides^2. Sums outside [2,2*sides] have zero mass.

### known-distribution-pilot-v1/sum_of_dice/02

Category: Composed events and mixtures; difficulty: derived.

A board game rolls two independent fair 6-sided dice. Each die has labels 1 through 6.

What is the sum of the two face values?

| Outcome | Exact probability |
|---|---:|
| 1 | 0 |
| 2 | 1/36 |
| 3 | 1/18 |
| 4 | 1/12 |
| 5 | 1/9 |
| 6 | 5/36 |
| 7 | 1/6 |
| 8 | 5/36 |
| 9 | 1/9 |
| 10 | 1/12 |
| 11 | 1/18 |
| 12 | 1/36 |
| 13 | 0 |

Enumerate every ordered pair of faces, each with mass 1/sides^2. Sums outside [2,2*sides] have zero mass.

### known-distribution-pilot-v1/independent_hits/01

Category: Composed events and mixtures; difficulty: derived.

In a game simulator, a player fires 2 shots. Each shot independently hits with probability 7/10. Hits do not change the probabilities of later shots.

How many shots hit?

| Outcome | Exact probability |
|---|---:|
| 0 | 9/100 |
| 1 | 21/50 |
| 2 | 49/100 |

Binomial law: P(k hits)=C(n,k)*p^k*(1-p)^(n-k).

### known-distribution-pilot-v1/independent_hits/02

Category: Composed events and mixtures; difficulty: derived.

In a game simulator, a player fires 3 shots. Each shot independently hits with probability 3/5. Hits do not change the probabilities of later shots.

How many shots hit?

| Outcome | Exact probability |
|---|---:|
| 0 | 8/125 |
| 1 | 36/125 |
| 2 | 54/125 |
| 3 | 27/125 |

Binomial law: P(k hits)=C(n,k)*p^k*(1-p)^(n-k).

### known-distribution-pilot-v1/backup_reliability/01

Category: Composed events and mixtures; difficulty: derived.

A building has two backup generators. During the next outage, A starts with probability 9/10, and B starts with probability 4/5. Their starts are independent. The building has power if at least one starts.

Will the building have power?

| Outcome | Exact probability |
|---|---:|
| yes | 49/50 |
| no | 1/50 |

Power fails only if both generators fail: P(power)=1-(1-pA)*(1-pB).

### known-distribution-pilot-v1/backup_reliability/02

Category: Composed events and mixtures; difficulty: derived.

A building has two backup generators. During the next outage, A starts with probability 3/4, and B starts with probability 3/4. Their starts are independent. The building has power if at least one starts.

Will the building have power?

| Outcome | Exact probability |
|---|---:|
| yes | 15/16 |
| no | 1/16 |

Power fails only if both generators fail: P(power)=1-(1-pA)*(1-pB).

### known-distribution-pilot-v1/delivery_mixture/01

Category: Composed events and mixtures; difficulty: derived.

A package uses route A with probability 3/10 and route B otherwise. Conditional on route A, it arrives on time with probability 9/10; conditional on route B, that probability is 1/5. You have not observed which route was selected.

Does the package arrive on time?

| Outcome | Exact probability |
|---|---:|
| yes | 41/100 |
| no | 59/100 |

Marginalize the unobserved route: P(on time)=w*pA+(1-w)*pB.

### known-distribution-pilot-v1/delivery_mixture/02

Category: Composed events and mixtures; difficulty: derived.

A package uses route A with probability 2/5 and route B otherwise. Conditional on route A, it arrives on time with probability 3/4; conditional on route B, that probability is 1/4. You have not observed which route was selected.

Does the package arrive on time?

| Outcome | Exact probability |
|---|---:|
| yes | 9/20 |
| no | 11/20 |

Marginalize the unobserved route: P(on time)=w*pA+(1-w)*pB.

### known-distribution-pilot-v1/independent_streak/01

Category: History, conditioning, and hidden state; difficulty: direct.

A verified fair coin is tossed repeatedly. Tosses are independent. The last 5 tosses were all heads. One more toss will now be made; there is no stopping or selection rule affecting that toss.

What is the next toss?

| Outcome | Exact probability |
|---|---:|
| heads | 1/2 |
| tails | 1/2 |

Independence means the observed streak cannot change the next-toss distribution.

### known-distribution-pilot-v1/independent_streak/02

Category: History, conditioning, and hidden state; difficulty: direct.

A verified fair coin is tossed repeatedly. Tosses are independent. The last 10 tosses were all heads. One more toss will now be made; there is no stopping or selection rule affecting that toss.

What is the next toss?

| Outcome | Exact probability |
|---|---:|
| heads | 1/2 |
| tails | 1/2 |

Independence means the observed streak cannot change the next-toss distribution.

### known-distribution-pilot-v1/without_replacement/01

Category: History, conditioning, and hidden state; difficulty: derived.

A drawer initially contained 3 red and 2 blue socks. Exactly 1 red socks were removed and not replaced; no other socks were removed or added. One of the remaining socks is selected uniformly.

What color is the selected sock?

| Outcome | Exact probability |
|---|---:|
| red | 1/2 |
| blue | 1/2 |

Update remaining counts, then divide each color's count by the remaining total.

### known-distribution-pilot-v1/without_replacement/02

Category: History, conditioning, and hidden state; difficulty: derived.

A drawer initially contained 6 red and 4 blue socks. Exactly 3 red socks were removed and not replaced; no other socks were removed or added. One of the remaining socks is selected uniformly.

What color is the selected sock?

| Outcome | Exact probability |
|---|---:|
| red | 3/7 |
| blue | 4/7 |

Update remaining counts, then divide each color's count by the remaining total.

### known-distribution-pilot-v1/restricted_observation/01

Category: History, conditioning, and hidden state; difficulty: derived.

A fair 6-sided die labeled 1 through 6 has already been rolled. A sensor reports only whether the value belongs to [2, 4, 6]. It reports yes, truthfully and without any additional selection process. No other information is revealed.

What value did the die show?

| Outcome | Exact probability |
|---|---:|
| 1 | 0 |
| 2 | 1/3 |
| 3 | 0 |
| 4 | 1/3 |
| 5 | 0 |
| 6 | 1/3 |

Condition the uniform prior on membership in the reported subset.

### known-distribution-pilot-v1/restricted_observation/02

Category: History, conditioning, and hidden state; difficulty: derived.

A fair 8-sided die labeled 1 through 8 has already been rolled. A sensor reports only whether the value belongs to [5, 6, 7, 8]. It reports yes, truthfully and without any additional selection process. No other information is revealed.

What value did the die show?

| Outcome | Exact probability |
|---|---:|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 1/4 |
| 6 | 1/4 |
| 7 | 1/4 |
| 8 | 1/4 |

Condition the uniform prior on membership in the reported subset.

### known-distribution-pilot-v1/hidden_state_sensor/01

Category: History, conditioning, and hidden state; difficulty: compositional.

A road is either busy or clear. Initially it is busy with probability 1/2. At each later time step, it stays in its current state with probability 4/5 and flips otherwise, independently of earlier history given the current state. At every time step a sensor reports the true state with probability 3/4, otherwise the opposite. Sensor errors are independent given the state sequence. The first reading is taken before any transition; one transition occurs before each later reading. The readings in order are ['busy', 'busy']. There is no transition after the final reading.

Is the road busy at the time of the final reading?

| Outcome | Exact probability |
|---|---:|
| yes | 39/46 |
| no | 7/46 |

Exact hidden Markov forward algorithm: transition, multiply observation likelihoods, then normalize.

### known-distribution-pilot-v1/hidden_state_sensor/02

Category: History, conditioning, and hidden state; difficulty: compositional.

A road is either busy or clear. Initially it is busy with probability 1/3. At each later time step, it stays in its current state with probability 3/4 and flips otherwise, independently of earlier history given the current state. At every time step a sensor reports the true state with probability 4/5, otherwise the opposite. Sensor errors are independent given the state sequence. The first reading is taken before any transition; one transition occurs before each later reading. The readings in order are ['busy', 'clear', 'busy']. There is no transition after the final reading.

Is the road busy at the time of the final reading?

| Outcome | Exact probability |
|---|---:|
| yes | 164/231 |
| no | 67/231 |

Exact hidden Markov forward algorithm: transition, multiply observation likelihoods, then normalize.

### known-distribution-pilot-v1/defect_screening/01

Category: Daily evidence and observation bias; difficulty: derived.

An item is selected uniformly from a production population where the defect probability is 1/100. A scanner flags a defective item with probability 9/10 and flags a good item with probability 1/20. The selected item was scanned exactly once and was flagged.

Is the selected item defective?

| Outcome | Exact probability |
|---|---:|
| yes | 2/13 |
| no | 11/13 |

Bayes: P(defect|flag)=prior*sensitivity / [prior*sensitivity+(1-prior)*false-positive].

### known-distribution-pilot-v1/defect_screening/02

Category: Daily evidence and observation bias; difficulty: derived.

An item is selected uniformly from a production population where the defect probability is 1/10. A scanner flags a defective item with probability 4/5 and flags a good item with probability 1/10. The selected item was scanned exactly once and was flagged.

Is the selected item defective?

| Outcome | Exact probability |
|---|---:|
| yes | 8/17 |
| no | 9/17 |

Bayes: P(defect|flag)=prior*sensitivity / [prior*sensitivity+(1-prior)*false-positive].

### known-distribution-pilot-v1/independent_repeated_tests/01

Category: Daily evidence and observation bias; difficulty: compositional.

A randomly selected transaction is fraudulent with probability 1/10. Each detector flags fraud with probability 4/5 and flags a legitimate transaction with probability 1/5. The detectors' outputs are independent conditional on the transaction's true status. Exactly 2 detectors were run, and all flagged the transaction.

Is the transaction fraudulent?

| Outcome | Exact probability |
|---|---:|
| yes | 16/25 |
| no | 9/25 |

Multiply conditionally independent likelihoods before applying Bayes; never multiply posterior probabilities.

### known-distribution-pilot-v1/independent_repeated_tests/02

Category: Daily evidence and observation bias; difficulty: compositional.

A randomly selected transaction is fraudulent with probability 1/10. Each detector flags fraud with probability 4/5 and flags a legitimate transaction with probability 1/5. The detectors' outputs are independent conditional on the transaction's true status. Exactly 3 detectors were run, and all flagged the transaction.

Is the transaction fraudulent?

| Outcome | Exact probability |
|---|---:|
| yes | 64/73 |
| no | 9/73 |

Multiply conditionally independent likelihoods before applying Bayes; never multiply posterior probabilities.

### known-distribution-pilot-v1/copied_correlated_tests/01

Category: Daily evidence and observation bias; difficulty: compositional.

A randomly selected transaction is fraudulent with probability 1/10. One detector flags fraud with probability 4/5 and flags a legitimate transaction with probability 1/5. Its single output is copied unchanged into 2 displays. All displays show a flag. No other detector ran and the displays add no new information.

Is the transaction fraudulent?

| Outcome | Exact probability |
|---|---:|
| yes | 4/13 |
| no | 9/13 |

Copied flags are one observation: (1/10*4/5)/(1/10*4/5+9/10*1/5)=4/13.

### known-distribution-pilot-v1/copied_correlated_tests/02

Category: Daily evidence and observation bias; difficulty: compositional.

A randomly selected transaction is fraudulent with probability 1/10. One detector flags fraud with probability 4/5 and flags a legitimate transaction with probability 1/5. Its single output is copied unchanged into 3 displays. All displays show a flag. No other detector ran and the displays add no new information.

Is the transaction fraudulent?

| Outcome | Exact probability |
|---|---:|
| yes | 4/13 |
| no | 9/13 |

Copied flags are one observation: (1/10*4/5)/(1/10*4/5+9/10*1/5)=4/13.

### known-distribution-pilot-v1/random_time_waiting/01

Category: Daily evidence and observation bias; difficulty: compositional.

A shuttle timetable repeats forever: a 5-minute gap, then a 15-minute gap, then the same cycle. You arrive at a time uniformly distributed over one complete 20-minute cycle, independently of the buses. Arrival exactly at a departure has probability zero.

Which kind of gap contains your arrival time?

| Outcome | Exact probability |
|---|---:|
| short_gap | 1/4 |
| long_gap | 3/4 |

Uniform time sampling weights intervals by duration, not by the equal number of gaps.

### known-distribution-pilot-v1/random_time_waiting/02

Category: Daily evidence and observation bias; difficulty: compositional.

A shuttle timetable repeats forever: a 10-minute gap, then a 20-minute gap, then the same cycle. You arrive at a time uniformly distributed over one complete 30-minute cycle, independently of the buses. Arrival exactly at a departure has probability zero.

Which kind of gap contains your arrival time?

| Outcome | Exact probability |
|---|---:|
| short_gap | 1/3 |
| long_gap | 2/3 |

Uniform time sampling weights intervals by duration, not by the equal number of gaps.

### known-distribution-pilot-v1/monty_informed_host/01

Category: Selective disclosure and probability puzzles; difficulty: compositional.

A game has 3 doors and one prize, placed uniformly. You select door 1 before receiving any information. A host who knows the prize location always opens exactly 1 unchosen doors that contain no prize, then always offers a switch to the only other unopened door. If more than one legal reveal set exists, the host chooses uniformly among them. You observe this protocol being completed, but are not told which door numbers were opened.

Where is the prize relative to the final two unopened doors?

| Outcome | Exact probability |
|---|---:|
| initial_door | 1/3 |
| other_unopened_door | 2/3 |

The host never reveals the prize: initial correctness stays 1/n; all other prior mass belongs to the switching door.

### known-distribution-pilot-v1/monty_informed_host/02

Category: Selective disclosure and probability puzzles; difficulty: compositional.

A game has 5 doors and one prize, placed uniformly. You select door 1 before receiving any information. A host who knows the prize location always opens exactly 3 unchosen doors that contain no prize, then always offers a switch to the only other unopened door. If more than one legal reveal set exists, the host chooses uniformly among them. You observe this protocol being completed, but are not told which door numbers were opened.

Where is the prize relative to the final two unopened doors?

| Outcome | Exact probability |
|---|---:|
| initial_door | 1/5 |
| other_unopened_door | 4/5 |

The host never reveals the prize: initial correctness stays 1/n; all other prior mass belongs to the switching door.

### known-distribution-pilot-v1/monty_uninformed_host/01

Category: Selective disclosure and probability puzzles; difficulty: compositional.

A game has 3 doors and one uniformly placed prize. You choose door 1. A host who does not know the prize location selects exactly 1 unchosen doors uniformly at random, independently of the prize, and opens them even if that would reveal the prize. In this game none of the opened doors contains the prize. There is no additional selection or offer rule. Only your door and one other door remain closed.

Conditional on that safe reveal, where is the prize?

| Outcome | Exact probability |
|---|---:|
| initial_door | 1/2 |
| other_unopened_door | 1/2 |

Surviving mass: initial door 1/n; unchosen prize survives with mass ((n-1)/n)*(1/(n-1))=1/n. Normalize.

### known-distribution-pilot-v1/monty_uninformed_host/02

Category: Selective disclosure and probability puzzles; difficulty: compositional.

A game has 5 doors and one uniformly placed prize. You choose door 1. A host who does not know the prize location selects exactly 3 unchosen doors uniformly at random, independently of the prize, and opens them even if that would reveal the prize. In this game none of the opened doors contains the prize. There is no additional selection or offer rule. Only your door and one other door remain closed.

Conditional on that safe reveal, where is the prize?

| Outcome | Exact probability |
|---|---:|
| initial_door | 1/2 |
| other_unopened_door | 1/2 |

Surviving mass: initial door 1/n; unchosen prize survives with mass ((n-1)/n)*(1/(n-1))=1/n. Normalize.

### known-distribution-pilot-v1/monty_selective_offer/01

Category: Selective disclosure and probability puzzles; difficulty: compositional.

Three doors hide one uniformly placed prize. You choose door 1. A knowledgeable host always opens one unchosen empty door, choosing uniformly if two are available. After opening it, the host offers a switch with probability 1 if your original door has the prize, and probability 1/4 if it does not. This offer decision depends on nothing else. You are offered a switch. The identity of the opened door is not provided.

Given the offer, where is the prize?

| Outcome | Exact probability |
|---|---:|
| initial_door | 2/3 |
| other_unopened_door | 1/3 |

Weight initial-correct and initial-wrong prior masses by their offer likelihoods, then normalize.

### known-distribution-pilot-v1/monty_selective_offer/02

Category: Selective disclosure and probability puzzles; difficulty: compositional.

Three doors hide one uniformly placed prize. You choose door 1. A knowledgeable host always opens one unchosen empty door, choosing uniformly if two are available. After opening it, the host offers a switch with probability 1/4 if your original door has the prize, and probability 1 if it does not. This offer decision depends on nothing else. You are offered a switch. The identity of the opened door is not provided.

Given the offer, where is the prize?

| Outcome | Exact probability |
|---|---:|
| initial_door | 1/9 |
| other_unopened_door | 8/9 |

Weight initial-correct and initial-wrong prior masses by their offer likelihoods, then normalize.

### known-distribution-pilot-v1/two_child_selection/01

Category: Selective disclosure and probability puzzles; difficulty: compositional.

In a toy population every family has exactly two children, distinguished as older and younger. Each child's recorded category is independently B or G with probability 1/2 each. A family is sampled uniformly. A truthful device reports only whether at least one child is B. It reports yes.

Are both children in this family category B?

| Outcome | Exact probability |
|---|---:|
| yes | 1/3 |
| no | 2/3 |

Of BB, BG, GB, GG, the observation retains BB, BG, GB equally. Only BB has two B children.

### known-distribution-pilot-v1/two_child_selection/02

Category: Selective disclosure and probability puzzles; difficulty: compositional.

In a toy population every family has exactly two children, distinguished as older and younger. Each child's recorded category is independently B or G with probability 1/2 each. A family is sampled uniformly, then one of its two children is selected uniformly and independently. The selected child is B.

Are both children in this family category B?

| Outcome | Exact probability |
|---|---:|
| yes | 1/2 |
| no | 1/2 |

BB produces the observation with likelihood 1; BG and GB with 1/2 each. Normalize BB mass 1/4 by evidence mass 1/2.

### known-distribution-pilot-v1/coupon_collection/01

Category: Sequential and combinatorial processes; difficulty: compositional.

Each cereal box independently contains one of 3 sticker types, with equal probability for every type. You initially have no stickers and buy exactly 3 boxes. Duplicates are allowed and you do not trade.

How many distinct sticker types do you have after opening all boxes?

| Outcome | Exact probability |
|---|---:|
| 0 | 0 |
| 1 | 1/9 |
| 2 | 2/3 |
| 3 | 2/9 |

Occupancy DP: with k types seen, the next draw repeats with probability k/m and adds a new type with (m-k)/m.

### known-distribution-pilot-v1/coupon_collection/02

Category: Sequential and combinatorial processes; difficulty: compositional.

Each cereal box independently contains one of 4 sticker types, with equal probability for every type. You initially have no stickers and buy exactly 5 boxes. Duplicates are allowed and you do not trade.

How many distinct sticker types do you have after opening all boxes?

| Outcome | Exact probability |
|---|---:|
| 0 | 0 |
| 1 | 1/256 |
| 2 | 45/256 |
| 3 | 75/128 |
| 4 | 15/64 |

Occupancy DP: with k types seen, the next draw repeats with probability k/m and adds a new type with (m-k)/m.

### known-distribution-pilot-v1/birthday_collision/01

Category: Sequential and combinatorial processes; difficulty: derived.

A toy calendar has exactly 10 possible birthday dates. Each of 4 people's birthdays is independently uniform across those dates. Ignore all real-world calendar effects.

Do at least two people share a birthday?

| Outcome | Exact probability |
|---|---:|
| yes | 62/125 |
| no | 63/125 |

Use the complement: P(no collision)=product over i=0..n-1 of (d-i)/d.

### known-distribution-pilot-v1/birthday_collision/02

Category: Sequential and combinatorial processes; difficulty: derived.

A toy calendar has exactly 30 possible birthday dates. Each of 8 people's birthdays is independently uniform across those dates. Ignore all real-world calendar effects.

Do at least two people share a birthday?

| Outcome | Exact probability |
|---|---:|
| yes | 108053/168750 |
| no | 60697/168750 |

Use the complement: P(no collision)=product over i=0..n-1 of (d-i)/d.

### known-distribution-pilot-v1/gambler_ruin/01

Category: Sequential and combinatorial processes; difficulty: compositional.

A simulated game starts with 2 tokens. Each round independently adds one token with probability 1/2 or removes one otherwise. Play stops immediately at 0 or 5 tokens. There is no time limit or other stopping rule.

Does the game reach 5 tokens before 0?

| Outcome | Exact probability |
|---|---:|
| yes | 2/5 |
| no | 3/5 |

Solve u(i)=p*u(i+1)+(1-p)*u(i-1), u(0)=0,u(N)=1. For p=1/2 use i/N; otherwise (1-r^i)/(1-r^N), r=(1-p)/p.

### known-distribution-pilot-v1/gambler_ruin/02

Category: Sequential and combinatorial processes; difficulty: compositional.

A simulated game starts with 2 tokens. Each round independently adds one token with probability 2/5 or removes one otherwise. Play stops immediately at 0 or 5 tokens. There is no time limit or other stopping rule.

Does the game reach 5 tokens before 0?

| Outcome | Exact probability |
|---|---:|
| yes | 40/211 |
| no | 171/211 |

Solve u(i)=p*u(i+1)+(1-p)*u(i-1), u(0)=0,u(N)=1. For p=1/2 use i/N; otherwise (1-r^i)/(1-r^N), r=(1-p)/p.

### known-distribution-pilot-v1/guaranteed_reward/01

Category: Sequential and combinatorial processes; difficulty: derived.

A game gives a reward on each ordinary attempt independently with probability 1/5. You stop at your first reward. If all first 2 attempts fail, attempt 3 gives the reward with certainty. You begin with no prior attempts and will continue until the first reward.

On which attempt do you first receive the reward?

| Outcome | Exact probability |
|---|---:|
| 1 | 1/5 |
| 2 | 4/25 |
| 3 | 16/25 |

For k below the guarantee use geometric mass (1-p)^(k-1)*p; all remaining mass is on the guaranteed attempt.

### known-distribution-pilot-v1/guaranteed_reward/02

Category: Sequential and combinatorial processes; difficulty: derived.

A game gives a reward on each ordinary attempt independently with probability 1/4. You stop at your first reward. If all first 3 attempts fail, attempt 4 gives the reward with certainty. You begin with no prior attempts and will continue until the first reward.

On which attempt do you first receive the reward?

| Outcome | Exact probability |
|---|---:|
| 1 | 1/4 |
| 2 | 3/16 |
| 3 | 9/64 |
| 4 | 27/64 |

For k below the guarantee use geometric mass (1-p)^(k-1)*p; all remaining mass is on the guaranteed attempt.
