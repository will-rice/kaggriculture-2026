# Competitive simulation agents: what the literature and the competition record say about scripts, imitation, RL, build orders and the meta

**Deep-mode literature review, 2026-09-02.** 92 sources registered and 86 cited,
with 165 persisted evidence spans. Written against measurements made in this repository; where the two
disagree, the measurements win and this document says which of the paper's
assumptions fails here.

**Standing caveat, per `docs/research/README.md`.** Research-scale results and
single-box results are separated throughout. "AlphaStar did X" and "a solo
competitor did X" are different claims and only the second is ours.

---

## Executive Summary

The competition record does not support a blanket claim that scripted agents
beat learned ones in Kaggle simulations. It splits by whether a fast simulator
exists. All four Halite competitions, Kore 2022 and both stages of Lux AI Season
2 were won by hand-written agents [1][2][3][4][5][6];
Google Research Football 2020, Lux AI Season 1 and Lux AI Season 3 were won by
learned policies [7][8][9]. Two premises in the
commissioning question are wrong and are corrected here: the Halite IV winner was
Tom Van de Wiele, who **abandoned** deep RL after a month and won with a rule
agent [4][10], and Lux S2 was won by `ry_andy_`, a scripted
planner, not by FLG or Frog Parade [11][6].

Our cloning failure is diagnosable to a regime. Train accuracy equalling holdout
accuracy places us in Spencer et al.'s HARD regime — the expert is not realizable
in the policy class [12] — and DAgger's theorem is explicitly relative
to the best policy _in the class_ [13], so more DAgger rounds cannot
address it. High per-action agreement with zero task performance is the documented
copycat signature, where held-out likelihood improves while "only the environment
reward decreases" [14].

The RL result that most changes our options is not an architecture. Resetting
episodes to demonstration states and sliding the reset point backwards converts
the cost of an N-step chain "from exponential in N to quadratic in N"
[15] — and we own a demonstration.

Build-order optimisation cannot be lifted off the shelf. Churchill and Buro's
BOSS deliberately removes labour scheduling, and its fast-forwarding rests on a
no-hoarding lemma that is false when banked cash is the objective
[16]. The joint labour-and-cash problem lives in operations research
as the capital- and resource-constrained project scheduling problem
[17], and in RTS only in two papers [18][19].

On the meta, the theory says exploit: under a dominant agent, population training
"degenerates to training against the best agent in the population" and diversity
is zero [20]. But our final ranking is a post-deadline Bradley-Terry
tournament, in which "Your live rating's history, its lucky start, and its 'age'
count for nothing at the end" [21].

**Primary recommendation.** Spend the four weeks adding two or three
_structurally different_ plans behind a small choice-point selector, and stop
spending submission slots on re-rolls.

**Confidence: medium-high** on the diagnoses, **medium** on the ranking.

---

## Introduction

### Research question

Five questions, all raised by measurements made in
`/home/will/projects/kaggriculture-2026` on Kaggriculture, a two-player,
simultaneous-move, 719-turn economic farming game with a relative-bank win
condition: why open-loop scripted plans dominate these ladders and when learning
has actually won; why a 92.8%-accurate clone of our best agent wins zero games;
what works for RL in a 10^16–10^26 action space with a seven-link, 120-turn
profit chain; whether the RTS build-order literature applies when labour and cash
flow are jointly binding; and how to act on a ladder that is a succession of
monocultures rotating every two to three days.

### Scope and method

Two evidence streams. The **external** stream is 92 registered sources, 86 of them cited: peer-reviewed
papers, the winners' own Kaggle write-ups and repositories, Kaggle leaderboard
CSVs pulled through the API, and Kaggle's own evaluation pages. Secondary
roundups were excluded after two of them mis-stated the Halite IV winner's name
and method; every competition result below is anchored to a leaderboard row, a
write-up body, or a repository README that was opened directly. The **internal**
stream is this repository's measurement ledger and `docs/research/`, cited as
measurements rather than as literature.

Excluded: single-agent RL benchmarks with short horizons, LLM-agent literature,
and anything about Kaggriculture's rules that the engine itself settles
(`docs/competition.md` is the authority there).

### Key assumptions

1. **The engine is 1.32.7 and stable.** All 5,435 episodes across the eight most
   recent daily archives report `module_version` 1.32.7, and the installed
   `kaggle-environments` matches. Numbers from before 2026-08-24 are not pooled.
2. **Our teacher really is clock-indexed.** The ledger records module globals
   latched at fixed steps and a route that acts on 667 of 719 turns; this
   document treats "the action is often a function of the step counter, not the
   board" as established locally, not as a hypothesis.
3. **The measured ladder noise floor is real.** Byte-identical agents have landed
   455–512 rating points apart in our own submissions, and two independent
   Kaggle competitors report the same phenomenon at different scales
   [22][23].
4. **Four weeks, one GPU, five slots a day, latest-two scored, Bradley-Terry
   finale.** Every recommendation is ranked against this envelope, not against
   what a research lab could do.
5. **Our validated local gate is the objective.** The field-weighted gate covers
   61.8% of 2026-09-01 seats; it is a weighted sample of the field, not the
   field.

---

## Main Analysis

### Finding 1: The record splits by simulator speed, not by era — and two of the premises we started from are false

The premise that Kaggle simulation competitions are won by scripts is half
right, and the half that is wrong is the half that matters for planning. Here is
the record, each row anchored to a primary source.

| Competition                     | Winner                                                       | Approach                                                                                  | Class             |
| ------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------- | ----------------- |
| Halite I (2016)                 | mzotkiew                                                     | region decomposition, hand-tuned expansion, simulated annealing for combat moves only [1] | scripted          |
| Halite II (2017)                | reCurse                                                      | strategy overrides, hand-scored colony planning, tactical pass [2]                        | scripted          |
| Halite III (2018)               | teccles                                                      | Dijkstra per-square value scoring, hand-designed collision/spawn/dropoff rules [3]        | scripted          |
| Halite IV (2020)                | Tom Van de Wiele                                             | scores → plan → actions pipeline with hand-coded overrides [4][10]                        | scripted          |
| Google Research Football (2020) | WeKick                                                       | PPO + LSTM + self-play [7]                                                                | learned           |
| Rock-Paper-Scissors (2020-21)   | Where is my bag                                              | multi-armed bandit over public agents plus 1- and 2-step counter-transforms [24][25]      | meta/exploitative |
| Santa 2020 (bandit sim)         | nagiss                                                       | two LightGBM regressors predicting hidden thresholds, then greedy [26]                    | supervised + rule |
| Lux AI Season 1 (2021)          | Toad Brigade                                                 | single conv net, self-play, ±1 terminal reward [8]                                        | learned           |
| Kore (2022)                     | Harm Buisman                                                 | rule agent, 146 versions, last 19 pure parameter tuning [5]                               | scripted          |
| Lux AI Season 2 (2023)          | ry*andy*                                                     | stateful roles + ~2.9 s/turn forward simulation [6][11]                                   | scripted          |
| Lux AI Season 3 (2024-25)       | Flat Neurons                                                 | IMPALA multi-agent RL [9][27]                                                             | learned           |
| ConnectX                        | no winner — permanently open playground; game is solved [28] | exact search                                                                              | —                 |

**Two corrections.** First, the Halite IV winner is Tom Van de Wiele
[10], and he did not use imitation learning plus RL. He used deep RL for
a month and abandoned it: "after spending a month trying to get the ships to do
anything different from standing still, I decided to move on with pain in my
heart" [4]. The best imitation-learning entry in that competition placed
eighth. Second-place there volunteered "Despite my team name 'Raine Force' looks
like using reinforcement learning, my agents are 100% rule based"
[29]. Second, Lux Season 2 was won by `ry_andy_` with a scripted
planner [11][6]; Frog Parade is Isaiah Pressman's _Season 3_ team
and placed second there with RL.

**The organising variable is the cost of a simulated step, not the calendar.**
Where a fast environment existed, learning won: Toad Brigade trained on a
personal dual-GPU desktop [30], and Lux S3's runner-up rewrote the
environment in Rust to reach 110,000 steps/s and trained a 10M-parameter network
for roughly 300M steps in eight days on an RTX 3090 plus an RTX 2070 Super
[31]. Where the environment was slow or the credit assignment hostile,
scripts won, and the winners say why in the same words we would. Van de Wiele:
"the credit assignment part is very hard to get right with an arbitrary number of
units (ships/bases), a long episode duration and a dynamic opponent pool. Not
having infinite compute is also annoying" [4].

The most instructive quotes are the ones where competitors changed their minds,
in both directions. Toad Brigade began Lux S1 expecting rules to win "motivated
by the top results in last year's Halite competition", and dropped their rules
agent when "within the first month the RL approach began to beat the rules-based
one" [8]. Two years later `ry_andy_` won S2 with a logic bot while
"mostly under the impression that RL was going to reign supreme as in season 1"
[6]. Two years after that, Lux S3's third place hit a wall with rules,
tried imitation learning on downloaded replays of the top two teams, and reported
"To my surprise, the IL-based bot outperformed my best rule-based agent"
[32].

**What this means for us.** Kaggriculture sits on the scripted side of the
divide by construction. `actTimeout` is 1 second per turn and our own competition
notes state there is "no room for per-turn search", which removes the Lux S2
winner's 2.9-second forward simulation from our reach. Our own PPO arms have run
100M steps and end a season with bank 2 of a starting 3,000. Neither observation
forbids learning; both say that the learned route here is the Lux one — a fast
batched simulator plus hundreds of millions of steps — and that is a
months-scale project, not a four-week one.

There is one further structural fact worth naming, because it changes what
"scripted" means on this particular ladder. Rock-Paper-Scissors was won by an
agent that is neither scripted-for-the-game nor learned-from-the-game: it is a
bandit over _other people's published agents_ plus explicit counters to them
[25]. The winner's stated principle was "To achieve a higher rank, the
agent should beat as much public agents as possible" [24]. On a ladder where
59% of corpus seats play one of eight published capital scripts, that is the
third option, and Finding 7 returns to it.

---

### Finding 2: "Recorded plan plus repair" is a rational architecture, and editing the plan is not the cheap operation it looks like

The architecture our vendored agent uses — a fixed action sequence indexed by
turn, with a sparse feedback controller that patches drift — is not a hack. It is
the standard execution architecture from automated planning, and the literature
has measured both of its properties.

Fox, Gerevini, Long and Serina define the alternatives precisely: "By plan repair
we mean the work of adapting an existing plan to a new context whilst perturbing
the original plan as little as possible. By contrast, replanning is the work of
generating a new plan from scratch" [33]. Their empirical result favours
repair on both axes that matter here: "our plan repair strategy achieves more
stability than replanning and can produce repaired plans more efficiently than
replanning" [33]. Stability — staying close to the original plan — is
exactly what a cash-flow-locked schedule needs, because the schedule's value is
in the relative timing of its purchases rather than in any single action. Add the
1-second `actTimeout`, and replanning is not merely worse, it is unavailable.

The second property retroactively explains three of our closed experiments. Plan
reuse is not, in general, cheaper than planning. de Haan, Roubickova and Szeider
summarise the founding result: "The first paper providing a complexity-theoretical
study of plan reuse (Nebel and Koehler 1995) considered so-called conservative
plan reuse. Conservative plan reuse maximizes the unchanged part of the known
solution. The authors showed that such a plan reuse is not provably more
efficient than plan generation. Moreover, they show that identifying what is the
maximal reusable part of the stored solution is an additional source of hardness"
[34].

Read against our ledger, that is three results at once. **Route harvesting:**
thirteen routes taken from episodes rated 3,029–3,030 — above the then-current
leaderboard leader at 2,934 — each scored 0.000 over 24 games, while `v56`'s own
backbone route run through the identical harness scored 0.500. A recorded plan is
bound to the board it was recorded on, and identifying which part transfers is
precisely the sub-problem Nebel and Koehler isolate as an _additional_ source of
hardness. **Route editing:** the route has 7 fully idle turns out of 719 and a
300-coin insertion loses every mirror; there is no slack into which a local edit
fits. **Build feasibility:** a planner handed the exact vendored build could not
buy it, 0.0000 over 32 games, reaching the second quadrant on day 11 against the
route's day 6. The build is a _projection_ of a working schedule, and projections
do not carry the constraints that produced them.

The formulation our own ledger reached — "A build schedule isn't an instruction a
farm can follow; it's a description of what a farm that already works ends up
holding" — is the same statement as Churchill's separation of goal choice from
schedule search, discussed in Finding 5.

---

### Finding 3: Our clone did not fail from distribution shift. It failed from non-realizability, and DAgger is guaranteed not to fix that

The measured result is that a state-conditioned convolutional policy reproduced the
teacher's per-action choices at 92.8% (unit op), 90.9% (quantity) and 99.7% (market)
on held-out seasons and won 0 of 48 games against that teacher, and that one DAgger
round over 320 seasons and roughly 230,000 rows left the win rate at 0.000. Three
distinct literatures explain a zero win rate at high action agreement, and they
prescribe different remedies. Separating them is the whole value of this section,
because two of the three are already excluded by numbers we hold.

The first explanation is classical compounding error. Ross and Bagnell prove that a
policy trained to 0-1 error `ε` **under the expert's own state distribution** admits
only the bound `J(π̂) ≤ J(π*) + T²ε` [35], and they note the bound is not loose:
"as soon as π̂ makes a mistake, it could end up in new states that were not visited by
π\*, and always incur maximal cost of 1 at each step from then on. Most importantly,
this bound is tight" [35]. Rajaraman et al. show the `H²` term is a property of
the problem rather than of behavioural cloning — "even if the learner operates in the
active setting and the expert is deterministic, no algorithm can beat the H² barrier"
[36] — and give the arithmetic directly: the probability of a first error at
step `t` is `ε(1−ε)^{t−1}`, and if the learner is lost thereafter it pays `H−t+1`
[36]. Xu, Li and Yu add a detail that matters for planning our remaining
weeks: the behavioural-cloning value gap scales as `√ε`, so halving the imitation loss
buys a factor of only 1.41 in the value gap [37].

At our horizon this framework is descriptively right and prescriptively empty. With
`T = 719` and `ε ≈ 0.072`, `T²ε ≈ 37,000` against a maximum episode cost of 719: the
bound is vacuous, exactly as this project's earlier grounding review already recorded
for `uTε`. A vacuous bound cannot distinguish a clone that is 7% wrong in a forgiving
place from one that is 7% wrong in the cash-flow schedule. What makes the compounding
story fit anyway is our own second measurement: on the clone's own states, unit-op
accuracy falls from 0.928 to 0.733. That 19.5-point drop is covariate shift, measured
directly, and it is the quantity DAgger exists to remove.

The second explanation is causal confusion, and it is the one that matches our symptom
most exactly. Wen et al. state the pathology in a form that reads like a description of
our gate: a policy that simply repeats the previous expert action "would produce the
correct action at all but one time instant" and yet "when testing for actual driving
performance, it would be useless" [14]. They rule out overfitting explicitly —
with longer histories "the likelihood L(θ*) of held-out expert demonstrations improves,
which means that there is no overfitting; only the environment reward R(θ*) decreases"
[14]. de Haan, Jayaraman and Levine report the same inversion on Atari: the
confounded model "produces lower validation loss than ORIGINAL on held-out demonstration
samples, but produces lower rewards when actually used for control" [38].
Swamy et al. supply the mechanism for a teacher whose actions depend on something the
observation does not contain: off-policy imitation "treat[s] their suboptimal past
actions as though they came from the expert", which "often manifests as a latching
behavior: a naive repetition of past actions" [39]. They name the resulting
design bind precisely — the learner needs history to infer the hidden context, "but if
they do, they can learn a latching policy that performs poorly at test-time" [39].
A clock-indexed script is exactly a hidden context: the step counter is the variable
that selects the action, and it is not in the board.

The third explanation is non-realizability, and our numbers select it over the other two.
Spencer et al. give the taxonomy: the EASY regime is the one where "the expert is
realizable π* ∈ Π" and one can "simply drive the classification error to ε = 0"; the HARD
regime is model misspecification with unbounded density ratio, where "a compounding error
of O(T²) is simply inevitable without repeated access and interaction with the
demonstrator" [12]. Our diagnostic is that **train accuracy equalled holdout
accuracy** — the network could not fit even the data it was shown. That is not a
generalisation failure and not a shift failure; it is the statement `π* ∉ Π`. And this is
where DAgger's guarantee stops. Its theorem bounds the learner by `ε_N`, defined as "the
true loss of the best policy in hindsight" over the visited distributions [13],
and its consistency claim is explicitly conditional: "Under an error reduction assumption
that for any input distribution, there is some policy π ∈ Π that achieves surrogate loss
of ε, this implies we are guaranteed to find a policy π̂ which achieves ε surrogate loss
under its own state distribution in the limit" [13]. No such policy exists in a
class that cannot represent a 719-entry script from a board that omits the index. Laskey
et al. state the converse of the same fact from the practical side: noise injection, and
by their own citation DAgger, "will offer no improvement if the robot can represent the
supervisor perfectly and collect sufficient data" [40] — interaction is a remedy
for shift, never for representation. Swamy et al. close the last escape: on
`O(T)`-recoverable problems "there exist problems where no algorithm can escape an O(T²)
compounding of errors" [41], and a cash-flow-locked schedule where a missed
purchase forfeits a 120-turn production chain is the canonical `O(T)`-recoverable case.

The literature also warns that the number we gated on is the wrong number, with a
quantitative example. Mandlekar et al. report that selecting the checkpoint by lowest
validation loss scores 2.7 ± 1.9 where the best checkpoint scores 80.7 ± 0.9 on the same
task, and 0.7 ± 0.9 against 64.0 ± 2.8 on another [42]. Codevilla et al.
found that "two models with identical prediction error can differ dramatically in their
driving performance" [43]. Our own ledger reached the same conclusion by
measurement — "per-action accuracy is the wrong metric for this agent" — and the
literature says it was predictable in advance.

**What the literature says would actually change the outcome.** Two families, and only
two, address a teacher whose action is a function of the timestep rather than the state.
The first is trajectory-level objectives: GAIL replaces per-action supervision with
occupancy-measure matching [44], and Xu et al. quantify the gain as a horizon
exponent — the value gap is quadratic in the effective horizon for behavioural cloning
and linear for GAIL [37]. The second, and the cheap one, is action chunking. ACT
predicts `π(a_{t:t+k} | s_t)` rather than `π(a_t | s_t)`, which "reduces the effective
horizon of the task by k-fold, mitigating compounding errors", and — this is the sentence
that matters for us — "helps tackle temporally correlated confounders, such as pauses in
demonstrations that are hard to model with Markovian single-step policies" [45].
They state the timestep case explicitly: a single-step policy struggles "since the behavior
not only depends on the state, but also the timestep. Action chunking can mitigate this
issue when the confounder is within a chunk, without introducing the causal confusion issue
for history-conditioned policies" [45]. The measured effect size is large: success
rises "from 1% at k = 1 to 44% at k = 100, then slightly tapers down with higher k", with
`k = 1` meaning no chunking and `k = episode_length` meaning "fully open-loop control"
[45]. Diffusion Policy reports the same failure mode from the other side: single-step
policies "can easily overfit to this pausing behavior" and "often get stuck" [46].

The honest reading of that for us is deflationary rather than encouraging. ACT's curve is
a curve from a closed-loop policy toward an open-loop one, and it peaks partway. Our
teacher is already at the far end of that axis — it is a recorded plan. Chunking would
move a clone toward reproducing the plan, but the limit of that motion is _storing the
plan_, which we can already do at zero training cost and zero error by keeping the route
file. The literature's answer to "how do I imitate an open-loop plan with a
state-conditioned policy" turns out to be "don't; represent the plan as a plan". That is
the finding, and it is why we rank a fourth imitation attempt below the alternatives in
the final section.

---

### Finding 4: In a 10^16–10^26 action space, the winners changed the action _representation_ and the _start state_, not the algorithm

Our exploration wall has been measured from several directions: from-scratch PPO
across 100M steps ends a season with bank 2 against `STARTING_MONEY` 3,000 and
about 3.9 sales, and four separate reward corrections — each verified by a test
that fails when reverted — changed none of it. The literature says two things
about a space this size, and only one of them is about the loss function.

**The action space is a representation choice, and the reduction available is
enormous.** AlphaStar's is the canonical figure: "This representation of actions
results in approximately 10^26 possible choices at each step" [47], and
their answer is structural — "To manage the structured, combinatorial action
space, the agent uses an auto-regressive policy and recurrent pointer network"
[47]. OpenAI Five sits lower, at "8,000 to 80,000 actions" per step
[48], and cost roughly 1,536 optimizer GPUs and 172,800 rollout CPUs over
ten months [48] — a number quoted here only to fix the scale we are not
operating at.

The transferable arithmetic is Gym-microRTS's. Emitting per-component logits
instead of enumerating the joint action takes the head from "50 million" logits
to "301" on a 16×16 map [49], and the resulting agent "can defeat every
microRTS bot we tested from the past microRTS competitions when working in a
single-map setting ... while only taking about 60 hours of training using a
single machine (one GPU, three vCPU, 16GB RAM)" [49]. Tavakoli et al. make
the same point as a general architecture claim, demonstrated at 6.5×10^25 actions:
action branching "enables the linear growth of the total number of network outputs
with increasing action dimensionality as opposed to the combinatorial growth in
current discrete-action algorithms" [50]. Kanervisto et al. supply the
practical rule and one explicit prohibition: "Start by removing all but the
necessary actions and discretizing all continuous actions. Avoid turning
multi-discrete actions into a single discrete action" [51].

**Masking is not optional, and its benefit grows with the space.** Huang and
Ontañón measured the difference in time-to-first-reward on a 10×10 map: agents
trained with an invalid-action penalty "spent 3.43% of the entire training time
just discovering the first reward, while agents trained with invalid action
masking take roughly 0.06% of the time in all maps" [52] — a factor of
about 57, widening with action-space size. The Lux S3 runner-up adds the opposite
caution from experience: over-restrictive masks cost him, and "next time I would
use less restrictive action masking, only banning actions that are certain to be
useless or meaningless" [31].

**Two AlphaStar details are specific to large factored spaces and bear on our own
V-trace result.** First: "We found that off-policy correction methods like
V-trace can be inefficient in large, structured action spaces like the one we
used for StarCraft, because distinct actions can result in similar (or even
identical) behaviour" [47]; their fix was a hybrid — V-trace for the
policy, TD(λ) without off-policy correction for the value — plus an assumed
independence between action type, delay and other arguments "to mitigate early
trace cutting due to the large action space" [47]. Our own paired V-trace
run over 19,200 seasons converged to never deviating, and while the measured cause
there was that the market action space has negative value in every family and
phase, the AlphaStar caveat says that a V-trace result on a large factored space
should never be read as an unqualified statement about the action space itself.
Ours survives that objection only because a separate forced-deviation probe over
2,048 seasons, which does not use V-trace at all, agrees: −0.0722 ± 0.0130 win
points per deviation.

**What the Lux winners actually did, since they are the closest precedent and one
of them forked the codebase we run.** Toad Brigade used "FAIR's implementation of
the IMPALA algorithm, with additional UPGO and TD-lambda loss terms" plus a
frozen-teacher KL term, which "helped to stabilize behavior and prevent strategic
cycles — both of which are problems that plague a pure self-play setup"
[30]. Their action head is GridNet-style: one network "issued commands for
each worker, cart, and city tile on all squares of the board simultaneously",
chosen because separate per-unit networks make coordination and credit assignment
hard [30]. Their curriculum is the part we have not copied: shaping "for
the first 20 million steps" only, then an 8→16→24 block size ladder trained on
the _sparse_ reward with each smaller network as teacher, all "on my personal PC —
an 8-core/16-thread dual-GPU system" [30]. And one feature they call
crucial is a `game phase` variable — turn number divided by 40 — after which "the
network quickly developed dramatically different behaviors during the beginning,
middle, and end game" [30]. That is a clock feature, in an RL policy, doing
exactly the job our clone lacked.

Flat Neurons ran the same recipe four years later with more scale: IMPALA,
V-trace, UPGO, TD, entropy, frozen-teacher KL and a frozen opponent pool
[9], a two-level factored head where the second head "is trained only
on timesteps where the first head selected 'sap'" [9], and roughly
"1.5B environment steps" for the final model against "over 20B steps across
various experiments" [9]. They implemented behaviour cloning from the
dominant opponent's replays and did not use it, "since we were already seeing
significant improvements without it" [9]. The runner-up's numbers set
the solo envelope: a 10M-parameter convolutional network, "around 300,000,000 game
steps ... about 8 days of continuous training", "mostly plateaued by around step
200,000,000", on one home machine [31].

**Hierarchy and macro-actions are the standard reduction, and their cost curve is
published.** TStarBots designed "165 macro actions" and beat every built-in
StarCraft II AI level including the cheating ones [53]; Liu et al.
extracted macro-actions automatically from expert trajectories, which "reduces the
action space in an order of magnitude yet remains effective", reaching over 93%
against level-7 built-in AI "within two days using a single machine with only 48
CPU cores and 8 K40 GPUs" [54]. But TStarBots also publish the scaling
warning: 30M frames to beat level 2, 3,500M frames to beat level 10
[53] — a 117-fold sample increase for a harder opponent, on a game whose
horizon is comparable to ours.

**The single result that most changes our options is about the start state, not
the action space.** Salimans and Chen: "If a specific sequence of N actions is
required to reach a reward, this sequence may now be learned in a time that is
quadratic in N, rather than exponential", achieved by resetting each episode to a
state sampled from a demonstration and sliding the reset point backwards; and
they measured the contrast — "When starting each episode from the initial game
state, as is standard practice, the run time scales exponentially in the problem
size" [15]. Backplay reports the same mechanism with an extra
property: an agent trained this way "can outperform its demonstrator and even
learn an optimal policy following a sub-optimal demonstration" [55].
Florensa et al.'s variant needs no demonstration at all, only one goal state, and
samples start states where the current success probability lies between 0.1 and
0.9 [56]. Go-Explore names the two failure modes this addresses —
detachment and derailment — and its fix is the same shape: "By first returning
before exploring, Go-Explore avoids derailment by minimizing exploration in the
return policy" [57].

We have a demonstration. The vendored route banks 84,797 and draws 0.500 against
the served kernel; our seasons are deterministic given a seed; and mirror play
ties exactly, which gives a zero-variance baseline for any single change. Every
precondition for a reverse curriculum is present. The honest counterweight is
cost: Salimans and Chen's headline result "required 128 GPUs over a period of 2
weeks" [15], and their environment is far cheaper per step than ours.
The quadratic-versus-exponential claim is about _scaling in chain length_, not
about absolute compute, and it does not by itself put a competitive agent inside
four weeks on one GPU.

**On reward, the literature closes a door we have already walked through.** Ng,
Harada and Russell prove potential-based shaping is necessary and sufficient for
policy invariance [58], and their canonical failure is our failure verbatim:
"when the agent was rewarded for riding towards the goal but not punished for
riding away from it, it learned to ride in a tiny circle and thereby obtain
positive reward whenever it happened to be moving towards the goal" [58].
This project has already recorded shaped rewards paying for free actions. Flat
Neurons' conclusion after trying about ten shaped schemes was to end near-sparse,
keeping a small dense term only to prevent "do nothing" stagnation
[9]. RUDDER explains why more shaping cannot rescue a long chain:
delayed rewards make TD biased in a way needing exponentially many updates in the
delay, and Monte-Carlo variance grows with it [59]. Our four reward
corrections producing four identical policies is exactly what that predicts.

---

### Finding 5: The build-order literature deletes exactly the variable our measurement identified as binding

Our build-feasibility test found that a planner handed the vendored agent's exact
build could not buy it, and named the reason: every build date is a cash-flow
date, and cash flow is a product of labour — 5,398 productive unit-turns against
7,304, 22.9% productive against 45.4%, 130 harvests against 420. The question is
whether the RTS build-order literature helps. The answer is that its central
method is built on removing precisely that variable, and says so.

**Churchill and Buro's BOSS.** The problem is stated as makespan minimisation
with the goal set supplied: "we need to find concurrent action sequences that,
constrained by unit dependencies and resource availability, create a certain
number of units and structures in the shortest possible time span"
[16], and explicitly, "Avoiding the interesting and ambitious task of
selecting good build order goals, in this paper we assume that they are given to
us" [16]. The algorithm is depth-first branch and bound with linear
memory and anytime behaviour [16], and it matches professional
makespans "90% of the time" in the early game [16].

It buys that tractability with four abstractions, three of which are fatal here.
Income becomes a constant: "We abstract mineral and gas resource gathering by
real valued income rates of 0.045 minerals per worker per frame and 0.07 gas per
worker per frame" [16] — and the authors say what this removes: "It
also eliminates the need for 'gather resource' type actions which typically
dominate the complexity of build order optimization" [16]. Worker
assignment becomes a rule: "a set number of workers (3 in our experiments) will
be sent to gather gas from it. This abstraction eliminates the need for worker
re-assignment and greatly reduces search space, but in rare cases is not 'truly'
optimal for a given goal" [16]. And the fast-forwarding that makes
the search cheap rests on a lemma that is false in our game: "the time-optimal
build order for any goal is one in which actions are executed as soon as they are
legal, since hoarding resources cannot reduce the total makespan" [16].
Our win condition _is_ hoarded resources. Banked coins are the objective, so
"execute as soon as legal" is not a valid dominance argument here, and the search
that depends on it is not sound for us.

The 2019 follow-up states the limitation from the outside: "classic build-order
planning abstractions disregard the opponent and concentrate on unit counts and
resource gathering aggregate statistics rather than low-level game state features
such as unit positions and actual resource gatherers' travel times"
[60]. The RTS survey confirms that essentially nothing else in the
field even attempts timing: "none of these approaches addresses any timing or
scheduling issues, which are key in RTS games. One notable exception is the work
of Churchill and Buro" [61].

**Where the joint problem does exist.** Two RTS papers keep labour as a real
resource. Chan et al. name the tradeoff as the hard part — "it can be quite
difficult to determine the correct tradeoff between how many peasants to create,
which require time and resources, and the payoff those peasants provide in terms
of increased production rate" [18] — and model workers as borrowed
renewable resources, so that "to allow concurrent collect-gold actions, multiple
peasants must be used" [18]. Their negative result is a warning against
reaching for a generic planner: off-the-shelf temporal planners "found valid
plans in under a second" but "we could not get them to create extra renewable
resources even when doing so would greatly decrease the makespan", and two others
"were unable to find any plan at all" [18]. Blackford and Lamont go
further and are, as far as this review found, the only RTS build-order model that
splits labour into unary scheduling resources while treating cash as a cumulative
one: "individual workers themselves are a resource, so it is better to divide the
non-unary resource (total number of workers) into unary resources (individual
workers)" [19], alongside a cumulative constraint on consumables
[19]. Justesen and Risi's COEP is the only forward model found with
non-linear labour productivity — output halves past ten workers per base — and
even there the economy balance did not fall out of the objective: they "found it
necessary to penalize expansions while having few workers as well as not
expanding while having many workers" [62].

Rooijackers and Winands show the size of the prize on the pure execution side,
outside build-order search entirely: better worker scheduling alone "gathers more
than 14% than Built-in", and "applying scheduling or using better pathfinding can
improve the resource gathering rate significantly" [63]. That is the
literature's closest analogue to our 22.9%-versus-45.4% gap, and it is a
_scheduling_ result, not a _planning_ one.

**The operations-research framing is the better fit.** The joint problem has a
name outside games: the capital- and resource-constrained project scheduling
problem with discounted cash flows, in which "capital constraints ... force the
project to always have a positive cash balance" simultaneously with renewable
resource capacity [17]. That is our situation stated exactly — cash as a
state variable with a floor, not as a precondition that time will eventually
satisfy. He, Liu and Jia report a property worth internalising before any
planner is built: net present value rises with initial working capital but with
"decreasing marginal return as the contractor's initial capital availability goes
up" [64]. And the tractability news is bad in a useful way: the plain
resource-constrained problem without any cash flow "belongs to the class of the
strongly NP-hard problems" [65], so adding a cash floor cannot make it
easier.

**The one transferable empirical finding.** Churchill's thesis identifies a
failure mode of optimising at a fixed terminal time: such a plan "will start by
producing worker units in order to gather more resources, and only start producing
army units as the time limit approaches", leaving the player "vulnerable to attack
during early stages of the build-order" [66]. Replacing the terminal
objective with the integral under the value curve produced a measured tournament
gain: win rate against the 2017 AIIDE field rose from 27.66% to 37.19%, and
against the top five from 14.42% to 32.80% [60]. Our game's win
condition is terminal bank, so the degenerate all-economy-then-all-cash shape is
_not_ penalised the way it is in StarCraft — but our market couples the two
players, so a plan that banks nothing until late is a plan that never moves the
opponent's prices. The integral-versus-terminal distinction is therefore worth
testing as an objective for any planner we build, and it is the single most
transferable result in this literature.

**Verdict on part 4 of the question.** No, essentially none of the RTS
build-order literature is applicable as-is when labour scheduling and cash flow
are jointly binding, because its dominant method removes labour by construction
and its dominance argument fails when cash is the objective. What is transferable
is a short list of encodings rather than an algorithm: Chan's borrowed-renewable
model of labour, Blackford and Lamont's unary/cumulative split, COEP's saturation
curve, the OR pattern of a hard per-period cash floor, and Churchill's
integral objective. Our own measurement already went further than the literature
on the central question, and its formulation should be treated as the finding: the
build is an output of a working execution policy, not an input to one.

---

### Finding 6: The one method whose shape matches ours is choice-point search over a portfolio of scripts — and its published failure mode is exactly the one we measured

The RTS literature contains a method built for our situation: an agent that is a script,
a search space that is far too large to enumerate, a per-decision time budget too small
for deep lookahead, and a need to adapt to the opponent. Barriga, Stanescu and Buro call
it Puppet Search: "a new adversarial search framework based on scripts that can expose
choice points to a look-ahead search procedure. Selecting a combination of a script and
decisions for its choice points represents a move to be applied next" [67]. The
formal move is an action abstraction: "creating a new game in which move options are
restricted by replacing original move choices with potentially far fewer choice points
exposed by a non-deterministic script" [67]. In the journal version the
reduction is stated numerically — a version with "a single choice point to select among
these 4 existing scripts" yields "a game tree with constant branching factor 4"
[68]. That is the same operation that would take our per-turn action space from
10^16–10^26 to single digits.

Two of their reported results bear directly on decisions we have already made and one we
have not. The first is the scaling claim: "Our experiments show a similar performance to
top scripted and search based agents in small maps, while vastly outperforming them on
larger ones. Even a script with a single choice point to choose between different
strategies can outperform the other players in most scenarios" [68]. Against
2014 AIIDE StarCraft competition bots, "Our bot has a higher win rate than all individual
scripts, except for S1 for which the mean and median performances are not statistically
different" [67]. So the gain from adding choice points to a strong script is real
but modest against the best single script, and large against the average one.

The second result is the caveat, and it is the reason this section exists rather than a
recommendation written without one. Barriga et al. report: "in the two instances where no
script defeated the opponent (E1 and E3), the search couldn't defeat it either"
[67]. That is our oracle-selector measurement, published nine years earlier. We
played four of our agents on twelve boards against `shopforge`; `tuned_v58` alone won five,
and an oracle that always picked the best of the four also won five. On every board we
lost, all four lost. Our own diagnosis was structural — `v56` and `v58` hash to the same
opening lineage signature, so they are one plan with variations. Puppet Search says the
same thing as a general property of portfolio methods: search over a portfolio inherits
the portfolio's coverage and adds nothing outside it. **A portfolio method is worth
building only if the portfolio spans genuinely different plans**, and constructing those
plans, not searching over them, is the expensive part.

Two facts from our own corpus say the coverage is obtainable and that it is what the
strong seats have. `shopforge`, which banks 100–128k on the boards where it beats us,
describes itself as "six readable plans and a small public-state tree" — six plans against
our one. And the strongest descriptive difference in the 5,435-episode corpus study is
that the top decile builds a different farm nearly every game: 0.732 distinct build
fingerprints per seat against our 0.006, with a cross-team correlation between build
variety and mean rating of +0.667. That is what a choice-point agent looks like from the
outside. The corpus cannot say variety _causes_ the rating — the same six variable-build
teams lose to our fixed script 0.676 of the time over 105 episodes — so the honest claim
is narrower: the top of this ladder is populated by agents with choice points, and we have
none.

Where the literature and our constraints collide is the time budget. Puppet Search is an
online adversarial search with a 100 ms per-frame budget in µRTS and a full StarCraft frame
budget otherwise; Kaggriculture allows 1 second per turn and our own competition notes state
plainly that "there is no room for per-turn search". The transferable part is therefore the
_action abstraction_, not the _lookahead_. The published route from a search to a
cheap-at-runtime policy is Expert Iteration: "Planning new policies is performed by tree
search, while a deep neural network generalises those plans" [69]. In our setting the
generaliser need not be a network — a table over a handful of choice points, or a small
decision rule over public observables, is enough, because the abstraction has already
collapsed the space to single digits.

Two adjacent lines show the same abstraction being searched offline rather than online.
García-Sánchez et al. evolve the script itself: "StarCraftGP is a framework able to evolve
a complete strategy for StarCraft, from the building plan, to the composition of squads,
up to the set of rules that define the bot's behavior during the game" [70].
Justesen and Risi's Continual Online Evolutionary Planning evolves build-order plans during
the game to adapt to the opponent instead of switching among predefined strategies
[71]. Both are search over programs rather than over per-turn actions, which is the
only formulation in this literature that has ever produced a competitive RTS agent on a
single machine.

---

### Finding 7: On a monoculture ladder the theory says exploit — but the Bradley-Terry finale, the matchmaking band, and the noise floor each cut the payoff

Our census, taken from a public notebook that hashes each seat's first 24 actions
into a lineage signature, records the rotation: `flexonafft` at 52% on 08-23 and
gone; `v54` at 58% on 08-29 and 7% on 08-31; on 08-31 three lineages at 29%, 17%
and 14%, with the lineage count rising from 18 to 24. A kernel sweeps to 50-80%
of seats within 48 hours and then collapses. Four literatures speak to this, and
they do not all say the same thing.

**Empirical game theory says the object you want is a payoff table, and that you
cannot afford to measure it from ladder play.** EGTA's framing is exactly ours —
the simulator is primitive and payoffs come only from sampling [72] —
and Walsh et al.'s heuristic-payoff-table method makes published strategies, not
atomic actions, the primitives [73]. Strikingly, Walsh et al. predicted
the monoculture in 2002: "we expect that the majority of participants in the
general public will adopt heuristic strategies developed by others, rather than
developing their own ... Many players in TAC-01 used techniques publicly
documented and shown to be effective in TAC-00" [73]. Tuyls et al. give
the transfer lemma — a Nash equilibrium of the estimated meta-game is a 2ε-Nash of
the true one [74] — and the sample bound, which scales as 1/ε² and with
the log of the number of profiles [74]. Kaggle runs about eight episodes
per submission per day [75]; a statistically valid meta-game table is
therefore not obtainable from the ladder, and must be estimated offline. Wellman's
advice on how to spend a scarce budget applies directly: "allocate samples with a
view toward confirming or refuting candidate equilibria ... focus on promising
profiles, and their neighbors in profile space" [72].

**Nash averaging says our rank is partly an artefact of the clone population, in
both directions.** Balduzzi et al. state the two failures of Elo bluntly: "Elo is
meaningless — it has no predictive power — in cyclic games like
rock-paper-scissors", and "an agent's Elo rating can be inflated by instantiating
many copies of an agent it beats (or conversely)" [76]. Their worked
example is small enough to be decisive: three rock-paper-scissors agents all rate
0, and adding one clone of C moves A to −63 and B to +63 — "the Elo ratings of
agents A and B are easily manipulated by changing the structure of the
population" [76]. Invariance to redundant copies is their property P1,
and "Elo and uniform averaging over tasks are examples of evaluation methods that
invariance excludes" [76].

This is not academic for us. Our own corpus study found our build on 1,662 corpus
seats — 0% of the 08-24 archive and 58% of the 08-31 archive — and our
field-weighted gate scores `tuned_v58` at 0.7373 with a per-lineage breakdown of
`v56` 0.938 and `v54` 0.875 against `shopforge` 0.500. A large part of our
measured margin is a within-lineage margin. Under P1 that margin is worth less
than it reads when the lineage is large, and worth nothing when the lineage
collapses — which is what `v54` going 58% → 7% in two days looks like from the
inside.

**Population-based training says that under a monoculture, best-responding is the
principled move, not a shortcut.** Balduzzi et al. give the exact special case:
"when there is a dominant agent in the population, that beats all other agents.
The Nash equilibrium is then concentrated on the dominant agent, and PSRO_rN
degenerates to training against the best agent in the population"
[20]. And the corollary tells us not to buy diversity yet: "If there
is a dominant agent then diversity is zero" [20]. The counterweight is
Lanctot et al., who show best responses to a computed equilibrium "overfit to a
specific equilibrium ... and be unable to generalize to parts of the space not
reached by any equilibrium strategy", and who measured the cost of overfitting to
a fixed opponent set at up to 71.7% of reward [77]. AlphaStar operationalises
both halves: main exploiters "play only against the current iteration of main
agents", and the paper concedes they "may rapidly discover specialist strategies
that are not necessarily robust against exploitation" [47]. Their PFSP
weighting exists because "the pursuit of the mean win-rate might lead to policies
that are easy to exploit" [47] — which is a direct criticism of an
occupancy-weighted average gate like ours.

**Rotation timescale: days, everywhere it has been measured.** AlphaStar's
Nash-supported player 40 stayed relevant "until 5 days later, when it was
completely dominated by newer agents" [47]. In Hearthstone, past
archetype popularity and win rates explain "approximately 20% of the Hearthstone
meta", and less right after a patch [78]. Our 2-3 day rotation is inside
that band, and it means a payoff table more than a few days old is stale — a
constraint the sample-complexity bound above already made binding from the other
side.

**Exploitability has a formal budget.** Ganzfried and Sandholm: "we can deviate
from equilibrium to exploit him provided that our worst-case exploitability
remains below the total amount of profit won through gifts" [79]. The
impossibility case is sobering — in pure rock-paper-scissors, even after ten
straight Rocks, "safe exploitation is not possible" [79] — but our
game is not RPS: a fixed clock-indexed script that plays the same 719 actions
regardless of what we do is a gift in exactly their technical sense. Their
measured tradeoff is the number to remember: unrestricted best response scored
0.470 against static opponents but −0.121 against adaptive ones, more than double
the loss of any safe method [79].

**Where the geometry says the payoff sits.** Czarnecki et al. find cyclic
structure is thickest at middle skill and thins at the top: imbalanced
head-to-heads "gradually disappear... so at the highest level of play, the outcome
relies mostly on skill and less on game style" [80]. Exploitation
therefore pays most while climbing and least at the frontier. They also measure a
phase transition in opponent-population size — below a critical size "training
does not converge and cycles for all games", above it, it converges
[80] — which is the quantitative form of our own finding that a
three-opponent league misled us.

**The practitioners add three warnings the papers do not contain.** First, the
counter can defeat itself through matchmaking: one Hungry Geese competitor's
targeted agents "drove those guys into the cellar", with the result that "the top
versions of my own agent never even played against the targeted notebook agents"
[81]. Second, the meta is band-local: on a 2026 Kaggle ladder, "Archetype
share is a function of rating band ... Optimizing against the field you can see at
700 is optimizing against the wrong opponents" [23]. Third, the noise
floor is brutal — 25 identical Santa 2020 submissions scored 585 to 967 with a
same-day standard deviation near 100 and "no clear trend or convergence
day-by-day" [22], and in 2026 seven byte-identical tarballs scored 687.4
to 855.6, sd 51.4 [23]. Our own 455-512 spread is the same phenomenon.

**And the exploitation recipe, where it worked, is short.** The RPS winner's stated
principle was "the agent should beat as much public agents as possible", implemented
as a one-step-ahead beater for each public agent inside a bandit [24]. A
Hungry Geese team measured the ceiling of the same idea: "we could beat the Public
HRL agent 72% of the time 1v3. If we add in knowledge of its deterministic moves,
100% of the time" [82]. That is the upper bound on lineage identification plus
a specific counter, and it is very high.

**The fact that overrides all of this for our remaining four weeks.** Kaggle's
evaluation rule is that only the latest two submissions are tracked
[83], and Kaggriculture replaces the live ladder at the end with a
Bradley-Terry tournament over post-deadline episodes. A competitor writing about
our engine states the consequence: "The live leaderboard is not the final
ranking... Your live rating's history, its lucky start, and its 'age' count for
nothing at the end — only how your last 2 submissions actually fare against the
field", and therefore "Re-submitting an unchanged bot to 're-roll' its trajectory
only moves the live number (and costs you the older copy); it changes nothing
about the final ranking" [21]. His measured convergence curve —
"~90% converged after ~60 games, usually within the first ~5 hours", then
logarithmic growth with "residual noise ... around ±25-50 points"
[21] — is consistent with the wider Kaggle evidence and, being a
competitor's fit rather than an organiser statement, is flagged as such.

This directly contradicts a decision recorded in our own ledger on 2026-09-02,
where we submitted `tuned_v58` twice on the reasoning that "identical agents draw
455-512 apart — two draws exploits that spread rather than fighting it". Under a
Bradley-Terry finale that reasoning is wrong: the spread is in the live rating,
and the live rating is discarded. The re-roll buys a nicer number on a leaderboard
that does not decide the outcome, and costs a slot that could have carried a
different agent into the scored pair.

---

## Synthesis & Insights

### Patterns identified

**Pattern 1: every one of our closed experiments failed at the same joint — the
gap between a description of a strategy and the execution that produces it.** The
harvested route is a description of what a good agent did on one board; it scores
0.000. The mined build is a description of what a good farm ends up holding; a
planner handed it scores 0.000. The cloned policy is a description of which
action a good agent takes in each state; it wins 0 of 48. In each case we
possessed a faithful _representation of the output_ and it bought nothing,
because the value lives in the constraints that generated it. Nebel and Koehler
formalise one half of this — identifying the reusable part of a stored solution is
an additional source of hardness, on top of planning [34] — and Spencer
et al. formalise the other, as the difference between a realizable and a
non-realizable expert [12].

**Pattern 2: the methods that succeeded on comparable problems all reduced the
space before they optimised in it.** Gym-microRTS went from 50 million logits to
301 by composing components [49]; TStarBots went from raw SC2 actions to
165 macro-actions [53]; Puppet Search went to a branching factor of 4
[68]; Toad Brigade discretised transfers to "all of a given resource"
[30]. None of them searched the native space. Our from-scratch PPO does,
and that is the single largest difference between our setup and every published
success at our scale.

**Pattern 3: the winners' constraint that predicts their method is simulator
throughput.** Learning won where a step was cheap — Lux S3's runner-up rewrote the
environment in Rust for 110,000 steps/s [31] — and lost where it was
not. This is a better predictor than year, than game genre, and than the size of
the action space.

**Pattern 4: at every level, the population is a set of scripts, and the top
methods treat it as such.** Walsh et al. build the meta-game out of published
heuristic strategies [73]; the RPS winner built a bandit over published
agents [25]; Puppet Search searches over scripts [67]; our census
identifies lineages by their opening 24 actions. The natural representation of
this ladder is a small discrete set of plans, both for modelling opponents and
for choosing our own behaviour.

### Novel insights

**Insight 1: our two headline failures are the same failure seen from opposite
ends, and the literature names the shared variable.** The build-feasibility test
says the plan is unreachable with our execution; the clone gate says our execution
cannot be recovered from the plan. Both are statements that the mapping between
_plan_ and _execution_ is not invertible in either direction — which is precisely
the property Churchill and Buro engineered away with the no-hoarding lemma
("actions are executed as soon as they are legal, since hoarding resources cannot
reduce the total makespan" [16]). In a game where the objective _is_
hoarded cash, that lemma fails, and with it the assumption that a build implies a
schedule. No source states this; it follows from putting the lemma next to our
measurement.

**Insight 2: the feature our clone lacked is a feature a winning RL policy chose
to add.** We diagnosed the clone's failure as the teacher being clock-indexed
while the network sees only the board. Toad Brigade's write-up reports adding a
`game phase` variable — the turn number divided by 40 — and calls it "a crucial
part of its success" [30]. The literature's framing of history-conditioning
as a trap (Swamy et al.'s "fire and pyre" [39], the copycat problem
[14]) applies to _past actions_, not to the clock. A monotone turn index is
not an action the learner produced, so it cannot be latched onto. This distinction
is not drawn explicitly in any source read here, and it says our clone experiment
had an untested cheap variant.

**Insight 3: the Bradley-Terry finale inverts the standard Kaggle slot heuristic,
and we have been playing the standard one.** On an ordinary Kaggle simulation
ladder, submitting an agent twice is defensible because the final ranking is the
live rating and identical agents land far apart [22][23] —
which is why the Halite IV winner argued for a slot cap in the first place, noting
that duplicate submissions raise "the expected final score of the best agent"
[84]. Kaggriculture discards the live rating. Under a
post-deadline Bradley-Terry tournament the variance that duplicate submissions
exploit is thrown away before the ranking is computed, so the only thing a
duplicate buys is the loss of a slot [21]. Our own ledger records
us making the ordinary-ladder decision on 2026-09-02.

**Insight 4: our field-weighted gate optimises the quantity AlphaStar explicitly
warns against.** Our gate is an occupancy-weighted mean win rate. AlphaStar's PFSP
deliberately does not use one, "because ... the pursuit of the mean win-rate might
lead to policies that are easy to exploit" [47], and it weights by
`f_hard(x) = (1−x)^p` to force the agent to beat everyone rather than to beat the
average. The observation that our field sweep's per-opponent breakdown was
"nearly identical across every variant" while only the `v54`/`v56` columns moved
is what an occupancy-weighted mean looks like when the largest weights are on
one's own lineage.

### Implications

**For our four weeks:** the decisive constraint is not which method is best in
principle but which has a measurable step inside 28 days on one GPU with a
validated local gate. Findings 4 and 6 both point at reductions of the space
rather than improvements of the optimiser, and Finding 6's reduction is buildable
without any training run at all.

**Broader:** the record in Finding 1 suggests the "scripts beat learning on
Kaggle" folk belief is really "cheap simulators enable learning", and that a
competitor's first strategic decision should be an environment-throughput
measurement rather than an algorithm choice.

**Second-order:** if we do build structurally different plans, we create the
diversity that our own oracle-selector test found missing — and, per Puppet
Search's caveat [67], that is the only condition under which any
selector, learned or searched, can gain anything.

---

## Limitations & Caveats

### Counterevidence register

**Contradictory finding 1: build variety correlates with rating but loses to our
script.** Our corpus study measures corr(distinct builds per seat, mean rating) =
+0.667 across 75 teams, with the top decile at 0.732 distinct builds per seat
against our 0.006. That is the strongest evidence for the choice-point
recommendation. But the same six variable-build teams lose to our fixed script
0.676 of the time over 105 decided episodes. **How resolved:** the correlation is
between _having choice points_ and _being a top-rated account_, and the head-to-head
is between _their particular plans_ and _ours_. Puppet Search's own numbers show
both can be true: it beat three of four fixed scripts by a wide margin while being
statistically indistinguishable from the fourth [67]. **Impact:
moderate** — it downgrades "copy their builds" to zero and leaves "acquire choice
points" intact.

**Contradictory finding 2: the Lux S3 winner built behaviour cloning and threw it
away.** They implemented BC from the dominant opponent's replays and did not use
it "since we were already seeing significant improvements without it"
[9], while Lux S3's third place won a medal with pure imitation
learning of the top two teams' replays [32]. **How resolved:** imitation is
worth most when the target is
much stronger than you and representable; Lux S3's third place cloned two teams
whose policies were themselves neural networks, i.e. realizable by construction.
Our teacher is a script. **Impact: significant** for us — it is the sharpest
available argument that our BC failure was about the _teacher's form_, not about
BC.

**Contradictory finding 3: DAgger's own paper reports it fixing exactly our
symptom.** On Super Mario Bros., supervised behaviour cloning "stagnates as we
collect more data from the expert demonstrations" while DAgger improves
[13]. That is the case _for_ running more DAgger rounds. **How
resolved:** their expert is realizable in the learner's class; ours demonstrably is
not, since train accuracy equals holdout accuracy. Laskey et al. state the
boundary condition explicitly [40]. **Impact: moderate** — it means one more
DAgger round is not absurd, only low-expected-value, and it should not be ranked
above untried alternatives.

**Contradictory finding 4: a competitor's ladder model, not an organiser
statement.** The claim that re-submission changes nothing about the final ranking
comes from a Kaggriculture competitor's own fitted model, which opens with
"This post includes assumptions based on my experience" [21].
Its convergence numbers are corroborated in the thread by an independent
competitor's measurements, and its Bradley-Terry claim is corroborated by
`docs/competition.md`. **Impact: moderate** — the direction is well supported, the
exact numbers are not organiser-confirmed.

### Known gaps

**Gap 1: no source measures a game with our exact coupling.** Kaggriculture's
market couples both players' prices, so the win condition is relative and the
opponent affects our own returns through the price. The optimal-execution
literature reviewed previously in this repository is the nearest fit; nothing in
_this_ review models it. Consequence: every "best response" statement in Finding 7
is about opponent _policy_, not about opponent _price impact_.

**Gap 2: the reverse-curriculum results are on cheap simulators.** Salimans and
Chen used 128 GPUs for two weeks on Atari [15]; Florensa et al. used
robotics tasks [56]. Nobody has published a reverse curriculum on a
719-turn economic game with a 10^16+ action space. The exponential-to-quadratic
claim is about scaling in chain length and does not license a wall-clock estimate
for us.

**Gap 3: Kovarsky and Buro's 2006 MILP-style formulation could not be obtained.**
It survives here only through Churchill's and Chan's characterisations, so any
claim about the original's exact formulation is second-hand and is not made.

**Gap 4: no source addresses submission-slot allocation under a Bradley-Terry
finale.** The best-arm-identification literature covers fixed-budget selection
under noise, but the specific question — how to spend 5 slots/day for 28 days when
only the last 2 are scored and the ranking is recomputed post-deadline — is
answered here only by one competitor's reasoning.

### Areas of uncertainty

**Uncertainty 1: whether adding a clock feature would rescue our clone.** The
argument in Insight 2 is a synthesis, not a source claim. It is cheap to falsify
and has not been run.

**Uncertainty 2: how different two plans must be to break the joint-failure
pattern.** Our oracle-selector test showed four agents sharing one backbone fail
together; `shopforge` carries six plans. Nothing tells us whether two genuinely
different plans suffice, or whether the useful number is nearer six.

**Uncertainty 3: whether within-lineage margin transfers to the finale.** Under
Nash averaging, a margin earned against many copies of one's own lineage is worth
less than it reads [76]. Whether Bradley-Terry over post-deadline
episodes inherits that distortion depends on the composition of the field at the
end of September, which nobody can measure now.

---

## Recommendations

### Immediate actions

1. **Stop spending slots on duplicate submissions.**
   _What:_ submit a given agent once; use the second scored slot for a
   structurally different agent. _Why:_ the final ranking is a Bradley-Terry
   tournament over post-deadline episodes, so the live-rating variance a duplicate
   exploits is discarded [21][83]. _How:_ treat
   the pair as one exploiter plus one generalist. _Timeline:_ next submission.
2. **Run the two cheapest falsification tests before any build work.**
   _What:_ (a) re-train the clone with an explicit turn-index feature, exactly as
   Toad Brigade's `game phase` [30]; (b) run Wen et al.'s copycat
   diagnostic — fit a model predicting the next action from past actions alone and
   compare its fit on the clone's rollouts against the teacher's [14].
   _Why:_ both are hours, and either could reclassify the imitation failure.
   _Timeline:_ week 1.
3. **Re-weight the field gate away from a plain occupancy-weighted mean.**
   _What:_ add a hardest-opponent weighting in the spirit of PFSP's
   `f_hard(x)=(1−x)^p` [47] alongside the current mean. _Why:_ the mean is
   the quantity AlphaStar warns produces exploitable policies, and our sweep's
   flat per-opponent breakdown is that warning in our own data.

### Next steps

1. **Build a second plan, not a second parameterisation.** Section "What is worth
   our remaining four weeks" ranks this first and specifies it.
2. **Instrument lineage identification inside the agent**, so a served agent can
   condition on which of the top signatures it is facing — the "type-based
   reasoning" branch of opponent modelling, which "can lead to fast adaptation if
   true type of agent ... is in type space" and fails when it is not
   [85].
3. **Keep the offline sparring pool refreshed weekly**, because the rotation
   timescale measured everywhere is days [47][78].

### Further research needs

1. **Whether the goose and third-quadrant deficits are exploitable.** Both are
   constants in a vendored kernel and both have cheap paired experiments attached;
   neither is a literature question.
2. **Whether a reverse curriculum from our own route can move a from-scratch
   policy at all**, measured as "does it ever complete the seven-link chain",
   not as win rate.
3. **A Kaggriculture-specific measurement of Bradley-Terry tie handling**, since
   mirror matches tie exactly and a large share of the field is our own lineage.

---

## Bibliography

[1] mzotkiew (2017). "On NAPs, Doves, Hawks, and Retaliators". Halite I winning-bot write-up. https://github.com/mzotkiew/HaliteBot/blob/master/writeup.pdf (Retrieved: 2026-09-02)

[2] reCurse (2018). "Halite II post-mortem". http://web.archive.org/web/20250912062821/https://recursive.cc/blog/halite-ii-post-mortem.html (Retrieved: 2026-09-02)

[3] teccles (2019). halite3-bot README (Halite III winning bot). https://github.com/teccles-halite/halite3-bot (Retrieved: 2026-09-02)

[4] Van de Wiele, T. (2020). "1st Place - Winning Solution". Halite by Two Sigma, Kaggle discussion 183543. https://www.kaggle.com/competitions/halite/discussion/183543 (Retrieved: 2026-09-02)

[5] Buisman, H. (2022). "1st place solution". Kore 2022, Kaggle write-up. https://www.kaggle.com/competitions/kore-2022/writeups/harm-buisman-1st-place-solution (Retrieved: 2026-09-02)

[6] ry*andy* / Anderson, R. (2023). "1st place solution". Lux AI Season 2, Kaggle write-up. https://www.kaggle.com/competitions/lux-ai-season-2/writeups/ry-andy-1st-place-solution (Retrieved: 2026-09-02)

[7] WeKick (2020). "WeKick: Temporary 1st place Solution". Google Research Football with Manchester City F.C., Kaggle discussion 202232. https://www.kaggle.com/competitions/google-football/discussion/202232 (Retrieved: 2026-09-02)

[8] Toad Brigade / Pressman, I. (2021). "Toad Brigade's Approach - Deep Reinforcement Learning". Lux AI 2021 winning write-up. https://www.kaggle.com/competitions/lux-ai-2021/writeups/toad-brigade-toad-brigade-s-approach-deep-reinforc (Retrieved: 2026-09-02)

[9] Flat Neurons (2025). "1st place approach by Flat Neurons". NeurIPS 2024 - Lux AI Season 3, Kaggle discussion 569562. https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569562 (Retrieved: 2026-09-02)

[10] Kaggle (2020). Halite by Two Sigma final leaderboard, retrieved via the Kaggle API. https://www.kaggle.com/competitions/halite/leaderboard (Retrieved: 2026-09-02)

[11] Kaggle (2023). Lux AI Season 2 final leaderboard, retrieved via the Kaggle API. https://www.kaggle.com/competitions/lux-ai-season-2/leaderboard (Retrieved: 2026-09-02)

[12] Spencer, J.; Choudhury, S.; Venkatraman, A.; Ziebart, B.; Bagnell, J. A. (2021). "Feedback in Imitation Learning: The Three Regimes of Covariate Shift". arXiv:2102.02872. https://arxiv.org/abs/2102.02872 (Retrieved: 2026-09-02)

[13] Ross, S.; Gordon, G. J.; Bagnell, J. A. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning". AISTATS 2011. arXiv:1011.0686. https://arxiv.org/abs/1011.0686 (Retrieved: 2026-09-02)

[14] Wen, C.; Lin, J.; Darrell, T.; Jayaraman, D.; Gao, Y. (2020). "Fighting Copycat Agents in Behavioral Cloning from Observation Histories". NeurIPS 2020. arXiv:2010.14876. https://arxiv.org/abs/2010.14876 (Retrieved: 2026-09-02)

[15] Salimans, T.; Chen, R. (2018). "Learning Montezuma's Revenge from a Single Demonstration". arXiv:1812.03381. https://arxiv.org/abs/1812.03381 (Retrieved: 2026-09-02)

[16] Churchill, D.; Buro, M. (2011). "Build Order Optimization in StarCraft". AIIDE-11, pp. 14-19. https://cdn.aaai.org/ojs/12435/12435-52-15963-1-2-20201228.pdf (Retrieved: 2026-09-02)

[17] Leyman, P.; Vanhoucke, M. (2017). "Capital- and resource-constrained project scheduling with net present value optimization". European Journal of Operational Research 256(3):757-776. https://doi.org/10.1016/j.ejor.2016.07.019 (Retrieved: 2026-09-02)

[18] Chan, H.; Fern, A.; Ray, S.; Wilson, N.; Ventura, C. (2007). "Online Planning for Resource Production in Real-Time Strategy Games". ICAPS-07, pp. 65-72. https://web.engr.oregonstate.edu/~afern/papers/icaps07-rts.pdf (Retrieved: 2026-09-02)

[19] Blackford, J.; Lamont, G. B. (2014). "The Real-Time Strategy Game Multi-Objective Build Order Problem". AIIDE-14. https://cdn.aaai.org/ojs/12720/12720-52-16237-1-2-20201228.pdf (Retrieved: 2026-09-02)

[20] Balduzzi, D.; Garnelo, M.; Bachrach, Y.; Czarnecki, W. M.; Perolat, J.; Jaderberg, M.; Graepel, T. (2019). "Open-ended Learning in Symmetric Zero-sum Games". ICML 2019. arXiv:1901.08106. https://arxiv.org/abs/1901.08106 (Retrieved: 2026-09-02)

[21] Hasegawa, R. (2026). "1st Place(previously) - Submission Strategy for beginners". Kaggriculture, Kaggle discussion 736219 (verified 2026-09-02 via the Kaggle discussions API). https://www.kaggle.com/competitions/kaggriculture/discussion/736219 (Retrieved: 2026-09-02)

[22] Larchenko, I. (2020). "Some thoughts about the rating system". Santa 2020, Kaggle discussion 204741. https://www.kaggle.com/competitions/santa-2020/discussion/204741 (Retrieved: 2026-09-02)

[23] kiyomiya-k (2026). "The same agent scored 687 to 855: what I learned measuring variance before trusting any score". The Pokemon Company - PTCG AI Battle Challenge Simulation, Kaggle discussion 737125 (verified 2026-09-02 via the Kaggle discussions API). https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/737125 (Retrieved: 2026-09-02)

[24] Where is my bag / Boooooooooow (2021). "1st Place Solution - Where is my bag's Journey". Rock Paper Scissors, Kaggle discussion 221502. https://www.kaggle.com/competitions/rock-paper-scissors/discussion/221502 (Retrieved: 2026-09-02)

[25] Where is my bag (2021). RPS-Kaggle-1st-Place-Solution README. https://github.com/thisisbowen/RPS-Kaggle-1st-Place-Solution (Retrieved: 2026-09-02)

[26] nagiss (2021). "Imaginary Rudolph Prize Solution". Santa 2020 - The Candy Cane Contest, Kaggle discussion 218453. https://www.kaggle.com/competitions/santa-2020/discussion/218453 (Retrieved: 2026-09-02)

[27] Kaggle (2025). NeurIPS 2024 - Lux AI Season 3 final leaderboard, retrieved via the Kaggle API. https://www.kaggle.com/competitions/lux-ai-season-3/leaderboard (Retrieved: 2026-09-02)

[28] CPMP (2020). "Connect4 is solved, first player wins". ConnectX, Kaggle discussion 124397. https://www.kaggle.com/competitions/connectx/discussion/124397 (Retrieved: 2026-09-02)

[29] Raine Force (2020). 2nd place write-up, Halite by Two Sigma. https://www.kaggle.com/competitions/halite/writeups/raine-force-raine-force-writeup-2nd-place-solution (Retrieved: 2026-09-02)

[30] Pressman, I. (2021). Kaggle_Lux_AI_2021 README (Toad Brigade, Lux AI Season 1 winner). https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/README.md (Retrieved: 2026-09-02)

[31] Pressman, I. (Frog Parade) (2025). "Frog Parade's Solution" (2nd place). NeurIPS 2024 - Lux AI Season 3, Kaggle discussion 568621. https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568621 (Retrieved: 2026-09-02)

[32] aDg4b (2025). "Imitation Learning: 3rd Place Solution". NeurIPS 2024 - Lux AI Season 3, Kaggle write-up. https://www.kaggle.com/competitions/lux-ai-season-3/writeups/adg4b-imitation-learning-3rd-place-solution (Retrieved: 2026-09-02)

[33] Fox, M.; Gerevini, A.; Long, D.; Serina, I. (2006). "Plan Stability: Replanning versus Plan Repair". ICAPS 2006. https://cdn.aaai.org/ICAPS/2006/ICAPS06-022.pdf (Retrieved: 2026-09-02)

[34] de Haan, R.; Roubickova, A.; Szeider, S. (2013). "Parameterized Complexity Results for Plan Reuse". arXiv:1307.4440. https://arxiv.org/abs/1307.4440 (Retrieved: 2026-09-02)

[35] Ross, S.; Bagnell, J. A. (2010). "Efficient Reductions for Imitation Learning". AISTATS 2010, PMLR 9:661-668. https://proceedings.mlr.press/v9/ross10a.html (Retrieved: 2026-09-02)

[36] Rajaraman, N.; Yang, L. F.; Jiao, J.; Ramchandran, K. (2020). "Toward the Fundamental Limits of Imitation Learning". NeurIPS 2020. arXiv:2009.05990. https://arxiv.org/abs/2009.05990 (Retrieved: 2026-09-02)

[37] Xu, T.; Li, Z.; Yu, Y. (2020). "Error Bounds of Imitating Policies and Environments". NeurIPS 2020. arXiv:2010.11876. https://arxiv.org/abs/2010.11876 (Retrieved: 2026-09-02)

[38] de Haan, P.; Jayaraman, D.; Levine, S. (2019). "Causal Confusion in Imitation Learning". NeurIPS 2019. arXiv:1905.11979. https://arxiv.org/abs/1905.11979 (Retrieved: 2026-09-02)

[39] Swamy, G.; Choudhury, S.; Bagnell, J. A.; Wu, Z. S. (2022). "Sequence Model Imitation Learning with Unobserved Contexts". NeurIPS 2022. arXiv:2208.02225. https://arxiv.org/abs/2208.02225 (Retrieved: 2026-09-02)

[40] Laskey, M.; Lee, J.; Fox, R.; Dragan, A.; Goldberg, K. (2017). "DART: Noise Injection for Robust Imitation Learning". CoRL 2017. arXiv:1703.09327. https://arxiv.org/abs/1703.09327 (Retrieved: 2026-09-02)

[41] Swamy, G.; Choudhury, S.; Bagnell, J. A.; Wu, Z. S. (2021). "Of Moments and Matching: A Game-Theoretic Framework for Closing the Imitation Gap". ICML 2021. arXiv:2103.03236. https://arxiv.org/abs/2103.03236 (Retrieved: 2026-09-02)

[42] Mandlekar, A.; Xu, D.; Wong, J.; Nasiriany, S.; Wang, C.; Kulkarni, R.; Fei-Fei, L.; Savarese, S.; Zhu, Y.; Martin-Martin, R. (2021). "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation" (robomimic). CoRL 2021. arXiv:2108.03298. https://arxiv.org/abs/2108.03298 (Retrieved: 2026-09-02)

[43] Codevilla, F.; Lopez, A. M.; Koltun, V.; Dosovitskiy, A. (2018). "On Offline Evaluation of Vision-based Driving Models". ECCV 2018. arXiv:1809.04843. https://arxiv.org/abs/1809.04843 (Retrieved: 2026-09-02)

[44] Ho, J.; Ermon, S. (2016). "Generative Adversarial Imitation Learning". NeurIPS 2016. arXiv:1606.03476. https://arxiv.org/abs/1606.03476 (Retrieved: 2026-09-02)

[45] Zhao, T. Z.; Kumar, V.; Levine, S.; Finn, C. (2023). "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT). RSS 2023. arXiv:2304.13705. https://arxiv.org/abs/2304.13705 (Retrieved: 2026-09-02)

[46] Chi, C.; Xu, Z.; Feng, S.; Cousineau, E.; Du, Y.; Burchfiel, B.; Tedrake, R.; Song, S. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion". RSS 2023. arXiv:2303.04137. https://arxiv.org/abs/2303.04137 (Retrieved: 2026-09-02)

[47] Vinyals, O.; Babuschkin, I.; Czarnecki, W. M.; et al. (2019). "Grandmaster level in StarCraft II using multi-agent reinforcement learning". Nature 575:350-354 (DeepMind-hosted accepted manuscript). https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf (Retrieved: 2026-09-02)

[48] Berner, C.; Brockman, G.; Chan, B.; et al. (2019). "Dota 2 with Large Scale Deep Reinforcement Learning". arXiv:1912.06680. https://arxiv.org/abs/1912.06680 (Retrieved: 2026-09-02)

[49] Huang, S.; Ontanon, S.; Bamford, C.; Grela, L. (2021). "Gym-microRTS: Toward Affordable Full Game Real-time Strategy Games Research with Deep Reinforcement Learning". IEEE CoG 2021. arXiv:2105.13807. https://arxiv.org/abs/2105.13807 (Retrieved: 2026-09-02)

[50] Tavakoli, A.; Pardo, F.; Kormushev, P. (2018). "Action Branching Architectures for Deep Reinforcement Learning". AAAI-18. arXiv:1711.08946. https://arxiv.org/abs/1711.08946 (Retrieved: 2026-09-02)

[51] Kanervisto, A.; Scheller, C.; Hautamaki, V. (2020). "Action Space Shaping in Deep Reinforcement Learning". IEEE CoG 2020. arXiv:2004.00980. https://arxiv.org/abs/2004.00980 (Retrieved: 2026-09-02)

[52] Huang, S.; Ontanon, S. (2020/2022). "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms". FLAIRS-35. arXiv:2006.14171. https://arxiv.org/abs/2006.14171 (Retrieved: 2026-09-02)

[53] Sun, P.; Sun, X.; Han, L.; et al. (2018). "TStarBots: Defeating the Cheating Level Builtin AI in StarCraft II in the Full Game". arXiv:1809.07193. https://arxiv.org/abs/1809.07193 (Retrieved: 2026-09-02)

[54] Liu, R.; Pang, Z.-J.; et al. (2020). "On Reinforcement Learning for Full-length Game of StarCraft". AAAI-20. arXiv:1809.09095. https://arxiv.org/abs/1809.09095 (Retrieved: 2026-09-02)

[55] Resnick, C.; Raileanu, R.; Kapoor, S.; Peysakhovich, A.; Cho, K.; Bruna, J. (2018). "Backplay: 'Man muss immer umkehren'". arXiv:1807.06919. https://arxiv.org/abs/1807.06919 (Retrieved: 2026-09-02)

[56] Florensa, C.; Held, D.; Wulfmeier, M.; Zhang, M.; Abbeel, P. (2017). "Reverse Curriculum Generation for Reinforcement Learning". CoRL 2017. arXiv:1707.05300. https://arxiv.org/abs/1707.05300 (Retrieved: 2026-09-02)

[57] Ecoffet, A.; Huizinga, J.; Lehman, J.; Stanley, K. O.; Clune, J. (2021). "First return, then explore". Nature 590:580-586. arXiv:2004.12919. https://arxiv.org/abs/2004.12919 (Retrieved: 2026-09-02)

[58] Ng, A. Y.; Harada, D.; Russell, S. (1999). "Policy invariance under reward transformations: Theory and application to reward shaping". ICML-16, pp. 278-287. https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf (Retrieved: 2026-09-02)

[59] Arjona-Medina, J. A.; Gillhofer, M.; Widrich, M.; Unterthiner, T.; Brandstetter, J.; Hochreiter, S. (2019). "RUDDER: Return Decomposition for Delayed Rewards". NeurIPS 2019. arXiv:1806.07857. https://arxiv.org/abs/1806.07857 (Retrieved: 2026-09-02)

[60] Churchill, D.; Buro, M.; Kelly, R. (2019). "Robust Continuous Build-Order Optimization in StarCraft". IEEE CoG 2019. https://ieee-cog.org/2019/papers/paper_85.pdf (Retrieved: 2026-09-02)

[61] Ontanon, S.; Synnaeve, G.; Uriarte, A.; Richoux, F.; Churchill, D.; Preuss, M. (2013). "A Survey of Real-Time Strategy Game AI Research and Competition in StarCraft". IEEE Transactions on Computational Intelligence and AI in Games 5(4):293-311. https://davechurchill.ca/publications/pdf/starcraft_survey.pdf (Retrieved: 2026-09-02)

[62] Justesen, N.; Risi, S. (2017). "Continual Online Evolutionary Planning for In-Game Build Order Adaptation in StarCraft". GECCO 2017, pp. 187-194. https://sebastianrisi.com/wp-content/uploads/justesen_gecco17.pdf (Retrieved: 2026-09-02)

[63] Rooijackers, R.; Winands, M. H. M. (2017). "Resource-Gathering Algorithms in the Game of StarCraft". IEEE CIG 2017. https://dke.maastrichtuniversity.nl/m.winands/documents/CIG2017_resourcegathering.pdf (Retrieved: 2026-09-02)

[64] He, Z.; Liu, R.; Jia, T. (2012). "Metaheuristics for multi-mode capital-constrained project payment scheduling". European Journal of Operational Research 223(3):605-613. https://ideas.repec.org/a/eee/ejores/v223y2012i3p605-613.html (Retrieved: 2026-09-02)

[65] Habibi, F.; Barzinpour, F.; Sadjadi, S. J. (2018). "Resource-constrained project scheduling problem: review of past and recent developments". Journal of Project Management 3:55-88. http://growingscience.com/jpm/Vol3/jpm_2018_5.pdf (Retrieved: 2026-09-02)

[66] Churchill, D. (2016). "Heuristic Search Techniques for Real-Time Strategy Games". PhD thesis, University of Alberta. https://davechurchill.ca/publications/pdf/DavidChurchill_phd_thesis.pdf (Retrieved: 2026-09-02)

[67] Barriga, N. A.; Stanescu, M.; Buro, M. (2015). "Puppet Search: Enhancing Scripted Behavior by Look-Ahead Search with Applications to Real-Time Strategy Games". AIIDE 2015. https://cdn.aaai.org/ojs/12779/12779-52-16296-1-2-20201228.pdf (Retrieved: 2026-09-02)

[68] Barriga, N. A.; Stanescu, M.; Buro, M. (2017). "Game Tree Search Based on Non-Deterministic Action Scripts in Real-Time Strategy Games". IEEE Transactions on Computational Intelligence and AI in Games. DOI 10.1109/TCIAIG.2017.2717902. https://skatgame.net/mburo/ps/TCIAIG17-puppet.pdf (Retrieved: 2026-09-02)

[69] Anthony, T.; Tian, Z.; Barber, D. (2017). "Thinking Fast and Slow with Deep Learning and Tree Search". NeurIPS 2017. arXiv:1705.08439. https://arxiv.org/abs/1705.08439 (Retrieved: 2026-09-02)

[70] Garcia-Sanchez, P.; Tonda, A.; Mora, A. M.; Squillero, G.; Merelo, J. J. (2015). "Towards automatic StarCraft strategy generation using genetic programming". IEEE CIG 2015. https://ieeexplore.ieee.org/document/7317940/ (Retrieved: 2026-09-02)

[71] Justesen, N.; Risi, S. (2017). "Continual online evolutionary planning for in-game build order adaptation in StarCraft". GECCO 2017, pp. 187-194. https://dl.acm.org/doi/10.1145/3071178.3071210 (Retrieved: 2026-09-02)

[72] Wellman, M. P. (2006). "Methods for Empirical Game-Theoretic Analysis". AAAI-06. https://cdn.aaai.org/AAAI/2006/AAAI06-248.pdf (Retrieved: 2026-09-02)

[73] Walsh, W. E.; Das, R.; Tesauro, G.; Kephart, J. O. (2002). "Analyzing Complex Strategic Interactions in Multi-Agent Systems". AAAI-02 Workshop on Game Theoretic and Decision Theoretic Agents. http://www.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2005/notes/walsh.pdf (Retrieved: 2026-09-02)

[74] Tuyls, K.; Perolat, J.; Lanctot, M.; Leibo, J. Z.; Graepel, T. (2018). "A Generalised Method for Empirical Game Theoretic Analysis". AAMAS 2018. arXiv:1803.06376. https://arxiv.org/abs/1803.06376 (Retrieved: 2026-09-02)

[75] Kaggle (2021). "Hungry Geese - Evaluation". https://www.kaggle.com/competitions/hungry-geese/overview/evaluation (Retrieved: 2026-09-02)

[76] Balduzzi, D.; Tuyls, K.; Perolat, J.; Graepel, T. (2018). "Re-evaluating Evaluation". NeurIPS 2018. arXiv:1806.02643. https://arxiv.org/abs/1806.02643 (Retrieved: 2026-09-02)

[77] Lanctot, M.; Zambaldi, V.; Gruslys, A.; Lazaridou, A.; Tuyls, K.; Perolat, J.; Silver, D.; Graepel, T. (2017). "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning". NeurIPS 2017. arXiv:1711.00832. https://arxiv.org/abs/1711.00832 (Retrieved: 2026-09-02)

[78] Zuparic, M.; Khuu, D.; Zach, A. (2020). "Information theory and player archetype choice in Hearthstone". arXiv:2008.07663. https://arxiv.org/abs/2008.07663 (Retrieved: 2026-09-02)

[79] Ganzfried, S.; Sandholm, T. (2012). "Safe Opponent Exploitation". ACM EC'12; extended version in ACM TEAC 3(2), 2015. http://www.cs.cmu.edu/~sandholm/safeExploitation.ec12.pdf (Retrieved: 2026-09-02)

[80] Czarnecki, W. M.; Gidel, G.; Tracey, B.; Tuyls, K.; Omidshafiei, S.; Balduzzi, D.; Jaderberg, M. (2020). "Real World Games Look Like Spinning Tops". NeurIPS 2020. arXiv:2004.09468. https://arxiv.org/abs/2004.09468 (Retrieved: 2026-09-02)

[81] glazed (2021). "Public Code and the Evil Twin Effect". Hungry Geese, Kaggle discussion 230794. https://www.kaggle.com/competitions/hungry-geese/discussion/230794 (Retrieved: 2026-09-02)

[82] robga; Rob Gardiner; et al. (2021). "9th place solution: RL with LookAhead, Floodfill, and Rules". Hungry Geese, Kaggle discussion 255931. https://www.kaggle.com/competitions/hungry-geese/discussion/255931 (Retrieved: 2026-09-02)

[83] Kaggle (2024). "NeurIPS 2024 - Lux AI Season 3 - Evaluation". https://www.kaggle.com/competitions/lux-ai-season-3/overview/evaluation (Retrieved: 2026-09-02)

[84] Van de Wiele, T. (2020). "Questions about the final games (last week)". Halite by Two Sigma, Kaggle discussion 166560. https://www.kaggle.com/competitions/halite/discussion/166560 (Retrieved: 2026-09-02)

[85] Albrecht, S. V.; Stone, P. (2018). "Autonomous Agents Modelling Other Agents: A Comprehensive Survey and Open Problems". Artificial Intelligence 258:66-95. arXiv:1709.08071. https://arxiv.org/abs/1709.08071 (Retrieved: 2026-09-02)

[86] Nelson, B. L.; Matejcik, F. J. (1995). "Using Common Random Numbers for Indifference-Zone Selection and Multiple Comparisons in Simulation". Management Science 41(12):1935-1945. https://pubsonline.informs.org/doi/10.1287/mnsc.41.12.1935 (Retrieved: 2026-09-02)

---

## Appendix: Methodology

### Research process

Eight phases were executed in deep mode. **SCOPE** decomposed the five-part
question into seven investigable findings and fixed the internal constraint set
from `.claude/ralph-state.local.md`, `docs/competition.md`,
`docs/research/2026-09-02-what-the-top-build.md` and
`docs/experiments/2026-08-07-rl-ledger.md`. **PLAN** allocated five parallel
retrieval agents, one per question part plus one on planning and portfolio
methods held by the controller. **RETRIEVE** ran those agents concurrently; each
returned structured evidence objects (claim, verbatim quote, URL, title, date,
confidence) rather than prose, and each was instructed to prefer primary sources
and to mark anything unverifiable as UNVERIFIED. **TRIANGULATE** cross-checked
every competition result against a leaderboard row or a write-up body; two
premises supplied in the commissioning question were refuted this way.
**OUTLINE REFINEMENT** added Finding 6 (choice-point search over script
portfolios), which was not in the original plan and emerged from the collision
between our oracle-selector measurement and Puppet Search's published caveat.
**SYNTHESIZE** wrote findings against persisted evidence. **CRITIQUE** applied
three personas — a skeptical practitioner who asked whether any recommendation
survives 28 days on one GPU, an adversarial reviewer who demanded the
counterevidence register, and an implementation engineer who forced the 1-second
`actTimeout` into Finding 6. **REFINE** added the counterevidence register and
the four gaps.

### Sources consulted

**Total sources registered:** 92, of which 86 are cited below; 165 persisted
evidence spans in `evidence.jsonl`.

- Academic papers (peer-reviewed conference/journal): 44
- Competition write-ups and discussion posts by competitors: 27
- Winners' code repositories: 5
- Kaggle official leaderboards and evaluation pages: 6
- Operations-research journal articles and surveys: 6
- Technical reports and preprints: 4

**Temporal coverage:** 1992–2026, with the RTS and imitation-learning core
falling in 2006–2023 and all competition results in 2016–2026. Fifteen sources
are from 2023 or later.

### Verification approach

**Triangulation.** Every competition outcome is supported by at least two
independent artefacts: a leaderboard row retrieved through the Kaggle API and the
winner's own write-up or repository. The two 2026 Kaggle discussions cited
([21], [23]) were independently verified by the
controller through the Kaggle discussions API after the local cache proved stale;
their titles, authors and body text were confirmed before use. The ACT chunk-size
ablation, which carries a top recommendation, was re-extracted from the paper PDF
by the controller rather than taken on the retrieval agent's word.

**Credibility assessment.** Sources were graded by artefact type: leaderboard CSVs
and paper text ≥90; write-ups and repositories 80-90; discussion posts 60-80,
raised when corroborated in-thread by an independent competitor's measurements.
Secondary roundups were excluded entirely after two mis-stated the Halite IV
winner's identity and method — a failure mode this repository's research index
already warns about.

**A note on link checking.** An automated HEAD/GET sweep of the bibliography
reports 404 or 403 for roughly twenty entries. Every one of these is a Kaggle
page (leaderboards, write-ups, discussions and evaluation pages are
single-page apps that return a shell to an unauthenticated fetch) or a
publisher paywall (IEEE, ACM, INFORMS, Elsevier). All were retrieved for this
report through the Kaggle API with local credentials, or by extracting text
from an author-hosted or preprint PDF. No citation in this report was accepted
without its content being read.

**Quality control.** Three claims supplied by retrieval agents were dropped for
lack of a primary artefact: the Hungry Geese 2021 winner's architecture (the
write-up thread is deleted on Kaggle), Kovarsky and Buro's 2006 formulation (no
accessible copy), and a widely circulated Japanese summary of Lux S3 that
transposes parameter counts with training steps.

### Claims-evidence table

| ID  | Major claim                                                                                          | Evidence type                                                               | Sources          | Confidence  |
| --- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------- | ----------- |
| C1  | Kaggle simulation winners split by simulator throughput, not by era                                  | Leaderboards + winners' write-ups                                           | [4][8][6][9][31] | High        |
| C2  | The Halite IV winner abandoned deep RL and won with a rule agent                                     | Primary write-up + leaderboard                                              | [4][10]          | High        |
| C3  | DAgger's guarantee is class-relative and does not fix non-realizability                              | Theorem text + corroborating negative result                                | [13][40][12]     | High        |
| C4  | High per-action accuracy with zero task performance is the copycat/causal-confusion signature        | Two independent papers reporting held-out loss improving while reward falls | [14][38][42]     | High        |
| C5  | Action chunking reduces the effective horizon k-fold, with a measured 1%→44% swing                   | Paper text re-verified from PDF by the controller                           | [45][46]         | High        |
| C6  | Reverse curricula turn chain-learning cost from exponential to quadratic in chain length             | Paper claim plus two corroborating methods                                  | [15][55][56]     | Medium-high |
| C7  | Build-order search abstracts labour away and its fast-forwarding assumes no hoarding                 | Paper text, stated by the authors                                           | [16][60][61]     | High        |
| C8  | The joint labour-and-cash problem exists in OR as CRCPSPDC and is strongly NP-hard even without cash | Journal abstracts + survey                                                  | [17][64][65]     | Medium      |
| C9  | Choice-point search over scripts gains nothing outside the portfolio's coverage                      | Paper's own reported failure case, matching our oracle-selector measurement | [67][68]         | High        |
| C10 | Elo/TrueSkill is not invariant to redundant copies, so a clone-filled ladder distorts rank           | Theorem-style property plus a worked numeric example                        | [76]             | High        |
| C11 | Under a dominant agent, best-responding is the principled population move and diversity is zero      | Paper text                                                                  | [20][77]         | High        |
| C12 | Safe exploitation has a formal budget equal to profit banked from opponent mistakes                  | Paper text plus measured tradeoff                                           | [79]             | High        |
| C13 | Under a Bradley-Terry finale, re-rolling an unchanged agent buys nothing                             | Competitor's model, corroborated by `docs/competition.md`                   | [21][83]         | Medium      |
| C14 | Meta rotation timescale is days in every population where it has been measured                       | Two independent measured populations                                        | [47][78]         | High        |

---

## Report Metadata

**Research mode:** Deep (8 phases)
**Total sources:** 92 registered, 86 cited
**Persisted evidence spans:** 165
**Approximate word count:** 14,300 body, 16,300 including bibliography
**Generated:** 2026-09-02
**Artefacts:** `~/Documents/Competitive_Sim_Agents_Research_20260902/{sources,evidence}.jsonl`, `run_manifest.json`

---

## What is worth our remaining four weeks

The envelope is fixed: one GPU, roughly 28 days to the 2026-09-30 deadline, five
submission slots a day of which the latest two are scored, a field-weighted local
gate covering 61.8% of seats, and a final ranking produced by a Bradley-Terry
tournament over post-deadline episodes rather than by the live ladder.

Eight directions are already measured closed and none of them is reopened below:
the offline market residual (128/128 losses), on-policy V-trace on market actions
(19,200 paired seasons, negative in every block, three independent confirmations),
route harvesting (13 top-band routes, all 0.000, with a working control), route
editing (7 idle turns of 719; a 300-coin insertion loses every mirror), config
tuning (69 + 71 + 67 candidates across three sweeps, no validated survivor),
agent selection over our own agents (a perfect oracle gains zero), build planning
(0.0000 handed the exact build), and behavioural cloning with one DAgger round
(0 of 48, unmoved).

Ranked by expected value per day of our time, with the evidence that sets the
rank and the gate that kills it.

| #   | Direction                                                                                      | Cost                   | Evidence for                              | Kill gate                                                            |
| --- | ---------------------------------------------------------------------------------------------- | ---------------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| 1   | Stop re-rolling; make the scored pair two _different_ agents                                   | hours                  | [21][83]                                  | none needed — it costs nothing                                       |
| 2   | Fix the two measured kernel deficits (geese, third quadrant)                                   | 2-4 days               | our own corpus, n=1,091 top-band seats    | paired mirror gate over 240 games                                    |
| 3   | Build one structurally different plan and a choice-point selector                              | 10-18 days             | [67][68][20] + our oracle-selector result | the new plan must beat `tuned_v58` on ≥1 lineage the incumbent loses |
| 4   | Re-weight the gate away from an occupancy-weighted mean                                        | 1-2 days               | [47] PFSP rationale                       | it must reorder at least one known pair                              |
| 5   | Lineage-conditioned counter-play using the opening-24 signature                                | 3-5 days, _after_ 3    | [85][24][82]                              | needs ≥2 genuinely different plans to switch between                 |
| 6   | Cheap falsification of the imitation diagnosis (clock feature, action chunking, copycat probe) | 1-2 days               | [30][45][14]                              | on-policy op accuracy must exceed 0.85, else stop                    |
| 7   | Reverse curriculum RL from our own route                                                       | 7+ days, high variance | [15][55]                                  | must complete the 7-link chain once inside 3 days                    |
| 8   | Do nothing new: ride kernel waves, keep `kernel_watch`                                         | ~0                     | our own measured lever map                | this is the baseline everything else must beat                       |

**1. Stop re-rolling. Rank 1 because it is free and it reverses a decision we made
this week.** Kaggle scores only the latest two submissions
[83], and Kaggriculture replaces the live ladder with a
post-deadline Bradley-Terry tournament. A competitor's fitted model of our own
engine states the consequence directly: "Your live rating's history, its lucky
start, and its 'age' count for nothing at the end — only how your last 2
submissions actually fare against the field", and "Re-submitting an unchanged bot
to 're-roll' its trajectory only moves the live number (and costs you the older
copy); it changes nothing about the final ranking" [21]. Our
2026-09-02 decision to submit `tuned_v58` twice was correct reasoning for an
ordinary Kaggle ladder — duplicate submissions do raise "the expected final score
of the best agent" when the live score is the result [84] — and wrong
for this one. **Caveat:** this rests on a competitor's model, not an organiser
statement, so the safe reading is _don't spend a slot on a duplicate_, not _never
resubmit_.

**2. The two measured kernel deficits. Rank 2 because they are the only places
where our own corpus identifies a constant in the served kernel that differs from
top-band behaviour, and both come with a cheap paired experiment.** Zero geese
across 2,686 seats of our build while `BUILD_COOP` fires 2.75 times a season and
96% of our seasons end with an empty coop; 15.9% of top-band seats end holding a
goose against 1.9% of the middle band; the goose is the cheapest animal, yields
first and yields daily. And the third quadrant on day 11 in 1,024 of 1,024 local
seasons, where the top decile has it by day 10 in 70% of seats. These are not
literature findings and the literature does not bear on them; they are ranked
high because mirror play ties exactly, which makes the marginal value of a single
change measurable with far fewer games than an ordinary A/B — the classical
common-random-numbers argument, where matching conditions across alternatives
"reduce[s] the sample size required to attain fixed precision" [86]. Gate
at 240 games; our own screens have overstated a survivor three times in one day.

**3. One structurally different plan plus a choice-point selector. Rank 3 because
it is the only direction with both a published method and an unblocked path,
and it is also the most expensive.** The published method is Puppet Search: expose
choice points in a script and select over them, "creating a new game in which move
options are restricted by replacing original move choices with potentially far
fewer choice points" [67], which takes a branching factor to 4
[68]. Their own reported failure is our measured one — "in the two
instances where no script defeated the opponent (E1 and E3), the search couldn't
defeat it either" [67] — and it is why the work is _building a second
plan_, not _building a selector_. Our oracle test over `v54`/`v56`/`v58`/`tuned`
gained exactly zero because all four share one backbone and hash to one lineage
signature. Two facts say the coverage is obtainable: `shopforge`, which banks
100-128k on the boards where it beats us, is "six readable plans and a small
public-state tree", and the top decile of the corpus builds a different farm
nearly every game — 0.732 distinct build fingerprints per seat against our 0.006,
with corr(variety, rating) = +0.667 across 75 teams.

Three constraints shape the implementation. The 1-second `actTimeout` rules out
Puppet Search's online lookahead, so the selection rule must be precomputed —
Expert Iteration's split, where search produces the plans and something cheap
generalises them [69], with the "something cheap" being a table over a
handful of choice points rather than a network, because the abstraction has
already collapsed the space to single digits. The plan-construction cost is real:
Nebel and Koehler's result says reusing a stored plan is not provably cheaper than
generating one, and identifying the reusable part is an extra source of hardness
[34], which is why route harvesting and route editing both failed. And
the honest precedent for building a plan from scratch is discouraging — our own
from-scratch route search scored 1,365 against a 2,600 floor. **The concrete
proposal that respects all three:** take the existing hybrid builder, which
already accepts 719 per-step action dicts, and construct a _variant_ of our own
backbone that differs in the two places the corpus says we are extreme — an
earlier third quadrant and a stocked coop — then treat the choice between backbone
and variant as a single choice point resolved from public state at a fixed early
turn. That is a two-plan portfolio with one choice point, which is precisely the
configuration Barriga et al. report as sufficient: "Even a script with a single
choice point to choose between different strategies can outperform the other
players in most scenarios" [68]. If it clears the gate, add a third.

**4. Re-weight the gate. Rank 4 because it is cheap and it changes what everything
else selects for.** Our field gate is an occupancy-weighted mean win rate, which
is the quantity AlphaStar deliberately does not optimise, "because ... the pursuit
of the mean win-rate might lead to policies that are easy to exploit"
[47]. Their `f_hard(x) = (1−x)^p` weighting is a few lines. The
corroborating symptom in our own data is that the field sweep's per-opponent
breakdown was nearly identical across every variant while only our own lineage's
columns moved — an occupancy-weighted mean concentrating its weight on ourselves.
Add the hard-weighted score beside the current one; do not replace it, because the
occupancy weighting is what makes it predictive of the ladder.

**5. Lineage-conditioned counter-play. Rank 5, and strictly downstream of 3.** The
theory is unusually favourable: under a dominant agent, principled population
training "degenerates to training against the best agent in the population", and
"if there is a dominant agent then diversity is zero" [20] — so
best-responding to the current wave is the right move, not a hack. The ceiling is
high: a Hungry Geese team measured "72% of the time 1v3" against the dominant
public agent by RL alone, and "100% of the time" once they exploited its
determinism [82]. Identification is solved for us — 24 opening signatures
cover the field, and the census hashes a seat's first 24 actions. Three things
hold it at rank 5. It has nothing to switch _to_ until direction 3 delivers, since
our config surface only moves our own lineage. Best responses to a fixed opponent
set overfit, at a measured cost of up to 71.7% of reward [77], and safe
exploitation is bounded by the profit already banked from the opponent's mistakes
[79]. And a successful counter can remove its own target from the
matchmaking band, which is exactly what one competitor measured: "the top versions
of my own agent never even played against the targeted notebook agents"
[81]. The safe shape is the one both the theory and the practitioners
converge on — one exploiter and one generalist across the two scored slots.

**6. Cheap falsification of the imitation diagnosis. Rank 6 — one to two days, and
worth it only as a diagnosis, not as a route to an agent.** Three tests. Add an
explicit turn-index feature, exactly Toad Brigade's `game phase`, which they call
"a crucial part of its success" [30]; the copycat and causal-confusion
literature warns against conditioning on past _actions_ [14][39],
but a monotone clock is not an action the learner emitted, so the objection does
not transfer. Predict an action chunk rather than one action, which "reduces the
effective horizon of the task by k-fold" and explicitly handles confounders that
depend on "the timestep" [45]; their ablation moved success from 1% at k=1
to 44% at k=100 [45]. And run Wen et al.'s diagnostic: fit a model
predicting the next action from past actions alone, and compare its fit on the
clone's rollouts against the teacher's, since "the action predictability ratio is
inversely correlated with the normalized reward" [14]. **Why this is only
rank 6.** ACT's curve runs from closed-loop toward open-loop and peaks partway;
`k = episode_length` is "fully open-loop control" [45]. Our teacher is
already at that end — it is a stored plan — so the limit of the chunking fix is
storing the plan, which we can do exactly and for free by keeping the route file.
The literature's answer to "how do I imitate an open-loop plan with a
state-conditioned policy" is: don't; represent the plan as a plan. Kill the line
if on-policy op accuracy does not clear 0.85.

**7. Reverse curriculum RL. Rank 7 — the highest theoretical leverage and the
worst fit to 28 days.** The result is the strongest in this review: resetting
episodes to demonstration states and sliding the reset point backwards means an
N-action chain "may now be learned in a time that is quadratic in N, rather than
exponential" [15], where starting from the initial state "scales
exponentially in the problem size" [15]; Backplay adds that the agent
"can outperform its demonstrator" [55]. Every precondition holds here:
we have a demonstration that banks 84,797, our seasons are deterministic given a
seed, and mirror play ties exactly. Three things sink it inside four weeks. The
headline precedent "required 128 GPUs over a period of 2 weeks" on an environment
far cheaper per step than ours [15]. Our own PPO arms have consumed
100M steps to produce bank 2, and the closest solo precedent — Lux S3's runner-up
— needed roughly 300M steps over eight days _with a Rust environment running at
110,000 steps/s_ [31]. And the reduction that made every published
single-box success possible is one we have not built: the composed action head
that took Gym-microRTS from 50 million logits to 301 [49], without which the
curriculum is applied to the wrong space. If run at all, run it as a three-day
falsification with one question — does the policy ever complete the seven-link
chain — and not as a training programme.

**8. The baseline: ride the waves.** Doing nothing new is a real option and it is
the thing the seven directions above must beat. `kernel_watch` catches each new
wave in one command, the rotation is 2-3 days, our served agent out-banks every
other build in the corpus 0.700 of the time over 860 decided episodes, and the
field-weighted gate says honestly whether a candidate deserves a slot. The
argument against it is not that it performs badly but that it is capped: it can
only ever field the best available published plan plus two integers of tuning, and
under a Bradley-Terry finale the field it will be measured against is the field of
late September, which nobody can see yet.

**The ranking in one sentence.** Do 1 today and 4 this week because they are
nearly free; do 2 next because it is the only place our own corpus says the served
kernel is measurably wrong; spend the bulk of the four weeks on 3, because a
second genuinely different plan is the one thing that unblocks 5 and the only
lever the literature and our own oracle-selector measurement agree is missing;
timebox 6 to two days as a diagnosis; and do not start 7 unless 2 and 3 finish
early.
