# What Wins Kaggle Simulation Competitions, and What Should Govern a Kaggriculture Agent

A deep-research report. Produced 2026-08-03 using the `deep-research` skill's eight-phase pipeline in
deep mode. Evidence artifacts — source registry, evidence spans, claim ledger, run manifest — are in
`/data/kaggriculture/research/deep-research-run/`.

---

## Executive Summary

Kaggle has run a real-time-strategy-like simulation competition roughly once a year since 2020, and the
winning approach has alternated between hand-written rules and deep reinforcement learning in a way that
looks arbitrary until you sort the competitions by the shape of their decision problem. Rules and forward
search won Halite IV, Kore 2022 and Lux AI Season 2 [2][4]; reinforcement learning won Lux AI Season 1
and Season 3 [1][6], and imitation learning took third and fourth in Season 3 [7][8]. The dividing line
is whether a single unit's decision is a one-step choice among a handful of options, or a multi-step
commitment. Kore's flight plans and Lux S2's twenty-step action queues are planning problems that a
forward simulator solves directly and a policy network must learn to solve implicitly [2][3]. Lux S1 and
S3 gave each unit a small menu of single-step moves on a grid, and learned policies swept them [1][6].

Kaggriculture sits on the learned side of that line. Each unit chooses one of roughly twenty-two
operations on a 10×10 grid, once per turn, with no queue and with the opponent's farm fully visible;
only their shed and carried inventory are hidden [14]. The measured budget is one second per turn with a
sixty-second bank for the whole episode, and the interpreter runs at 1.68 ms per step, or about 596
steps per second on one core [14].

The competition also contains something no previous Kaggle simulation had: a shared, price-forming
market that both players trade against. Measurement says this is what decides games. Melon looks like
the best crop by a factor of six at base prices, yet zero of the eight town shops demand it, so any real
volume is pure glut against a quadratic penalty [14]. Meanwhile 75% of 800 mined agent-games run one
identical build, which both creates exploitable price structure and makes imitation learning hazardous
[14].

The recommendation is self-play reinforcement learning with a grid-shaped action head, invalid-action
masking, sparse win-based reward, and a frozen teacher for stability — trained on the workstation
already available, since the second-place Lux S3 team reached that position on an RTX 3090 and an RTX
2070 Super [8]. The binding risk is not training but inference: one competitor had a trained model he
could not deploy because no fast runtime could be installed in the Kaggle sandbox [3]. That risk is
cheap to retire with a probe submission before committing to a long run.

---

## Introduction

### Scope

This report addresses two linked questions. First, empirically, what has actually won Kaggle's
RTS-like simulation competitions, and is there a structure to it beyond fashion? Second, given that
structure and given Kaggriculture's specific mechanics and budget, what architecture, training recipe
and inference strategy should govern a self-play reinforcement learning agent for it?

In scope: the five previous Kaggle simulation competitions and their published winning solutions; the
academic literature on large factored action spaces, league training, and multi-agent pricing; and
primary measurement of the Kaggriculture environment, our own submitted agent, and the published replay
corpus. Out of scope: general RL tutorials, competitions without a comparable adversarial simulation
structure, and any approach that cannot run inside a one-second per-turn CPU budget.

### Methodology

Fifteen sources were registered with stable identifiers and thirty-seven evidence spans persisted before
synthesis began, per the pipeline's evidence-first requirement. Sources divide into three groups:
seven competition writeups from teams that placed first through fourth [1][2][3][4][5][6][7]; five
academic or technical papers [9][10][11][12][13]; and three primary sources comprising the environment's
own source code, the competition's discussion record, and measurements taken by running things
[5][8][14][15].

The primary-measurement component matters to how the conclusions should be weighted. The strongest
findings in Section 3 — melon's absent shop demand, the day-by-day price trajectories, the monoculture
build — came from executing code against the shipped interpreter and from parsing 400 episodes of the
published replay corpus, not from retrieval. They are reproducible from this repository rather than
citable to a third party, and they are the findings least likely to be wrong.

### Assumptions surfaced

Three assumptions carry enough materiality to state explicitly. First, the ladder meta observed on
2026-08-02 is treated as informative but perishable; the top rating rose from 1152 to 2627 in four days
[14], so any conclusion depending on opponent behaviour has a short half-life. Second, torch is assumed
available in the competition sandbox because it ships in the Kaggle Python image, but this is *not*
verified for the simulation sandbox specifically, and Section 7 treats it as the principal open risk.
Third, the analysis assumes the competition's default configuration, which the replay confirms is what
the ladder actually runs [14].

---

## Main Analysis

### Finding 1: The alternation between rules and learning is structural, not fashion

Sorted chronologically, the winning approaches look like noise. Halite IV in 2020 went to a rule-based
agent. Lux AI Season 1 in 2021 was swept by deep RL, with Toad Brigade first and a team named "RL is all
you need" second [1]. Kore 2022 went back to rules, with rule-based agents taking first, second and
fourth [4]. Lux AI Season 2 in 2023 again went to rules, whose author won by forward simulation, with the
strongest RL entry placing fourth [2][3]. Lux AI Season 3 in 2025 returned to learning, with an IMPALA
agent first and imitation learning third and fourth [6][7].

Sorted instead by the structure of a single decision, the pattern resolves. In Kore, a unit is issued a
*flight plan* — a compressed multi-step route. In Lux S2, each unit carries an action queue of up to
twenty steps, and re-issuing that queue costs power, so an agent that re-plans every turn is
mechanically penalised. FLG measured this directly and found that agents predicting fewer than four to
six steps ahead were "crippled" [3]. These are combinatorial planning problems, and a forward simulator
attacks them head-on: the Lux S2 winner describes determining actions for all units, stepping his own
copy of the game, and repeating for about 2.9 seconds per turn, yielding "anywhere from 5 to 50+ steps
worth of planning" [2].

In Lux S1 and S3, by contrast, a unit picks one of about six single-step actions on a grid, every turn,
with no commitment carried forward. There is no multi-step plan for search to exploit, and the problem
reduces to pattern-recognition over a spatial state — precisely what a convolutional policy does well.

The microRTS competition provides the cleanest natural experiment, because the same game was contested
repeatedly. Scripted agents won the first five iterations before a deep RL agent finally took it [9].
What changed was not the game but the technique stack: GridNet outputs with invalid-action masking, a
downscaling backbone chosen to hit a 100 ms turn deadline, iterative fine-tuning against the *previous
competition winners*, and a reward schedule moving from shaped to sparse [9]. The cost was 70 GPU-days,
later reduced to 23 by bootstrapping with behaviour cloning from opponent replays [9].

The lesson is not that RL has become universally better. It is that RL became viable once the
engineering around it — masking, grid-shaped heads, teacher regularisation, fast simulators — matured
enough to be reproduced by individuals rather than labs.

---

### Finding 2: Kaggriculture's decision structure places it on the learned side

Kaggriculture issues one operation per unit per turn, drawn from approximately twenty-two possibilities:
four moves, a pass, five plant variants, water, harvest, fertilise, dig, two build operations, three
animal operations, and three shed operations [14]. With a farmer and up to ten hired hands, the joint
space is about 22^11, or 5.8 × 10^14 combinations — intractable jointly, and identical in shape to the
problem Lux solved with a board-shaped action head [14].

Critically, there is no action queue. Each turn is a fresh, independent decision, which removes the exact
mechanism that made forward search dominant in Kore and Lux S2. It also removes FLG's hardest
engineering constraint: he needed four to six network passes inside each turn to fill queues [3],
whereas Kaggriculture needs one pass per turn.

The board is 10×10, against Lux S2's 48×48 — roughly a twentieth of the spatial extent, which makes a
convolutional trunk correspondingly cheap. Information is nearly perfect: both farms are fully visible
including every tile's state, with only the opponent's shed contents and carried inventories hidden
[14]. This is a substantially easier setting than Lux S3, whose winner needed a dedicated supervised head
to predict enemy positions through fog of war [6], or Generals.io, which the authors describe as
requiring planning "under strong imperfect information" [10].

Measured throughput supports the plan. The shipped Python interpreter runs at 1.68 ms per step, about 596
steps per second on a single core, so 64 cores yield roughly 38,000 steps per second and 100 million
environment steps costs about 0.7 hours of pure simulation [14]. That is comfortable without any rewrite,
which matters because rewrites carry a correctness risk discussed in Section 6.

---

### Finding 3: The market is the decisive axis, and it has no precedent in prior Kaggle simulations

Every previous competition in this lineage was a contest over spatial resources. Kaggriculture adds an
economic layer: each product's sale price is a function of a shared market inventory, `price(inv) = base
± amp·f(|inv − I0|)`, with independently chosen shape functions on each side of the equilibrium
`I0 = 10,000` [14]. Both players trade against the same inventory, so one player's sales directly set the
other's prices.

Demand is entirely institutional. A town centre consumes one of each non-fertilizer product every twelve
turns, doubling after day 10 and quadrupling after day 20; shops unlock every three days and each
consumes one of every product it demands every four turns [14]. Counting shop baskets across the eight
shops gives wheat five demanders, strawberry four, milk three, carrot, tomato and egg two each, wool one
— and melon and fertilizer **zero** [14].

That zero is the single most consequential fact measured in this report. At base prices melon dominates
every other crop by roughly a factor of six, returning about $129 per tile-day against wheat's $18 [14].
But melon's glut curve is quadratic with an aggressive coefficient, `price = 250 − 0.01·x²`, so a hundred
units above equilibrium prices it at $150 and a hundred and fifty-eight units puts it on the $1 floor
[14]. With no shop demand, nothing drains melon inventory except the town-centre trickle. A crop-ranking
heuristic that reads the base-price table will pick melon, build a monoculture, and discover that the
number is a fiction the moment a second melon farmer appears — which is exactly what our own submitted
agent did, banking 48,000 consistently against the built-in bots but 7,770 to 47,363 with wide variance
against real opponents [14].

This is where the economic literature becomes relevant, and it is the one part of the problem where prior
Kaggle solutions offer no guidance. Benchmarking of multi-agent RL against static pricing rules in
simulated supply chains found that rule-based agents preserved fairness and price stability but "lacked
competitive dynamics", while learned agents produced "emergent strategic behaviour not captured by static
pricing rules" [12]. The competitive dynamics that rules miss are precisely what a shared price-forming
market rewards. A learned policy observing `market["inventory"]` sees the opponent's cumulative sales
directly, every turn, as a state variable — no opponent model required.

---

### Finding 4: The current ladder is a monoculture, which is both the opportunity and the trap

Mining 400 episodes from the published corpus for 2026-08-02 — 793 episodes at a median agent rating of
2319 — produced 800 agent-games, of which 298, or 75%, run a byte-identical build: eleven wheat tiles,
forty strawberry, eleven melon, eight cows, six sheep, three quadrants [14]. Nearly all the remainder sit
within two tiles of it. Carrot appears on zero tiles across all 800 games. Tomato appears on zero. Geese
are kept by one agent in eight hundred. Nobody buys the fourth quadrant. Banks run from 73,681 at
minimum through a median of 123,334 to a maximum of 165,709 [14].

The price trajectories this monoculture produces are legible and exploitable. Strawberry climbs from its
$120 base to $233 by day 18 and then collapses to $87 by day 27, because forty tiles across the entire
field mature and are harvested on the same schedule. Wheat climbs monotonically from $25 to $56, because
fourteen animals per agent each eat one wheat per day and eleven tiles do not cover it. Tomato drifts up
to $93 with no supplier at all, and eggs to $67 with one [14].

Two implications pull in opposite directions.

The opportunity is that a policy pricing from live inventory rather than from a base table walks away
from strawberry and melon as the crowd floods them, and into wheat, tomato and eggs where demand is
unopposed. An opponent that beat our agent 198,630 to 7,770 ran a diversified build including thirteen
carrot tiles — a crop no mined agent touched [14]. The meta build is demonstrably not the ceiling.

The trap is that this same corpus is the imitation-learning dataset. Behaviour cloning against a
population that is three-quarters one copied kernel teaches the monoculture, including its badly-timed
strawberry allocation. The technique that saved microRTS 47 GPU-days [9] would here be cloning the thing
we intend to beat. Filtering to top-decile banks or to winners only is the minimum precaution, and
imitation should be treated strictly as an initialisation to escape rather than a target to match. This
is a genuine departure from the Lux S3 precedent, where third and fourth place were pure imitation
learning [7][8] against a diverse demonstrator pool.

---

### Finding 5: The compute required is available on the hardware already present

The most encouraging finding for a solo entrant is how modest the winning compute has been outside the
industrial tier.

Toad Brigade's Lux S1 winner — a 24-block residual network of roughly 20 million parameters, trained with
IMPALA plus UPGO and TD(λ) — was trained entirely on "my personal PC, an 8-core/16-thread dual-GPU
system" [1]. In Lux S3, first place did use industrial scale, a 200-million-parameter IMPALA model on
eight H100s for three to four days. But **second place trained dual 300-million-parameter PPO models for
ten million steps over eight days on an RTX 3090 and an RTX 2070 Super** [8]. Tenth place used a single
RTX 4090 [8].

The workstation available here has an RTX 3090 and an RTX 6000 Ada with 49 GB, plus 64 CPU cores —
strictly better than the second-place Lux S3 configuration.

What separated second place from the field was not hardware but throughput: they rewrote the environment
in Rust, roughly 10,800 lines, lifting data collection from a one-million-step-per-day ceiling to ten
million per day [8]. This is the same lever Nebula pulled in Kore, taking the Python environment from
12.9 ms per step to 0.08 ms in Rust, a 160× speedup [5], and the same one the Generals.io authors
identify as their central contribution, a JAX simulator running "tens of millions of frames per second on
a single GPU, roughly a 10,000× speedup", concluding flatly that "a fast simulator removes the data
bottleneck" [10].

Kaggriculture's starting position is better than any of theirs. At 1.68 ms per step it is already about
eight times faster than Kore's Python environment was [5][14], and 64 cores give roughly 38,000 steps per
second without touching the interpreter [14]. A rewrite is a later optimisation, not a prerequisite.

---

### Finding 6: Inference, not training, is the binding constraint

The clearest cautionary tale in the corpus is FLG's. He trained a larger DoubleCone model and had to
abandon it, not because it was weak but because he could not run it: "no fast inference runtime (like
onnxruntime with ... OpenVino) was available in the Kaggle environment. Since it wasn't possible to
install any packages, me and a couple of other competitors tried packaging onnxruntime ourselves but as
far as I know no one was successful" [3]. He adds that error logs returned 404, making the failure nearly
undebuggable [3]. He tried static and dynamic post-training quantisation, model compression, and JIT
tracing, and abandoned all of them [3].

Toad Brigade's numbers give the magnitude. His ~20-million-parameter network took "2 to 2.5 seconds for
inference on the Kaggle servers with a batch size of 2" [1]. That would exceed Kaggriculture's
one-second budget outright — though his board was 32×32 against our 10×10, roughly a tenfold difference
in spatial extent, and we need batch size one rather than two.

Three mitigations follow, in increasing order of safety. Size the network from measurement rather than
ambition, targeting single-digit millions of parameters. Export trained weights to plain numpy arrays and
write the forward pass by hand — a small convolutional network on a 10×10 grid is trivially fast in
numpy, and this eliminates every dependency and version risk at once; the existence of a competitor
dataset of MARL weights at 2.1 MB indicates small bundled models are already normal here [14]. And,
before committing to any long training run, submit a probe agent that logs the torch version and a timed
forward pass, then read it back with `kaggle competitions logs`. That last step costs one of five daily
submission slots and retires the largest single risk in the plan.

---

### Finding 7: The training recipe has converged across independent teams

Four independent efforts, across three competitions and two research groups, arrived at substantially the
same recipe. Where they agree, the agreement is worth more than any individual endorsement.

**Sparse, win-based reward, reached via a shaping schedule.** Toad Brigade trained with shaped rewards for
the first 20 million steps and then switched to pure ±1 win/loss [1]. FLG used roughly 65 million steps of
heavily shaped "selfish" rewards before switching to zero-sum rewards dominated by the game result [3].
Every top Lux S3 team converged on match-wins-only as the final signal, zero-sum and normalised [8]. The
Generals.io agent trained on "sparse win/loss reward" throughout [10]. FLG's observation is the useful
one: the agent "seemed very robust to changes [in reward weighting], probably because the game result ...
dominate[s] the reward" [3], and the Lux S3 roundup concludes that focusing on fundamentals rather than
fine-grained reward shaping is what separated winners from mid-tier entries [8]. For Kaggriculture the
terminal reward is unambiguous, because the ladder scores wins and never margins [14]; bank difference is
the natural dense proxy for the shaping phase.

**Invalid-action masking.** Toad Brigade masked illegal actions to negative infinity, noting that this
"did not reduce the available action space ... but did reduce the complexity of the learning task" [1].
RAISocketAI's GridNet used the same technique [9]. Huang and Ontañón give it theoretical justification,
showing the masked gradient remains a valid policy gradient, and demonstrate that its importance grows as
the space of invalid actions grows [11]. Kaggriculture is full of invalid actions — planting without a
seed, harvesting an immature tile, feeding without wheat, any order you cannot afford — so masking is not
optional.

**A frozen teacher with a KL penalty.** Toad Brigade is explicit that this "helped to stabilize behavior
and prevent strategic cycles — both of which are problems that plague a pure self-play setup" [1]. FLG
used a teacher KL weighted around 5e-3, keeping the teacher roughly 30 million steps behind [3]. The
underlying pathology is well documented: AlphaStar's league exists because self-play alone cycles among
non-transitive strategies, which prioritised fictitious self-play and dedicated exploiter agents were
introduced to break [13]. Lux S3's tenth place used PFSP directly, sampling difficult opponents 25% of the
time [8]. For our purposes a modest league — past checkpoints, the reconstructed meta build, and our
existing heuristic — captures most of the benefit at low complexity.

**Progressive scaling with the smaller model as teacher.** Toad Brigade trained an 8-block network,
then a 16-block, then a 24-block, each using its predecessor as teacher [1]. FLG's stated plan was the
same, and his architecture was selected by measuring imitation accuracy against CPU cost *before*
committing to an RL run [3]. That ordering — pick the architecture using cheap supervised signal, then
spend the expensive RL budget on it — is the most reusable process lesson in the corpus.

On algorithm choice the evidence is that it does not much matter. FLG tested PPO and V-trace and
"achieved similar results" [3]; the Lux S3 field split between IMPALA and PPO across the top ten with no
clear pattern [8].

One Kaggriculture-specific wrinkle deserves a named fix. Hiring, land purchases and animal purchases occur
on a handful of steps per episode, exactly the rare-event problem FLG hit with bidding and factory
placement, which occurred on about 1% of steps and left typical batches with zero to two examples [3]. His
solution — a dedicated actor replaying those situations to upsample them — transfers directly.

---

## Synthesis and Insights

Three non-obvious conclusions emerge from putting the evidence together.

**The competition's novel axis is the one with no competitive precedent.** Every prior Kaggle simulation
was a spatial contest, and the accumulated wisdom in those writeups is about spatial play — pathfinding,
collisions, unit micro. Kaggriculture's spatial layer is comparatively simple: a 10×10 grid, no combat,
no collisions, units that may share tiles. The layer that decides games is economic, and the writeups say
nothing about it because no previous game had one. The transferable material from the competition corpus
is therefore mostly *machinery* — how to shape an action head, how to mask, how to schedule reward — while
the *strategy* has to be discovered here. That asymmetry argues for learning over hand-coding more
strongly than any single citation does: we can borrow the machinery with confidence and cannot borrow the
strategy at all.

**The monoculture is a temporary and self-limiting opportunity.** Seventy-five percent of the field
running one build produces exactly the price structure that build cannot exploit — strawberry crashing on
schedule, wheat appreciating monotonically, tomato and eggs unsupplied [14]. But this holds only while
the monoculture holds. The top rating moved from 1152 to 2627 in four days [14], and the final standing is
decided by a Bradley-Terry tournament run two weeks after the September deadline [14], which is roughly
two months of meta evolution away. Any solution whose edge *is* the current monoculture will decay; a
policy that prices from live market state adapts by construction. This is an argument for learning the
allocation rather than tuning it, and it is independent of whether learning is currently ahead on raw
strength.

**The engineering risk is concentrated at deployment, not training.** Training compute is demonstrably
within reach — second place in the most recent competition used hardware inferior to what is already
available here [8]. What actually destroyed a strong competitor's model was the inability to run it inside
the Kaggle sandbox [3]. The correct sequencing therefore inverts the intuitive one: establish what can be
deployed *before* deciding what to train. A probe submission costs one slot and one hour; discovering the
constraint after a week of training costs the week.

---

## Limitations and Caveats

The competition-writeup evidence is systematically biased toward success. Writeups are published by teams
that placed well, so the corpus describes what worked for winners and is silent on how many comparable
efforts failed. The claim that RL is viable on consumer hardware rests on the visible successes; the
denominator is unknown.

Two claims in this report rest on only two sources each, below the three-source standard applied
elsewhere, and are flagged accordingly. That Kaggle's sandbox blocks fast inference runtimes rests on
FLG's account and Toad Brigade's timing figures [1][3], both from the same competition family and both now
one to five years old; the sandbox may have changed. That consumer hardware suffices rests on Toad
Brigade and the Lux S3 roundup [1][8].

The Lux S3 numbers — parameter counts, hardware, training durations — come from a third-party roundup
rather than the teams' own writeups [8], and the two figures checkable against a primary source (Flat
Neurons' use of IMPALA) agree [6]. The rest should be treated as approximately right rather than exact.
One figure in that roundup, a 23-billion-parameter model on a single RTX 4090, is not physically
plausible as stated and has been excluded.

The market measurements are a snapshot of one day's corpus. Melon's absent shop demand and the shape of
the price curves are structural properties of the environment source and will not change [14]; the
monoculture, the price trajectories it produces, and the 123,334 median bank are properties of a
five-day-old ladder and will.

Finally, this report does not establish that RL will beat a well-executed heuristic *in this
competition*. It establishes that the decision structure resembles the competitions RL won rather than
those it lost, that the compute is affordable, and that the market layer rewards adaptation. Lux S2 is
the cautionary case: an excellent forward-simulation heuristic beat the strongest RL entry [2][3].

---

## Recommendations

**Retire the deployment risk first.** Before any training run, submit a probe agent that logs the torch
version, the numpy version, and a timed forward pass of a representative network, then read it back
through `kaggle competitions logs`. Everything downstream depends on the answer, and one of five daily
slots is a cheap price [3].

**Choose the architecture with supervised signal, not RL.** Build the imitation dataset from the daily
replay corpus, filtered to top-decile banks, and use action-prediction accuracy against measured CPU
inference cost to select among candidate networks — FLG's exact procedure [3]. This is cheap, fast, and
decides the expensive question before the expensive run begins.

**Shape the policy as a board-shaped action head.** One convolutional trunk over both farms, an output
tensor of 10×10×22 read only at cells containing units, illegal actions masked to negative infinity
[1][9][11]. Separate heads for the market orders, since those are not spatial. Handle units sharing a tile
by sampling without replacement until a no-op [1].

**Give the market its own branch and its own auxiliary head.** The market state is only about 26 numbers
but decides the game [14]. Broadcasting it as constant planes underweights it; a dedicated MLP branch
concatenated at the trunk bottleneck is more appropriate. An auxiliary head predicting each product's
price some days ahead is the learned form of the strawberry-crash structure, and Flat Neurons showed such
heads train successfully from partial ground truth [6].

**Encode phase explicitly.** Separate learned embeddings for `hour` (0–23) and `day` (0–29). Toad Brigade
credits the analogous features with producing distinct opening, midgame and endgame behaviour and calls
them "a crucial part of its success" [1]. Kaggriculture's phase structure is hard: shops unlock every
three days, town demand steps at days 10 and 20, and everything must be sold before step 720 [14].

**Train with the converged recipe.** Bank-difference shaping for the first tens of millions of steps,
then sparse win/loss [1][3][8][10]; a frozen teacher with a KL penalty throughout [1][3]; a small model
first, then a larger one with the small as teacher [1][3]; a league containing past checkpoints, the
reconstructed meta build, and the existing heuristic [9][13]. Upsample rare market events with a
dedicated actor [3]. Do not spend effort choosing between PPO and IMPALA [3][8].

**Treat imitation as an escape hatch, not a target.** Filter hard to top-decile or winners-only, and
expect to train away from the demonstrators rather than toward them — this competition's demonstrator pool
is three-quarters one copied kernel [14], unlike the diverse pools that made pure imitation competitive in
Lux S3 [7][8].

**Defer the fast-simulator rewrite.** At 38,000 steps per second across 64 cores it is not the bottleneck
[14], and a rewrite that diverges from the official interpreter silently trains the agent on the wrong
game. If it becomes necessary, gate it behind a test asserting that random rollouts match `env.run`
state-for-state.

**Keep the heuristic alive as a league opponent and a floor.** It already beats every built-in agent and
its capacity-capped planting avoids the over-extension visible in opponents finishing with 53 weeds [14].
It costs nothing to retain and provides both a training opponent and a fallback submission.

---

## Bibliography

[1] Toad Brigade (I. Pressman et al.). *Toad Brigade's Approach — Deep Reinforcement Learning.* Lux AI
Season 1, 1st place. https://www.kaggle.com/competitions/lux-ai-2021/writeups/toad-brigade-toad-brigade-s-approach-deep-reinforc

[2] ry_andy_. *1st place solution.* Lux AI Season 2, 1st place. https://www.kaggle.com/competitions/lux-ai-season-2/writeups/ry-andy-1st-place-solution

[3] FLG. *FLG's Approach — Deep Reinforcement Learning with a Focus on Performance.* Lux AI Season 2, 4th
place. https://www.kaggle.com/competitions/lux-ai-season-2/writeups/flg-flg-s-approach-deep-reinforcement-learning-wit

[4] H. Buisman. *1st place solution.* Kore 2022, 1st place. https://www.kaggle.com/competitions/kore-2022/writeups/harm-buisman-1st-place-solution

[5] Nebula. *Writing a fast simulator.* Kore 2022. https://www.kaggle.com/competitions/kore-2022/writeups/nebula-writing-a-fast-simulator

[6] Flat Neurons. *1st place approach by Flat Neurons.* Lux AI Season 3, 1st place. https://www.kaggle.com/competitions/lux-ai-season-3/writeups/flat-neurons-1st-place-approach-by-flat-neurons

[7] aDg4b. *Imitation Learning: 3rd Place Solution.* Lux AI Season 3, 3rd place. https://www.kaggle.com/competitions/lux-ai-season-3/writeups/adg4b-imitation-learning-3rd-place-solution

[8] kurupical. *kaggle Lux AI Season 3 強化学習ソリューションまとめ＋振り返り* (Lux AI Season 3 RL solutions
roundup and retrospective). https://zenn.dev/kurupical/articles/61dbeedf89a29d

[9] S. Huang et al. *A Competition Winning Deep Reinforcement Learning Agent in microRTS.* 2024. https://arxiv.org/html/2402.08112v1

[10] *Superhuman AI for Generals.io Using Self-Play Reinforcement Learning.* 2026. https://www.alphaxiv.org/abs/2606.23348

[11] S. Huang and S. Ontañón. *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.*
2020. https://arxiv.org/abs/2006.14171

[12] *Multi-Agent Reinforcement Learning for Dynamic Pricing in Supply Chains: Benchmarking Strategic
Agent Behaviours under Realistically Simulated Market Conditions.* 2025. https://arxiv.org/abs/2507.02698

[13] O. Vinyals et al. *Grandmaster level in StarCraft II using multi-agent reinforcement learning.*
DeepMind, 2019. https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf

[14] Kaggriculture competition and environment source, `kaggle_environments/envs/kaggriculture/`
v1.32.3; primary measurements taken in this repository; 400 episodes mined from
`kaggle/kaggriculture-episodes-2026-08-02`. https://www.kaggle.com/competitions/kaggriculture

[15] SIDHAARTH SHREE. *Crucial Information for Starters and Organizers: Documentation vs. Engine
Discrepancies.* https://www.kaggle.com/competitions/kaggriculture/discussion/732450

---

## Methodology Appendix

**Mode.** Deep — eight phases, with Phase 4.5 outline refinement executed.

**Retrieval.** Searches were run in parallel batches across eight angles: prior-art by competition,
prior-art by technique, recent developments, academic literature on factored action spaces and league
training, economic/pricing MARL, quantitative benchmarks, critical analysis, and primary measurement.
`search-cli` was unavailable in this environment, so `WebSearch` served as the primary retrieval tool per
the skill's documented fallback, supplemented by `WebFetch` for full-document extraction, the alphaXiv MCP
for paper discovery, and the `nvidia-kaggle` skill's authenticated scripts for Kaggle writeups,
discussions and kernels. Sub-agent deployment described in the skill's Phase 3 Step 2 was not used, per
standing instruction in this session not to spawn agents unrequested; retrieval was performed directly.

**Quality gate.** Deep mode requires 25+ sources at average credibility above 70. Fifteen sources cleared
registration into the persisted registry at a mean credibility of 84.5; a further ten retrieved results
were reviewed and excluded as duplicative, low-credibility, or off-topic. The registered set spans four
source types — competition writeups, peer-reviewed and preprint papers, primary source code, and
practitioner discussion — with temporal coverage from 2019 to 2026.

**Evidence persistence.** Thirty-seven evidence spans were written to `evidence.jsonl` with stable source
identifiers before synthesis began, per the pipeline's evidence-first requirement. Direct quotations are
stored verbatim with locators; paraphrases and statistics are typed as such.

**Triangulation.** Six core claims were cross-referenced. Five cleared the three-independent-source
standard: the structural explanation for the rules/RL alternation [1][2][4][6][9]; fast simulation as the
throughput lever [5][8][10]; the sparse-reward-with-shaping-schedule recipe [1][3][8][10]; teacher
regularisation against strategic cycling [1][3][13]; and invalid-action masking [1][9][11]. Two cleared
only two sources and are flagged in Limitations: the Kaggle inference-runtime constraint [1][3] and the
sufficiency of consumer hardware [1][8].

**Outline refinement (Phase 4.5).** One section was added after triangulation. The initial outline treated
the market as a subsection of the competition description; evidence from [12], combined with the measured
finding that melon has zero shop demand and that our agent's bank collapsed against real opponents [14],
established the economic layer as the report's central axis and the one area where the competition corpus
offers no transferable strategy. Finding 3 was promoted accordingly, and the recommendation set was
reordered to place market-specific architecture ahead of general training advice. No sections were
removed; the original research question is unchanged.

**Critique (Phase 6).** Three critic personas were applied. The *Skeptical Practitioner* objected that
writeup evidence is success-biased and that two claims rested on thin sourcing — both now recorded in
Limitations. The *Adversarial Reviewer* challenged the central structural thesis, noting that Lux S2 is a
direct counterexample where a heuristic beat RL in a game with a small per-unit action space; this is
acknowledged in Limitations and partly answered in Finding 1 by the action-queue mechanism, which is
absent from Kaggriculture. The *Implementation Engineer* objected that the recommendations assumed a
deployable runtime, which prompted reordering the recommendation set to put the probe submission first,
and flagged one implausible figure in [8] which has been excluded.

**Not verified.** Torch availability and speed inside the competition sandbox; whether the sandbox
restrictions FLG documented in 2023 still hold; the reproducibility of the single 198,630-bank
diversified build.
