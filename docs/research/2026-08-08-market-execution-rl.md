# Selling into a market you move: what the execution literature gives us

Research run 2026-08-08. The two prior passes framed this as a self-play problem
(`2026-08-07-self-play-rl-findings.md`) and a reproduction problem
(`2026-08-07-fable-rl-grounding.md`). This one takes the reframing seriously:
**our agent's core economic problem is selling a position into a price-forming
market it moves itself, against an opponent doing the same.** That is optimal
execution with market impact, in a two-player game.

Primary sources only. Every number carries its artifact or an explicit
UNVERIFIED label. All measurements labelled "measured" were taken today from the
installed engine or from `/data/kaggriculture/episodes/`, read-only; no training
arm or evaluator was touched.

---

# Part 0 — What our own engine says, measured today

The literature is only useful against a correct picture of our market. Four
facts, all read out of
`kaggle_environments/envs/kaggriculture/kaggriculture.py` or measured from the
replay archives.

## 0.1 Price is a deterministic function of one shared inventory

`market_price(item, inventory)` (engine, `kaggriculture.py:177`) is closed-form:

```
amp   = target * base / shape(func, T)
price = base + amp * shape(below_func, I0 - inv)      if inv <  I0     # scarcity
      = base - amp * shape(above_func, inv - I0)      if inv >= I0     # glut
price = max(1, round(price))
```

Every seat's `SELL` increments that inventory by one **per unit**, and
`_process_market` re-quotes between units, so a multi-unit order walks its own
price down inside a single turn. There is no separate order book: **impact is
permanent in inventory and shared between the players.**

## 0.2 The market is a reservoir with a fixed refill rate, and it is small

`_town_consume` removes inventory on a fixed schedule: unlocked shops every 4
steps, the town centre every 12 steps with a day-dependent multiplier
(`TOWN_CENTER_DEMAND_SCHEDULE = [(20, 4), (10, 2), (0, 1)]`). Shops unlock one
every 3 days from day 3, so all 8 are open by day 24; which shop opens when is
seeded, so per-product totals vary by episode.

Season demand across 200 simulated unlock schedules (measured):

| item       | min | median | max | revenue at base (median) |
| ---------- | --: | -----: | --: | -----------------------: |
| WHEAT      | 500 |    644 | 770 |                   16,100 |
| CARROT     | 266 |    446 | 608 |                   15,610 |
| TOMATO     | 230 |    338 | 446 |                   20,280 |
| STRAWBERRY | 392 |    536 | 680 |                   64,320 |
| MELON      | 140 |    140 | 140 |                   35,000 |
| EGG        | 230 |    338 | 446 |                   16,900 |
| MILK       | 302 |    446 | 572 |                   71,360 |
| WOOL       | 212 |    338 | 464 |                   67,600 |
| FERTILIZER |   0 |      0 |   0 |                        0 |
| **total**  |     |        |     | **307,170** (both seats) |

Halved, that is **~153,585 per seat of base-price absorption for a whole
season**, against a measured corpus median seat of 117,779 and a max of 158,603.
The top of the ladder is running at close to the town's entire capacity to
absorb goods at base price. This is not a production-limited game at the top; it
is an absorption-limited one.

**MELON is fixed at 140 units a season in every seed** — town centre only, no
shop ever buys it. **FERTILIZER has zero town demand at all**, so its inventory
only ever rises and its price only ever falls.

## 0.3 Prices _rise_ when the market is starved — and that inverts the usual execution problem

Because the town consumes continuously, inventory falls below `I0` whenever the
players under-supply, and `market_price` pays a scarcity premium. Quoted price by
day if **neither** seat sells anything (measured):

| item       | base |  d0 |  d5 | d10 | d15 | d20 | d25 | d29 |
| ---------- | ---: | --: | --: | --: | --: | --: | --: | --: |
| WHEAT      |   25 |  26 |  30 |  35 |  40 |  45 |  49 |  52 |
| CARROT     |   35 |  36 |  38 |  39 |  40 |  41 |  42 |  42 |
| TOMATO     |   60 |  60 |  61 |  63 |  69 |  77 |  89 |  99 |
| STRAWBERRY |  120 | 128 | 148 | 182 | 224 | 257 | 292 | 317 |
| MELON      |  250 | 256 | 272 | 277 | 283 | 287 | 291 | 293 |
| EGG        |   50 |  50 |  51 |  55 |  60 |  65 |  71 |  76 |
| MILK       |  160 | 169 | 189 | 201 | 228 | 258 | 298 | 324 |
| WOOL       |  200 | 206 | 221 | 227 | 232 | 236 | 241 | 245 |

Strawberry 2.6×, milk 2.0×, wheat 2.1×. Melon 1.17× — melon has a `log` scarcity
curve (`below_target 0.2`) and a `sq` glut curve (`above_target 3.6`): negligible
upside, brutal convex downside. It is the worst asset in the game to be long.

**Every paper in the execution literature assumes a driftless price** (Almgren &
Chriss's arithmetic random walk with zero drift; every RL successor inherits it).
Ours has a strong, deterministic, _upward_ drift whenever supply is short. That
single difference flips the sign of the classic advice and is the main reason
"liquidate promptly" does not transfer. See Part 3.

## 0.4 What the winners actually do at the market

Measured over 120 episodes (240 seats) of
`kaggriculture-episodes-2026-08-07.zip`, pricing each `SELL` at the market quote
recorded on the preceding step:

- median final bank **117,779**, mean 118,074.

| item       |   units |    revenue | % of revenue | base | median realised | ratio |
| ---------- | ------: | ---------: | -----------: | ---: | --------------: | ----: |
| STRAWBERRY |  88,858 | 14,163,834 |        26.9% |  120 |             186 | 1.55× |
| MILK       |  66,459 | 10,477,613 |        19.9% |  160 |             181 | 1.13× |
| WOOL       |  48,556 |  9,189,979 |        17.5% |  200 |             214 | 1.07× |
| WHEAT      | 178,134 |  7,911,821 |        15.0% |   25 |              46 | 1.84× |
| MELON      |  45,048 |  7,825,961 |        14.9% |  250 |             186 | 0.74× |
| FERTILIZER |  54,857 |  3,054,097 |         5.8% |  100 |              55 | 0.55× |
| CARROT     |      12 |        456 |         0.0% |   35 |              38 | 1.09× |

Share of units sold at or above base: wheat 100%, strawberry 67.9%, milk 55.3%,
wool 54.4%, melon 17.4%, **fertilizer 1.2%**.

Execution style: **median `SELL` order size 4 units** (mean 6.3, p90 12, p99 42,
max 76), on a **median of 245 of 719 turns**. Terminal unsold shed value: **median
0**. So the winning population splits its sales into small slices across a third
of the season and liquidates completely — the behaviour Almgren–Chriss predicts,
arrived at without any of this literature.

Volume is heavily back-loaded: days 0–9 carry ~47k units across the sample, days
20–29 carry ~265k. They accumulate into the scarcity premium and sell late.

---

# Part 1 — What applies to us

## 1.1 Our shaping term is a mis-specified benchmark, and it is charging a corpus-shaped agent 32,768 coins a season

This is the finding with the most leverage, and it is a code-level fact rather
than an analogy.

`progress.py` builds its potential from `VALUE`, which prices every product at
`MARKET_PARAMS[item]["base"]`. The module states the reasoning explicitly and it
is worth quoting, because it is the assumption that fails:

> Deliberately a table of engine constants and not a reading of
> `market["prices"]`: the live quote moves on the town's consumption schedule and
> on the opponent's sales, so a potential built from it would hand us a reward
> for the opponent flooding the market … It would also stop being a function of
> _our_ state, which is the property the whole construction rests on.

The second sentence is not correct. Ng, Harada & Russell's Theorem 1 (ICML 1999)
requires `Φ : S → ℝ` over the **environment state**, not over one agent's
sub-state; market inventory is in `observation["market"]` and is part of `S`.
Devlin & Kudenko (AAMAS 2012, pp. 433–440) extend the invariance result to the
multi-agent case — the shaped game has _consistent Nash equilibria_ — and Grześ
(AAMAS 2017, §4.2) shows the finite-horizon caveat is exactly the terminal-zero
condition `progress.py` already enforces. **There is no theorem forbidding a
live-price potential.** The objection is a variance argument, and variance is
measurable.

What the base-price choice actually does. With `rewards` = per-turn bank delta
and shaping `γΦ(s') − Φ(s)`, a single `SELL` of one unit at realised price `p`
carries immediate reward `≈ p − base`, and a `BUY_PRODUCT` at `p` carries
`≈ base − p`. Applying that to the corpus's own transactions gives the shaped
reward our current potential would pay a median seat, per season:

| item       | from SELL | from BUY_PRODUCT |
| ---------- | --------: | ---------------: |
| STRAWBERRY |   +14,587 |                — |
| WHEAT      |   +14,410 |      **−34,496** |
| MILK       |      −649 |                — |
| WOOL       |    −2,172 |                — |
| FERTILIZER |   −10,132 |                — |
| MELON      |   −14,317 |                — |
| **net**    |           |      **−32,768** |

against a bank of ~118,000. Three consequences, each of which matches something
we have already observed and failed to explain:

1. **The animal chain is priced as a 34,496-coin mistake.** Wheat trades at 1.84×
   base all season because 14 animals a seat eat it and nobody plants enough
   (`STRATEGY.md`). Our potential credits bought wheat at 25 while the agent pays
   ~46, so every unit of feed is a −21 shaped hit. The prior research pass
   recorded that "**`livestock` is 0 in every one of the 31 logged iterations**"
   and could not explain it, since animals are carried at cost and therefore
   supposedly neutral. They are neutral; **their feed is not.** This is a
   mechanical explanation for the observation, and it is testable.
2. **Selling melon is punished at −14,317 a season**, selling wheat rewarded at
   +14,410. The shaping is silently imposing a per-product opinion worth ±12% of
   a whole bank, which nobody chose and nobody tuned.
3. **There is a live reward pump in the shipped potential.** `BUY_PRODUCT` is
   legal for `WHEAT` and `FERTILIZER` (engine `_process_market`; our
   `encoding.py:904` `_BUY_PRODUCT_ITEMS = ("FERTILIZER", "WHEAT")`). Fertilizer
   has **zero town demand**, so the field dumps it to a median $55 against a base
   of 100 — and our potential credits each bought unit at 100. That is **+45 of
   shaped reward per fertilizer bought**, repeatable until buying pushes the price
   back to base (≈225 units, ≈5,062 shaped coins per cycle). Potential-based
   shaping guarantees this nets to zero _over a whole episode_; it does not stop a
   GAE-λ-0.8 advantage from seeing +45 now and the offset 700 turns later.

Wiewiora (JAIR 19:205–208, 2003) is the reason this matters even though the
optimum is provably unchanged: **potential-based shaping is exactly equivalent to
initialising Q with Φ.** A potential that is wrong by −45% (fertilizer) to +84%
(wheat) is a Q-initialisation that is wrong by the same amount, in a direction
that specifically discourages the highest-revenue behaviour in the corpus. His
own conclusion: "care must be taken in choosing a potential function that does
not lead to poor learning performance."

## 1.2 The fix the algebra picks out: mark the potential to liquidation value

Replace the constant `VALUE[item]` with what the shed would actually fetch if it
were dumped into the current market:

```
Φ_stored = Σ_item Σ_{j=0}^{n_item − 1}  market_price(item, I_item + j)
```

`I_item` is the live `market["inventory"]`, straight from the observation;
`market_price` is already imported in `constants.py`. Carried inventory the same;
`growing` and `livestock` at `GROWING ×` the _marginal_ live price rather than
base; seeds, land and animal purchase prices stay at cost (they are not market
goods and are already neutral).

Two exact identities, both checkable by hand:

- **Selling `q` units.** Cash received is `Σ_{j<q} price(I+j)`. The potential
  loses exactly the same sum, because the units that remain are re-marked at the
  new, lower prices and the telescoping cancels: `ΔΦ = −Σ_{j<q} price(I+j)`. Net
  shaped reward **exactly 0**.
- **Buying one unit** at the engine's post-buy quote `price(I−1)`: cash `−price(I−1)`,
  `ΔΦ = +price(I−1)`. Net shaped reward **exactly 0** — which mirrors the engine's
  own comment on that line, "_Quote at post-buy inventory so a buy/sell
  round-trip against an unchanged market nets zero._"

So **every market transaction becomes reward-neutral at the instant it happens,
by construction.** The fertilizer pump closes exactly. The 34,496-coin feed
penalty disappears exactly. No clamp, no coefficient, nothing to tune.

What the shaping then pays for is only three things, and they are the right
three: **production** (a harvest adds units and Φ rises by their marginal
liquidation value), **price drift** (holding while the town starves the market
pays `n × Δp` per turn; holding while the opponent floods it costs the same), and
**the Grześ terminal zero** (everything unsold at turn 719 is forfeited, which is
also the game's own scoring rule).

The literature converges on this object from four directions:

- **Ning, Lin & Jaimungal** (arXiv:1812.06600, eq. 3.1) use
  `Ř = q_t (p_{t+1} − p_t) − a (x/M)²` — a per-step mark-to-market on held
  inventory — and prove by summation by parts (eq. 3.4) that it telescopes into
  implementation shortfall. Our `γΦ' − Φ` with a live-price Φ _is_ that first
  term, with the impact penalty supplied by the engine rather than by a
  hand-set `a`.
- **Schnaubelt** (EJOR 296(3):993–1006, 2022, eq. 3–4) designs his per-step
  reward so that "the sum of per-step rewards is exactly the relative total
  implementation shortfall", normalised by arrival mid so it survives price-level
  drift between train and test.
- **Koulouris & Campajola** (arXiv:2605.20348, §4) give the amortised form
  `r_t = −S₀q₀/N + S̃_t v_t − a v_t²`, "equivalent, up to sign and normalisation
  convention, to minimising implementation shortfall" — a dense signal with the
  same optimum as the terminal objective.
- **Grześ** (AAMAS 2017) supplies the terminal condition and, independently,
  **Hafsi & Vittori** (arXiv:2411.06389, eq. 4) and **Karpe et al.**
  (arXiv:2006.05574, ICAIF'20, §4.2) supply the same object as a hand-set terminal
  inventory penalty `β·I_T`. Ours is derived rather than tuned.

Notably, the net per-sale reward under a live-price potential is `q(p − mark)` —
**realised price relative to a benchmark**, which is precisely the "sell well, not
just sell" term the brief asked about. It arrives here as a consequence of the
potential rather than as an extra term, which is what makes it unfarmable.

## 1.3 Our degenerate mirror equilibrium is the predicted attractor, and our fix is the published fix — with a published expiry date

Two primary sources, and one is structurally almost our game.

**Lillo & Macrì**, "Deviations from the Nash equilibrium in a two-player optimal
execution game with reinforcement learning" (arXiv:2408.11773, v1 21 Aug 2024, v2
13 Feb 2026). Two DDQN agents liquidate the _same_ asset into a _shared_ price:
`S_t = S_{t−1} − g(V_t/τ)τ + σ√τ ξ` with `V_t = v_t^(1) + v_t^(2)`, while each
agent's fill price is hit only by its own volume. Neither observes the other's
inventory or actions — the coupling is entirely through the price. Verbatim from
the abstract: "the strategies learned by the agents deviate significantly from the
Nash equilibrium … the learned strategies exhibit supra-competitive solution,
{which might be compatible with a tacit collusive behaviour}, closely aligning
with the Pareto-optimal solution." They under-trade relative to Nash. **This is
our mirror self-play result — both copies under-sell, prices stay high — obtained
independently in the cleanest possible version of our game.**

**Calvano, Calzolari, Denicolò & Pastorello**, "Artificial Intelligence,
Algorithmic Pricing, and Collusion", _AER_ 110(10):3267–97, 2020 (verified: title,
authors, volume, issue, pages, and the abstract's "punishment phases and gradual
returns"). Two tabular Q-learners in a repeated Bertrand duopoly reach a profit
gain `Δ = (π̄ − π^N)/(π^M − π^N)` of **70–90% across the whole (α, β) grid,
≈85% at baseline**, with no communication. Deviations are punished in >95% of
cases, the punishment lasts 5–7 periods and returns gradually, and grim trigger is
never observed. Δ falls to 64% at n=3 and 56% at n=4 — **two players is the worst
case**.

Their §6.2 is our fix, published: re-matching converged agents into new pairs with
exploration off drops "the average profit gain … from 85% to about 20%, confirming
that coordination is almost completely pair-specific." That is why putting a
scripted opponent into our mix moved the real-opponent number 35 → 158 in eight
updates.

**The caveat is the part we do not yet have.** In the same experiment the
re-matched pair re-adapts to a collusive level in **less than one tenth** of the
original learning time, and re-enabling exploration restores the original
collusion. Against a _fixed_ script, expect the degenerate equilibrium to creep
back. Ganesh et al. (arXiv:1911.05892, JPMorgan) explain the mechanism from the
other side: their Theorem 1 makes the optimal quote a best response to the
competitor's price CDF `F^comp`; a scripted opponent pins `F^comp` down and makes
the best response unique, whereas under mirror self-play **any (high, high) pair
is self-consistent**. Their own market maker is never trained in mirror self-play
for exactly this reason.

Our market is a **quantity** game, not a price game, so the Cournot version is the
closer economic model. **Martin, Normann, Püplichhuisen & Werner**
(arXiv:2501.07178) port Calvano's design to a repeated Cournot duopoly (inverse
demand `p(Q) = max{a − bQ, 0}`, `a=91, b=1, c=19`, quantity grid `{0,3,…,45}`,
memory k=1): total quantity converges to **≈40 against a Cournot-Nash of 48 —
about 17% below Nash**, two-thirds of the way to joint monopoly, with profits
≈1275 vs Nash 1152 vs monopoly 1296. (That is Δ ≈ 0.854 — _our arithmetic from
their reported profits, not a number they state_.) Their memoryless control sits
essentially on Cournot-Nash, and in **>60%** of runs neither agent has a
profitable one-shot deviation, so it is a genuine equilibrium rather than a
learning failure.

## 1.3b But it may not be collusion at all — three explanations, three different fixes

This is the most important open question in this document, and it is cheap to
settle. Three primary sources make three incompatible predictions about our
degenerate equilibrium.

**(a) Tacit collusion.** Requires memory _and_ patience. Calvano's own control:
with `k = 0` punishment is unrepresentable and Δ drops below 5%; at `δ = 0.35`,
Δ = 16%. Martin et al. agree in Cournot. Koulouris & Campajola localise it
further: **ex-ante schedule learners converge stably to Nash; DDQN agents that see
only the current price stay near Nash; only agents given recent prices _and their
own past actions_ go supra-competitive.**

**(b) An absorbing rest point from asynchronous value updating — no memory and no
discounting required.** **Asker, Fershtman & Pakes**, "Artificial Intelligence,
Algorithm Design and Pricing", _AEA Papers and Proceedings_ 112:452–56, 2022 (read
in full from the authors' PDF). Bertrand duopoly, `c = 2`, 100-price grid on
[0.1, 10], static Nash at 2.0282 and 2.1291, `α = 0.1`, initial values drawn
i.i.d. `U[10,20]`. Critically, verbatim: _"The AIAs in this paper do not value the
future (that is, they only value profit in the current period). **This rules out
'collusive'-style equilibria** in which play is shaped by perceptions of the
future returns flowing from a present action."_ Their results, 100 simulations
each:

| updating rule                                                   |    min |   25th |     median |   75th |    max |        converged by |
| --------------------------------------------------------------- | -----: | -----: | ---------: | -----: | -----: | ------------------: |
| **asynchronous** (only the played action's value revised)       |   5.06 |   7.33 |   **8.34** |   9.14 |     10 |      ~4,600 periods |
| **perfect synchronous** (exact counterfactual for every action) | 2.1291 | 2.1291 | **2.1291** | 2.1291 | 2.1291 | **~105 iterations** |
| **synchronous using only "demand slopes downward"**             | 2.0282 |  2.129 | **2.1291** |   2.23 |   3.84 |        <500 periods |

An asynchronous median of 8.34 is more than four times marginal cost, with
collusion structurally impossible. The mechanism is that values attached to
_unplayed_ actions are never revised, so optimistic initial beliefs about high
prices are never corrected, and the system is absorbed. Their own explanation of
the fix: _"imposing downwardly sloping demand insures that the W's attached to a
higher price than the price actually played can only be updated downward."_

**(c) Neither — policy gradient with two players should converge to Nash.**
**Shi & Zhang**, "Multi-Agent Reinforcement Learning in Cournot Games"
(arXiv:2009.06224, 14 Sep 2020; title, authors, date and abstract verified).
Verbatim from the abstract: _"We prove the convergence of policy gradient dynamics
to the Nash equilibrium when the price function is linear **or the number of
agents is two**."_ Continuous actions, agents observe only their own realised
payoff. **We run PPO — a policy gradient method — with two players.** Their
conditions do not strictly hold for us (our sell head is a bucketed discrete
quantity, our game is a 719-step sequential MDP rather than a repeated static
game), but the tension is real and should be stated rather than glossed: the
theory closest to our algorithm and player count predicts Nash-like racing, and
supra-competitive restraint would be the anomaly.

**How to tell them apart.** (a) predicts under-selling that _disappears_ if we
strip opponent price history from the state, and that returns as history
accumulates within an episode. (b) predicts under-selling that is **insensitive to
memory and to γ**, that depends on value-head initialisation, and that is killed
by counterfactual updating (item 2 in Part 4). (c) predicts we should not see it
at all, which would mean our low banking is an exploration or credit-assignment
failure rather than an equilibrium.

Run the memory ablation first: it is a state-encoder change, no new machinery. If
under-selling survives with no price history in the state, it is not collusion and
the opponent-pool work in item 3 is not where the gains are.

## 1.4 Clamping was the documented wrong cap

Our reward-farming pump — the agent bought and re-sold at a loss because a clamped
per-sale reward paid for the sale and forgave the purchase — has two exact
diagnoses in primary sources:

- **Amodei et al.** (arXiv:1606.06565, §4), on reward capping: "while capping can
  prevent extreme low-probability, high-payoff strategies, it can't prevent
  strategies like the cleaning robot closing its eyes to avoid seeing dirt. Also,
  the correct capping strategy could be subtle as **we might need to cap total
  reward rather than reward per timestep**." We capped per timestep, so the agent
  scaled the number of timesteps.
- **van Hasselt et al.** (PopArt, arXiv:1602.07714): DQN-style clipping "results in
  **optimizing the frequency of rewards, rather than their sum**". We optimised the
  frequency of sales, and the agent manufactured sales by buying.
- **Ng et al.'s necessity direction** (Lemma 3, appendix) is the structural
  statement: if `F(s, a, s') ≠ F(s, a', s')` for some pair of actions, there exist
  `(T, R)` for which _no_ optimal policy in the shaped MDP is optimal in the real
  one. A clamped per-sale term makes `F` depend on whether the action was BUY or
  SELL. A price-forming market with a buyable good supplies the counterexample.
  **The pump was guaranteed, not unlucky.**

Both prior arms — unclamped and clamped, 1× and 10× — are also consistent with the
reward simply not being the binding constraint. Three primary results say to rule
other things out first. Johanson et al. (arXiv:2205.06760): in an identical
bartering environment with an identical reward, **V-MPO learned to trade and A2C
did not** — 8 of 16 A2C agents scored at or below −970, which is exactly the
return of an agent that never moves and never eats. Zheng et al. (arXiv:2004.13332
and the `ai-economist` repo) needed a **labour-cost curriculum**, not a trade
reward: without it "a suboptimal policy can experience too much labor cost and
converge to doing nothing." Fang et al. (arXiv:2103.10860, AAAI 2021) found their
benchmark-relative reward insufficient end-to-end and had to add oracle-policy
distillation — that was the paper's entire motivation.

## 1.5 State: a price _level_ is not enough; the literature says you need history

**Koulouris & Campajola** (arXiv:2605.20348, verified title/authors/date/abstract)
run the two-agent Almgren–Chriss game specifically to isolate what causes
supra-competitive behaviour. Their results, in their own framing: multi-agent
learning alone does not produce it; agents that see only the _current_ price stay
near Nash; **"when agents are given access to intra-episode history, especially
recent prices and own past actions, supra-competitive outcomes become
substantially more frequent and more persistent."** They also report an
integrated-gradients diagnostic that the learned Q-function assigns little
effective importance to the price coordinate after early training — it is driven
by inventory and time.

We currently feed the market as ~26 numbers, all _levels_. Three additions are
cheap and each is named in a primary source:

- per-product **price change over the last k turns** and our own **cumulative units
  sold per product** (Koulouris & Campajola's "recent prices and own past actions");
- the **town's absorption rate for the current day**, which is fully deterministic
  given the day and the unlocked shop set and which the agent currently has to
  infer from inventory deltas — this is the resilience parameter of the impact
  model and there is no reason to hide it;
- Nevmyvaka et al. (ICML 2006) measured what each market feature is worth on top of
  the private `(time, inventory)` pair: bid-ask spread **7.97%**, immediate
  market-order cost **4.26%**, signed transaction volume **2.81%**, the three
  together **12.85%** additional cost reduction. Their private-variables-only
  agent already captured **27.16–35.50%** over an optimised submit-and-leave
  baseline. The lesson transfers as a ranking: **time-remaining and
  inventory-remaining do most of the work; market features are a 10%-scale
  refinement on top.**

## 1.6 The negative result, stated plainly: no Kaggle simulation competition has ever had endogenous prices

Asked for anything closer to our structure than Lux, in the competition world the
answer is a clean negative. Kaggle's simulation category is **20 competitions**
(ConnectX, Halite + Playground, Google Football, Rock Paper Scissors, Santa 2020,
Hungry Geese, LLM 20 Questions, FIDE Chess, Orbit Wars, Maze Crawler, Pokémon TCG,
Lux S1/S2/S3 and their beta stages, Kore 2022 + beta, Kaggriculture). None of the
other 19 has a price that the players' own actions move. Lux S2's turn-1 sealed
bid for factory-placement order is the only auction in Kaggle simulation history,
and it is an auction over turn order, not a price your selling moves. **We are the
first.** There is no competitor writeup to borrow a market policy from, in any
competition, ever — which is consistent with `STRATEGY.md`'s finding that the
public kernels are exhausted.

The one structural near-miss is **Santa 2020 — The Candy Cane Contest**: two
players share 100 machines, and _every_ pull decays that machine's payout
probability by 3% regardless of who pulled it or whether it paid. That is
two-player shared-resource depletion with a terminal objective — the same shape as
price depression. **The disanalogy is decisive**: Santa's decay is irreversible
(the environment source notes "_When a threshold is not selected it is reduced by
(decay_rate) ^ 0 (i.e. no recovery)_"), so candy left on the table is simply candy
the opponent takes and restraint is never rational. **Our price recovers** — the
town drains inventory every 4 and 12 steps and the scarcity branch pays a premium.
That single difference is what makes restraint potentially rational here and
provably irrational there, and it is why the Santa writeups (unanimously "estimate
the true value, act greedily") do not transfer. Worth noting anyway: 2nd place
(kibuna) reports a ladder of rule-based UCB 951.9 → **RL 997.9** → LightGBM greedy
1442.1 → LightGBM quantile 1521.1. **RL lost badly to a GBDT in the closest Kaggle
analogue we have.**

Two more warnings from adjacent literatures, both about what our training curve
will look like:

- **Perolat et al.** (arXiv:1707.06600), 12 independent DQN agents in a commons
  game with density-dependent respawn, pass through a long **"tragedy" phase** in
  which learning _decreases_ returns — agents get good at harvesting, deplete the
  stock by roughly half the episode, and collect **less than half** what random
  play collected. They escape it not by learning restraint but by learning to
  **tag each other**, reducing the effective population. We have no tagging beam
  and no territory. Expect the tragedy phase; do not expect their exit from it.
- **Klein** (_RAND J. Econ._ 52(3):538–558, 2021) finds that a _finer_ action grid
  does not restore competition — it destroys symmetry. At k=24 the outcome is a
  deterministic asymmetric Edgeworth cycle in which **the same firm always pays
  the reset cost** while the other free-rides, because pure-strategy learners
  cannot mix. **If we see persistent bank asymmetry between two identical
  self-play seats, that is a documented signature, not a bug in our harness.**
  (Flag: the "k=12 or 100" figure that circulates for Klein comes from Calvano's
  description of Klein's 2018 working paper; the published RAND version uses k=6
  and k=24. Klein's often-quoted "≈50% profit gain" is likewise Calvano's
  characterisation — UNVERIFIED against Klein's own text.)

---

# Part 2 — What does not apply, and why

**Exogenous inventory.** Every execution paper is handed a position `X` (or `q₀`,
or `v₀`) and must reach zero by `T`. Almgren–Chriss, Nevmyvaka, Ning, Lin &
Beling, Schnaubelt, Karpe, Macrì & Lillo, Koulouris & Campajola — all of them. Our
inventory is _produced_, and the production decision is the same policy's. Every
closed-form trajectory in that literature (including AC's `sinh` solution, eq. 17)
is therefore inapplicable as a target; only the reward _shapes_ transfer.

**The completion constraint, and why we cannot borrow the standard anti-inaction
device.** The literature's answer to "a relative-price reward incentivises never
trading" is never a reward term — it is an environment constraint. Fang et al.
(arXiv:2103.10860) force `a_{T−1} = max(1 − Σa_i, π(s_{T−1}))` **in the action
mapping**: "the agent will have to propose … to fulfill the order target." Ning et
al. (§4.5) append an extra timestep in which everything is liquidated. Hafsi &
Vittori add `β·I_T` and deliberately arrange the sign of the per-step reward to be
negative-dominated so that "a 0 reward obtained consistently once the execution is
finished is preferable to a negative reward." **We have no order to fill and no
target quantity**, so none of these are available. Our substitute is the Grześ
terminal zero, which is the same object arrived at from theory.

**Zero market impact.** Nevmyvaka trains against replayed order books where "own
actions do not affect market variables". Ning et al. state it: "we assume the
trader's actions do not directly affect the price process during training".
Spooner et al.: "simulated orders placed by an agent cannot impact the market".
Tesauro & Bredin: "it assumes the agent has negligible market impact… a reasonable
approximation for sufficiently large markets." **With two players in a market whose
melon demand is 140 units a season, each player is the market.** This kills the
whole GD/MGD/GDX/ZIP bidding lineage as a source of ready-made strategy, and it is
why Lillo & Macrì (who model impact explicitly, and shared) is the only close
analogue.

**The AI Economist's gains from trade.** Their agents earn coin from building
houses, **paid from outside the simulated economy**, and the double auction is a
complementary market between low-skill gatherers and a high-skill builder. Value
flows into the system, so trade is Pareto-improving and emerged with no shaping at
all. Our two seats are pure competitors for one price with no gains from trade
between them. Their "trade emerged for free" result does **not** transfer;
Calvano's does. What does transfer is their curriculum philosophy — they annealed
the cost of _acting_, not the reward of _trading_.

**Anything needing a cluster.** Separating honestly:

| source                                              | scale, as stated                                                                                                                                                                                            | ours?                                                  |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Calvano et al. (AER 2020)                           | "a few seconds of CPU time in any session"; 3,375-entry Q-table; 10⁵–10⁶ _periods_                                                                                                                          | yes, trivially                                         |
| Schnaubelt (EJOR 2022)                              | **one workstation: Threadripper 1950X + 1× GTX 1080**, TensorFlow; could not afford to tune DDQN/PPO                                                                                                        | yes — the only paper here that names its hardware      |
| Ning et al.                                         | 6 layers × 20 nodes, replay 10k; hardware UNVERIFIED                                                                                                                                                        | yes                                                    |
| Macrì & Lillo; Lillo & Macrì; Koulouris & Campajola | 5 layers × 30–128 nodes, 5k–10k episodes, 10–20 seeds                                                                                                                                                       | yes                                                    |
| Spooner et al. (AAMAS 2018)                         | linear function approx over tile codings, 1,000 episodes; hardware UNVERIFIED                                                                                                                               | yes                                                    |
| Ganesh et al.                                       | PPO/RLlib, 2×256 MLP, 5 seeds; hardware UNVERIFIED                                                                                                                                                          | yes                                                    |
| AI Economist                                        | arXiv Table 2: **15 CPUs, 2 GPUs**, 60 replicas, 50M + 400M steps. Repo tutorial: one 16-CPU GCP box, `gpus: 0`. _Science Advances_: 50M + **1 billion** steps. The three disagree; all three are reported. | borderline — the box yes, the step count is 450M–1.05B |
| Johanson et al. (DeepMind)                          | **800 parallel env processes**, 16 agents × 8×10⁸ steps each                                                                                                                                                | **no — cluster**                                       |

**Parameter count is a red herring.** Not one of these papers is near our ~10M:
the AI Economist agent is 2 conv + 2 FC(128) + LSTM(128); Johanson's is Conv(24) +
MLP(256) + LSTM(128); Ganesh's is 2×256; Calvano's is a table. What separated the
papers that produced sane economies from the ones that did not was **environment
steps and opponent diversity**, never width. If we are compute-bound, that is where
to spend.

---

# Part 3 — Where the literature contradicts our measurements, and which assumption of theirs fails

**Claim: hold inventory as little as possible.** Almgren–Chriss's objective is
`E(x) + λV(x)` with `V(x) = σ² Σ τ x_k²` — a running penalty on _held_ inventory,
which is the only thing standing between `λ = 0` (linear/TWAP) and hoarding. Karpe
et al. state the rule directly: "if there is no penalty for any running inventory,
but a significant penalty for the ending inventory, the TWAP strategy is also
optimal." Every RL successor carries some version of it.

**Our measurement says the opposite.** Holding is not a risk here, it is a return.
Prices rise monotonically with scarcity (Part 0.3: strawberry 120 → 317, milk 160
→ 324), the corpus back-loads its volume heavily into days 20–29, and its median
realised price is _above_ base on four of six traded products. **The assumption
that fails is zero drift.** AC's `V(x)` is a variance penalty derived from a
driftless arithmetic random walk; with a deterministic positive drift the same
term points the wrong way. Do **not** add a running inventory carrying cost. The
`(1 − γ)Φ` per-turn cost that potential-based shaping already supplies (≈0.001 ×
Φ) is the right order of magnitude, and the two real penalties on hoarding are
already in the engine: the **100-unit shed cap with overflow discarded** at
`_end_of_day`, and the terminal confiscation.

**Claim: mirror self-play under-selling is a pathology to be eliminated.** Half of
it is correct play. The self-play equilibrium discovered the scarcity premium; what
it failed to discover is the liquidation. Corpus terminal unsold value is a median
of **0**. That is the number to hold the checkpoint to, not "sells more often".

**Claim (ours): melon dominates.** Already retired in `STRATEGY.md`, and Part 0
adds the mechanism: melon's town demand is **exactly 140 units a season in every
seed**, it is bought by no shop, its scarcity curve is `log` and its glut curve is
`sq`. It has the least upside and the most downside of any product in the game.

---

# Part 4 — Ranked changes

Each has its primary citation, its expected failure mode, and the measurement that
confirms or refutes it.

### 1. Mark the potential to liquidation value instead of base price

`Φ_stored = Σ_item Σ_{j<n} market_price(item, I_item + j)`, live `I` from the
observation; `carried` the same; `growing`/`livestock` at `GROWING ×` marginal live
price; capital at cost; terminal zero unchanged.

_Citations._ Ng, Harada & Russell (ICML 1999) Theorem 1 for the form; Wiewiora
(JAIR 19:205–208, 2003) for why a wrong Φ is a wrong Q-init even though the optimum
is preserved; Ning et al. (arXiv:1812.06600, eq. 3.1/3.4) and Schnaubelt (EJOR
296(3), eq. 3–4) for the mark-to-market-that-telescopes-to-shortfall construction;
Grześ (AAMAS 2017) for the terminal zero, already in place.

_Expected failure mode._ Variance. Φ now moves `n × Δp` per turn on events we did
not cause. Spooner et al. (arXiv:1804.04216) found the inventory term "the leading
driver of agent behaviour" and "also the main source of instability", with episodic
reward std diverging at `η = 0`. With a 100-unit shed and melon moving tens of
coins in a flood, expect ±1,000–3,000 of per-turn reward noise against a terminal
signal of ~118,000. Secondary risk: it teaches holding, which is correct here but
is the same direction as the degenerate equilibrium.

_Measurement._ Primary: mean realised sell price ÷ base, per product, against the
corpus table in Part 0.4 (wheat 1.84×, strawberry 1.55×, milk 1.13×, wool 1.07×,
melon 0.74×). Confirmed if our ratios move toward those. Secondary: `BUY_PRODUCT
FERTILIZER` count per episode — under the current potential this is a live exploit
and should be _rising_; under the new one it should collapse to whatever fertiliser
is genuinely worth. Tertiary: wheat bought per episode and `livestock > 0` — the
prior pass measured livestock at 0 in all 31 logged iterations, and §1.1 predicts
the feed penalty is why. Also log advantage std before and after; if it blows up,
fall back to Spooner's asymmetric dampening (`r = Ψ − max[0, η·Inv·Δp]`) and sweep
η — his protocol: divergence at η=0, exposure falls at η≈0.05, policy character
changes at η≈0.1.

_Falsifier._ If `BUY_PRODUCT` and `BUY_ANIMAL` counts stay at ~0 after the change,
the barrier is exploration, not reward, and no further reward reweighting will
help. That is the same falsification test the prior pass already specified.

### 2. Use the price function we already know: counterfactual (synchronous) targets for the market head

We are in a position essentially nobody in this literature is in. `market_price` is
**closed-form and exported by our own `constants.py`**, and `market["inventory"]`
is in the observation. So the immediate revenue of selling _every_ quantity, of
_every_ product, at the observed inventory, is computable exactly, for free, with
no extra simulation:

```
revenue(item, q) = Σ_{j<q} market_price(item, I_item + j)
```

Our PPO update currently revises only the quantity it sampled. That is precisely
Asker, Fershtman & Pakes's **asynchronous** case. Feed the closed-form
counterfactual into the market head instead — as an all-actions target for the
value/advantage on the sell dimension, or at minimum as an auxiliary regression
loss on the market head's logits.

_Citation._ Asker, Fershtman & Pakes, _AEA P&P_ 112:452–56, 2022 (verified from
the authors' PDF): asynchronous updating converges to a median price of **8.34**
against a Nash of 2.13 after ~4,600 periods; perfect synchronous updating reaches
**2.1291 in every one of 100 simulations in ~105 iterations**, with initial
conditions irrelevant. Their weaker variant — using _only_ the knowledge that
demand slopes downward, which costs nothing but observing the quantity sold —
recovers almost the whole effect (median 2.1291, max 3.84). **This changed the
equilibrium reached, not merely the speed.**

_Expected failure mode._ Our sell decision is not a one-shot static payoff: the
counterfactual revenue is exact for _this turn_ but ignores the continuation value
of not having sold, which is exactly the thing that matters here given the scarcity
premium. So this must be an **auxiliary/critic-side** target, never a replacement
for the policy gradient — otherwise it trains a myopic dumper. It also assumes the
opponent's simultaneous order does not land between our units, which
`_process_market` shows it can (both players' orders interleave unit-by-unit); the
counterfactual is therefore an approximation whose error grows with the opponent's
order size on the same product in the same turn.

_Measurement._ Advantage variance on the market head before and after (this is
primarily a variance-reduction change, and it should show there first). Then the
equilibrium test: units sold per episode and realised price ÷ base in **mirror
self-play**, which is where the asynchronous artifact would live. Asker et al.
predict the supra-competitive rest point should collapse. If mirror self-play
under-selling survives counterfactual targets _and_ survives the memory ablation
in item 3, then explanations (a) and (b) in §1.3b are both refuted and the problem
is exploration or credit assignment, not equilibrium.

### 3. Check whether our policy can even represent a punishment, before assuming collusion

_Citation._ Calvano et al. (AER 110(10):3267–97) — with memory `k = 0` the profit
gain collapses to the discretisation floor, so collusion requires the opponent's
recent behaviour to be in the state. Koulouris & Campajola (arXiv:2605.20348) —
ex-ante schedule learners and current-price-only DDQN agents both stay at Nash;
only intra-episode history produces supra-competitive play.

_Expected failure mode._ None; this is a read of our own encoder plus one ablation
run. The risk is learning that it _is_ representable and having to keep the
opponent pool diverse permanently.

_Measurement._ Does the recent price _series_ (not just the level) reach the trunk,
and does the recurrent state carry it across turns? Then ablate it and re-run
mirror self-play. If under-selling disappears without price history, it is
collusion and item 4 is load-bearing. If it survives, it is not, and we are chasing
the wrong bug.

### 4. Keep the opponent pool changing, and instrument for collusion creeping back

_Citations._ Calvano et al. §6.2 — re-matching drops Δ from 85% to ~20%, "coordination
is almost completely pair-specific", but the pair re-adapts in **under one tenth**
of the original learning time and re-enabling exploration restores the original
level. Ganesh et al. (arXiv:1911.05892) Theorem 1 for why a pinned `F^comp` makes
the best response unique.

_Expected failure mode._ A fixed script gets co-opted; also, beating an open-loop
tape can be achieved by exploiting its blindness rather than by playing well —
`STRATEGY.md` already flags this.

_Measurement._ Track (realised sell price ÷ base) and total units sold per episode
_against the fixed script_ across training. **Rising price ratio with falling
volume is the collusion signature returning.** Rotate the pool when it appears.

### 5. Add intra-episode history and the known absorption rate to the state

_Citations._ Koulouris & Campajola (arXiv:2605.20348) — behaviour changes only with
"recent prices and own past actions", and their integrated-gradients result that a
price _level_ gets little effective attribution. Nevmyvaka et al. (ICML 2006,
Table 1) for the expected magnitude: market features are worth ~2.8–12.9% on top of
`(time, inventory)`, which themselves are worth 27–36%.

_Expected failure mode._ Cheap and low-yield. Nevmyvaka's own numbers say this is a
refinement, not a fix; do it after item 1, not instead of it.

_Measurement._ Feature-ablation on the sell logits: zero the price-history channel
at inference and measure the change in realised price ÷ base. If it does not move,
the policy is not using it and the extra channels are dead weight.

### 6. Do **not** add a running inventory carrying cost, and do **not** add a

separate benchmark-relative reward term

_Citations._ Almgren & Chriss (_J. Risk_ 3(2):5–39, 2000) `V(x) = σ²Στx_k²`, derived
under zero drift; our Part 0.3 measurement of strong positive drift. And: no paper
in this literature adds a VWAP/arrival-relative term as _shaping on top of_ a
terminal-wealth objective — in all of them the benchmark-relative quantity **is**
the objective, and it is always paired with a completion constraint we cannot
build. Item 1 already delivers the "realised price vs benchmark" signal as
`q(p − mark)`, unfarmably.

_Expected failure mode of doing it anyway._ Reward by inaction: never sell, never a
bad price. Published instances — Pan et al. (ICLR 2022, arXiv:2201.03544): replacing
"minimise commute" with "maximise velocity" made larger policy models **stop the
autonomous vehicles entirely** so they never merge; in Riverraid, agents "exploited
a bug… that halts the plane… thereby achieving high pacifist reward". Ganesh et al.
report the same shape: "a very large penalty term results in the RL agent choosing
not to trade at all by setting a high spread."

_Measurement._ Pan et al. §4.3 is the warning to instrument against: reward hacking
occurred **even where proxy and true reward were strongly positively correlated**
early in training. Plot terminal coins (true) and the shaping sum (proxy) on the
same axes and watch for Skalse et al.'s signature (NeurIPS 2022, arXiv:2209.13085,
Fig. 2): proxy continues up while true turns over.

---

# Part 5 — Sources

Verified today by fetching the primary artifact: Lillo & Macrì arXiv:2408.11773
(title/authors/dates/abstract — note the v2 title drops "and emergence of tacit
collusion", which appeared in v1; the abstract retains the claim); Koulouris &
Campajola arXiv:2605.20348 (title/authors/date/abstract); Calvano et al. AER
110(10):3267–97 (title/authors/volume/issue/pages/abstract); Shi & Zhang
arXiv:2009.06224 (title/authors/date/abstract, including the two-player
condition); Asker, Fershtman & Pakes AEA P&P (full text read from the authors'
PDF — the δ=0 statement, the parameters, and every number in the updating table
above). Engine facts and all corpus measurements re-derived from
`kaggle_environments` and `/data/kaggriculture/episodes/` in this session.

**Execution.** Almgren & Chriss, _J. Risk_ 3(2):5–39, 2000 (cited inconsistently as
2001, 3:5–40; same paper). Nevmyvaka, Feng & Kearns, ICML 2006, pp. 673–680. Ning,
Lin & Jaimungal, arXiv:1812.06600 (_Applied Mathematical Finance_, 2021). Lin &
Beling, IJCAI-20, pp. 4548–4554. Schnaubelt, _EJOR_ 296(3):993–1006, 2022. Karpe,
Fang, Ma & Wang, arXiv:2006.05574 / ICAIF'20. Macrì & Lillo, arXiv:2402.12049.
Lillo & Macrì, arXiv:2408.11773. Koulouris & Campajola, arXiv:2605.20348. Fang et
al., arXiv:2103.10860 (AAAI 2021), code `seqml.github.io/opd/`. Hafsi & Vittori,
arXiv:2411.06389.

**Market-forming MARL.** Zheng et al., arXiv:2004.13332; _Science Advances_ 8(18),
eabk2607, 2022; `github.com/salesforce/ai-economist`. Johanson, Hughes, Timbers &
Leibo, arXiv:2205.06760. Spooner, Fearnley, Savani & Koukorinis, arXiv:1804.04216
(AAMAS 2018). Ganesh et al., arXiv:1911.05892. Tesauro & Bredin, AAMAS'02,
pp. 591–598. Perolat, Leibo, Zambaldi, Beattie, Tuyls & Graepel, arXiv:1707.06600;
Leibo et al., arXiv:1702.03037.

**Algorithmic pricing / learning in duopoly.** Calvano, Calzolari, Denicolò &
Pastorello, _AER_ 110(10):3267–97, 2020; data and code openICPSR 119462. Asker,
Fershtman & Pakes, "Artificial Intelligence, Algorithm Design and Pricing", _AEA
Papers and Proceedings_ 112:452–56, 2022 (`johnasker.com/AIAEA.pdf`). Shi & Zhang,
arXiv:2009.06224. Martin, Normann, Püplichhuisen & Werner, arXiv:2501.07178. Klein,
_RAND J. Econ._ 52(3):538–558, 2021. Deshpande & Jacobson, arXiv:2601.17263
(LLM agents, not RL). Schlechtinger et al., IJCAI-24, arXiv:2406.02650.

**Kaggle.** Santa 2020 environment source `kaggle_environments/envs/mab/mab.py`
and the competition's Environment Rules page; winner writeups in the competition
discussion forum (nagiss 218453, kibuna 221030, Coffee Candy 220634, Guller
216536).

**Shaping and reward hacking.** Ng, Harada & Russell, ICML 1999 (use the Berkeley
mirror `people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf`; the
`~pabbeel/cs287-fa09` mirror is a broken Type-3 font and extracts as garbage).
Wiewiora, _JAIR_ 19:205–208, 2003 (arXiv:1106.5267). Devlin & Kudenko, AAMAS 2012,
pp. 433–440. Grześ, AAMAS 2017, pp. 565–573. Skalse, Howe, Krasheninnikov & Krueger,
NeurIPS 2022, arXiv:2209.13085. Pan, Bhatia & Steinhardt, ICLR 2022,
arXiv:2201.03544, code `github.com/aypan17/reward-misspecification`. Amodei et al.,
arXiv:1606.06565. van Hasselt et al., PopArt, arXiv:1602.07714.

**Explicitly UNVERIFIED.** Hardware/compute for Nevmyvaka 2006, Ning et al., Lin &
Beling, Karpe et al., Spooner et al., Ganesh et al. — none of these papers names
its hardware. The AI Economist's three primary sources disagree on scale (15 CPU +
2 GPU / one 16-CPU box / 400M vs 1B phase-2 steps); all three are quoted above
rather than reconciled. Lin & Beling's IJCAI-20 paper is internally inconsistent on
the sign of IS (footnote 4 vs. Table 2); the operative semantics are higher IS =
better. The shaped reward of Lin & Beling (2019) is characterised but never
reproduced in the IJCAI paper and the ECML version is paywalled — formula
UNVERIFIED. The CoastRunners boat: the qualitative behaviour is attested in Skalse
et al. §1, but the OpenAI post is Cloudflare-403 from this environment and the
commonly-quoted score figures could not be verified — do not cite them. Tesauro &
Das (EC'01) full text not retrieved; numeric details UNVERIFIED. Waltman & Kaymak
(_JEDC_ 32(10):3275–3293, 2008) is paywalled with no OA copy; only the publisher
abstract is confirmed and the circulating Δ≈60% is second-hand via Calvano —
Martin et al. 2025 supersedes it here. Klein's "≈50% profit gain" and the "k=12 or
100" grid figure are Calvano's characterisations of Klein's 2018 working paper,
not the published RAND text (k=6 baseline, k=24). Martin et al.'s Δ = 0.854 is our
arithmetic from their reported profits, not a number they state. Asker et al.'s
JEMS 33(2):276–304 / NBER WP 28535 versions were not read; the P&P forward-refers
to them as "qualitatively similar" with positive discounting. The Kaggle
"no endogenous prices" verdict for the other 19 simulation competitions rests on a
full-text term scan of their official specification pages, not a line-by-line read
of each specification. The Santa 2020 discussion enumeration returned 20 of 134
topics; all high-vote threads and the staff index are covered, but one or two
further writeups may exist.

---

# Appendix A — Scope correction: `toad_reward`, not `progress` (added 2026-08-08)

## A.0 The correction is accepted, and here is the verification

Confirmed, and it is worse than a mislabel: **`progress.py` does not exist on the
branch the live arms run.** `src/kaggriculture/learn/` on the working branch
(HEAD `1dfc27b`, "fix: the shed holds animals, and the market does not price
them") holds `toad_reward.py`, `toad_loss.py` and `sales.py` and **no
`progress.py`**; the file I analysed lives on the main line (HEAD `8e53c1e`),
where `ppo.py`, `rollout.py` and `selfplay.py` still import it. So §1.1's
32,768-coin figure describes the closed line and **must not be used to justify
touching a live arm**.

What survives the correction unchanged, because none of it depends on which
reward module is in play: the corpus price table (§0.4), the starvation
price-rise table (§0.3), the absorption ceiling (§0.2), and the general claim
that a shed valued at anything other than its live liquidation price imposes a
per-product bias. What does **not** survive: the specific −32,768 number, the
"−34,496 for wheat feed" number, and the fertilizer-at-base pump. `toad_reward`
prices nothing, so none of those three mechanisms exist there.

## A.1 Does the pathology exist in `toad_reward`? Yes, in a purer form — and it is not material _today_

The term is `FUEL_WEIGHT * max(after.fuel − before.fuel, 0) / NORMALISER` with
`fuel = sum(observation["private"]["shed"].values())`, i.e. **1e-5 per unit of
shed count acquired**, clamped, price-blind.

Three structural facts, read from the engine:

1. **`_commit_unit` credits the shed on `BUY_PRODUCT` and `BUY_ANIMAL` with no
   capacity check** — our own `encoding.py:218` already notes this. The 100-unit
   shed cap is applied only in `_drop_inventories_to_shed` at end of day, to
   _carried_ inventory. So bought stock both pays `fuel` and bypasses the cap.
2. In the baseline arm (`money_weight = 0.0`) the coins spent are **never
   charged**, and the clamp means the disposal is **never penalised**. There is
   no offsetting cost anywhere in the reward. The coordinator's reading is
   correct.
3. Price-blindness makes it worse than a uniform bias: a fertilizer unit (corpus
   median $55) and a melon unit ($186) pay identically, so the cheapest unit in
   the game is the most reward-efficient thing to buy.

**Sizing, in the same units as `shaped_mean`** (which is `t.shaped.sum()` — a
per-_episode_ total, `toad_phase1.py:324`). Measured over 50 real corpus seats by
running their observations through `toad_reward.counts`:

| component   |    mean / episode |   median |
| ----------- | ----------------: | -------: |
| city        |          +0.10368 | +0.10000 |
| **fuel**    |      **+0.01197** | +0.01213 |
| step        |          +0.00719 | +0.00719 |
| unit        |          +0.00590 | +0.00600 |
| research    |          +0.00160 | +0.00160 |
| game_result | 0 (mirror sample) |        0 |
| **total**   |      **+0.13034** | +0.12692 |

So `fuel` is **9.2%** of the shaped signal, and `city` dominates at 80%. Of the
fuel credit a corpus seat earns, **21% falls on turns where it also made a
shed-crediting purchase** (per-turn attribution, respecting the clamp) — buying
already contaminates a fifth of the term even for an agent not optimising it.

**Units, stated explicitly.** `gross_purchases` is logged in **coins**
(`(-t.own.clamp(max=0.0)).sum()`), and it counts _all_ spending — seeds, hires,
land — of which only `BUY_PRODUCT`/`BUY_ANIMAL` credit `fuel`. A corpus seat
spends **42,987 coins on 923 shed-crediting units (46.6 coins/unit)** plus 7,030
coins on seeds, which credit `private["seeds"]` and not the shed. Converting the
live arms' 3,400–8,600 coins at that rate gives an **upper bound of 73–185
shed-crediting units, i.e. 0.0007–0.0019 of shaped reward — 0.6–1.5% of the
0.130 total.** The true figure is lower, because arms banking ~0 are spending
mostly on seeds and hires.

**But the ceiling is not small.** `MARKET_SLOTS` gives two `BUY_PRODUCT` slots
(`_BUY_PRODUCT_ITEMS = ("FERTILIZER", "WHEAT")`) and `QUANTITIES` tops out at 64,
so 128 shed-crediting units per turn are expressible. The clamp forces the buy
and the sell onto different turns, halving it to ~64/turn sustained:

```
64 units/turn × 719 turns × 0.005 / 500 = 0.460 per episode
```

against a legitimate total of 0.130 — **3.5× the entire rest of the shaped
signal.** The coin cost is negligible: deep in a wheat glut the marginal spread
between adjacent units is `0.834/(1+Δ)`, about 0.002 coins at Δ=400, and in the
baseline arm coins are not charged at all.

**Verdict.** Not material at the currently logged volumes — **do not restart four
arms for a 1% term.** But it is not a 1% term by design, it is a 1% term by the
policy not having found it yet, and **the reward's argmax is the pump**: the
gradient points uphill from 0.002 to 0.460 the whole way, with no barrier. This
is the same shape as arm W, which climbed `money_term` 0.0064 → 0.028 in 34
updates. The proportionate response is **instrumentation, not a restart**: log
`BUY_PRODUCT` units per episode (not coins) as a first-class metric alongside
`gross_purchases`, and set a kill threshold. At 1,300 units/episode the fuel term
doubles; that is the number to alarm on.

Citation for the general form, already in §1.4: Amodei et al. (arXiv:1606.06565,
§4), "we might need to cap total reward rather than reward per timestep"; and
van Hasselt et al. (arXiv:1602.07714), clipping "results in optimizing the
frequency of rewards, rather than their sum". A clamped per-turn delta on an
acquirable quantity optimises the _number of acquisitions_.

## A.2 Does `Φ_stored` port? Yes — and the clamp must go, or the port is strictly worse than doing nothing

**The clamp is exactly what breaks the neutrality, and your suspicion is right.**
The proof in §1.2 is that the shaping is a _difference_: selling `q` units yields
cash `Σ_{j<q} price(I+j)` and `ΔΦ = −Σ_{j<q} price(I+j)`, which cancel. Apply
`max(·, 0)` and the negative half is deleted: the cash is credited and the
potential drop is forgiven. That is the pump, definitionally.

**Porting it with the clamp retained would be far worse than today.** Today the
clamp forgives _one unit of count_ (1e-5). Under a liquidation-value potential at
a coin-denominated weight it would forgive _the unit's price_ — melon at $186
median realised, roughly **190× the current per-unit leak**. Do not port the
potential and keep the clamp; that combination is the worst option on the table.

Three further conditions the port needs, all visible in the current file:

1. **One shared coin-denominated weight.** The cancellation is exact only if the
   coins term and the potential term enter at the _same_ weight. Today `fuel` is
   `FUEL_WEIGHT = 0.005` per unit and money is `money_weight = 0.001` per coin —
   different scales on different quantities. The port must replace _both_ with a
   single block `w · [Δmoney + (γΦ(s′) − Φ(s))]`. Any residual `|w_money − w_Φ|`
   is a per-unit pump of exactly that size times the price.
2. **`γ` must be threaded in.** `shaped()` takes no discount and forms plain
   undiscounted deltas. Ng, Harada & Russell's Theorem 1 is `γΦ(s′) − Φ(s)`, and
   §1.4 above is the record of what dropping the `γ` costs. `progress.py`'s
   `progress_reward` — on the dead line — already implements this correctly and
   is worth transcribing rather than rewriting.
3. **The terminal potential must be zero** (Grześ, AAMAS 2017), which is also the
   game's own scoring rule: unsold stock is worth nothing at turn 719. The corpus
   confirms the target behaviour — terminal unsold shed value has a **median of
   0** (§0.4).

Noted for the record, and it strengthens the argument rather than duplicating it:
arm W's death at update 34 on a confirmed buy-sell pump, with the signed version
pump-free by construction, is an _independent measurement of the same theorem_ on
this codebase. `toad_reward.money_signed`'s own docstring already reaches the
right conclusion — "this also makes the component potential-based with Phi =
money" — and the proposal in §1.2 is precisely that docstring's Φ extended from
`money` to `money + liquidation value of stock`. That is the smallest change that
makes the market branch potential-based end to end.

One caveat that does not apply to the count-based term but does apply to the
ported one, repeated from §1.2: Φ then moves when the _opponent_ sells and when
the town consumes, at `n × Δp` per turn. Log advantage standard deviation across
the switch, and fall back to Spooner et al.'s asymmetric dampening only if it
diverges.
