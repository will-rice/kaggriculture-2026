"""Deterministic CPU-only progressive evolution over hybrid-policy logits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Self, cast

import kaggle_environments

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.search import arena
from kaggriculture.search.arena import HybridOpponent, Opponent
from kaggriculture.search.fitness import Fitness, StrengthWeights, score_fitness
from kaggriculture.search.genome import GenomeCodec
from kaggriculture.search.scripts.frontier_round_robin import FRONTIER_SEEDS

SCREENING_SEEDS = tuple(range(820_000, 820_008))
DEVELOPMENT_SEEDS = tuple(range(830_000, 830_032))
PROMOTION_SEEDS = tuple(range(840_000, 840_128))
_SEED_SETS = (FRONTIER_SEEDS, SCREENING_SEEDS, DEVELOPMENT_SEEDS, PROMOTION_SEEDS)
assert all(
    set(left).isdisjoint(right)
    for index, left in enumerate(_SEED_SETS)
    for right in _SEED_SETS[index + 1 :]
), "frontier, screening, development, and promotion seeds must be disjoint"

STATE_SCHEMA_VERSION = 1
_UNSPECIFIED_MANIFEST_SHA256 = "0" * 64


@dataclass(frozen=True)
class EvolutionConfig:
    """Validated resource limits and mutation settings for one search run."""

    population: int = 32
    elites: int = 4
    generations: int = 40
    mutation_sigma: float = 0.35
    seed: int = 20_260_825
    workers: int = 16
    engine: str = kaggle_environments.__version__
    manifest_sha256: str = _UNSPECIFIED_MANIFEST_SHA256

    def __post_init__(self) -> None:
        """Reject unsafe CPU settings and ambiguous optimizer shapes."""
        for value, label in (
            (self.population, "population"),
            (self.elites, "elites"),
            (self.generations, "generations"),
            (self.seed, "seed"),
            (self.workers, "workers"),
        ):
            if type(value) is not int:
                raise TypeError(f"{label} must be an integer")
        if self.population < 1:
            raise ValueError("population must be at least one")
        if not 1 <= self.elites <= self.population:
            raise ValueError("elites must be between one and population")
        if self.generations < 0:
            raise ValueError("generations must be non-negative")
        if (
            not isinstance(self.mutation_sigma, (int, float))
            or isinstance(self.mutation_sigma, bool)
            or not math.isfinite(self.mutation_sigma)
            or self.mutation_sigma < 0.0
        ):
            raise ValueError("mutation_sigma must be finite and non-negative")
        if not 1 <= self.workers <= 16:
            raise ValueError("workers must be between 1 and 16 while Toad is running")
        if type(self.engine) is not str or not self.engine:
            raise ValueError("engine must be a non-empty string")
        if (
            type(self.manifest_sha256) is not str
            or len(self.manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.manifest_sha256
            )
        ):
            raise ValueError("manifest_sha256 must be 64 lowercase hexadecimal digits")


@dataclass(frozen=True)
class CandidateEvaluation:
    """One decoded candidate and all telemetry used to order it."""

    genome: tuple[float, ...]
    config: HybridConfig
    matchup_rates: Mapping[str, float]
    fitness: Fitness | None
    paired_normalized_margin: float
    failures: tuple[str, ...]
    runtime_seconds: float


@dataclass(frozen=True)
class GenerationRecord:
    """Every screening result and every developed survivor in one generation."""

    generation: int
    screened: tuple[CandidateEvaluation, ...]
    developed: tuple[CandidateEvaluation, ...]


@dataclass(frozen=True)
class SearchIdentity:
    """Every immutable input that gives a search state meaning."""

    manifest_sha256: str
    engine: str
    frontier_seeds: tuple[int, ...]
    screening_seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    promotion_seeds: tuple[int, ...]
    genome_schema_sha256: str
    league_identity: tuple[tuple[str, str, str], ...]
    strength_weights: tuple[tuple[str, int], ...]
    evolution_config: EvolutionConfig


RandomState = tuple[int, tuple[int, ...], float | None]


@dataclass(frozen=True)
class SearchState:
    """Canonical resumable state after zero or more complete generations."""

    identity: SearchIdentity
    generation: int
    rng_state: RandomState
    elites: tuple[CandidateEvaluation, ...]
    history: tuple[GenerationRecord, ...]

    def advance(
        self,
        generation: int,
        screened: Sequence[CandidateEvaluation],
        developed: Sequence[CandidateEvaluation],
        rng_state: object,
    ) -> "SearchState":
        """Record one complete generation and retain its strongest elites."""
        if generation != self.generation:
            raise ValueError("generation advance must be contiguous")
        elite_count = self.identity.evolution_config.elites
        elites = top_eligible(developed, count=elite_count)
        if not elites:
            raise RuntimeError("generation produced no eligible development survivor")
        record = GenerationRecord(generation, tuple(screened), tuple(developed))
        return SearchState(
            identity=self.identity,
            generation=generation + 1,
            rng_state=_validate_random_state(rng_state),
            elites=elites,
            history=(*self.history, record),
        )

    def to_payload(self) -> dict[str, object]:
        """Return the exact versioned JSON payload written by atomic saves."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "identity": _identity_payload(self.identity),
            "generation": self.generation,
            "rng_state": _random_state_payload(self.rng_state),
            "elites": [_candidate_payload(candidate) for candidate in self.elites],
            "history": [_generation_payload(record) for record in self.history],
        }

    def to_json(self) -> str:
        """Serialize canonically so equivalent resumed runs are byte-identical."""
        return (
            json.dumps(
                self.to_payload(),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_payload(cls, value: object) -> Self:
        """Strictly validate a decoded state artifact before it can resume."""
        payload = _dictionary(
            value,
            {
                "schema_version",
                "identity",
                "generation",
                "rng_state",
                "elites",
                "history",
            },
            "search state",
        )
        if payload["schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"search state schema_version must be {STATE_SCHEMA_VERSION}"
            )
        generation = _integer(payload["generation"], "search generation", minimum=0)
        elites = tuple(
            _candidate_from_payload(candidate)
            for candidate in _list(payload["elites"], "search elites")
        )
        history = tuple(
            _generation_from_payload(record)
            for record in _list(payload["history"], "search history")
        )
        if generation != len(history):
            raise ValueError("search generation must equal the complete history length")
        if tuple(record.generation for record in history) != tuple(range(generation)):
            raise ValueError("search history generations must be contiguous from zero")
        return cls(
            identity=_identity_from_payload(payload["identity"]),
            generation=generation,
            rng_state=_random_state_from_payload(payload["rng_state"]),
            elites=elites,
            history=history,
        )

    @classmethod
    def from_json(cls, source: str) -> Self:
        """Parse a strict state from JSON without accepting non-finite numbers."""
        return cls.from_payload(
            json.loads(source, parse_constant=_reject_json_constant)
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load one complete state artifact from disk."""
        return cls.from_json(path.read_text())


def mutate_genome(
    parent: Sequence[float], sigma: float, rng: random.Random
) -> tuple[float, ...]:
    """Add deterministic Gaussian noise to every logit and continuous gene."""
    return tuple(float(value) + rng.gauss(0.0, sigma) for value in parent)


def initial_state(
    codec: GenomeCodec,
    initial_configs: Sequence[HybridConfig],
    league: Mapping[str, Opponent],
    weights: StrengthWeights,
    config: EvolutionConfig,
) -> SearchState:
    """Create the unevaluated generation-zero state and bind all run identity."""
    if not initial_configs:
        raise ValueError("initial_configs must contain at least one candidate")
    genomes = tuple(codec.encode(candidate) for candidate in initial_configs)
    parents = tuple(
        CandidateEvaluation(
            genome=genome,
            config=candidate,
            matchup_rates={},
            fitness=None,
            paired_normalized_margin=0.0,
            failures=(),
            runtime_seconds=0.0,
        )
        for genome, candidate in zip(genomes, initial_configs, strict=True)
    )
    return SearchState(
        identity=_search_identity(codec, league, weights, config),
        generation=0,
        rng_state=_validate_random_state(random.Random(config.seed).getstate()),
        elites=parents[: config.population],
        history=(),
    )


def spawn_population(
    parents: Sequence[CandidateEvaluation],
    config: EvolutionConfig,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    """Keep current parents and fill the population with deterministic mutations."""
    if not parents:
        raise ValueError("at least one parent is required to spawn a population")
    genomes = [parent.genome for parent in parents[: config.population]]
    while len(genomes) < config.population:
        parent = parents[rng.randrange(len(parents))]
        genomes.append(mutate_genome(parent.genome, config.mutation_sigma, rng))
    return tuple(genomes)


def evaluate_population(
    genomes: Sequence[Sequence[float]],
    codec: GenomeCodec,
    seeds: Sequence[int],
    league: Mapping[str, Opponent],
    weights: StrengthWeights,
    workers: int,
) -> tuple[CandidateEvaluation, ...]:
    """Evaluate each candidate against every member, continuing after failures."""
    seed_tuple = tuple(int(seed) for seed in seeds)
    expected_games = 2 * len(seed_tuple)
    evaluated: list[CandidateEvaluation] = []
    for raw_genome in genomes:
        genome = tuple(float(value) for value in raw_genome)
        candidate_config = codec.decode(genome)
        candidate = HybridOpponent(to_runtime(candidate_config))
        rates: dict[str, float] = {}
        normalized_margins: list[float] = []
        failures: list[str] = []
        runtime_seconds = 0.0
        for name, opponent in league.items():
            started = perf_counter()
            try:
                scores = arena.outcomes(
                    candidate, {name: opponent}, seed_tuple, workers
                )
            except Exception as error:
                rates[name] = 0.0
                failures.append(f"{name}: {type(error).__name__}: {error}")
                runtime_seconds += perf_counter() - started
                continue
            runtime_seconds += float(
                getattr(scores, "runtime_seconds", perf_counter() - started)
            )
            matchup_failures = tuple(getattr(scores, "failures", ()))
            failures.extend(f"{name}: {failure}" for failure in matchup_failures)
            if len(scores) != expected_games:
                rates[name] = 0.0
                failures.append(
                    f"{name}: arena returned {len(scores)} scores for "
                    f"{expected_games} games"
                )
                continue
            rates[name] = fmean(float(score) for score in scores)
            margins = tuple(getattr(scores, "normalized_margins", ()))
            if margins and len(margins) != expected_games:
                failures.append(
                    f"{name}: arena returned {len(margins)} normalized margins "
                    f"for {expected_games} games"
                )
            elif margins:
                normalized_margins.extend(float(margin) for margin in margins)
            else:
                normalized_margins.extend(0.0 for _ in scores)
        fitness = score_fitness(rates, weights, failures=len(failures))
        evaluated.append(
            CandidateEvaluation(
                genome=genome,
                config=candidate_config,
                matchup_rates=rates,
                fitness=fitness,
                paired_normalized_margin=(
                    fmean(normalized_margins) if normalized_margins else 0.0
                ),
                failures=tuple(failures),
                runtime_seconds=runtime_seconds,
            )
        )
    return tuple(evaluated)


def top_eligible(
    candidates: Sequence[CandidateEvaluation], count: int
) -> tuple[CandidateEvaluation, ...]:
    """Rank by fitness, normalized margin, runtime, then canonical genome."""
    if type(count) is not int or count < 0:
        raise ValueError("eligible count must be a non-negative integer")
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.fitness is not None and candidate.fitness.eligible
    )
    return tuple(
        sorted(
            eligible,
            key=lambda candidate: (
                -cast(Fitness, candidate.fitness).value,
                -candidate.paired_normalized_margin,
                candidate.runtime_seconds,
                candidate.genome,
            ),
        )[:count]
    )


def evolve(
    *,
    codec: GenomeCodec,
    initial_configs: Sequence[HybridConfig],
    league: Mapping[str, Opponent],
    weights: StrengthWeights,
    config: EvolutionConfig,
    output: Path,
    resume: SearchState | None = None,
) -> SearchState:
    """Run progressive evolution, atomically saving every complete generation."""
    expected_identity = _search_identity(codec, league, weights, config)
    if resume is None:
        state = initial_state(codec, initial_configs, league, weights, config)
    else:
        _validate_resume(resume, expected_identity)
        _validate_resume_contents(resume, codec, weights)
        state = resume
    if state.generation > config.generations:
        raise ValueError("resume generation exceeds configured generations")
    rng = random.Random()
    rng.setstate(state.rng_state)
    for generation in range(state.generation, config.generations):
        genomes = spawn_population(state.elites, config, rng)
        screened = evaluate_population(
            genomes, codec, SCREENING_SEEDS, league, weights, config.workers
        )
        survivors = top_eligible(screened, count=config.elites * 2)
        if not survivors:
            raise RuntimeError("screening produced no eligible survivor")
        developed = evaluate_population(
            tuple(candidate.genome for candidate in survivors),
            codec,
            DEVELOPMENT_SEEDS,
            league,
            weights,
            config.workers,
        )
        state = state.advance(generation, screened, developed, rng.getstate())
        save_state_atomic(state, output)
    return state


def save_state_atomic(state: SearchState, output: Path) -> None:
    """Replace a state only after its complete canonical bytes reach disk."""
    _write_atomic(output, state.to_json())


def write_finalists(state: SearchState, output: Path) -> None:
    """Write decoded finalist configurations with the exact search identity."""
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "identity": _identity_payload(state.identity),
        "generation": state.generation,
        "finalists": [_candidate_payload(candidate) for candidate in state.elites],
    }
    source = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    _write_atomic(output, source)


def genome_schema_sha256(codec: GenomeCodec) -> str:
    """Hash every path, domain, offset, bound, and template in the codec."""
    payload = {
        "template": codec.template,
        "width": codec.width,
        "discrete_groups": [
            {
                "path": group.path,
                "values": group.values,
                "start": group.start,
                "width": group.width,
            }
            for group in codec.discrete_groups
        ],
        "continuous_genes": [asdict(gene) for gene in codec.continuous_genes],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _search_identity(
    codec: GenomeCodec,
    league: Mapping[str, Opponent],
    weights: StrengthWeights,
    config: EvolutionConfig,
) -> SearchIdentity:
    if not league:
        raise ValueError("league must contain at least one opponent")
    if set(league) != set(weights.values):
        raise ValueError("league names must exactly match strength weight names")
    return SearchIdentity(
        manifest_sha256=config.manifest_sha256,
        engine=config.engine,
        frontier_seeds=tuple(FRONTIER_SEEDS),
        screening_seeds=tuple(SCREENING_SEEDS),
        development_seeds=tuple(DEVELOPMENT_SEEDS),
        promotion_seeds=tuple(PROMOTION_SEEDS),
        genome_schema_sha256=genome_schema_sha256(codec),
        league_identity=tuple(
            (name, *_opponent_identity(opponent)) for name, opponent in league.items()
        ),
        strength_weights=tuple(weights.values.items()),
        evolution_config=config,
    )


def _opponent_identity(opponent: Opponent) -> tuple[str, str]:
    """Return an exact kind and content digest for one league member."""
    if isinstance(opponent, HybridOpponent):
        source = json.dumps(
            opponent.runtime.to_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return ("hybrid_runtime", hashlib.sha256(source).hexdigest())
    if isinstance(opponent, str):
        path = Path(opponent)
        if path.is_file():
            return ("agent_source", hashlib.sha256(path.read_bytes()).hexdigest())
        source = json.dumps(
            {"unresolved_agent_path": opponent}, sort_keys=True, separators=(",", ":")
        ).encode()
        return ("agent_path", hashlib.sha256(source).hexdigest())
    source = json.dumps(opponent, sort_keys=True, separators=(",", ":")).encode()
    return ("route", hashlib.sha256(source).hexdigest())


def _validate_resume(state: SearchState, expected: SearchIdentity) -> None:
    """Name the first identity mismatch instead of accepting a partial resume."""
    checks = (
        ("manifest sha256", state.identity.manifest_sha256, expected.manifest_sha256),
        ("engine", state.identity.engine, expected.engine),
        ("frontier seeds", state.identity.frontier_seeds, expected.frontier_seeds),
        ("screening seeds", state.identity.screening_seeds, expected.screening_seeds),
        (
            "development seeds",
            state.identity.development_seeds,
            expected.development_seeds,
        ),
        ("promotion seeds", state.identity.promotion_seeds, expected.promotion_seeds),
        (
            "genome schema",
            state.identity.genome_schema_sha256,
            expected.genome_schema_sha256,
        ),
        ("league identity", state.identity.league_identity, expected.league_identity),
        (
            "strength weights",
            state.identity.strength_weights,
            expected.strength_weights,
        ),
        (
            "evolution config",
            state.identity.evolution_config,
            expected.evolution_config,
        ),
    )
    for label, actual, wanted in checks:
        if actual != wanted:
            raise ValueError(f"resume {label} differs from the current search")


def _validate_resume_contents(
    state: SearchState, codec: GenomeCodec, weights: StrengthWeights
) -> None:
    """Reject internally inconsistent candidates before restoring the RNG."""
    if state.generation != len(state.history):
        raise ValueError("resume generation differs from its history length")
    if tuple(record.generation for record in state.history) != tuple(
        range(state.generation)
    ):
        raise ValueError("resume history generations are not contiguous")
    if state.generation == 0:
        _validate_initial_resume(state, codec, weights)
        return
    _validate_evaluated_resume(state, codec, weights)


def _validate_initial_resume(
    state: SearchState, codec: GenomeCodec, weights: StrengthWeights
) -> None:
    if state.history or not state.elites:
        raise ValueError("resume generation zero must hold only initial parents")
    for candidate in state.elites:
        _validate_resume_candidate(candidate, codec, weights, evaluated=False)


def _validate_evaluated_resume(
    state: SearchState, codec: GenomeCodec, weights: StrengthWeights
) -> None:
    config = state.identity.evolution_config
    if any(len(record.screened) != config.population for record in state.history):
        raise ValueError("resume screening population differs from evolution config")
    if any(
        not 1 <= len(record.developed) <= config.elites * 2 for record in state.history
    ):
        raise ValueError("resume development population differs from evolution config")
    for record in state.history:
        for candidate in (*record.screened, *record.developed):
            _validate_resume_candidate(candidate, codec, weights, evaluated=True)
    for candidate in state.elites:
        _validate_resume_candidate(candidate, codec, weights, evaluated=True)
    expected_elites = top_eligible(state.history[-1].developed, count=config.elites)
    if state.elites != expected_elites:
        raise ValueError("resume elites differ from the last development ranking")


def _validate_resume_candidate(
    candidate: CandidateEvaluation,
    codec: GenomeCodec,
    weights: StrengthWeights,
    *,
    evaluated: bool,
) -> None:
    try:
        decoded = codec.decode(candidate.genome)
    except (TypeError, ValueError) as error:
        raise ValueError("resume candidate genome/config is invalid") from error
    if decoded != candidate.config:
        raise ValueError("resume candidate genome/config do not match")
    if not evaluated:
        if (
            candidate.fitness is not None
            or candidate.matchup_rates
            or candidate.failures
            or candidate.paired_normalized_margin != 0.0
            or candidate.runtime_seconds != 0.0
        ):
            raise ValueError("resume initial parent contains evaluation telemetry")
        return
    if candidate.fitness is None:
        raise ValueError("resume evaluated candidate has no fitness")
    try:
        expected = score_fitness(
            candidate.matchup_rates, weights, failures=len(candidate.failures)
        )
    except ValueError as error:
        raise ValueError("resume candidate fitness inputs are invalid") from error
    if candidate.fitness != expected:
        raise ValueError("resume candidate fitness differs from its matchups")
    if (
        not math.isfinite(candidate.paired_normalized_margin)
        or not math.isfinite(candidate.runtime_seconds)
        or candidate.runtime_seconds < 0.0
    ):
        raise ValueError(
            "resume candidate telemetry must be finite and runtime non-negative"
        )


def _write_atomic(output: Path, source: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _identity_payload(identity: SearchIdentity) -> dict[str, object]:
    return {
        "manifest_sha256": identity.manifest_sha256,
        "engine": identity.engine,
        "frontier_seeds": list(identity.frontier_seeds),
        "screening_seeds": list(identity.screening_seeds),
        "development_seeds": list(identity.development_seeds),
        "promotion_seeds": list(identity.promotion_seeds),
        "genome_schema_sha256": identity.genome_schema_sha256,
        "league_identity": [list(item) for item in identity.league_identity],
        "strength_weights": [list(item) for item in identity.strength_weights],
        "evolution_config": asdict(identity.evolution_config),
    }


def _identity_from_payload(value: object) -> SearchIdentity:
    payload = _dictionary(
        value,
        {
            "manifest_sha256",
            "engine",
            "frontier_seeds",
            "screening_seeds",
            "development_seeds",
            "promotion_seeds",
            "genome_schema_sha256",
            "league_identity",
            "strength_weights",
            "evolution_config",
        },
        "search identity",
    )
    manifest_sha256 = _string(payload["manifest_sha256"], "manifest sha256")
    engine = _string(payload["engine"], "engine")
    schema = _string(payload["genome_schema_sha256"], "genome schema sha256")
    league_identity = tuple(
        (
            _string_row(item, 3, "league identity")[0],
            _string_row(item, 3, "league identity")[1],
            _string_row(item, 3, "league identity")[2],
        )
        for item in _list(payload["league_identity"], "league identity")
    )
    weight_rows = _list(payload["strength_weights"], "strength weights")
    weight_mapping: dict[str, int] = {}
    for item in weight_rows:
        row = _list(item, "strength weight")
        if len(row) != 2:
            raise ValueError("strength weight rows must have two values")
        name = _string(row[0], "strength weight name")
        if name in weight_mapping:
            raise ValueError("strength weight names must be unique")
        weight_mapping[name] = _integer(row[1], "strength weight", minimum=1)
    weights = StrengthWeights(weight_mapping)
    config_payload = _dictionary(
        payload["evolution_config"],
        {
            "population",
            "elites",
            "generations",
            "mutation_sigma",
            "seed",
            "workers",
            "engine",
            "manifest_sha256",
        },
        "evolution config",
    )
    config = EvolutionConfig(**cast(dict[str, Any], config_payload))
    return SearchIdentity(
        manifest_sha256=manifest_sha256,
        engine=engine,
        frontier_seeds=_integer_tuple(payload["frontier_seeds"], "frontier seeds"),
        screening_seeds=_integer_tuple(payload["screening_seeds"], "screening seeds"),
        development_seeds=_integer_tuple(
            payload["development_seeds"], "development seeds"
        ),
        promotion_seeds=_integer_tuple(payload["promotion_seeds"], "promotion seeds"),
        genome_schema_sha256=schema,
        league_identity=league_identity,
        strength_weights=tuple(weights.values.items()),
        evolution_config=config,
    )


def _candidate_payload(candidate: CandidateEvaluation) -> dict[str, object]:
    fitness = candidate.fitness
    return {
        "genome": list(candidate.genome),
        "config": candidate.config.model_dump(mode="json"),
        "matchup_rates": dict(candidate.matchup_rates),
        "fitness": None
        if fitness is None
        else {
            "value": fitness.value if fitness.eligible else None,
            "weighted_mean": fitness.weighted_mean,
            "worst": fitness.worst,
            "eligible": fitness.eligible,
            "failures": fitness.failures,
        },
        "paired_normalized_margin": candidate.paired_normalized_margin,
        "failures": list(candidate.failures),
        "runtime_seconds": candidate.runtime_seconds,
    }


def _candidate_from_payload(value: object) -> CandidateEvaluation:
    payload = _dictionary(
        value,
        {
            "genome",
            "config",
            "matchup_rates",
            "fitness",
            "paired_normalized_margin",
            "failures",
            "runtime_seconds",
        },
        "candidate",
    )
    config_value = payload["config"]
    if type(config_value) is not dict:
        raise TypeError("candidate config must be a dictionary")
    config = HybridConfig.model_validate_json(
        json.dumps(config_value, allow_nan=False, separators=(",", ":"))
    )
    rates_payload = _dictionary_open(payload["matchup_rates"], "matchup rates")
    rates = {
        _string(name, "matchup name"): _finite_float(rate, "matchup rate")
        for name, rate in rates_payload.items()
    }
    fitness_value = payload["fitness"]
    fitness = None if fitness_value is None else _fitness_from_payload(fitness_value)
    failures = tuple(
        _string(item, "candidate failure")
        for item in _list(payload["failures"], "candidate failures")
    )
    return CandidateEvaluation(
        genome=tuple(
            _finite_float(item, "genome value")
            for item in _list(payload["genome"], "candidate genome")
        ),
        config=config,
        matchup_rates=rates,
        fitness=fitness,
        paired_normalized_margin=_finite_float(
            payload["paired_normalized_margin"], "paired normalized margin"
        ),
        failures=failures,
        runtime_seconds=_finite_float(payload["runtime_seconds"], "runtime seconds"),
    )


def _fitness_from_payload(value: object) -> Fitness:
    payload = _dictionary(
        value,
        {"value", "weighted_mean", "worst", "eligible", "failures"},
        "fitness",
    )
    eligible = payload["eligible"]
    if type(eligible) is not bool:
        raise TypeError("fitness eligible must be a boolean")
    raw_value = payload["value"]
    if eligible:
        fitness_value = _finite_float(raw_value, "fitness value")
    else:
        if raw_value is not None:
            raise ValueError("ineligible fitness value must be null")
        fitness_value = -math.inf
    return Fitness(
        value=fitness_value,
        weighted_mean=_finite_float(payload["weighted_mean"], "weighted mean"),
        worst=_finite_float(payload["worst"], "worst matchup"),
        eligible=eligible,
        failures=_integer(payload["failures"], "fitness failures", minimum=0),
    )


def _generation_payload(record: GenerationRecord) -> dict[str, object]:
    return {
        "generation": record.generation,
        "screened": [_candidate_payload(candidate) for candidate in record.screened],
        "developed": [_candidate_payload(candidate) for candidate in record.developed],
    }


def _generation_from_payload(value: object) -> GenerationRecord:
    payload = _dictionary(
        value, {"generation", "screened", "developed"}, "generation record"
    )
    return GenerationRecord(
        generation=_integer(payload["generation"], "generation", minimum=0),
        screened=tuple(
            _candidate_from_payload(candidate)
            for candidate in _list(payload["screened"], "screened candidates")
        ),
        developed=tuple(
            _candidate_from_payload(candidate)
            for candidate in _list(payload["developed"], "developed candidates")
        ),
    )


def _random_state_payload(state: RandomState) -> list[object]:
    return [state[0], list(state[1]), state[2]]


def _random_state_from_payload(value: object) -> RandomState:
    values = _list(value, "random state")
    if len(values) != 3:
        raise ValueError("random state must contain version, words, and gaussian cache")
    gaussian = values[2]
    if gaussian is not None:
        gaussian = _finite_float(gaussian, "random gaussian cache")
    return _validate_random_state(
        (
            _integer(values[0], "random state version", minimum=0),
            _integer_tuple(values[1], "random state words"),
            gaussian,
        )
    )


def _validate_random_state(value: object) -> RandomState:
    if type(value) is not tuple or len(value) != 3:
        raise TypeError("random state must be a three-item tuple")
    version, words, gaussian = value
    if type(version) is not int or type(words) is not tuple:
        raise TypeError("random state has invalid version or word storage")
    if any(type(word) is not int for word in words):
        raise TypeError("random state words must be integers")
    if gaussian is not None and type(gaussian) is not float:
        raise TypeError("random gaussian cache must be a float or null")
    state = cast(RandomState, value)
    probe = random.Random()
    probe.setstate(state)
    return state


def _dictionary(value: object, expected: set[str], label: str) -> dict[str, object]:
    payload = _dictionary_open(value, label)
    if set(payload) != expected:
        raise ValueError(f"{label} keys must be {sorted(expected)}")
    return payload


def _dictionary_open(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a dictionary")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a list")
    return cast(list[object], value)


def _integer(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    return tuple(_integer(item, label, minimum=0) for item in _list(value, label))


def _finite_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _string_row(value: object, length: int, label: str) -> tuple[str, ...]:
    row = _list(value, label)
    if len(row) != length:
        raise ValueError(f"{label} rows must contain {length} strings")
    return tuple(_string(item, label) for item in row)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
