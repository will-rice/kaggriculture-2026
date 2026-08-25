"""Deterministic CPU-only progressive evolution over hybrid-policy logits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Self, cast

import kaggle_environments

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.search import arena
from kaggriculture.search.arena import HybridOpponent, Opponent
from kaggriculture.search.fitness import Fitness, StrengthWeights, score_fitness
from kaggriculture.search.frontier import FrontierManifest, VerifiedFrontier
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

STATE_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class EvolutionConfig:
    """Validated resource limits and mutation settings for one search run."""

    artifact_mode: str
    manifest_sha256: str | None
    population: int = 32
    elites: int = 4
    generations: int = 40
    mutation_sigma: float = 0.35
    seed: int = 20_260_825
    workers: int = 16
    engine: str = kaggle_environments.__version__

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
        _validate_artifact_identity(self.artifact_mode, self.manifest_sha256)


def _validate_artifact_identity(mode: str, manifest_sha256: str | None) -> None:
    if mode not in {"certified", "test"}:
        raise ValueError("artifact_mode must be 'certified' or 'test'")
    if mode == "test":
        if manifest_sha256 is not None:
            raise ValueError("test artifact mode cannot assert a manifest sha256")
        return
    if (
        type(manifest_sha256) is not str
        or len(manifest_sha256) != 64
        or manifest_sha256 == "0" * 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise ValueError(
            "certified manifest_sha256 must be 64 nonzero lowercase hexadecimal digits"
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    """One decoded candidate and all telemetry used to order it."""

    genome: tuple[float, ...]
    config: HybridConfig
    stage: str
    seeds: tuple[int, ...]
    matchup_rates: Mapping[str, float]
    matchup_games: Mapping[str, int]
    fitness: Fitness | None
    paired_normalized_margin: float
    failures: tuple[str, ...]


RandomState = tuple[int, tuple[int, ...], float | None]


@dataclass(frozen=True)
class GenerationRecord:
    """Every screening result and every developed survivor in one generation."""

    generation: int
    screened: tuple[CandidateEvaluation, ...]
    developed: tuple[CandidateEvaluation, ...]
    parent_digest: str
    spawned_population: tuple[tuple[float, ...], ...]
    survivor_genomes: tuple[tuple[float, ...], ...]
    elites: tuple[CandidateEvaluation, ...]
    rng_state: RandomState
    integrity_digest: str


@dataclass(frozen=True)
class SearchIdentity:
    """Every immutable input that gives a search state meaning."""

    certified: bool
    manifest_sha256: str | None
    engine: str
    frontier_seeds: tuple[int, ...]
    screening_seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    promotion_seeds: tuple[int, ...]
    genome_schema_sha256: str
    league_identity: tuple[tuple[str, str, str], ...]
    league_snapshots: tuple[tuple[str, str, str, str], ...]
    strength_weights: tuple[tuple[str, int], ...]
    initial_genomes: tuple[tuple[float, ...], ...]
    evolution_config: EvolutionConfig


@dataclass(frozen=True, init=False)
class _SearchCertification:
    """Verified production provenance accepted by evolution."""

    manifest_sha256: str
    engine: str
    league_identity: tuple[tuple[str, str, str], ...]
    league_snapshots: tuple[tuple[str, str, str, str], ...]
    snapshot: SnapshotLeague


@dataclass(frozen=True)
class SnapshotSource:
    """One immutable source copy and its original-to-snapshot mapping."""

    name: str
    source_path: str
    snapshot_path: str
    sha256: str


@dataclass(frozen=True)
class SnapshotLeague:
    """Content-addressed runnable sources rooted beside a search output."""

    root: str
    sources: tuple[SnapshotSource, ...]

    @property
    def opponents(self) -> Mapping[str, str]:
        """Return the exact worker league rooted only in snapshot paths."""
        return {source.name: source.snapshot_path for source in self.sources}


@dataclass(frozen=True)
class SearchState:
    """Canonical resumable state after zero or more complete generations."""

    identity: SearchIdentity
    generation: int
    rng_state: RandomState
    elites: tuple[CandidateEvaluation, ...]
    history: tuple[GenerationRecord, ...]
    integrity_digest: str

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
        screened_tuple = tuple(screened)
        developed_tuple = tuple(developed)
        terminal_rng = _validate_random_state(rng_state)
        spawned_population = tuple(candidate.genome for candidate in screened_tuple)
        survivor_genomes = tuple(candidate.genome for candidate in developed_tuple)
        digest = _generation_integrity_digest(
            identity=self.identity,
            generation=generation,
            parent_digest=self.integrity_digest,
            spawned_population=spawned_population,
            screened=screened_tuple,
            survivor_genomes=survivor_genomes,
            developed=developed_tuple,
            elites=elites,
            rng_state=terminal_rng,
        )
        record = GenerationRecord(
            generation=generation,
            screened=screened_tuple,
            developed=developed_tuple,
            parent_digest=self.integrity_digest,
            spawned_population=spawned_population,
            survivor_genomes=survivor_genomes,
            elites=elites,
            rng_state=terminal_rng,
            integrity_digest=digest,
        )
        return SearchState(
            identity=self.identity,
            generation=generation + 1,
            rng_state=terminal_rng,
            elites=elites,
            history=(*self.history, record),
            integrity_digest=digest,
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
            "integrity_digest": self.integrity_digest,
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
                "integrity_digest",
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
            integrity_digest=_sha256_string(
                payload["integrity_digest"], "search integrity digest"
            ),
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
    certification: _SearchCertification | None = None,
) -> SearchState:
    """Create the unevaluated generation-zero state and bind all run identity."""
    if not initial_configs:
        raise ValueError("initial_configs must contain at least one candidate")
    genomes = tuple(codec.encode(candidate) for candidate in initial_configs)
    parents = tuple(
        CandidateEvaluation(
            genome=genome,
            config=candidate,
            stage="initial",
            seeds=(),
            matchup_rates={},
            matchup_games={},
            fitness=None,
            paired_normalized_margin=0.0,
            failures=(),
        )
        for genome, candidate in zip(genomes, initial_configs, strict=True)
    )
    identity = _search_identity(
        codec, initial_configs, league, weights, config, certification
    )
    return SearchState(
        identity=identity,
        generation=0,
        rng_state=_validate_random_state(random.Random(config.seed).getstate()),
        elites=parents[: config.population],
        history=(),
        integrity_digest=_initial_integrity_digest(identity),
    )


def spawn_population(
    parents: Sequence[CandidateEvaluation],
    config: EvolutionConfig,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    """Keep current parents and fill the population with deterministic mutations."""
    if not parents:
        raise ValueError("at least one parent is required to spawn a population")
    return _spawn_population_genomes(
        tuple(parent.genome for parent in parents), config, rng
    )


def _spawn_population_genomes(
    parents: Sequence[Sequence[float]],
    config: EvolutionConfig,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    """Replayable population spawn over canonical parent genomes."""
    if not parents:
        raise ValueError("at least one parent is required to spawn a population")
    parent_genomes = tuple(
        tuple(float(value) for value in parent) for parent in parents
    )
    genomes = list(parent_genomes[: config.population])
    while len(genomes) < config.population:
        parent = parent_genomes[rng.randrange(len(parent_genomes))]
        genomes.append(mutate_genome(parent, config.mutation_sigma, rng))
    return tuple(genomes)


def evaluate_population(
    genomes: Sequence[Sequence[float]],
    codec: GenomeCodec,
    seeds: Sequence[int],
    league: Mapping[str, Opponent],
    weights: StrengthWeights,
    workers: int,
    *,
    stage: str,
) -> tuple[CandidateEvaluation, ...]:
    """Evaluate each candidate against every member, continuing after failures."""
    seed_tuple = tuple(int(seed) for seed in seeds)
    expected_seeds = {
        "screening": SCREENING_SEEDS,
        "development": DEVELOPMENT_SEEDS,
    }
    if stage not in expected_seeds:
        raise ValueError("evaluation stage must be screening or development")
    if seed_tuple != expected_seeds[stage]:
        raise ValueError(f"{stage} evaluation must use its fixed seed set")
    expected_games = 2 * len(seed_tuple)
    evaluated: list[CandidateEvaluation] = []
    for raw_genome in genomes:
        genome = tuple(float(value) for value in raw_genome)
        candidate_config = codec.decode(genome)
        candidate = HybridOpponent(to_runtime(candidate_config))
        rates: dict[str, float] = {}
        matchup_games: dict[str, int] = {}
        normalized_margins: list[float] = []
        failures: list[str] = []
        for name, opponent in league.items():
            try:
                scores = arena.outcomes(
                    candidate, {name: opponent}, seed_tuple, workers
                )
            except Exception as error:
                rates[name] = 0.0
                matchup_games[name] = 0
                failures.append(f"{name}: {type(error).__name__}: {error}")
                continue
            matchup_games[name] = len(scores)
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
                stage=stage,
                seeds=seed_tuple,
                matchup_rates=rates,
                matchup_games=matchup_games,
                fitness=fitness,
                paired_normalized_margin=(
                    fmean(normalized_margins) if normalized_margins else 0.0
                ),
                failures=tuple(failures),
            )
        )
    return tuple(evaluated)


def top_eligible(
    candidates: Sequence[CandidateEvaluation], count: int
) -> tuple[CandidateEvaluation, ...]:
    """Rank by fitness, normalized margin, then canonical genome."""
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
    certification: _SearchCertification | None = None,
) -> SearchState:
    """Run progressive evolution, atomically saving every complete generation."""
    expected_identity = _search_identity(
        codec, initial_configs, league, weights, config, certification
    )
    _validate_output_outside_snapshot(expected_identity, output)
    if resume is None:
        state = initial_state(
            codec, initial_configs, league, weights, config, certification
        )
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
            genomes,
            codec,
            SCREENING_SEEDS,
            league,
            weights,
            config.workers,
            stage="screening",
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
            stage="development",
        )
        state = state.advance(generation, screened, developed, rng.getstate())
        save_state_atomic(state, output)
    return state


def save_state_atomic(state: SearchState, output: Path) -> None:
    """Replace a state only after its complete canonical bytes reach disk."""
    _validate_output_outside_snapshot(state.identity, output)
    _write_atomic(output, state.to_json())


def write_finalists(
    state: SearchState,
    output: Path,
    *,
    certification: object | None = None,
    codec: GenomeCodec | None = None,
) -> None:
    """Validate a completed production state before writing its finalists."""
    _validate_identity_cross_fields(state.identity)
    _validate_output_outside_snapshot(state.identity, output)
    _validate_certification(state.identity, certification)
    config = state.identity.evolution_config
    if state.generation == 0 or not state.history:
        raise ValueError("finalists require a nonzero evaluated history")
    if state.generation != config.generations:
        raise ValueError("finalists require the configured generation to be complete")
    finalist_codec = GenomeCodec.default() if codec is None else codec
    if genome_schema_sha256(finalist_codec) != state.identity.genome_schema_sha256:
        raise ValueError("finalist codec differs from the search genome schema")
    weights = StrengthWeights(dict(state.identity.strength_weights))
    _validate_resume_contents(state, finalist_codec, weights)
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "identity": _identity_payload(state.identity),
        "generation": state.generation,
        "finalists": [_candidate_payload(candidate) for candidate in state.elites],
        "integrity_digest": state.integrity_digest,
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


def _integrity_sha256(payload: Mapping[str, object]) -> str:
    """Hash canonical structural evidence; this is tamper-evident, not signed."""
    source = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(source).hexdigest()


def _initial_integrity_digest(identity: SearchIdentity) -> str:
    return _integrity_sha256(
        {"kind": "hybrid-search-initial", "identity": _identity_payload(identity)}
    )


def _generation_integrity_digest(
    *,
    identity: SearchIdentity,
    generation: int,
    parent_digest: str,
    spawned_population: Sequence[Sequence[float]],
    screened: Sequence[CandidateEvaluation],
    survivor_genomes: Sequence[Sequence[float]],
    developed: Sequence[CandidateEvaluation],
    elites: Sequence[CandidateEvaluation],
    rng_state: RandomState,
) -> str:
    return _integrity_sha256(
        {
            "kind": "hybrid-search-generation",
            "static_identity_digest": _initial_integrity_digest(identity),
            "generation": generation,
            "parent_digest": parent_digest,
            "spawned_population": [list(genome) for genome in spawned_population],
            "screened": [_candidate_payload(candidate) for candidate in screened],
            "survivor_genomes": [list(genome) for genome in survivor_genomes],
            "developed": [_candidate_payload(candidate) for candidate in developed],
            "elites": [_candidate_payload(candidate) for candidate in elites],
            "terminal_rng_state": _random_state_payload(rng_state),
        }
    )


def snapshot_frontier(
    frontier: VerifiedFrontier,
    output: Path,
    *,
    require_existing: bool = False,
) -> SnapshotLeague:
    """Atomically copy verified sources into a content-addressed run league."""
    if frontier.manifest_sha256 is None:
        raise ValueError("frontier has no verified manifest sha256")
    artifact_names = tuple(artifact.name for artifact in frontier.artifacts)
    if artifact_names != tuple(frontier.opponents):
        raise ValueError("frontier artifacts and opponents must have identical order")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.with_name(f"{output.name}.league").absolute()
    sources = tuple(
        SnapshotSource(
            name=artifact.name,
            source_path=str(Path(frontier.opponents[artifact.name]).absolute()),
            snapshot_path=str(
                root
                / (
                    f"{artifact.name}-{artifact.sha256}"
                    f"{artifact.relative_path.suffix or '.py'}"
                )
            ),
            sha256=artifact.sha256,
        )
        for artifact in frontier.artifacts
    )
    snapshot = SnapshotLeague(str(root), sources)
    if root.exists() or root.is_symlink():
        _verify_snapshot_league(snapshot)
        return snapshot
    if require_existing:
        raise ValueError(f"snapshot league is missing: {root}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.", suffix=".tmp", dir=root.parent)
    )
    published = False
    try:
        for source in sources:
            original = Path(source.source_path).read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            if digest != source.sha256:
                raise ValueError(
                    f"{source.name}: source sha256 changed before snapshot"
                )
            destination = temporary / Path(source.snapshot_path).name
            with destination.open("xb") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
            destination.chmod(0o444)
        _fsync_directory(temporary)
        temporary.replace(root)
        published = True
        _fsync_directory(root.parent)
        _verify_snapshot_league(snapshot)
    except BaseException:
        cleanup = root if published else temporary
        shutil.rmtree(cleanup, ignore_errors=True)
        if published:
            try:
                _fsync_directory(root.parent)
            except OSError:
                pass
        raise
    return snapshot


def restore_snapshot_frontier(
    manifest_path: Path,
    artifact_root: Path,
    state: SearchState,
    output: Path,
) -> tuple[VerifiedFrontier, SnapshotLeague, _SearchCertification]:
    """Restore a certified league without trusting mutable original bytes."""
    _validate_identity_cross_fields(state.identity)
    _validate_output_outside_snapshot(state.identity, output)
    manifest_source = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_source).hexdigest()
    manifest = FrontierManifest.model_validate_json(manifest_source)
    if manifest_sha256 != state.identity.manifest_sha256:
        raise ValueError("resume manifest sha256 differs from search identity")
    if (
        manifest.engine != state.identity.engine
        or manifest.engine != kaggle_environments.__version__
    ):
        raise ValueError("resume manifest engine differs from search identity")
    root = artifact_root.resolve()
    opponents: dict[str, str] = {}
    for artifact in manifest.artifacts:
        source = (root / artifact.relative_path).resolve()
        if root not in source.parents:
            raise ValueError(
                f"{artifact.name}: resume source path escapes artifact root"
            )
        opponents[artifact.name] = str(source)
    frontier = VerifiedFrontier(
        manifest.engine, opponents, manifest.artifacts, manifest_sha256
    )
    snapshot = _snapshot_league_from_identity(state.identity)
    if snapshot is None:
        raise ValueError("certified resume identity has no snapshot league")
    certification = certify_frontier(frontier, snapshot)
    if certification.league_snapshots != state.identity.league_snapshots:
        raise ValueError("resume snapshot mapping differs from search identity")
    return frontier, snapshot, certification


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_snapshot_league(snapshot: SnapshotLeague) -> None:
    root = Path(snapshot.root)
    try:
        root_status = root.lstat()
    except OSError as error:
        raise ValueError(f"snapshot league is missing: {root}") from error
    if root.is_symlink() or not stat.S_ISDIR(root_status.st_mode):
        raise ValueError(f"snapshot league root is not a regular directory: {root}")
    expected_paths = {Path(source.snapshot_path) for source in snapshot.sources}
    try:
        actual_paths = set(root.iterdir())
    except OSError as error:
        raise ValueError(f"snapshot league cannot be read: {root}") from error
    if actual_paths != expected_paths:
        raise ValueError("snapshot league paths differ from certified mapping")
    for source in snapshot.sources:
        path = Path(source.snapshot_path)
        if path.parent != root or source.sha256 not in path.name:
            raise ValueError(f"{source.name}: snapshot path is not content-addressed")
        try:
            status = path.lstat()
            content = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"{source.name}: snapshot is missing or unreadable"
            ) from error
        if (
            path.is_symlink()
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or status.st_mode & 0o222
        ):
            raise ValueError(
                f"{source.name}: snapshot must be a read-only unaliased regular file"
            )
        if hashlib.sha256(content).hexdigest() != source.sha256:
            raise ValueError(f"{source.name}: snapshot sha256 changed")


def certify_frontier(
    frontier: VerifiedFrontier, snapshot: SnapshotLeague
) -> _SearchCertification:
    """Bind a production search to freshly reverified snapshot bytes."""
    if frontier.manifest_sha256 is None:
        raise ValueError("frontier has no verified manifest sha256")
    _verify_snapshot_league(snapshot)
    expected_sources = tuple(
        (
            artifact.name,
            str(Path(frontier.opponents[artifact.name]).absolute()),
            artifact.sha256,
        )
        for artifact in frontier.artifacts
    )
    actual_sources = tuple(
        (source.name, source.source_path, source.sha256) for source in snapshot.sources
    )
    if actual_sources != expected_sources:
        raise ValueError("snapshot source mapping differs from verified frontier")
    league_identity = tuple(
        (name, *_opponent_identity(opponent))
        for name, opponent in snapshot.opponents.items()
    )
    league_snapshots = _snapshot_identity(snapshot)
    certification = object.__new__(_SearchCertification)
    object.__setattr__(certification, "manifest_sha256", frontier.manifest_sha256)
    object.__setattr__(certification, "engine", frontier.engine)
    object.__setattr__(certification, "league_identity", league_identity)
    object.__setattr__(certification, "league_snapshots", league_snapshots)
    object.__setattr__(certification, "snapshot", snapshot)
    return certification


def _snapshot_identity(
    snapshot: SnapshotLeague,
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            source.name,
            source.source_path,
            source.snapshot_path,
            source.sha256,
        )
        for source in snapshot.sources
    )


def _snapshot_league_from_identity(
    identity: SearchIdentity,
) -> SnapshotLeague | None:
    """Reconstruct the one immutable league root bound into an identity."""
    if not identity.league_snapshots:
        return None
    sources = tuple(SnapshotSource(*row) for row in identity.league_snapshots)
    roots = {Path(source.snapshot_path).parent for source in sources}
    if len(roots) != 1:
        raise ValueError("certified snapshot mapping has multiple league roots")
    return SnapshotLeague(str(roots.pop()), sources)


def _same_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_output_outside_snapshot(identity: SearchIdentity, output: Path) -> None:
    """Reject lexical, canonical, and inode aliases into a bound snapshot league."""
    snapshot = _snapshot_league_from_identity(identity)
    if snapshot is None:
        return
    # Keep a lexical normalized form as well as the canonical symlink-resolved
    # form below; ``Path.resolve`` would collapse the two checks into one.
    root = Path(os.path.abspath(snapshot.root))  # noqa: PTH100
    candidate = Path(os.path.abspath(output))  # noqa: PTH100
    canonical_root = Path(os.path.realpath(root))
    canonical_candidate = Path(os.path.realpath(candidate))
    if _same_or_below(candidate, root) or _same_or_below(
        canonical_candidate, canonical_root
    ):
        raise ValueError("output path must be outside the certified snapshot league")
    try:
        candidate_status = candidate.lstat()
    except OSError:
        return
    if not stat.S_ISREG(candidate_status.st_mode):
        return
    for source in snapshot.sources:
        try:
            snapshot_status = Path(source.snapshot_path).lstat()
        except OSError:
            continue
        if os.path.samestat(candidate_status, snapshot_status):
            raise ValueError("output path aliases a certified snapshot file")


def _validate_certification(
    identity: SearchIdentity, certification: object | None
) -> None:
    if not identity.certified or identity.evolution_config.artifact_mode != "certified":
        raise ValueError("non-certifying search state cannot write finalists")
    if type(certification) is not _SearchCertification:
        raise ValueError("finalists require verified production certification")
    verified = certification
    _verify_snapshot_league(verified.snapshot)
    if (
        identity.manifest_sha256 != verified.manifest_sha256
        or identity.engine != verified.engine
        or identity.league_identity != verified.league_identity
        or identity.league_snapshots != verified.league_snapshots
    ):
        raise ValueError("finalist certification differs from search identity")


def _search_identity(
    codec: GenomeCodec,
    initial_configs: Sequence[HybridConfig],
    league: Mapping[str, Opponent],
    weights: StrengthWeights,
    config: EvolutionConfig,
    certification: _SearchCertification | None,
) -> SearchIdentity:
    if not league:
        raise ValueError("league must contain at least one opponent")
    if set(league) != set(weights.values):
        raise ValueError("league names must exactly match strength weight names")
    league_identity = tuple(
        (name, *_opponent_identity(opponent)) for name, opponent in league.items()
    )
    if config.artifact_mode == "certified":
        if certification is None:
            raise ValueError("certified evolution requires verified certification")
        _verify_snapshot_league(certification.snapshot)
        if (
            config.manifest_sha256 != certification.manifest_sha256
            or config.engine != certification.engine
            or league_identity != certification.league_identity
            or dict(league) != dict(certification.snapshot.opponents)
        ):
            raise ValueError(
                "verified certification differs from current search inputs"
            )
    elif certification is not None:
        raise ValueError("test artifact mode cannot accept production certification")
    initial_genomes = tuple(codec.encode(candidate) for candidate in initial_configs)[
        : config.population
    ]
    if not initial_genomes:
        raise ValueError("initial_configs must contain at least one candidate")
    identity = SearchIdentity(
        certified=certification is not None,
        manifest_sha256=config.manifest_sha256,
        engine=config.engine,
        frontier_seeds=tuple(FRONTIER_SEEDS),
        screening_seeds=tuple(SCREENING_SEEDS),
        development_seeds=tuple(DEVELOPMENT_SEEDS),
        promotion_seeds=tuple(PROMOTION_SEEDS),
        genome_schema_sha256=genome_schema_sha256(codec),
        league_identity=league_identity,
        league_snapshots=(
            certification.league_snapshots if certification is not None else ()
        ),
        strength_weights=tuple(weights.values.items()),
        initial_genomes=initial_genomes,
        evolution_config=config,
    )
    _validate_identity_cross_fields(identity)
    return identity


def _validate_identity_cross_fields(identity: SearchIdentity) -> None:
    config = identity.evolution_config
    if identity.engine != config.engine:
        raise ValueError("search identity engine differs from evolution config")
    if identity.manifest_sha256 != config.manifest_sha256:
        raise ValueError("search identity manifest differs from evolution config")
    production_mode = config.artifact_mode == "certified"
    if identity.certified != production_mode:
        raise ValueError("search identity certification differs from artifact mode")
    snapshot_names = tuple(row[0] for row in identity.league_snapshots)
    league_names = tuple(row[0] for row in identity.league_identity)
    if production_mode and (
        not identity.league_snapshots or snapshot_names != league_names
    ):
        raise ValueError("certified search identity has no complete snapshot mapping")
    if not production_mode and identity.league_snapshots:
        raise ValueError("test search identity cannot contain snapshot mapping")


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
        ("certification", state.identity.certified, expected.certified),
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
            "league snapshots",
            state.identity.league_snapshots,
            expected.league_snapshots,
        ),
        (
            "strength weights",
            state.identity.strength_weights,
            expected.strength_weights,
        ),
        ("initial genomes", state.identity.initial_genomes, expected.initial_genomes),
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
    """Reject inconsistent or locally tampered state before external effects."""
    _validate_identity_cross_fields(state.identity)
    if state.generation != len(state.history):
        raise ValueError("resume generation differs from its history length")
    if tuple(record.generation for record in state.history) != tuple(
        range(state.generation)
    ):
        raise ValueError("resume history generations are not contiguous")
    config = state.identity.evolution_config
    rng = random.Random(config.seed)
    parent_genomes = state.identity.initial_genomes
    if state.generation == 0:
        _validate_initial_resume(state, codec, weights)
    for record in state.history:
        parent_genomes = _validate_generation_record(
            record, parent_genomes, state.identity, codec, weights, rng
        )
    if state.history:
        for candidate in state.elites:
            _validate_resume_candidate(
                candidate,
                codec,
                weights,
                stage="development",
                seeds=state.identity.development_seeds,
            )
        expected_elites = top_eligible(state.history[-1].developed, count=config.elites)
        if state.elites != expected_elites:
            raise ValueError("resume elites differ from the last development ranking")
    expected_rng_state = _validate_random_state(rng.getstate())
    if state.rng_state != expected_rng_state:
        raise ValueError("resume RNG state differs from replayed history")
    _validate_integrity_chain(state)


def _validate_integrity_chain(state: SearchState) -> None:
    parent_digest = _initial_integrity_digest(state.identity)
    for record in state.history:
        if record.parent_digest != parent_digest:
            raise ValueError("resume generation parent integrity digest differs")
        expected = _generation_integrity_digest(
            identity=state.identity,
            generation=record.generation,
            parent_digest=record.parent_digest,
            spawned_population=record.spawned_population,
            screened=record.screened,
            survivor_genomes=record.survivor_genomes,
            developed=record.developed,
            elites=record.elites,
            rng_state=record.rng_state,
        )
        if record.integrity_digest != expected:
            raise ValueError("resume generation integrity digest differs")
        parent_digest = expected
    if state.integrity_digest != parent_digest:
        raise ValueError("resume terminal integrity digest differs")


def _validate_generation_record(
    record: GenerationRecord,
    parent_genomes: tuple[tuple[float, ...], ...],
    identity: SearchIdentity,
    codec: GenomeCodec,
    weights: StrengthWeights,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    config = identity.evolution_config
    _validate_generation_spawn(record, parent_genomes, config, rng)
    return _validate_generation_outcomes(record, identity, codec, weights, rng)


def _validate_generation_spawn(
    record: GenerationRecord,
    parent_genomes: tuple[tuple[float, ...], ...],
    config: EvolutionConfig,
    rng: random.Random,
) -> None:
    expected_spawn = _spawn_population_genomes(parent_genomes, config, rng)
    if record.spawned_population != expected_spawn:
        raise ValueError("resume recorded spawned population differs from RNG replay")
    actual_spawn = tuple(candidate.genome for candidate in record.screened)
    if actual_spawn != record.spawned_population:
        raise ValueError("resume spawned population differs from RNG replay")
    if len(record.screened) != config.population:
        raise ValueError("resume screening population differs from evolution config")


def _validate_generation_outcomes(
    record: GenerationRecord,
    identity: SearchIdentity,
    codec: GenomeCodec,
    weights: StrengthWeights,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    config = identity.evolution_config
    for candidate in record.screened:
        _validate_resume_candidate(
            candidate,
            codec,
            weights,
            stage="screening",
            seeds=identity.screening_seeds,
        )
    survivors = top_eligible(record.screened, count=config.elites * 2)
    survivor_genomes = tuple(candidate.genome for candidate in survivors)
    if record.survivor_genomes != survivor_genomes:
        raise ValueError("resume survivor lineage differs from screening ranking")
    for candidate in record.developed:
        _validate_resume_candidate(
            candidate,
            codec,
            weights,
            stage="development",
            seeds=identity.development_seeds,
        )
    if tuple(candidate.genome for candidate in record.developed) != (
        record.survivor_genomes
    ):
        raise ValueError(
            "resume development survivor set differs from screening ranking"
        )
    parents = top_eligible(record.developed, count=config.elites)
    if not parents:
        raise ValueError("resume development contains no eligible parent")
    if record.elites != parents:
        raise ValueError("resume generation elites differ from development ranking")
    if record.rng_state != _validate_random_state(rng.getstate()):
        raise ValueError("resume generation RNG transition differs from replay")
    return tuple(candidate.genome for candidate in parents)


def _validate_initial_resume(
    state: SearchState, codec: GenomeCodec, weights: StrengthWeights
) -> None:
    if state.history or not state.elites:
        raise ValueError("resume generation zero must hold only initial parents")
    if tuple(candidate.genome for candidate in state.elites) != (
        state.identity.initial_genomes
    ):
        raise ValueError("resume initial parents differ from initial genomes")
    for candidate in state.elites:
        _validate_resume_candidate(candidate, codec, weights, stage="initial", seeds=())


def _validate_resume_candidate(
    candidate: CandidateEvaluation,
    codec: GenomeCodec,
    weights: StrengthWeights,
    *,
    stage: str,
    seeds: tuple[int, ...],
) -> None:
    try:
        decoded = codec.decode(candidate.genome)
    except (TypeError, ValueError) as error:
        raise ValueError("resume candidate genome/config is invalid") from error
    if decoded != candidate.config:
        raise ValueError("resume candidate genome/config do not match")
    if candidate.stage != stage:
        raise ValueError(f"resume candidate differs from {stage} stage")
    if candidate.seeds != seeds:
        raise ValueError(f"resume candidate differs from {stage} seeds")
    if stage == "initial":
        _validate_initial_candidate_telemetry(candidate)
        return
    _validate_evaluated_candidate(candidate, weights, seeds)


def _validate_initial_candidate_telemetry(candidate: CandidateEvaluation) -> None:
    if (
        candidate.fitness is not None
        or candidate.matchup_rates
        or candidate.matchup_games
        or candidate.failures
        or candidate.paired_normalized_margin != 0.0
    ):
        raise ValueError("resume initial parent contains evaluation telemetry")


def _validate_evaluated_candidate(
    candidate: CandidateEvaluation,
    weights: StrengthWeights,
    seeds: tuple[int, ...],
) -> None:
    if set(candidate.matchup_rates) != set(weights.values):
        raise ValueError("resume candidate matchup names differ from strength weights")
    if set(candidate.matchup_games) != set(weights.values):
        raise ValueError(
            "resume candidate game-count names differ from strength weights"
        )
    expected_games = 2 * len(seeds)
    for name, games in candidate.matchup_games.items():
        if type(games) is not int or games < 0 or games > expected_games:
            raise ValueError(f"resume candidate {name} game count is invalid")
        failed = any(failure.startswith(f"{name}:") for failure in candidate.failures)
        if not failed and games != expected_games:
            raise ValueError(
                f"resume candidate {name} game count must be {expected_games}"
            )
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
    if not math.isfinite(candidate.paired_normalized_margin):
        raise ValueError("resume candidate normalized margin must be finite")


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
        "certified": identity.certified,
        "manifest_sha256": identity.manifest_sha256,
        "engine": identity.engine,
        "frontier_seeds": list(identity.frontier_seeds),
        "screening_seeds": list(identity.screening_seeds),
        "development_seeds": list(identity.development_seeds),
        "promotion_seeds": list(identity.promotion_seeds),
        "genome_schema_sha256": identity.genome_schema_sha256,
        "league_identity": [list(item) for item in identity.league_identity],
        "league_snapshots": [list(item) for item in identity.league_snapshots],
        "strength_weights": [list(item) for item in identity.strength_weights],
        "initial_genomes": [list(genome) for genome in identity.initial_genomes],
        "evolution_config": asdict(identity.evolution_config),
    }


def _identity_from_payload(value: object) -> SearchIdentity:
    payload = _dictionary(
        value,
        {
            "certified",
            "manifest_sha256",
            "engine",
            "frontier_seeds",
            "screening_seeds",
            "development_seeds",
            "promotion_seeds",
            "genome_schema_sha256",
            "league_identity",
            "league_snapshots",
            "strength_weights",
            "initial_genomes",
            "evolution_config",
        },
        "search identity",
    )
    certified = payload["certified"]
    if type(certified) is not bool:
        raise TypeError("certified must be a boolean")
    manifest_value = payload["manifest_sha256"]
    manifest_sha256 = (
        None if manifest_value is None else _string(manifest_value, "manifest sha256")
    )
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
    league_snapshots = tuple(
        (
            _string_row(item, 4, "league snapshot")[0],
            _string_row(item, 4, "league snapshot")[1],
            _string_row(item, 4, "league snapshot")[2],
            _sha256_string(
                _string_row(item, 4, "league snapshot")[3],
                "league snapshot sha256",
            ),
        )
        for item in _list(payload["league_snapshots"], "league snapshots")
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
            "artifact_mode",
            "manifest_sha256",
            "population",
            "elites",
            "generations",
            "mutation_sigma",
            "seed",
            "workers",
            "engine",
        },
        "evolution config",
    )
    config = EvolutionConfig(**cast(dict[str, Any], config_payload))
    identity = SearchIdentity(
        certified=certified,
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
        league_snapshots=league_snapshots,
        strength_weights=tuple(weights.values.items()),
        initial_genomes=tuple(
            tuple(
                _finite_float(item, "initial genome value")
                for item in _list(genome, "initial genome")
            )
            for genome in _list(payload["initial_genomes"], "initial genomes")
        ),
        evolution_config=config,
    )
    _validate_identity_cross_fields(identity)
    return identity


def _candidate_payload(candidate: CandidateEvaluation) -> dict[str, object]:
    fitness = candidate.fitness
    return {
        "genome": list(candidate.genome),
        "config": candidate.config.model_dump(mode="json"),
        "stage": candidate.stage,
        "seeds": list(candidate.seeds),
        "matchup_rates": dict(candidate.matchup_rates),
        "matchup_games": dict(candidate.matchup_games),
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
    }


def _candidate_from_payload(value: object) -> CandidateEvaluation:
    payload = _dictionary(
        value,
        {
            "genome",
            "config",
            "stage",
            "seeds",
            "matchup_rates",
            "matchup_games",
            "fitness",
            "paired_normalized_margin",
            "failures",
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
    games_payload = _dictionary_open(payload["matchup_games"], "matchup games")
    games = {
        _string(name, "matchup name"): _integer(value, "matchup games", minimum=0)
        for name, value in games_payload.items()
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
        stage=_string(payload["stage"], "candidate stage"),
        seeds=_integer_tuple(payload["seeds"], "candidate seeds"),
        matchup_rates=rates,
        matchup_games=games,
        fitness=fitness,
        paired_normalized_margin=_finite_float(
            payload["paired_normalized_margin"], "paired normalized margin"
        ),
        failures=failures,
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
        "parent_digest": record.parent_digest,
        "spawned_population": [list(genome) for genome in record.spawned_population],
        "survivor_genomes": [list(genome) for genome in record.survivor_genomes],
        "elites": [_candidate_payload(candidate) for candidate in record.elites],
        "rng_state": _random_state_payload(record.rng_state),
        "integrity_digest": record.integrity_digest,
    }


def _generation_from_payload(value: object) -> GenerationRecord:
    payload = _dictionary(
        value,
        {
            "generation",
            "screened",
            "developed",
            "parent_digest",
            "spawned_population",
            "survivor_genomes",
            "elites",
            "rng_state",
            "integrity_digest",
        },
        "generation record",
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
        parent_digest=_sha256_string(
            payload["parent_digest"], "generation parent digest"
        ),
        spawned_population=_genome_rows(
            payload["spawned_population"], "spawned population"
        ),
        survivor_genomes=_genome_rows(payload["survivor_genomes"], "survivor genomes"),
        elites=tuple(
            _candidate_from_payload(candidate)
            for candidate in _list(payload["elites"], "generation elites")
        ),
        rng_state=_random_state_from_payload(payload["rng_state"]),
        integrity_digest=_sha256_string(
            payload["integrity_digest"], "generation integrity digest"
        ),
    )


def _genome_rows(value: object, label: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            _finite_float(item, f"{label} value")
            for item in _list(row, f"{label} genome")
        )
        for row in _list(value, label)
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


def _sha256_string(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def _string_row(value: object, length: int, label: str) -> tuple[str, ...]:
    row = _list(value, label)
    if len(row) != length:
        raise ValueError(f"{label} rows must contain {length} strings")
    return tuple(_string(item, label) for item in row)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
