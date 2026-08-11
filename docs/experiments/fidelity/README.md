# Simulator fidelity records

No acceptance record is current yet. The 2026-08-10 preflight establishes exact
codec parity and mutation sensitivity against `kaggle-environments==1.32.6`,
but it is deliberately marked unaccepted until the 10,000-episode campaign and
CUDA throughput matrix have run.

The installed reference contains 107 `if` nodes across the twelve functions
named by the simulator spec. The spec's prose says 63; its ground-truth rule
makes the parsed 107-node manifest authoritative. The CI tripwire pins that
manifest as well as both installed reference-file hashes.
