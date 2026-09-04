//! Command-line front end.
//!
//! `run` replays an action tape read from stdin and writes one JSON line per
//! state (the reset state first), which is how the Python differential test
//! compares this engine against the reference. `bench` measures raw
//! throughput with the built-in agents.

use std::io::{self, BufWriter, Read, Write};
use std::time::Instant;

use serde_json::{json, Value};

use kaggriculture_engine::agents::{random_agent, starter_agent};
use kaggriculture_engine::{Action, Config, Engine, PyRandom};

fn usage() -> ! {
    eprintln!(
        "usage:\n  kaggriculture-engine run [--no-observations] < tape.json\n  \
         kaggriculture-engine bench [--episodes N] [--seed S]\n  \
         kaggriculture-engine render [--seed S] [--steps N]\n\n\
         tape.json: {{\"configuration\": {{...}}, \"actions\": [[player0, player1], ...]}}"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("render") => render(&args[1..]),
        _ => usage(),
    }
}

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn run(args: &[String]) {
    let emit_observations = !args.iter().any(|arg| arg == "--no-observations");
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("read stdin");
    let tape: Value = serde_json::from_str(&input).unwrap_or_else(|error| {
        eprintln!("invalid tape JSON: {error}");
        std::process::exit(1);
    });
    let config = tape
        .get("configuration")
        .map(|value| {
            Config::from_json(value).unwrap_or_else(|error| {
                eprintln!("invalid configuration: {error}");
                std::process::exit(1);
            })
        })
        .unwrap_or_default();
    let players = tape.get("players").and_then(Value::as_u64).unwrap_or(2) as usize;
    let turns: Vec<Vec<Action>> = tape
        .get("actions")
        .and_then(Value::as_array)
        .map(|turns| {
            turns
                .iter()
                .map(|turn| {
                    turn.as_array()
                        .map(|players| players.iter().map(Action::from_json).collect())
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default();

    let mut engine = Engine::with_players(config, players);
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let emit = |out: &mut BufWriter<_>, engine: &Engine| {
        let observations: Vec<Value> = if emit_observations {
            (0..engine.players())
                .map(|player| engine.observation(player))
                .collect()
        } else {
            Vec::new()
        };
        let line = json!({
            "seed": engine.seed(),
            "step": engine.state().step,
            "done": engine.done(),
            "rewards": engine.rewards(),
            "observations": observations,
        });
        serde_json::to_writer(&mut *out, &line).expect("write");
        out.write_all(b"\n").expect("write");
    };
    emit(&mut out, &engine);
    for actions in &turns {
        if engine.done() {
            break;
        }
        engine.step(actions);
        emit(&mut out, &engine);
    }
    out.flush().expect("flush");
}

fn bench(args: &[String]) {
    let episodes: u64 = flag_value(args, "--episodes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let seed: i64 = flag_value(args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let started = Instant::now();
    let mut steps = 0u64;
    let mut checksum = 0.0f64;
    for episode in 0..episodes {
        let mut engine = Engine::new(Config::with_seed(seed + episode as i64));
        let mut rng = PyRandom::new((seed + episode as i64) as i128);
        while !engine.done() {
            let actions = [
                starter_agent(engine.state(), 0),
                random_agent(engine.state(), 1, &mut rng),
            ];
            engine.step(&actions);
            steps += 1;
        }
        checksum += engine.rewards().iter().sum::<f64>();
    }
    let elapsed = started.elapsed();
    let per_step = elapsed.as_secs_f64() / steps as f64;
    println!(
        "{episodes} episodes, {steps} steps in {:.3}s: {:.2} us/step, {:.0} steps/s (bank checksum {checksum})",
        elapsed.as_secs_f64(),
        per_step * 1e6,
        1.0 / per_step
    );
}

fn render(args: &[String]) {
    let seed: i64 = flag_value(args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let steps: usize = flag_value(args, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let mut engine = Engine::new(Config::with_seed(seed));
    let mut rng = PyRandom::new(seed as i128);
    for _ in 0..steps {
        if engine.done() {
            break;
        }
        let actions = [
            starter_agent(engine.state(), 0),
            random_agent(engine.state(), 1, &mut rng),
        ];
        engine.step(&actions);
    }
    print!("{}", kaggriculture_engine::render(engine.state()));
}
