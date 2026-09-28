//! Run the full pipeline (parse → typecheck → translate → verify) on a file.
//!
//! ## What a failure costs
//!
//! Only a *parse* failure is fatal to the whole file: with no AST there is
//! nothing to attribute an error to. Everything after that is reported per
//! **unit** — one method, function, predicate, field, `domain`, `adt`, macro or
//! `import`, as split by [`viper::units`].
//!
//! A unit is rejected when it mentions a construct this verifier does not
//! implement ([`viper::unsupported::scan_declaration`]), or when typechecking or
//! translation refuses it. Every unit that uses a name the rejected unit
//! provided is then reported as skipped, transitively: a method calling a
//! rejected method cannot be said to verify, and claiming otherwise is the
//! failure mode this design exists to rule out.
//!
//! Because a translation failure leaves an unfilled slot in the builder, the
//! typecheck/translate stages run in a loop: reject the named units, drop them
//! and their dependents, and lower again. The loop runs at most once per
//! rejected unit and terminates because every round removes at least one.

use crate::dhash::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::viper::{
    GlobalsCollector, IdentCollector, disambiguate_reporting, inline_macros,
    typecheck_program_reporting, units::Units, unsupported::scan_declaration, walk::AstWalkable,
};
use crate::{translate, verify, viper_parser, vmir};

#[derive(Debug)]
pub enum PipelineError {
    Io(std::io::Error),
    Parse(String),
    Typecheck(String),
    Translate(String),
    Analyze(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO: {e}"),
            Self::Parse(e) => write!(f, "parse: {e}"),
            Self::Typecheck(e) => write!(f, "typecheck: {e}"),
            Self::Translate(e) => write!(f, "translate: {e}"),
            Self::Analyze(e) => write!(f, "analyze: {e}"),
        }
    }
}

/// What became of one unit.
#[derive(Debug)]
pub enum MemberStatus {
    /// Every proof obligation in it was discharged.
    Verified,
    /// It was verified and an obligation was not discharged.
    Failed(verify::VerifyError),
    /// It uses a Viper construct this verifier does not implement, named here
    /// as it is written in the source. Never verified.
    Unsupported(String),
    /// Typecheck or translation refused it — an error in the program, not a
    /// missing feature.
    Rejected(String),
    /// Not verified because the named unit it depends on was not.
    Skipped { on: String },
}

impl MemberStatus {
    /// Whether this counts as a clean result. Anything else makes the run fail.
    pub fn is_ok(&self) -> bool {
        matches!(self, MemberStatus::Verified)
    }

    /// The tag this status is printed under.
    pub fn tag(&self) -> &'static str {
        match self {
            MemberStatus::Verified => "OK",
            MemberStatus::Failed(_) => "FAIL",
            MemberStatus::Unsupported(_) => "UNSUPPORTED",
            MemberStatus::Rejected(_) => "ERROR",
            MemberStatus::Skipped { .. } => "SKIP",
        }
    }
}

impl std::fmt::Display for MemberStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemberStatus::Verified => Ok(()),
            MemberStatus::Failed(e) => write!(f, "{e}"),
            MemberStatus::Unsupported(msg) => write!(f, "{msg}"),
            MemberStatus::Rejected(msg) => write!(f, "{msg}"),
            MemberStatus::Skipped { on } => write!(f, "depends on `{on}`, which was not verified"),
        }
    }
}

/// One reported row: the unit's name and what became of it.
pub type MemberResult = (String, MemberStatus);

/// Wall-clock duration of each pipeline phase, in execution order.
#[derive(Debug, Default)]
pub struct PhaseTimings {
    pub phases: Vec<(&'static str, Duration)>,
    pub total: Duration,
}

impl std::fmt::Display for PhaseTimings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (name, dur) in &self.phases {
            writeln!(f, "  {name:<12} {:>10.3?}", dur)?;
        }
        write!(f, "  {:<12} {:>10.3?}", "total", self.total)
    }
}

/// Run the full pipeline on a `.vpr` file. Returns one row per unit, or a
/// `PipelineError` if the file could not be parsed at all.
pub fn run_file(path: &Path) -> Result<Vec<MemberResult>, PipelineError> {
    run_file_timed(path).map(|(results, _, _, _)| results)
}

/// Like [`run_file`] but also returns the [`PhaseTimings`] for each phase and the
/// verifier cost metrics ([`verify::VerifyStats`]) for the run. Timings are for
/// the pass that reached the verifier; earlier passes discarded because a unit
/// was rejected are not counted.
pub fn run_file_timed(
    path: &Path,
) -> Result<
    (
        Vec<MemberResult>,
        PhaseTimings,
        Vec<(String, Duration)>,
        verify::VerifyStats,
    ),
    PipelineError,
> {
    let mut timings = PhaseTimings::default();
    let overall = Instant::now();

    // Time a phase, recording its duration under `name`. A repeated phase (the
    // lowering loop below runs again after dropping a rejected unit) overwrites
    // its earlier entry, so the reported timing is the pass that got through.
    macro_rules! phase {
        ($name:literal, $body:expr) => {{
            let start = Instant::now();
            let out = $body;
            let elapsed = start.elapsed();
            match timings.phases.iter_mut().find(|(n, _)| *n == $name) {
                Some(slot) => slot.1 = elapsed,
                None => timings.phases.push(($name, elapsed)),
            }
            out
        }};
    }

    let input = std::fs::read_to_string(path).map_err(PipelineError::Io)?;

    let parsed = phase!(
        "parse",
        viper_parser::vpr_program(&input).map_err(|e| PipelineError::Parse(e.to_string()))?
    );

    // ── Reject the units we cannot support, before anything else looks at them ──
    let units = Units::of(&parsed);
    let mut rejected: Vec<(usize, MemberStatus)> = Vec::new();
    for (idx, unit) in units.0.iter().enumerate() {
        if let Some(what) = unit
            .decls
            .iter()
            .find_map(|&d| scan_declaration(&parsed.0[d]))
        {
            rejected.push((
                idx,
                MemberStatus::Unsupported(format!("unsupported construct: {what}")),
            ));
        }
    }

    // The lowering loop: typecheck and translate what is left, rejecting any
    // unit those stages refuse and trying again without it. Dependents of a
    // rejected unit are dropped too — their names really are gone, so leaving
    // them in would produce a cascade of "undefined" errors that name the
    // wrong thing.
    let analyzed = loop {
        let dropped = dropped_units(&units, &rejected);
        let mut program = filtered(&parsed, &units, &dropped);

        let interner = phase!("idents", {
            let mut ident_collector = IdentCollector::default();
            program.walk_mut(&mut ident_collector);
            ident_collector.finalize()
        });

        let globals = phase!("globals", {
            let mut globals_collector = GlobalsCollector::new(&interner);
            program.walk(&mut globals_collector);
            globals_collector
                .finalize()
                .map_err(|e| PipelineError::Typecheck(format!("{e:?}")))?
        });

        let disambiguation_errors = phase!(
            "disambiguate",
            disambiguate_reporting(&mut program, &interner, &globals)
        );
        if !disambiguation_errors.is_empty() {
            if reject(
                &units,
                &mut rejected,
                disambiguation_errors
                    .iter()
                    .map(|(n, e)| (n.as_str(), MemberStatus::Rejected(e.to_string()))),
            ) {
                continue;
            }
            return Err(PipelineError::Typecheck(format!(
                "{disambiguation_errors:?}"
            )));
        }

        phase!(
            "macros",
            inline_macros(&mut program, &interner)
                .map_err(|e| PipelineError::Typecheck(format!("{e:?}")))?
        );

        let (typed, type_errors) = phase!(
            "typecheck",
            typecheck_program_reporting(&mut program, interner, &globals)
        );
        if !type_errors.is_empty() {
            if reject(
                &units,
                &mut rejected,
                type_errors.iter().map(named_type_error),
            ) {
                continue;
            }
            // Nothing could be attributed to a unit: report the file.
            return Err(PipelineError::Typecheck(format!("{type_errors:?}")));
        }

        let vmir = match phase!("translate", translate::translate_reporting(&typed)) {
            Ok(vmir) => vmir,
            Err(errs) => {
                if reject(
                    &units,
                    &mut rejected,
                    errs.iter().map(named_translation_error),
                ) {
                    continue;
                }
                return Err(PipelineError::Translate(format!("{errs:?}")));
            }
        };

        break phase!(
            "analyze",
            vmir::analyze(vmir).map_err(|e| PipelineError::Analyze(e.to_string()))?
        );
    };

    let (verified, member_times, stats) = {
        let start = Instant::now();
        let out = verify::verify_with_stats(&analyzed);
        let elapsed = start.elapsed();
        match timings.phases.iter_mut().find(|(n, _)| *n == "verify") {
            Some(slot) => slot.1 = elapsed,
            None => timings.phases.push(("verify", elapsed)),
        }
        out
    };

    // ── Rows ──
    // Rejected units first, in source order, then the units that were actually
    // verified. A unit that depends on a rejected one is reported as skipped.
    let mut rows: Vec<MemberResult> = Vec::new();
    let rejected_idx: HashSet<usize> = rejected.iter().map(|(idx, _)| *idx).collect();
    let skipped = units.dependents_of(&rejected_idx);
    let mut all: Vec<(usize, MemberStatus)> = rejected;
    all.extend(
        skipped
            .into_iter()
            .map(|(idx, on)| (idx, MemberStatus::Skipped { on })),
    );
    all.sort_by_key(|(idx, _)| *idx);
    rows.extend(
        all.into_iter()
            .map(|(idx, status)| (units.0[idx].name.clone(), status)),
    );
    rows.extend(verified.into_iter().map(|(name, outcome)| {
        let status = match outcome {
            Ok(()) => MemberStatus::Verified,
            Err(e) => MemberStatus::Failed(e),
        };
        (name, status)
    }));

    timings.total = overall.elapsed();
    Ok((rows, timings, member_times, stats))
}

/// The units that must not reach the verifier: those rejected outright and
/// everything that depends on one.
fn dropped_units(units: &Units, rejected: &[(usize, MemberStatus)]) -> HashSet<usize> {
    let direct: HashSet<usize> = rejected.iter().map(|(idx, _)| *idx).collect();
    let mut all = direct.clone();
    all.extend(units.dependents_of(&direct).into_iter().map(|(idx, _)| idx));
    all
}

/// The parsed program with every declaration of a dropped unit removed.
fn filtered(
    parsed: &crate::viper::Program,
    units: &Units,
    dropped_units: &HashSet<usize>,
) -> crate::viper::Program {
    let dropped: HashSet<usize> = dropped_units
        .iter()
        .flat_map(|&u| units.0[u].decls.iter().copied())
        .collect();
    crate::viper::Program(
        parsed
            .0
            .iter()
            .enumerate()
            .filter(|(i, _)| !dropped.contains(i))
            .map(|(_, d)| d.clone())
            .collect(),
    )
}

/// Add newly named failures to `rejected`, returning whether any were
/// added — i.e. if lowering should be retried without them. A failure naming a
/// unit that is already rejected adds nothing, which is what stops the loop.
fn reject<'a>(
    units: &Units,
    rejected: &mut Vec<(usize, MemberStatus)>,
    errors: impl Iterator<Item = (&'a str, MemberStatus)>,
) -> bool {
    let mut added = false;
    for (name, status) in errors {
        let Some(idx) = units.index_of(name) else {
            continue;
        };
        if rejected.iter().any(|(i, _)| *i == idx) {
            continue;
        }
        rejected.push((idx, status));
        added = true;
    }
    added
}

/// A typecheck error, split into "we do not implement this" and "this program
/// is wrong" — the distinction `TypeError::is_unsupported` exists to draw.
fn named_type_error(e: &(String, crate::viper::TypeError)) -> (&str, MemberStatus) {
    let (name, err) = e;
    let status = if err.is_unsupported() {
        MemberStatus::Unsupported(err.to_string())
    } else {
        MemberStatus::Rejected(err.to_string())
    };
    (name.as_str(), status)
}

/// As [`named_type_error`], for translation.
fn named_translation_error(e: &(String, translate::TranslationError)) -> (&str, MemberStatus) {
    let (name, err) = e;
    let status = if err.is_unsupported() {
        MemberStatus::Unsupported(err.to_string())
    } else {
        MemberStatus::Rejected(err.to_string())
    };
    (name.as_str(), status)
}
