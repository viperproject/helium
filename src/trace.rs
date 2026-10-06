//! Opt-in, structured tracing of the verifier's work, for diagnosing where a
//! run's time goes and why an obligation failed.
//!
//! A trace session ([`with_trace`]) enables a set of [`Category`]s for the code
//! it runs, on the current thread. Each event of an enabled category is written
//! as one JSON object per line:
//!
//! ```text
//! {"cat":"block","ev":"end","member":"m","block":3,"nodes_before":120,"nodes":131,...}
//! ```
//!
//! `cat` and `ev` name the event, `member` is the verification unit it happened
//! in (absent outside one), and the remaining fields are the event's own, in a
//! fixed order. A field whose value is `null` is left out.
//!
//! **Deterministic.** Events are written in execution order and their fields
//! are functions of the input, so two runs of the same binary on the same input
//! write the same bytes. Only the `fail` term dumps name e-classes (`@n`); every
//! other event can be diffed across builds. Wall-clock fields (`secs`, `*_ns`)
//! are the exception and appear only when the `time` category is enabled.
//!
//! **Free when off.** An event site tests one bit of a thread-local mask and
//! computes its fields only when the bit is set. The always-on, run-wide
//! aggregates stay in [`VerifyStats`](crate::verify::VerifyStats); the trace
//! says where inside the run they were spent.
//!
//! **Adding an event.** `trace_event!(Block, "end", block = id, nodes = n)`,
//! with any field value `Json` converts from. Totals that would be too many
//! events (one per rule application, per quantifier match) go through
//! `trace_tally!`, which sums them per verification unit and writes one event
//! per key when the unit ends. A new category is a [`Category`] variant plus its
//! row in [`Category::ALL`].

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use crate::json::Json;

/// What a trace can report. Selected by name on the `verify` command line
/// (`--trace=block,sat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    Member,
    Block,
    Sat,
    Rule,
    Quant,
    Scratch,
    Heap,
    Fail,
    Time,
}

impl Category {
    /// Every category, with its name and what it reports, indexed by the
    /// variant.
    pub const ALL: [(Category, &'static str, &'static str); 9] = [
        (
            Category::Member,
            "member",
            "each verification unit: begin; end with its verdict and work counters",
        ),
        (
            Category::Block,
            "block",
            "each method CFG block: begin; end with ground growth and work counters",
        ),
        (
            Category::Sat,
            "sat",
            "each rule run: graph, rule set, iterations, stop reason, size before/after",
        ),
        (
            Category::Rule,
            "rule",
            "per unit: applications of each rule",
        ),
        (
            Category::Quant,
            "quant",
            "quantifier recipes; per unit: visits, trigger matches and new instances of each",
        ),
        (
            Category::Scratch,
            "scratch",
            "block scratch builds, and each probe-tier obligation proved against one",
        ),
        (
            Category::Heap,
            "heap",
            "per unit: conditional permission trees built by join merges, and zero leaves",
        ),
        (
            Category::Fail,
            "fail",
            "the terms at a failed obligation, insufficient permission or framing miss",
        ),
        (
            Category::Time,
            "time",
            "adds wall-clock fields to the events above (output is then not deterministic)",
        ),
    ];

    pub fn name(self) -> &'static str {
        Self::ALL[self as usize].1
    }

    const fn bit(self) -> u32 {
        1 << self as u32
    }
}

/// A set of [`Category`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Categories(u32);

impl Categories {
    pub const NONE: Self = Self(0);

    pub fn contains(self, c: Category) -> bool {
        self.0 & c.bit() != 0
    }

    pub fn with(self, c: Category) -> Self {
        Self(self.0 | c.bit())
    }

    /// A comma-separated list of category names. `all` is every category but
    /// `time`, which has to be asked for by name since it makes the output
    /// non-deterministic.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut set = Self::NONE;
        for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if name == "all" {
                for (c, _, _) in Category::ALL {
                    if c != Category::Time {
                        set = set.with(c);
                    }
                }
                continue;
            }
            match Category::ALL.iter().find(|(_, n, _)| *n == name) {
                Some((c, _, _)) => set = set.with(*c),
                None => return Err(format!("unknown trace category `{name}`")),
            }
        }
        Ok(set)
    }
}

/// How a trace session is set up.
pub struct TraceConfig {
    pub categories: Categories,
    /// Where event lines go.
    pub out: Box<dyn Write>,
    /// If set, write Graphviz snapshots of the e-graph and heap, one PDF per
    /// verified member, into this directory (see `verify::viz`).
    pub viz_dir: Option<PathBuf>,
}

/// A summed per-unit total, keyed by its event and key fields.
type TallyKey = (Category, &'static str, Vec<(&'static str, Key)>);

struct Session {
    out: Box<dyn Write>,
    viz_dir: Option<PathBuf>,
    member: Option<String>,
    tallies: BTreeMap<TallyKey, Vec<(&'static str, u64)>>,
}

impl Session {
    fn write(&mut self, cat: Category, ev: &'static str, fields: Vec<(&'static str, Json)>) {
        let mut obj: Vec<(&str, Json)> = vec![("cat", cat.name().into()), ("ev", ev.into())];
        if let Some(m) = &self.member {
            obj.push(("member", m.as_str().into()));
        }
        obj.extend(fields.into_iter().filter(|(_, v)| *v != Json::Null));
        // One write per line, so a trace streamed to a terminal or a file is
        // whole up to the last event even if the run never finishes.
        let line = format!("{}\n", Json::obj(obj));
        // Best-effort: a diagnostic that cannot be written must not fail the run.
        let _ = self.out.write_all(line.as_bytes());
    }

    fn flush_tallies(&mut self) {
        for ((cat, ev, key), values) in std::mem::take(&mut self.tallies) {
            let fields = key
                .into_iter()
                .map(|(k, v)| (k, Json::from(v)))
                .chain(values.into_iter().map(|(k, v)| (k, Json::from(v))))
                .collect();
            self.write(cat, ev, fields);
        }
    }
}

thread_local! {
    /// The enabled categories, read by every event site. Kept apart from the
    /// session so the off path is one `Cell` load.
    static MASK: Cell<u32> = const { Cell::new(0) };
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

/// Whether events of `c` are being recorded.
#[inline]
pub fn enabled(c: Category) -> bool {
    MASK.with(|m| m.get() & c.bit() != 0)
}

/// Run `f` with tracing set up as `config` on this thread. The previous setup
/// (usually none) is restored afterwards, also if `f` panics.
pub fn with_trace<T>(config: TraceConfig, f: impl FnOnce() -> T) -> T {
    struct Restore(u32, Option<Session>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SESSION.with(|s| {
                if let Ok(mut s) = s.try_borrow_mut() {
                    if let Some(s) = s.as_mut() {
                        s.flush_tallies();
                        let _ = s.out.flush();
                    }
                    *s = self.1.take();
                }
            });
            MASK.with(|m| m.set(self.0));
        }
    }
    let session = Session {
        out: config.out,
        viz_dir: config.viz_dir,
        member: None,
        tallies: BTreeMap::new(),
    };
    let prev = SESSION.with(|s| s.replace(Some(session)));
    let _restore = Restore(MASK.with(|m| m.replace(config.categories.0)), prev);
    f()
}

/// Run `f` with `categories` traced, returning the trace text alongside its
/// result. For tests and in-process tools.
pub fn capture<T>(categories: Categories, f: impl FnOnce() -> T) -> (T, String) {
    #[derive(Clone, Default)]
    struct Buf(Rc<RefCell<Vec<u8>>>);
    impl Write for Buf {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let buf = Buf::default();
    let config = TraceConfig {
        categories,
        out: Box::new(buf.clone()),
        viz_dir: None,
    };
    let out = with_trace(config, f);
    let text = String::from_utf8(buf.0.take()).expect("trace lines are UTF-8");
    (out, text)
}

/// Write one event. Prefer `trace_event!`, which skips building the fields when
/// the category is off.
pub fn emit(cat: Category, ev: &'static str, fields: Vec<(&'static str, Json)>) {
    SESSION.with(|s| {
        if let Some(s) = s.borrow_mut().as_mut() {
            s.write(cat, ev, fields);
        }
    });
}

/// A tally's key value.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Key {
    Num(u64),
    Str(String),
}

impl From<u64> for Key {
    fn from(v: u64) -> Self {
        Key::Num(v)
    }
}

impl From<usize> for Key {
    fn from(v: usize) -> Self {
        Key::Num(v as u64)
    }
}

impl From<u32> for Key {
    fn from(v: u32) -> Self {
        Key::Num(v.into())
    }
}

impl From<&str> for Key {
    fn from(v: &str) -> Self {
        Key::Str(v.to_string())
    }
}

impl From<Key> for Json {
    fn from(k: Key) -> Self {
        match k {
            Key::Num(n) => Json::UInt(n),
            Key::Str(s) => Json::Str(s),
        }
    }
}

/// Add `values` to the current unit's total for `(cat, ev, key)`. Prefer
/// `trace_tally!`, which skips this when the category is off.
pub fn tally(
    cat: Category,
    ev: &'static str,
    key: Vec<(&'static str, Key)>,
    values: &[(&'static str, u64)],
) {
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
        let Some(s) = s.as_mut() else {
            return;
        };
        let sums = s.tallies.entry((cat, ev, key)).or_default();
        for &(name, v) in values {
            match sums.iter_mut().find(|(n, _)| *n == name) {
                Some((_, sum)) => *sum += v,
                None => sums.push((name, v)),
            }
        }
    });
}

/// Attribute the events `f` emits to the verification unit `name`, and write
/// the unit's tallies when it returns.
pub fn in_member<T>(name: &str, f: impl FnOnce() -> T) -> T {
    if MASK.with(Cell::get) == 0 {
        return f();
    }
    let set = |member: Option<String>| {
        SESSION.with(|s| {
            if let Some(s) = s.borrow_mut().as_mut() {
                s.flush_tallies();
                s.member = member;
            }
        })
    };
    set(Some(name.to_string()));
    let out = f();
    set(None);
    out
}

/// Start a wall-clock measurement for a `time` field: `None` when `time` is
/// off, so the clock is not even read.
pub fn clock() -> Option<Instant> {
    enabled(Category::Time).then(Instant::now)
}

/// Seconds since [`clock`], or `None` (so the field is left out) when `time` is
/// off.
pub fn secs(start: Option<Instant>) -> Option<f64> {
    start.map(|t| t.elapsed().as_secs_f64())
}

/// The directory Graphviz snapshots go to, if the session asked for them.
pub fn viz_dir() -> Option<PathBuf> {
    SESSION.with(|s| s.borrow().as_ref()?.viz_dir.clone())
}

/// Write one trace event if its category is enabled; the field expressions are
/// evaluated only then.
///
/// `trace_event!(Block, "end", block = id, nodes = n)`
macro_rules! trace_event {
    ($cat:ident, $ev:literal $(, $k:ident = $v:expr)* $(,)?) => {
        if $crate::trace::enabled($crate::trace::Category::$cat) {
            $crate::trace::emit(
                $crate::trace::Category::$cat,
                $ev,
                vec![$((stringify!($k), $crate::json::Json::from($v))),*],
            );
        }
    };
}

/// Add to a per-unit total if its category is enabled: the bracketed fields
/// are the key, the rest are summed. Written as one event per key when the unit
/// ends, in key order.
///
/// `trace_tally!(Quant, "inst", [recipe = id], matches = m, new = n)`
macro_rules! trace_tally {
    ($cat:ident, $ev:literal, [$($k:ident = $kv:expr),* $(,)?] $(, $v:ident = $vv:expr)+ $(,)?) => {
        if $crate::trace::enabled($crate::trace::Category::$cat) {
            $crate::trace::tally(
                $crate::trace::Category::$cat,
                $ev,
                vec![$((stringify!($k), $crate::trace::Key::from($kv))),*],
                &[$((stringify!($v), ($vv) as u64)),+],
            );
        }
    };
}

pub(crate) use {trace_event, trace_tally};

/// For a path that names the trace output: `-` is stderr.
pub fn open_output(path: &str) -> std::io::Result<Box<dyn Write>> {
    if path == "-" {
        return Ok(Box::new(std::io::stderr()));
    }
    let file = std::fs::File::create(Path::new(path))?;
    Ok(Box::new(std::io::LineWriter::new(file)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cats(spec: &str) -> Categories {
        Categories::parse(spec).unwrap()
    }

    #[test]
    fn category_table_is_indexed_by_variant() {
        for (i, (c, _, _)) in Category::ALL.iter().enumerate() {
            assert_eq!(*c as usize, i);
        }
    }

    #[test]
    fn parses_category_lists() {
        assert_eq!(cats(""), Categories::NONE);
        let s = cats("block, sat");
        assert!(s.contains(Category::Block) && s.contains(Category::Sat));
        assert!(!s.contains(Category::Member));
        let all = cats("all");
        assert!(
            Category::ALL
                .iter()
                .all(|(c, _, _)| all.contains(*c) == (*c != Category::Time))
        );
        assert!(cats("all,time").contains(Category::Time));
        assert!(Categories::parse("block,bogus").is_err());
    }

    #[test]
    fn writes_only_enabled_categories_as_json_lines() {
        let ((), text) = capture(cats("block"), || {
            trace_event!(
                Block,
                "end",
                block = 3usize,
                dead = false,
                gone = None::<u64>
            );
            trace_event!(Sat, "run", iterations = 1u64);
        });
        assert_eq!(
            text,
            "{\"cat\":\"block\",\"ev\":\"end\",\"block\":3,\"dead\":false}\n"
        );
    }

    #[test]
    fn off_evaluates_no_fields() {
        let mut evaluated = false;
        let ((), text) = capture(cats("sat"), || {
            trace_event!(
                Block,
                "end",
                x = {
                    evaluated = true;
                    1u64
                }
            );
        });
        assert!(text.is_empty());
        assert!(!evaluated);
        // And no session at all: nothing recorded, nothing evaluated.
        trace_event!(
            Block,
            "end",
            x = {
                evaluated = true;
                1u64
            }
        );
        assert!(!evaluated);
    }

    #[test]
    fn tallies_are_summed_per_member_and_written_in_key_order() {
        let ((), text) = capture(cats("quant"), || {
            in_member("m", || {
                trace_tally!(
                    Quant,
                    "inst",
                    [recipe = 10usize],
                    matches = 2u64,
                    new = 1u64
                );
                trace_tally!(Quant, "inst", [recipe = 2usize], matches = 1u64, new = 1u64);
                trace_tally!(
                    Quant,
                    "inst",
                    [recipe = 10usize],
                    matches = 3u64,
                    new = 0u64
                );
                trace_event!(Quant, "recipe", recipe = 2usize);
            });
            trace_event!(Quant, "after");
        });
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines,
            [
                "{\"cat\":\"quant\",\"ev\":\"recipe\",\"member\":\"m\",\"recipe\":2}",
                "{\"cat\":\"quant\",\"ev\":\"inst\",\"member\":\"m\",\"recipe\":2,\"matches\":1,\"new\":1}",
                "{\"cat\":\"quant\",\"ev\":\"inst\",\"member\":\"m\",\"recipe\":10,\"matches\":5,\"new\":1}",
                "{\"cat\":\"quant\",\"ev\":\"after\"}",
            ]
        );
    }

    #[test]
    fn sessions_nest_and_restore() {
        let ((), outer) = capture(cats("block"), || {
            let ((), inner) = capture(cats("sat"), || {
                trace_event!(Block, "x");
                trace_event!(Sat, "y");
            });
            assert_eq!(inner, "{\"cat\":\"sat\",\"ev\":\"y\"}\n");
            trace_event!(Block, "z");
            trace_event!(Sat, "w");
        });
        assert_eq!(outer, "{\"cat\":\"block\",\"ev\":\"z\"}\n");
        assert!(!enabled(Category::Block));
    }

    #[test]
    fn time_fields_only_when_time_is_enabled() {
        let ((), text) = capture(cats("block"), || {
            let t = clock();
            trace_event!(Block, "end", secs = secs(t));
        });
        assert_eq!(text, "{\"cat\":\"block\",\"ev\":\"end\"}\n");
        let ((), text) = capture(cats("block,time"), || {
            let t = clock();
            trace_event!(Block, "end", secs = secs(t));
        });
        assert!(text.starts_with("{\"cat\":\"block\",\"ev\":\"end\",\"secs\":"));
    }

    /// Case files with blocks, quantifiers, join merges and a failure.
    const CASES: [&str; 3] = [
        "passing/interplay/param_point_conversion_cycle.vpr",
        "passing/predicates/domain_cons_tower_inversion.vpr",
        "failing/insufficient_perm.vpr",
    ];

    fn run(case: &str) -> (Vec<String>, crate::verify::VerifyStats) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/cases")
            .join(case);
        let (rows, _, _, stats) = crate::pipeline::run_file_timed(&path).expect("pipeline");
        let rows = rows
            .iter()
            .map(|(n, s)| format!("{n}: {} {s}", s.tag()))
            .collect();
        (rows, stats)
    }

    /// Tracing observes and never steers: with every category on, the verdicts
    /// and every deterministic work counter are exactly those of an untraced run.
    #[test]
    fn tracing_changes_no_verdict_and_no_work() {
        for case in CASES {
            let plain = run(case);
            let (traced, text) = capture(cats("all,time"), || run(case));
            assert!(!text.is_empty());
            assert_eq!(plain, traced, "{case}");
        }
    }

    /// Without `time` a trace is a function of the input: byte-identical across
    /// runs, one well-formed event per line, only the selected categories.
    #[test]
    fn trace_is_deterministic_and_selective() {
        for case in CASES {
            let (_, a) = capture(cats("all"), || run(case));
            let (_, b) = capture(cats("all"), || run(case));
            assert_eq!(a, b, "{case}");
            let begins = a.matches(r#"{"cat":"member","ev":"begin""#).count();
            let ends = a.matches(r#"{"cat":"member","ev":"end""#).count();
            assert!(begins > 0 && begins == ends, "{case}");
            assert!(
                a.lines()
                    .all(|l| l.starts_with(r#"{"cat":""#) && l.ends_with('}'))
            );
            let (_, only) = capture(cats("block,sat"), || run(case));
            assert!(
                only.lines().all(|l| l.starts_with(r#"{"cat":"block""#)
                    || l.starts_with(r#"{"cat":"sat""#)),
                "{case}"
            );
        }
    }

    #[test]
    fn failure_trace_names_the_terms() {
        let (_, text) = capture(cats("fail"), || run(CASES[2]));
        let event = r#"{"cat":"fail","ev":"insufficient","member":"client""#;
        assert!(text.lines().any(|l| l.starts_with(event)), "{text}");
    }
}
