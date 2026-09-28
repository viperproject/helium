//! Per-block reaching-condition algebra for the method-body CFG linearizer.
//!
//! A block is lowered under its *reaching condition* — the disjunction of the
//! path conditions of its incoming edges. These helpers build that condition as
//! a `(PathConds, Val)` pair: boolean `Val`s are desugared to ternaries (the IR
//! has no `Not`/`And`/`Or`), constant-folded over `TRUE`/`FALSE`, and the OR of
//! edge cubes is minimized by adjacency so straight-line code keeps a trivial
//! `<>` guard. Also builds the phi (`ite`) environment merge at joins.

use crate::dhash::{HashMap, HashSet};

use lasso::Spur;

use crate::translate::sink::Sink;
use crate::vmir::{FALSE, PathConds, Polarity, PureInst, TRUE, Type, Val};

/// `!v`, constant-folded over `true`/`false`.
pub(crate) fn not_val(sink: &mut Sink, v: Val) -> Val {
    if v == TRUE {
        FALSE
    } else if v == FALSE {
        TRUE
    } else {
        sink.emit_pure(Type::Bool, PureInst::Ternary(v, FALSE, TRUE))
    }
}

/// `a && b` as `a ? b : false`, constant-folded.
pub(crate) fn and_val(sink: &mut Sink, a: Val, b: Val) -> Val {
    if a == TRUE {
        b
    } else if b == TRUE {
        a
    } else if a == FALSE || b == FALSE {
        FALSE
    } else {
        sink.emit_pure(Type::Bool, PureInst::Ternary(a, b, FALSE))
    }
}

/// `a || b` as `a ? true : b`, constant-folded.
fn or_val(sink: &mut Sink, a: Val, b: Val) -> Val {
    if a == FALSE {
        b
    } else if b == FALSE {
        a
    } else if a == TRUE || b == TRUE {
        TRUE
    } else {
        sink.emit_pure(Type::Bool, PureInst::Ternary(a, TRUE, b))
    }
}

/// Materialize a path condition into a single boolean `Val` (conjunction of its
/// literals; `true` for the empty pc).
fn reach_val_of(sink: &mut Sink, pc: &PathConds) -> Val {
    let mut acc = TRUE;
    for (v, pol) in &pc.conds {
        let lit = match pol {
            Polarity::Positive => v.clone(),
            Polarity::Negative => not_val(sink, v.clone()),
        };
        acc = and_val(sink, acc, lit);
    }
    acc
}

/// The reaching condition of a block, from the `pool` of incoming edge cubes
/// (each predecessor's reach DNF, with the taken branch literal already pushed
/// onto every cube). Returns `(dnf, pc, reach_val)`:
///
/// - `dnf` — the reach as a **minimized cube set**, kept so successors can pool
///   the raw cubes and reduce further. This is why a DNF is threaded instead of a
///   single pc: a chain-decoded `n`-way `match` reaches a join with unequal-length
///   cubes that [`merge_cubes`] cannot merge in isolation, but once a later join
///   also pools the `else` cube the partition completes and telescopes to `<>`.
///   Materializing the reach into one literal here destroys that structure.
/// - `pc` — the block's own lowering guard for `with_conds`. When the DNF is one
///   cube that cube *is* the conjunctive pc (a diamond → its prefix; a full
///   split → `<>`); otherwise the cubes are OR'd into a single materialized
///   literal (a conjunctive pc cannot express a genuine disjunction).
/// - `reach_val` — the materialized boolean, for edge/phi construction.
///
/// [`merge_cubes`] only applies the value-preserving adjacency law
/// (`P∧x ∨ P∧!x ⇒ P`), so the result is always exactly the block's reach
/// condition, hence sound.
pub(crate) fn block_reach(sink: &mut Sink, pool: &[PathConds]) -> (Vec<PathConds>, PathConds, Val) {
    if pool.is_empty() {
        // Unreachable (filtered out before lowering); keep it fully gated.
        let pc = PathConds {
            conds: vec![(FALSE, Polarity::Positive)],
        };
        return (vec![pc.clone()], pc, FALSE);
    }

    let mut cubes: Vec<PathConds> = Vec::new();
    for epc in pool {
        if !cubes.contains(epc) {
            cubes.push(epc.clone());
        }
    }
    merge_cubes(&mut cubes);

    if let [only] = cubes.as_slice() {
        let pc = only.clone();
        let rv = reach_val_of(sink, &pc);
        return (cubes, pc, rv);
    }
    let mut rv = FALSE;
    for cube in &cubes {
        let cv = reach_val_of(sink, cube);
        rv = or_val(sink, rv, cv);
    }
    let pc = PathConds {
        conds: vec![(rv.clone(), Polarity::Positive)],
    };
    (cubes, pc, rv)
}

/// Boolean cube minimization: while two cubes are *adjacent* (identical literals
/// except one variable at opposite polarity), replace the pair with the shared
/// sub-cube. Value-preserving (`P∧x ∨ P∧!x = P`), so the disjunction is unchanged
/// — it just shrinks. Only equal-length cubes merge, so the innermost differing
/// literal collapses first and cascades outward, giving the LIFO/innermost-first
/// reduction a structured nest (or a goto split re-covering a subcube) expects.
fn merge_cubes(cubes: &mut Vec<PathConds>) {
    loop {
        let mut found = None;
        'search: for i in 0..cubes.len() {
            for j in (i + 1)..cubes.len() {
                if let Some(m) = merge_adjacent(&cubes[i], &cubes[j]) {
                    found = Some((i, j, m));
                    break 'search;
                }
            }
        }
        let Some((i, j, m)) = found else { break };
        cubes.remove(j); // j > i, so remove it first to keep index `i` valid
        cubes.remove(i);
        if !cubes.contains(&m) {
            cubes.push(m);
        }
    }
}

/// Two cubes merge iff they share every literal except exactly one variable that
/// appears with opposite polarity; the result drops that variable.
fn merge_adjacent(a: &PathConds, b: &PathConds) -> Option<PathConds> {
    if a.conds.len() != b.conds.len() {
        return None;
    }
    let a_only: Vec<(Val, Polarity)> = a
        .conds
        .iter()
        .filter(|l| !b.conds.contains(l))
        .cloned()
        .collect();
    let b_only: Vec<(Val, Polarity)> = b
        .conds
        .iter()
        .filter(|l| !a.conds.contains(l))
        .cloned()
        .collect();
    if a_only.len() != 1 || b_only.len() != 1 {
        return None;
    }
    let (va, pa) = &a_only[0];
    let (vb, pb) = &b_only[0];
    if va != vb || pa == pb {
        return None;
    }
    let conds = a
        .conds
        .iter()
        .filter(|(v, p)| !(v == va && p == pa))
        .cloned()
        .collect();
    Some(PathConds { conds })
}

/// Phi-merge two predecessor environments under a binary join guard: `then_env`
/// is selected when `cond` holds, `els_env` otherwise (the **unguarded**
/// fall-through, so the select reads `cond ? then : els`). A variable that
/// agrees on both sides passes through unchanged; one defined on a single side
/// inherits from that side. Binary because the block IR normalises every join to
/// a chain of these (a diamond is one; an n-way merge nests them).
///
/// Equivalent to a two-edge `ite(cond, then_v, els_v)` phi; the block lowerer
/// folds it right-to-left over an n-ary merge's arms.
pub(crate) fn merge_two_envs(
    sink: &mut Sink,
    cond: Val,
    then_env: &HashMap<Spur, Val>,
    els_env: &HashMap<Spur, Val>,
    var_types: &HashMap<Spur, Type>,
) -> HashMap<Spur, Val> {
    let mut names: HashSet<Spur> = HashSet::default();
    names.extend(then_env.keys().copied());
    names.extend(els_env.keys().copied());
    let mut out: HashMap<Spur, Val> = HashMap::default();
    for name in names {
        let merged = match (then_env.get(&name), els_env.get(&name)) {
            (Some(t), Some(e)) if t == e => t.clone(),
            (Some(t), Some(e)) => {
                let ty = var_types.get(&name).cloned().unwrap_or(Type::Int);
                sink.emit_pure(ty, PureInst::Ternary(cond.clone(), t.clone(), e.clone()))
            }
            (Some(v), None) | (None, Some(v)) => v.clone(),
            (None, None) => continue,
        };
        out.insert(name, merged);
    }
    out
}
