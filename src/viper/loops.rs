//! Natural-loop detection over a block CFG.
//!
//! A loop is *any* back edge — an edge `n → h` where `h` dominates `n`. Source
//! `while` is only a special case: Prusti emits goto CFGs with no `while` at
//! all, carrying invariants on the loop head's `label`, so back-edge detection
//! is the primary path rather than a fallback. This mirrors Silver's
//! `LoopDetector`, which likewise runs dominator-based detection over the goto
//! CFG and only then classifies edges.
//!
//! The analysis is generic over the node type so it can be unit-tested against
//! a bare `DiGraphMap` without building a whole method body.
//!
//! # What the caller gets
//!
//! - [`Loops::back_edges`] — the edges to cut. Removing them leaves a DAG, which
//!   keeps the topological block order (and the `preds precede` invariant the VMIR
//!   lowering and verifier both assume) valid.
//! - [`Loops::loops`] — one [`Loop`] per head, with its natural-loop body.
//! - [`Loops::classify`] — per edge: which loops it leaves and which it enters,
//!   innermost-first and outermost-first respectively. An edge leaving two loops
//!   at once (a `break` out of a nested loop) must have each loop's frame
//!   restored in order, so the order is part of the contract, not incidental.

use crate::dhash::{HashMap, HashSet};
use std::hash::Hash;

use petgraph::algo::{dominators, tarjan_scc};
use petgraph::prelude::DiGraphMap;

/// A natural loop: a header plus every block that can reach one of the header's
/// back-edge tails without passing through the header.
#[derive(Debug, Clone)]
pub struct Loop<N> {
    /// The single entry point. Natural loops are single-entry by construction —
    /// a multi-entry cycle is irreducible and rejected by [`Loops::detect`].
    pub head: N,
    /// Tails of the back edges targeting `head`. A `continue` inside a
    /// conditional gives more than one; each becomes its own exhale leaf.
    pub back_edge_tails: Vec<N>,
    /// Every block in the loop, including `head`.
    pub body: HashSet<N>,
}

/// How an edge relates to the loop nesting structure.
///
/// An edge can both leave and enter loops (a `goto` from inside one loop into
/// another). Both are recorded; [`EdgeClass::is_exit`] follows Silver's
/// precedence, where leaving takes priority over entering.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EdgeClass {
    /// Loop indices this edge leaves, **innermost first**. Frames must be
    /// unioned back in this order.
    pub exits: Vec<usize>,
    /// Loop indices this edge enters, **outermost first**.
    pub entries: Vec<usize>,
    /// This edge is a back edge to `loops[head_of].head`.
    pub back: Option<usize>,
}

impl EdgeClass {
    /// A normal intra-block edge: neither a back edge nor crossing any loop
    /// boundary.
    pub fn is_normal(&self) -> bool {
        self.exits.is_empty() && self.entries.is_empty() && self.back.is_none()
    }

    /// Leaving at least one loop. Silver gives this precedence over entering.
    pub fn is_exit(&self) -> bool {
        !self.exits.is_empty()
    }
}

/// Why a CFG cannot be handled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopError<N> {
    /// A cycle with more than one entry point from outside. Rust MIR from
    /// `loop`/`while`/`for` is always reducible, so this only arises from
    /// hand-written `goto` spaghetti. Carries the entry blocks.
    Irreducible(Vec<N>),
}

/// The loop structure of a CFG.
#[derive(Debug, Clone)]
pub struct Loops<N> {
    /// One entry per loop header, ordered **outermost first** (by body size
    /// descending), so a prefix of the list is always a valid nesting order.
    pub loops: Vec<Loop<N>>,
    /// Every back edge, as `(tail, head)`.
    pub back_edges: HashSet<(N, N)>,
    /// Index into `loops` for each header, for O(1) lookup.
    head_index: HashMap<N, usize>,
}

impl<N> Loops<N>
where
    N: Copy + Ord + Hash,
{
    /// Detect every natural loop reachable from `entry`.
    ///
    /// Unreachable blocks are ignored: they are dead code a `goto` left behind
    /// and are dropped before lowering anyway, and `simple_fast` reports no
    /// dominators for them.
    pub fn detect(graph: &DiGraphMap<N, ()>, entry: N) -> Result<Self, LoopError<N>> {
        check_reducible(graph, entry)?;

        let doms = dominators::simple_fast(graph, entry);

        // A back edge is `n → h` with `h` dominating `n`. `dominators` yields
        // `None` for an unreachable node, which correctly excludes dead code.
        let mut back_edges: HashSet<(N, N)> = HashSet::default();
        let mut tails_of: HashMap<N, Vec<N>> = HashMap::default();
        for (u, v, _) in graph.all_edges() {
            let Some(mut ds) = doms.dominators(u) else {
                continue; // unreachable tail
            };
            if ds.any(|d| d == v) {
                back_edges.insert((u, v));
                tails_of.entry(v).or_default().push(u);
            }
        }

        let mut loops: Vec<Loop<N>> = tails_of
            .into_iter()
            .map(|(head, mut tails)| {
                tails.sort();
                let body = natural_loop_body(graph, head, &tails);
                Loop {
                    head,
                    back_edge_tails: tails,
                    body,
                }
            })
            .collect();

        // Outermost first: an enclosing loop's body strictly contains a nested
        // one's, so descending size is a valid nesting order. Tie-break on the
        // head to keep the result deterministic.
        loops.sort_by(|a, b| {
            b.body
                .len()
                .cmp(&a.body.len())
                .then_with(|| a.head.cmp(&b.head))
        });

        let head_index = loops.iter().enumerate().map(|(i, l)| (l.head, i)).collect();

        Ok(Self {
            loops,
            back_edges,
            head_index,
        })
    }

    /// No loops at all — the common case, and the one where every downstream
    /// stage must behave exactly as it did before loops existed.
    pub fn is_empty(&self) -> bool {
        self.loops.is_empty()
    }

    /// The loop headed by `head`, if any.
    pub fn at_head(&self, head: N) -> Option<&Loop<N>> {
        self.head_index.get(&head).map(|&i| &self.loops[i])
    }

    /// Whether `(tail, head)` is a back edge, i.e. an edge to cut.
    pub fn is_back_edge(&self, tail: N, head: N) -> bool {
        self.back_edges.contains(&(tail, head))
    }

    /// Classify an edge against the loop nesting structure.
    ///
    /// `exits` is innermost-first and `entries` outermost-first, so a caller
    /// restoring frames on an out edge can simply iterate `exits` in order.
    pub fn classify(&self, from: N, to: N) -> EdgeClass {
        let mut exits: Vec<usize> = Vec::new();
        let mut entries: Vec<usize> = Vec::new();
        for (i, l) in self.loops.iter().enumerate() {
            let inside_from = l.body.contains(&from);
            let inside_to = l.body.contains(&to);
            // A back edge stays within its own loop, so it never counts as an
            // exit of that loop — `to` is the head, which is in the body.
            if inside_from && !inside_to {
                exits.push(i);
            }
            if !inside_from && inside_to {
                entries.push(i);
            }
        }
        // `loops` is outermost-first, so `entries` already is; reverse for
        // innermost-first exits.
        exits.reverse();
        EdgeClass {
            exits,
            entries,
            back: self
                .back_edges
                .contains(&(from, to))
                .then(|| self.head_index[&to]),
        }
    }
}

/// The natural loop of `head`: every node that reaches one of `tails` without
/// passing through `head`, plus `head` itself.
///
/// Standard backward worklist from the tails. `head` is seeded as visited so the
/// walk stops there, which is exactly what makes the result single-entry.
fn natural_loop_body<N>(graph: &DiGraphMap<N, ()>, head: N, tails: &[N]) -> HashSet<N>
where
    N: Copy + Ord + Hash,
{
    let mut body: HashSet<N> = HashSet::default();
    body.insert(head);
    let mut stack: Vec<N> = Vec::new();
    for &t in tails {
        if body.insert(t) {
            stack.push(t);
        }
    }
    while let Some(n) = stack.pop() {
        for p in graph.neighbors_directed(n, petgraph::Direction::Incoming) {
            if body.insert(p) {
                stack.push(p);
            }
        }
    }
    body
}

/// Reject irreducible CFGs: a cycle entered at more than one point.
///
/// Tested per non-trivial strongly-connected component — an SCC is reducible iff
/// exactly one of its nodes is reachable from outside it (counting the CFG entry
/// as an outside reference when it lies inside the component). Nested loops share
/// one SCC and still have a single entry, so this does not reject them.
///
/// **Only live nodes count.** Dead code is not control flow, so a `goto` from an
/// unreachable block into a loop is not a second door. Prusti emits exactly that
/// shape: an `if` whose arms both `goto` leaves an unreachable fall-through
/// continuation, and that continuation flows into the next labelled block, which
/// may be a loop header. Everything downstream agrees — [`Loops::detect`] skips a
/// back edge with an unreachable tail, and the lowering never walks a block
/// `Cfg::reachable` excludes.
fn check_reducible<N>(graph: &DiGraphMap<N, ()>, entry: N) -> Result<(), LoopError<N>>
where
    N: Copy + Ord + Hash,
{
    let mut live: HashSet<N> = HashSet::default();
    let mut stack = vec![entry];
    while let Some(n) = stack.pop() {
        if live.insert(n) && graph.contains_node(n) {
            stack.extend(graph.neighbors(n));
        }
    }
    for scc in tarjan_scc(graph) {
        let members: HashSet<N> = scc.into_iter().filter(|n| live.contains(n)).collect();
        let cyclic = members.len() > 1
            || members
                .iter()
                .next()
                .is_some_and(|&n| graph.contains_edge(n, n));
        if !cyclic {
            continue;
        }
        let mut doors: Vec<N> = Vec::new();
        for &n in &members {
            let entered_from_outside = graph
                .neighbors_directed(n, petgraph::Direction::Incoming)
                .any(|p| live.contains(&p) && !members.contains(&p));
            if entered_from_outside || n == entry {
                doors.push(n);
            }
        }
        if doors.len() > 1 {
            doors.sort();
            return Err(LoopError::Irreducible(doors));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `DiGraphMap` from an edge list over `usize` nodes.
    fn g(edges: &[(usize, usize)]) -> DiGraphMap<usize, ()> {
        let mut graph = DiGraphMap::new();
        for &(a, b) in edges {
            graph.add_edge(a, b, ());
        }
        graph
    }

    fn body(l: &Loop<usize>) -> Vec<usize> {
        let mut v: Vec<usize> = l.body.iter().copied().collect();
        v.sort();
        v
    }

    #[test]
    fn straight_line_has_no_loops() {
        let ls = Loops::detect(&g(&[(0, 1), (1, 2)]), 0).unwrap();
        assert!(ls.is_empty());
        assert!(ls.back_edges.is_empty());
    }

    #[test]
    fn diamond_has_no_loops() {
        let ls = Loops::detect(&g(&[(0, 1), (0, 2), (1, 3), (2, 3)]), 0).unwrap();
        assert!(ls.is_empty());
    }

    /// `0 → 1 ⇄ 2`, exit `1 → 3`. The classic while loop.
    #[test]
    fn simple_loop() {
        let ls = Loops::detect(&g(&[(0, 1), (1, 2), (2, 1), (1, 3)]), 0).unwrap();
        assert_eq!(ls.loops.len(), 1);
        let l = &ls.loops[0];
        assert_eq!(l.head, 1);
        assert_eq!(l.back_edge_tails, vec![2]);
        assert_eq!(body(l), vec![1, 2]);
        assert!(ls.is_back_edge(2, 1));
        assert!(!ls.is_back_edge(1, 2));
    }

    /// Cutting the back edges must leave a DAG — the invariant the whole
    /// lowering strategy rests on.
    #[test]
    fn cutting_back_edges_leaves_a_dag() {
        let graph = g(&[(0, 1), (1, 2), (2, 1), (1, 3), (3, 4), (4, 3)]);
        let ls = Loops::detect(&graph, 0).unwrap();
        let mut cut = graph.clone();
        for &(u, v) in &ls.back_edges {
            cut.remove_edge(u, v);
        }
        assert!(petgraph::algo::toposort(&cut, None).is_ok());
    }

    #[test]
    fn self_loop() {
        let ls = Loops::detect(&g(&[(0, 1), (1, 1), (1, 2)]), 0).unwrap();
        assert_eq!(ls.loops.len(), 1);
        assert_eq!(ls.loops[0].head, 1);
        assert_eq!(body(&ls.loops[0]), vec![1]);
    }

    /// A `continue` inside a conditional: two back edges, one head, one loop.
    #[test]
    fn multiple_back_edges_to_one_head() {
        let ls = Loops::detect(&g(&[(0, 1), (1, 2), (2, 1), (2, 3), (3, 1), (1, 4)]), 0).unwrap();
        assert_eq!(ls.loops.len(), 1);
        assert_eq!(ls.loops[0].back_edge_tails, vec![2, 3]);
        assert_eq!(body(&ls.loops[0]), vec![1, 2, 3]);
    }

    /// Outer `1 ⇄ 4`, inner `2 ⇄ 3`. Must be ordered outermost-first.
    #[test]
    fn nested_loops_are_outermost_first() {
        let ls = Loops::detect(
            &g(&[(0, 1), (1, 2), (2, 3), (3, 2), (2, 4), (4, 1), (1, 5)]),
            0,
        )
        .unwrap();
        assert_eq!(ls.loops.len(), 2);
        assert_eq!(ls.loops[0].head, 1, "outer loop first");
        assert_eq!(ls.loops[1].head, 2, "inner loop second");
        assert_eq!(body(&ls.loops[0]), vec![1, 2, 3, 4]);
        assert_eq!(body(&ls.loops[1]), vec![2, 3]);
    }

    /// A `break` out of the inner loop only leaves the inner loop.
    #[test]
    fn break_out_of_inner_loop() {
        let ls = Loops::detect(
            &g(&[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 2),
                (3, 4),
                (2, 4),
                (4, 1),
                (1, 5),
            ]),
            0,
        )
        .unwrap();
        let c = ls.classify(3, 4);
        assert_eq!(c.exits.len(), 1);
        assert_eq!(ls.loops[c.exits[0]].head, 2);
        assert!(c.is_exit());
    }

    /// A `break` escaping BOTH loops must report them innermost-first, because
    /// frames are restored in that order.
    #[test]
    fn break_out_of_both_loops_is_innermost_first() {
        let ls = Loops::detect(
            &g(&[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 2),
                (3, 9),
                (2, 4),
                (4, 1),
                (1, 5),
            ]),
            0,
        )
        .unwrap();
        let c = ls.classify(3, 9);
        assert_eq!(c.exits.len(), 2, "leaves both loops");
        assert_eq!(ls.loops[c.exits[0]].head, 2, "inner first");
        assert_eq!(ls.loops[c.exits[1]].head, 1, "outer second");
    }

    #[test]
    fn entering_a_loop_is_classified() {
        let ls = Loops::detect(&g(&[(0, 1), (1, 2), (2, 1), (1, 3)]), 0).unwrap();
        let c = ls.classify(0, 1);
        assert_eq!(c.entries.len(), 1);
        assert!(c.exits.is_empty());
        assert!(c.back.is_none());
    }

    #[test]
    fn back_edge_is_not_an_exit_of_its_own_loop() {
        let ls = Loops::detect(&g(&[(0, 1), (1, 2), (2, 1), (1, 3)]), 0).unwrap();
        let c = ls.classify(2, 1);
        assert_eq!(c.back, Some(0));
        assert!(c.exits.is_empty(), "the head is inside the body");
    }

    #[test]
    fn normal_edge_is_normal() {
        let ls = Loops::detect(&g(&[(0, 1), (1, 2), (2, 1), (1, 3), (3, 4)]), 0).unwrap();
        assert!(ls.classify(3, 4).is_normal());
    }

    /// Two entries into the same cycle: `1 → 2 → 1` reachable at both 1 and 2.
    #[test]
    fn irreducible_is_rejected() {
        let r = Loops::detect(&g(&[(0, 1), (0, 2), (1, 2), (2, 1)]), 0);
        assert!(matches!(r, Err(LoopError::Irreducible(_))));
    }

    /// Nested loops share one SCC but have a single entry — must NOT be
    /// mistaken for irreducible.
    #[test]
    fn nested_loops_are_not_irreducible() {
        assert!(
            Loops::detect(
                &g(&[(0, 1), (1, 2), (2, 3), (3, 2), (2, 4), (4, 1), (1, 5)]),
                0
            )
            .is_ok()
        );
    }

    /// A `goto` from *dead* code into a loop is not a second entry. This is the
    /// shape Prusti emits — the unreachable fall-through continuation of an `if`
    /// whose arms both `goto` flows into the next labelled block, here the loop
    /// header `1`. The loop is still single-entry and must be accepted.
    #[test]
    fn dead_edge_into_a_loop_is_not_a_second_entry() {
        // live: 0 → 1 → 2 → 1 (back edge), 1 → 3.  dead: 4 → 2.
        let r = Loops::detect(&g(&[(0, 1), (1, 2), (2, 1), (1, 3), (4, 2)]), 0);
        let ls = r.expect("a dead in-edge does not make the loop irreducible");
        assert_eq!(ls.loops.len(), 1);
        assert_eq!(ls.loops[0].head, 1);
    }

    /// The liveness filter must not let a *genuinely* irreducible cycle through:
    /// both doors of `1 ⇄ 2` are reachable here.
    #[test]
    fn irreducible_with_dead_code_present_is_still_rejected() {
        let r = Loops::detect(&g(&[(0, 1), (0, 2), (1, 2), (2, 1), (9, 1)]), 0);
        assert!(matches!(r, Err(LoopError::Irreducible(_))));
    }

    /// Dead code behind a `goto` has no dominators; it must be skipped rather
    /// than panicking or inventing a loop.
    #[test]
    fn unreachable_cycle_is_ignored() {
        let ls = Loops::detect(&g(&[(0, 1), (7, 8), (8, 7)]), 0).unwrap();
        assert!(ls.is_empty());
    }
}
