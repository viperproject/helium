//! Control-flow analysis of a typed Viper method body.
//!
//! Builds a basic-block CFG from the structured `If`/`Block` statements and the
//! `Label`/`Goto` jumps. We analyse Viper here; VMIR is always emitted flat, so
//! a later linearization pass walks this CFG (topologically — it is a DAG) to
//! emit PathCond-gated straight-line VMIR.
//!
//! Loops are **detected**, not rejected: a `goto` forming a back edge (and a
//! `while`, which is desugared into that same shape here) makes the block graph
//! cyclic, and [`viper::loops`](crate::viper::loops) identifies the back edges.
//! Cutting them leaves a DAG, so [`Cfg::topo_order`] and [`Cfg::predecessors`]
//! — and hence the `preds precede` invariant the VMIR lowering and the verifier
//! both assume — stay valid whether or not the method loops. Every loop head is
//! normalised to a single in-edge by a pre-header.
//!
//! Only genuinely irreducible control flow (a cycle entered at more than one
//! point) is rejected, as [`CfgError::Irreducible`].

use crate::dhash::{HashMap, HashSet};

use derive_more::{From, Into};
use lasso::Spur;
use petgraph::algo::toposort;
use petgraph::prelude::DiGraphMap;
use typed_index_collections::TiVec;

use crate::viper::loops::LoopError;
use crate::viper::typed::{PureMethodExp, SpatialMethodExp, Statement, StmtBlock};

/// The loop structure of a method body, over its block ids.
pub type Loops = crate::viper::loops::Loops<BlockId>;

/// One natural loop of a method body.
pub type Loop = crate::viper::loops::Loop<BlockId>;

/// Stable identity of an invariant-bearing loop head, in source order.
///
/// Loop invariants become their own `Resource` declarations, and slots can only
/// be allocated at *declare* time — before any CFG exists. So `declare` and
/// `build_cfg` must agree on which head is which without sharing a `BlockId`.
/// Both derive this key from the same source-order traversal
/// ([`loop_head_keys`] and `Builder::process`), and `build_cfg` asserts the two
/// agree, so a divergence is a loud failure rather than a mismatched resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LoopHeadKey {
    /// `label L invariant …` — labels are unique within a method.
    Label(Spur),
    /// `while (c) invariant … { … }` — the head has no name, so it is keyed by
    /// its index among the method's `while`s in source order.
    While(usize),
}

/// Every invariant-bearing loop head of `body`, in source order.
///
/// The traversal must match `Builder::process` exactly: statements in order,
/// an `If`'s then-arm before its else-arm, a `Block`'s contents inline, and a
/// `While`'s head *before* its body (so an outer loop precedes the loops nested
/// inside it).
pub fn loop_head_keys(body: &StmtBlock) -> Vec<LoopHeadKey> {
    let mut out = Vec::new();
    let mut whiles = 0usize;
    collect_head_keys(&body.0, &mut whiles, &mut out);
    out
}

fn collect_head_keys(stmts: &[Statement], whiles: &mut usize, out: &mut Vec<LoopHeadKey>) {
    for s in stmts {
        match s {
            Statement::Label(l, invs) if !invs.is_empty() => out.push(LoopHeadKey::Label(*l)),
            Statement::While(_, _, body) => {
                out.push(LoopHeadKey::While(*whiles));
                *whiles += 1;
                collect_head_keys(&body.0, whiles, out);
            }
            Statement::If(_, then, els) => {
                collect_head_keys(&then.0, whiles, out);
                if let Some(e) = els {
                    collect_head_keys(&e.0, whiles, out);
                }
            }
            Statement::Block(inner) => collect_head_keys(&inner.0, whiles, out),
            _ => {}
        }
    }
}

/// Index of a basic block within a [`Cfg`].
#[derive(Debug, From, Into, Eq, PartialEq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct BlockId(pub usize);

/// How a basic block transfers control to its successor(s).
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Unconditional successor (a fall-through or an explicit `goto`).
    Goto(BlockId),
    /// Two-way branch on a pure condition (from an `if`).
    Branch {
        cond: PureMethodExp,
        then_: BlockId,
        else_: BlockId,
    },
    /// Method exit — control falls off the end of the body.
    Return,
}

/// A maximal straight-line run of statements with a single entry and a single
/// terminator.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub stmts: Vec<Statement>,
    pub term: Terminator,
    /// The source label this block is the target of, if any (also an `old[L]`
    /// heap-capture point).
    pub label: Option<Spur>,
    /// The invariants declared on that label (`label L invariant A`). Empty
    /// unless this block is a labelled loop head.
    pub invs: Vec<SpatialMethodExp>,
    /// Source-order identity of this block as an invariant-bearing loop head.
    /// `None` for every other block. Lets translation match a head to the
    /// `Resource` slot allocated for it at declare time.
    pub head_key: Option<LoopHeadKey>,
}

/// The basic-block control-flow graph of a method body. Cyclic exactly when the
/// method loops; `loops.back_edges` are the edges to cut to get a DAG back.
#[derive(Debug, Clone)]
pub struct Cfg {
    pub blocks: TiVec<BlockId, BasicBlock>,
    pub entry: BlockId,
    pub labels: HashMap<Spur, BlockId>,
    /// Loop structure. Empty for a loop-free method, which is the case every
    /// ordering question below degenerates to.
    pub loops: Loops,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgError {
    /// A cycle entered at more than one point, carrying those entries. Rust MIR
    /// from `loop`/`while`/`for` is always reducible, so this only arises from
    /// hand-written `goto` spaghetti.
    Irreducible(Vec<BlockId>),
    /// A `goto` targets a label that is never declared.
    UndefinedLabel(Spur),
}

/// Which outgoing edge of a predecessor reaches a block: an unconditional
/// `goto`/fall-through, or the then/else arm of a `Branch` (the branch
/// condition must hold / must not hold to take it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeSide {
    Goto,
    Then,
    Else,
}

impl Cfg {
    /// Blocks in topological (dependency) order — every block precedes its
    /// successors.
    ///
    /// Taken over the graph with back edges cut, so this is total even for a
    /// looping method: a loop head precedes its body, and the body's back-edge
    /// tail is a leaf. Infallible, since cutting the back edges of a reducible
    /// CFG always leaves a DAG.
    pub fn topo_order(&self) -> Vec<BlockId> {
        toposort(&forward_graph(self), None).expect("cutting back edges leaves a DAG")
    }

    /// For each block, its predecessors paired with the edge that reaches it.
    ///
    /// **Back edges are excluded.** A loop head's entry state is the cut state —
    /// the havoc plus the invariant — not a merge of where control came from, so
    /// a back edge contributes nothing to reach it. That is what lets a head be
    /// lowered from its single forward predecessor.
    pub fn predecessors(&self) -> TiVec<BlockId, Vec<(BlockId, EdgeSide)>> {
        let mut preds: TiVec<BlockId, Vec<(BlockId, EdgeSide)>> =
            self.blocks.iter().map(|_| Vec::new()).collect();
        for (id, blk) in self.blocks.iter_enumerated() {
            let mut push = |t: BlockId, side| {
                if !self.loops.is_back_edge(id, t) {
                    preds[t].push((id, side));
                }
            };
            match &blk.term {
                Terminator::Goto(t) => push(*t, EdgeSide::Goto),
                Terminator::Branch { then_, else_, .. } => {
                    push(*then_, EdgeSide::Then);
                    push(*else_, EdgeSide::Else);
                }
                Terminator::Return => {}
            }
        }
        preds
    }

    /// The set of blocks reachable from the entry (the rest are dead code an
    /// `if`/`goto` left behind and need not be lowered).
    pub fn reachable(&self) -> HashSet<BlockId> {
        let mut seen = HashSet::default();
        let mut stack = vec![self.entry];
        while let Some(b) = stack.pop() {
            if seen.insert(b) {
                stack.extend(successors(&self.blocks[b].term));
            }
        }
        seen
    }
}

/// Build the basic-block CFG of a method body, rejecting loops (back-edge
/// `goto`s) and `goto`s to undefined labels.
pub fn build_cfg(body: &StmtBlock) -> Result<Cfg, CfgError> {
    let mut builder = Builder::default();
    let entry = builder.new_block();
    // Control reaching the end of the body (any still-open block) returns.
    if let Some(tail) = builder.process(&body.0, Some(entry)) {
        builder.open_seal(tail, Terminator::Return);
    }
    builder.finish(entry, &loop_head_keys(body))
}

#[derive(Default)]
struct Builder {
    blocks: TiVec<BlockId, BlockData>,
    /// Block for each label seen (declared or merely referenced by a `goto`).
    labels: HashMap<Spur, BlockId>,
    /// Labels actually declared by a `label` statement (a subset of `labels`).
    defined: HashSet<Spur>,
    /// `while`s seen so far, for [`LoopHeadKey::While`] numbering.
    whiles: usize,
    /// Head keys in assignment order, checked against [`loop_head_keys`].
    assigned_keys: Vec<LoopHeadKey>,
}

#[derive(Default)]
struct BlockData {
    stmts: Vec<Statement>,
    term: Option<Terminator>,
    label: Option<Spur>,
    invs: Vec<SpatialMethodExp>,
    head_key: Option<LoopHeadKey>,
}

impl Builder {
    fn new_block(&mut self) -> BlockId {
        self.blocks.push_and_get_key(BlockData::default())
    }

    /// The block for label `l`, created on first reference (a `goto` may precede
    /// the `label` declaration).
    fn label_block(&mut self, l: Spur) -> BlockId {
        if let Some(&b) = self.labels.get(&l) {
            return b;
        }
        let b = self.new_block();
        self.blocks[b].label = Some(l);
        self.labels.insert(l, b);
        b
    }

    /// Set a block's terminator, but only if it is still open (the first
    /// terminator wins).
    fn open_seal(&mut self, b: BlockId, t: Terminator) {
        if self.blocks[b].term.is_none() {
            self.blocks[b].term = Some(t);
        }
    }

    /// The current open block, creating a fresh one at an unreachable point
    /// (`*cur == None`, e.g. just after a `goto`) so a statement is not dropped.
    fn ensure(&mut self, cur: &mut Option<BlockId>) -> BlockId {
        if let Some(b) = *cur {
            return b;
        }
        let b = self.new_block();
        *cur = Some(b);
        b
    }

    /// Process a statement sequence starting in `cur` (an open block, or `None`
    /// at an unreachable point). Returns the open block control reaches after
    /// the sequence, or `None` if it ends unreachable (e.g. after a `goto`).
    fn process(&mut self, stmts: &[Statement], mut cur: Option<BlockId>) -> Option<BlockId> {
        for s in stmts {
            match s {
                Statement::If(cond, then, els) => {
                    let b = self.ensure(&mut cur);
                    let then_e = self.new_block();
                    let else_e = self.new_block();
                    self.open_seal(
                        b,
                        Terminator::Branch {
                            cond: cond.clone(),
                            then_: then_e,
                            else_: else_e,
                        },
                    );
                    let then_x = self.process(&then.0, Some(then_e));
                    let else_stmts = els.as_ref().map(|b| b.0.as_slice()).unwrap_or(&[]);
                    let else_x = self.process(else_stmts, Some(else_e));
                    // Arms that fall through merge; arms ending in goto/return
                    // (returned `None`) do not.
                    let merge = self.new_block();
                    if let Some(tx) = then_x {
                        self.open_seal(tx, Terminator::Goto(merge));
                    }
                    if let Some(ex) = else_x {
                        self.open_seal(ex, Terminator::Goto(merge));
                    }
                    cur = Some(merge);
                }
                Statement::Block(inner) => {
                    cur = self.process(&inner.0, cur);
                }
                Statement::Label(l, invs) => {
                    self.defined.insert(*l);
                    let lb = self.label_block(*l);
                    // The block may already exist (a `goto` referenced the label
                    // before its declaration); the declaration is what carries
                    // the invariants.
                    self.blocks[lb].invs = invs.clone();
                    if !invs.is_empty() {
                        let key = LoopHeadKey::Label(*l);
                        self.assigned_keys.push(key.clone());
                        self.blocks[lb].head_key = Some(key);
                    }
                    if let Some(b) = cur {
                        self.open_seal(b, Terminator::Goto(lb));
                    }
                    cur = Some(lb);
                }
                Statement::Goto(l) => {
                    let lb = self.label_block(*l);
                    let b = self.ensure(&mut cur);
                    self.open_seal(b, Terminator::Goto(lb));
                    cur = None;
                }
                // `while (c) invariant A { BODY }` becomes the block structure a
                // hand-written `goto` loop already produces, so everything past
                // this point sees exactly one loop shape:
                //
                //   pred ──> head[invs] ──c──> body ──> (back edge) ──> head
                //                        └─!c─> exit
                //
                // Desugared here rather than upstream because the head needs no
                // name: the CFG identifies blocks by `BlockId` and invariants ride
                // on the block, so a `label`+`goto` rewrite would mint a synthetic
                // identifier for nothing.
                Statement::While(cond, invs, body) => {
                    let b = self.ensure(&mut cur);
                    let head = self.new_block();
                    self.blocks[head].invs = invs.clone();
                    let key = LoopHeadKey::While(self.whiles);
                    self.whiles += 1;
                    self.assigned_keys.push(key.clone());
                    self.blocks[head].head_key = Some(key);
                    self.open_seal(b, Terminator::Goto(head));

                    let body_e = self.new_block();
                    let exit = self.new_block();
                    self.open_seal(
                        head,
                        Terminator::Branch {
                            cond: cond.clone(),
                            then_: body_e,
                            else_: exit,
                        },
                    );
                    // The back edge — the whole point of the construct. A body
                    // that ends unreachable (`goto`/`return`) has none, exactly
                    // as with an `if` arm.
                    if let Some(bx) = self.process(&body.0, Some(body_e)) {
                        self.open_seal(bx, Terminator::Goto(head));
                    }
                    cur = Some(exit);
                }
                // Everything else is straight-line: append to the current block.
                _ => {
                    let b = self.ensure(&mut cur);
                    self.blocks[b].stmts.push(s.clone());
                }
            }
        }
        cur
    }

    fn finish(self, entry: BlockId, expected_keys: &[LoopHeadKey]) -> Result<Cfg, CfgError> {
        let Builder {
            blocks,
            labels,
            defined,
            assigned_keys,
            whiles: _,
        } = self;

        // `declare` allocates one Resource slot per key from `loop_head_keys`;
        // this walk assigns keys to blocks independently. They must agree, or a
        // head would be matched to another head's invariant resource.
        debug_assert_eq!(
            assigned_keys, expected_keys,
            "loop_head_keys and Builder::process disagree on loop head identity"
        );

        // Every referenced label must be declared.
        for &l in labels.keys() {
            if !defined.contains(&l) {
                return Err(CfgError::UndefinedLabel(l));
            }
        }

        // Finalize: a still-open block falls off the end → Return.
        let blocks: TiVec<BlockId, BasicBlock> = blocks
            .into_iter()
            .map(|bd| BasicBlock {
                stmts: bd.stmts,
                term: bd.term.unwrap_or(Terminator::Return),
                label: bd.label,
                invs: bd.invs,
                head_key: bd.head_key,
            })
            .collect();

        // Detect loops rather than reject them. Cutting the back edges leaves a
        // DAG, which is what keeps `topo_order` (and the `preds precede`
        // invariant the VMIR lowering and verifier both assume) valid.
        let mut blocks = blocks;
        let mut loops = detect(&blocks, entry)?;

        // Give every loop head a single in-edge, so the lowering has one place
        // to establish the invariant and one frame to restore on exit. Distinct
        // in-edges would otherwise each carry their own residual heap.
        if insert_preheaders(&mut blocks, &loops, entry) {
            // Block ids moved; re-derive rather than patching the structure.
            loops = detect(&blocks, entry)?;
        }

        Ok(Cfg {
            blocks,
            entry,
            labels,
            loops,
        })
    }
}

/// Natural-loop detection over the block graph, mapping the analysis' error into
/// a [`CfgError`].
fn detect(blocks: &TiVec<BlockId, BasicBlock>, entry: BlockId) -> Result<Loops, CfgError> {
    Loops::detect(&block_graph(blocks), entry).map_err(|e| match e {
        LoopError::Irreducible(doors) => CfgError::Irreducible(doors),
    })
}

/// Insert a pre-header before every loop head reached by more than one forward
/// (non-back) edge: a fresh block that all those edges are redirected to, which
/// then falls through to the head.
///
/// Returns whether anything was inserted.
///
/// A head is left alone when it already has a single in-edge — the common case,
/// and the one where adding a block would only cost a redundant join.
fn insert_preheaders(
    blocks: &mut TiVec<BlockId, BasicBlock>,
    loops: &Loops,
    entry: BlockId,
) -> bool {
    let mut inserted = false;
    for l in &loops.loops {
        let head = l.head;
        // Forward predecessors only: a back edge is the loop repeating, not an
        // entry into it.
        let in_edges: Vec<BlockId> = blocks
            .iter_enumerated()
            .filter(|(id, b)| successors(&b.term).contains(&head) && !loops.is_back_edge(*id, head))
            .map(|(id, _)| id)
            .collect();
        // The entry block being the head means control also arrives from outside
        // the graph, which counts as an in-edge.
        if in_edges.len() + usize::from(head == entry) <= 1 {
            continue;
        }
        let pre = blocks.push_and_get_key(BasicBlock {
            stmts: Vec::new(),
            term: Terminator::Goto(head),
            label: None,
            invs: Vec::new(),
            head_key: None,
        });
        for p in in_edges {
            retarget(&mut blocks[p].term, head, pre);
        }
        inserted = true;
    }
    inserted
}

/// Repoint every `from` successor of a terminator at `to`.
fn retarget(term: &mut Terminator, from: BlockId, to: BlockId) {
    match term {
        Terminator::Goto(b) => {
            if *b == from {
                *b = to;
            }
        }
        Terminator::Branch { then_, else_, .. } => {
            if *then_ == from {
                *then_ = to;
            }
            if *else_ == from {
                *else_ = to;
            }
        }
        Terminator::Return => {}
    }
}

/// The blocks a terminator transfers control to.
pub fn successors_of(t: &Terminator) -> Vec<BlockId> {
    successors(t)
}

fn successors(t: &Terminator) -> Vec<BlockId> {
    match t {
        Terminator::Goto(b) => vec![*b],
        Terminator::Branch { then_, else_, .. } => vec![*then_, *else_],
        Terminator::Return => vec![],
    }
}

fn block_graph(blocks: &TiVec<BlockId, BasicBlock>) -> DiGraphMap<BlockId, ()> {
    let mut g = DiGraphMap::new();
    for (id, _) in blocks.iter_enumerated() {
        g.add_node(id);
    }
    for (id, b) in blocks.iter_enumerated() {
        for succ in successors(&b.term) {
            g.add_edge(id, succ, ());
        }
    }
    g
}

/// The block graph with every back edge removed — a DAG, even when the method
/// loops. This is the graph every ordering question is asked of.
fn forward_graph(cfg: &Cfg) -> DiGraphMap<BlockId, ()> {
    let mut g = block_graph(&cfg.blocks);
    for &(u, v) in &cfg.loops.back_edges {
        g.remove_edge(u, v);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viper::typed::{Literal, PureExpKind, Statement, StmtBlock, Type, TypedPureExp};
    use lasso::Rodeo;

    /// A trivial straight-line statement (contents irrelevant to the CFG).
    fn nop() -> Statement {
        Statement::Var(vec![], None)
    }

    /// A trivial pure condition for `if`s.
    fn cond() -> PureMethodExp {
        TypedPureExp {
            ty: Type::Bool,
            exp: Box::new(PureExpKind::Const(Literal::Bool(true))),
        }
    }

    fn block(stmts: Vec<Statement>) -> StmtBlock {
        StmtBlock(stmts)
    }

    #[test]
    fn straight_line_is_one_block() {
        let cfg = build_cfg(&block(vec![nop(), nop()])).unwrap();
        assert_eq!(cfg.blocks.len(), 1);
        let entry = &cfg.blocks[cfg.entry];
        assert_eq!(entry.stmts.len(), 2);
        assert!(matches!(entry.term, Terminator::Return));
    }

    #[test]
    fn if_else_branches_and_merges() {
        let body = block(vec![Statement::If(
            cond(),
            block(vec![nop()]),
            Some(block(vec![nop()])),
        )]);
        let cfg = build_cfg(&body).unwrap();
        // entry(branch) + then + else + merge.
        assert_eq!(cfg.blocks.len(), 4);
        let Terminator::Branch { then_, else_, .. } = cfg.blocks[cfg.entry].term else {
            panic!("entry should branch");
        };
        // Both arms fall through to the same merge block, which returns.
        let Terminator::Goto(tm) = cfg.blocks[then_].term else {
            panic!("then arm should fall through")
        };
        let Terminator::Goto(em) = cfg.blocks[else_].term else {
            panic!("else arm should fall through")
        };
        assert_eq!(tm, em);
        assert!(matches!(cfg.blocks[tm].term, Terminator::Return));
    }

    #[test]
    fn if_without_else_routes_else_to_merge() {
        let body = block(vec![Statement::If(cond(), block(vec![nop()]), None)]);
        let cfg = build_cfg(&body).unwrap();
        assert_eq!(cfg.blocks.len(), 4);
        let Terminator::Branch { then_, else_, .. } = cfg.blocks[cfg.entry].term else {
            panic!("entry should branch");
        };
        // Both arms reach the same merge block.
        let Terminator::Goto(tm) = cfg.blocks[then_].term else {
            panic!()
        };
        let Terminator::Goto(em) = cfg.blocks[else_].term else {
            panic!()
        };
        assert_eq!(tm, em);
    }

    #[test]
    fn forward_goto_is_acyclic() {
        let mut r = Rodeo::default();
        let l = r.get_or_intern("L");
        let body = block(vec![
            Statement::Goto(l),
            nop(),
            Statement::Label(l, vec![]),
            nop(),
        ]);
        let cfg = build_cfg(&body).unwrap();
        assert_eq!(cfg.labels.get(&l).copied(), Some(BlockId(1)));
        assert!(matches!(cfg.blocks[cfg.entry].term, Terminator::Goto(b) if b == BlockId(1)));
    }

    #[test]
    fn backward_goto_is_detected_as_a_loop() {
        let mut r = Rodeo::default();
        let l = r.get_or_intern("L");
        let body = block(vec![Statement::Label(l, vec![]), nop(), Statement::Goto(l)]);
        let cfg = build_cfg(&body).expect("a back edge is a loop, not an error");
        assert_eq!(cfg.loops.loops.len(), 1);
        let head = cfg.loops.loops[0].head;
        assert_eq!(cfg.blocks[head].label, Some(l), "the label is the head");
        assert_eq!(cfg.loops.back_edges.len(), 1);
    }

    /// The property the whole lowering strategy rests on: whatever the method
    /// does, ordering questions are asked of a DAG.
    #[test]
    fn topo_order_is_total_over_a_loop() {
        let mut r = Rodeo::default();
        let l = r.get_or_intern("L");
        let body = block(vec![Statement::Label(l, vec![]), nop(), Statement::Goto(l)]);
        let cfg = build_cfg(&body).unwrap();
        assert_eq!(cfg.topo_order().len(), cfg.blocks.len());
    }

    /// A loop head's entry state is the cut state, so a back edge is not one of
    /// its predecessors.
    #[test]
    fn back_edge_is_not_a_predecessor() {
        let mut r = Rodeo::default();
        let l = r.get_or_intern("L");
        let body = block(vec![Statement::Label(l, vec![]), nop(), Statement::Goto(l)]);
        let cfg = build_cfg(&body).unwrap();
        let head = cfg.loops.loops[0].head;
        let preds = cfg.predecessors();
        let (tail, _) = *cfg.loops.back_edges.iter().next().unwrap();
        assert!(
            !preds[head].iter().any(|(p, _)| *p == tail),
            "the back-edge tail must not reach the head as a predecessor"
        );
    }

    /// `while` becomes head/body/back-edge blocks, with the invariants on the
    /// head and no label minted for it.
    #[test]
    fn while_lowers_to_a_loop_head_without_a_label() {
        let body = block(vec![Statement::While(cond(), vec![], block(vec![nop()]))]);
        let cfg = build_cfg(&body).unwrap();
        assert_eq!(cfg.loops.loops.len(), 1);
        let head = cfg.loops.loops[0].head;
        assert!(
            cfg.blocks[head].label.is_none(),
            "a while head needs no name"
        );
        assert!(matches!(cfg.blocks[head].term, Terminator::Branch { .. }));
        assert!(cfg.labels.is_empty());
    }

    /// Two forward edges into one head collapse onto a single pre-header, so the
    /// lowering has one place to establish the invariant and one frame.
    #[test]
    fn multiple_in_edges_get_a_preheader() {
        let mut r = Rodeo::default();
        let l = r.get_or_intern("L");
        // if (c) { goto L } ; label L ; nop ; goto L
        let body = block(vec![
            Statement::If(cond(), block(vec![Statement::Goto(l)]), None),
            Statement::Label(l, vec![]),
            nop(),
            Statement::Goto(l),
        ]);
        let cfg = build_cfg(&body).unwrap();
        let head = cfg.loops.loops[0].head;
        let preds = cfg.predecessors();
        assert_eq!(
            preds[head].len(),
            1,
            "every forward edge into the head goes through one pre-header"
        );
    }

    #[test]
    fn irreducible_control_flow_is_rejected() {
        let mut r = Rodeo::default();
        let (a, b) = (r.get_or_intern("A"), r.get_or_intern("B"));
        // if (c) { goto A } else { goto B } ; label A ; goto B ; label B ; goto A
        let body = block(vec![
            Statement::If(
                cond(),
                block(vec![Statement::Goto(a)]),
                Some(block(vec![Statement::Goto(b)])),
            ),
            Statement::Label(a, vec![]),
            Statement::Goto(b),
            Statement::Label(b, vec![]),
            Statement::Goto(a),
        ]);
        assert!(matches!(build_cfg(&body), Err(CfgError::Irreducible(_))));
    }

    /// The Prusti-emitted goto shape: an `if` whose arms both `goto` leaves an
    /// unreachable fall-through continuation, and the block it flows into sits in
    /// a loop. That dead edge is not a second entry, so the loop is reducible.
    #[test]
    fn dead_if_continuation_flowing_into_a_loop_is_accepted() {
        let mut r = Rodeo::default();
        let (h, x, y, e) = (
            r.get_or_intern("H"),
            r.get_or_intern("X"),
            r.get_or_intern("Y"),
            r.get_or_intern("E"),
        );
        // label H ; if (c) { goto E } else { goto X } ; label X ; goto Y ;
        // label Y ; goto H ; label E
        //
        // The block after the `if` is unreachable and falls through into `X`,
        // which is inside the `H` loop.
        let body = block(vec![
            Statement::Label(h, vec![]),
            Statement::If(
                cond(),
                block(vec![Statement::Goto(e)]),
                Some(block(vec![Statement::Goto(x)])),
            ),
            Statement::Label(x, vec![]),
            Statement::Goto(y),
            Statement::Label(y, vec![]),
            Statement::Goto(h),
            Statement::Label(e, vec![]),
        ]);
        let cfg = build_cfg(&body).expect("a dead in-edge does not make the loop irreducible");
        assert_eq!(cfg.loops.loops.len(), 1);
    }

    #[test]
    fn goto_undefined_label_rejected() {
        let mut r = Rodeo::default();
        let l = r.get_or_intern("L");
        let body = block(vec![Statement::Goto(l)]);
        assert!(matches!(build_cfg(&body), Err(CfgError::UndefinedLabel(ll)) if ll == l));
    }

    #[test]
    fn arm_ending_in_goto_leaves_one_merge_predecessor() {
        let mut r = Rodeo::default();
        let l = r.get_or_intern("L");
        // if (c) { goto L } ; label L ; nop
        let body = block(vec![
            Statement::If(cond(), block(vec![Statement::Goto(l)]), None),
            Statement::Label(l, vec![]),
            nop(),
        ]);
        let cfg = build_cfg(&body).unwrap();
        // No loop, label defined, builds fine.
        assert!(cfg.labels.contains_key(&l));
        let Terminator::Branch { then_, .. } = cfg.blocks[cfg.entry].term else {
            panic!("entry should branch");
        };
        // The then-arm jumps to the label, not the merge.
        assert!(
            matches!(cfg.blocks[then_].term, Terminator::Goto(b) if Some(&b) == cfg.labels.get(&l))
        );
    }
}
