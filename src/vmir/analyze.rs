//! Static analyses over a raw `vmir::Program`, producing an
//! [`AnalyzedProgram`].
//!
//! The main analysis is the verification order: dependencies are scheduled
//! before dependents, and methods (always sinks) fall out last. Strongly
//! connected components (SCCs) of the dependency graph are computed so that
//! (mutually) recursive **functions** can be verified as a group via the
//! limited-function encoding. A cyclic SCC containing anything other than
//! functions (a recursive resource/method) is still rejected.

use petgraph::algo::tarjan_scc;
use petgraph::prelude::DiGraphMap;

use crate::vmir::{
    Declaration, HeapInst, Inst, InstKind, MemberId, Method, Precond, Program, PureInst,
    ResourceBody, Type,
};

/// Dependency graph: node = schedulable `MemberId`, edge dependency ->
/// dependent. May contain cycles among functions (recursion); those are grouped
/// into SCCs rather than rejected.
pub type DepGraph = DiGraphMap<MemberId, ()>;

/// A `Program` augmented with the results of static analyses. Extend with
/// further analysis fields as they are added.
#[derive(Debug, Clone)]
pub struct AnalyzedProgram {
    pub program: Program,
    /// Dependency graph over schedulable members. Edge dependency -> dependent.
    /// May contain function recursion cycles (see [`AnalyzedProgram::scc_order`]).
    pub dep_graph: DepGraph,
    /// The dependency graph's SCCs in **dependency-first** order: each inner
    /// `Vec` is one SCC (a singleton for a non-recursive member; a multi-member
    /// group or a self-looping singleton for (mutual) recursion). Verifying the
    /// groups in this order schedules every dependency before its dependents; a
    /// recursive group is verified as a unit (all members' certificates become
    /// available together). The single source of truth for scheduling.
    pub scc_order: Vec<Vec<MemberId>>,
}

impl AnalyzedProgram {
    /// The set of members co-recursive with `m` (the members of `m`'s SCC),
    /// **only if** that SCC is a genuine recursion cycle (multi-member, or a
    /// self-looping singleton); `None` for an ordinary non-recursive member.
    /// Call sites within a recursive function retarget in-set callees to their
    /// limited twin.
    pub fn recursive_scc(&self, m: MemberId) -> Option<crate::dhash::HashSet<MemberId>> {
        let scc = self.scc_order.iter().find(|scc| scc.contains(&m))?;
        let recursive = scc.len() > 1 || self.dep_graph.contains_edge(m, m);
        recursive.then(|| scc.iter().copied().collect())
    }
}

#[derive(Debug)]
pub enum AnalysisError {
    /// A cyclic dependency that cannot be ordered for verification: either a
    /// resource/method recursion, or a function recursion that drags in a
    /// non-function member. (Function-only cycles are permitted — the
    /// limited-function encoding handles them.) Carries the member names.
    CircularDependency(Vec<String>),
    /// A permission temp read outside the block that defines it. Permissions are
    /// **block-local** by construction: a permission has no join (heaps own
    /// those), so nothing may carry one across a block boundary. Carries the
    /// member name and the offending `p` index.
    PermTempEscapesBlock(String, usize),
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CircularDependency(names) => {
                write!(f, "circular resource dependency: {}", names.join(" -> "))
            }
            Self::PermTempEscapesBlock(name, i) => {
                write!(f, "{name}: permission temp p{i} read outside its block")
            }
        }
    }
}

/// Run all analyses over `program`, producing an [`AnalyzedProgram`].
pub fn analyze(program: Program) -> Result<AnalyzedProgram, AnalysisError> {
    let dep_graph = build_dep_graph(&program);
    // SCCs in reverse-topological order (petgraph's `tarjan_scc` contract);
    // reversing yields dependency-first order for scheduling.
    let mut scc_order: Vec<Vec<MemberId>> = tarjan_scc(&dep_graph);
    scc_order.reverse();
    // A genuine recursion cycle (multi-member SCC, or a self-looping singleton)
    // is permitted only if every member is a function — the limited-function
    // encoding handles those. Anything else (recursive resource/method, or a
    // function cycle dragging in a non-function) is rejected.
    for scc in &scc_order {
        let cyclic = scc.len() > 1 || dep_graph.contains_edge(scc[0], scc[0]);
        if cyclic
            && !scc
                .iter()
                .all(|&id| matches!(program.decls[id], Declaration::Function(_)))
        {
            let names = scc.iter().map(|&id| program.name(id).to_string()).collect();
            return Err(AnalysisError::CircularDependency(names));
        }
    }
    for (id, decl) in program.decls.iter_enumerated() {
        check_perm_locality(decl, || program.name(id).to_string())?;
    }
    dump_callgraph(&dep_graph, &program);
    Ok(AnalyzedProgram {
        program,
        dep_graph,
        scc_order,
    })
}

/// Check that every `PermVal::Temp` is defined **earlier in the same block**.
///
/// Permission temps are numbered per declaration (one sink), so "same block" is
/// "index at or after the count of perm insts preceding this block". The
/// def-before-use half also rules out a cycle, which is why a permission can
/// never feed itself.
///
/// The remaining permission invariants need no check: a `PermVal` is reachable
/// only from a `PermInst` arm or a heap instruction's `perm` field, because those
/// are the only places the Rust types admit one. That is the point of giving
/// permissions their own namespace rather than a `Val` type tag.
fn check_perm_locality(decl: &Declaration, name: impl Fn() -> String) -> Result<(), AnalysisError> {
    use crate::vmir::{HeapInst, PermInst, PermVal};

    // `(first p index of this block, count after this phase)`, threaded across
    // phases and blocks in stored order — the same numbering the display walker
    // and the verifier's eval state use.
    fn phase(
        insts: &[Inst],
        block_base: usize,
        count: &mut usize,
        name: &impl Fn() -> String,
    ) -> Result<(), AnalysisError> {
        for inst in insts {
            let check = |pv: &PermVal| match pv {
                PermVal::Temp(i) if *i < block_base || *i >= *count => {
                    Err(AnalysisError::PermTempEscapesBlock(name(), *i))
                }
                _ => Ok(()),
            };
            match &inst.kind {
                InstKind::Perm(PermInst::Ite(_, t, e)) => {
                    check(t)?;
                    check(e)?;
                    *count += 1;
                }
                InstKind::Heap(
                    HeapInst::Add { perm, .. }
                    | HeapInst::Sub { perm, .. }
                    | HeapInst::Inhale { perm, .. }
                    | HeapInst::Exhale { perm, .. },
                ) => check(perm)?,
                _ => {}
            }
        }
        Ok(())
    }

    match decl {
        Declaration::Method(m) => {
            let mut count = 0usize;
            for blk in &m.blocks {
                let base = count;
                phase(&blk.join, base, &mut count, &name)?;
                phase(&blk.body, base, &mut count, &name)?;
            }
        }
        // Every other body is a single straight-line stream — one block.
        Declaration::Function(f) => {
            if let Some(b) = &f.body {
                phase(&b.insts, 0, &mut 0, &name)?;
            }
        }
        Declaration::Resource(r) => phase(&r.body.insts, 0, &mut 0, &name)?,
        _ => {}
    }
    Ok(())
}

/// Build the dependency graph over `program`'s schedulable members
/// (`Resource | Function | Method`). Edges run dependency -> dependent;
/// references to non-schedulable declarations (domains, ADTs) are dropped.
fn build_dep_graph(program: &Program) -> DepGraph {
    let mut graph = DepGraph::new();

    // One node per schedulable declaration.
    for (id, decl) in program.decls.iter_enumerated() {
        if is_schedulable(decl) {
            graph.add_node(id);
        }
    }

    // Edges: dependency -> dependent. Skip references to non-schedulable
    // declarations (which are not nodes).
    let mut deps = Vec::new();
    for (id, decl) in program.decls.iter_enumerated() {
        if !graph.contains_node(id) {
            continue;
        }
        deps.clear();
        decl_deps(decl, &mut deps);
        for &dep in &deps {
            if graph.contains_node(dep) {
                graph.add_edge(dep, id, ());
            }
        }
    }

    graph
}

/// With a trace session's `viz_dir` (`verify --viz=DIR`), dump the dependency graph to
/// `<dir>/callgraph.dot` for debugging. Node labels are member names;
/// edges are unlabeled.
fn dump_callgraph(graph: &DepGraph, program: &Program) {
    use petgraph::dot::{Config, Dot};

    let Some(dir) = crate::trace::viz_dir() else {
        return;
    };
    let edge_attr = |_, _| String::new();
    let node_attr = |_, (id, _): (MemberId, &MemberId)| format!("label = \"{}\"", program.name(id));
    let dot = Dot::with_attr_getters(
        graph,
        &[Config::EdgeNoLabel, Config::NodeNoLabel],
        &edge_attr,
        &node_attr,
    );
    let path = dir.join("callgraph.dot");
    if let Err(e) =
        std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, format!("{dot:?}")))
    {
        eprintln!("failed to write {}: {e}", path.display());
    }
}

fn is_schedulable(decl: &Declaration) -> bool {
    matches!(
        decl,
        Declaration::Resource(_) | Declaration::Function(_) | Declaration::Method(_)
    )
}

/// Collect the `MemberId`s a declaration depends on (must be verified
/// first). Domain/ADT references are included verbatim; the caller drops
/// any that aren't schedulable nodes.
fn decl_deps(decl: &Declaration, out: &mut Vec<MemberId>) {
    match decl {
        Declaration::Resource(r) => {
            if let Precond::Ctx(req, _) = &r.precond {
                out.push(*req);
            }
            resource_body_deps(&r.body, out);
        }
        Declaration::Method(m) => method_deps(m, out),
        Declaration::Function(f) => {
            // A function depends on every function it calls — including its own
            // `f#requires`/`f#ensures`, which are ordinary `Function` decls. This
            // orders callees before callers (so their bodies inline) and turns any
            // (mutual) recursion into a dependency cycle, rejected by `analyze`.
            if let Some(body) = &f.body {
                inst_deps(&body.insts, out);
            }
            // Contract links: an abstract function has no body insts, but its
            // synthesized post axiom still needs the contract decls verified
            // first. (For a concrete function these edges duplicate the body's
            // call edges — harmless.)
            if let Some(rq) = &f.requires {
                out.push(rq.member());
            }
            if let Some(en) = &f.ensures {
                out.push(en.member);
            }
        }
        Declaration::Axiom(ax) => inst_deps(&ax.body.insts, out),
        // Leaf declarations: nothing to depend on.
        Declaration::Domain(_) | Declaration::Adt(_) => {}
    }
}

/// The members a trigger pattern mentions: the function at each application head
/// it matches on (an ADT head is not a declaration the verifier schedules — its
/// ids are minted on demand).
fn trig_term_deps(term: &crate::vmir::TrigTerm, out: &mut Vec<MemberId>) {
    if let crate::vmir::TrigTerm::App { head, args, .. } = term {
        if let crate::vmir::TrigHead::Func(id) = head {
            out.push(*id);
        }
        for a in args.iter() {
            trig_term_deps(a, out);
        }
    }
}

/// Collect the schedulable members referenced by an instruction stream: the
/// callee of each non-address `FunctionCall`, and the resource of each
/// inhale/exhale. Shared by resource, method, and function bodies.
///
/// An **address-typed** `FunctionCall` (result `Type::Addr`) is NOT a dependency:
/// forming an address needs no certificate, and a predicate's address function is
/// the predicate's own id, so treating it as a dependency would make a recursive
/// predicate (`acc(P(this.next))` in `P`'s body) a self-cycle.
///
/// This is what keeps a desugared `fold`/`unfold` scheduled correctly now that
/// neither is a variant: the pair's resource half is an inhale/exhale naming the
/// predicate, so the dependency is recorded there, while its `P@addr(args)` half
/// is address-typed and skipped by the rule above.
fn inst_deps(insts: &[Inst], out: &mut Vec<MemberId>) {
    for inst in insts {
        match &inst.kind {
            InstKind::Pure(ty, PureInst::FunctionCall(fc)) if !matches!(ty, Type::Addr { .. }) => {
                out.push(fc.function)
            }
            InstKind::Heap(HeapInst::Inhale { call, .. } | HeapInst::Exhale { call, .. }) => {
                out.push(call.resource)
            }
            // An inline `forall` depends on whatever its body calls and its
            // triggers match on — the body is an ordinary inst stream one scope
            // down, so recurse (nested `forall`s included).
            InstKind::Pure(_, PureInst::Forall(q)) => {
                inst_deps(&q.body.insts, out);
                for group in q.triggers.iter() {
                    for term in group.terms.iter() {
                        trig_term_deps(term, out);
                    }
                }
            }
            _ => {}
        }
    }
}

fn resource_body_deps(body: &ResourceBody, out: &mut Vec<MemberId>) {
    inst_deps(&body.insts, out);
}

fn method_deps(m: &Method, out: &mut Vec<MemberId>) {
    for b in &m.blocks {
        inst_deps(&b.join, out);
        inst_deps(&b.body, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dhash::HashSet;
    use crate::vmir::{
        Block, BlockId, Function, FunctionBody, FunctionCall, HeapInst, HeapVal, Inst, InstKind,
        PathConds, Precond, Preds, Resource, ResourceCall, Val,
    };
    use lasso::{Key, Rodeo};
    use typed_index_collections::TiVec;

    /// A function whose body calls `callee` (a plain, non-address `FunctionCall`),
    /// creating a dependency edge `callee -> this`. `callee == self` yields a
    /// self-recursive function.
    fn function_calling(callee: MemberId) -> Declaration {
        let call = FunctionCall {
            function: callee,
            type_args: vec![],
            args: vec![].into(),
            export: true,
        };
        let inst = Inst {
            pc: PathConds::default(),
            heap: None,
            kind: InstKind::Pure(Type::Int, PureInst::FunctionCall(call)),
        };
        Declaration::Function(Function {
            name: lasso::Spur::try_from_usize(0).unwrap(),
            params: vec![].into(),
            ret: Type::Int,
            body: Some(FunctionBody {
                insts: vec![inst],
                res: Val::Temp(0),
            }),
            requires: None,
            ensures: None,
        })
    }

    fn resource_requiring(req: Option<MemberId>) -> Declaration {
        Declaration::Resource(Resource {
            name: lasso::Spur::try_from_usize(0).unwrap(),
            params: vec![],
            precond: match req {
                Some(r) => Precond::Ctx(r, vec![]),
                None => Precond::SelfFramed,
            },
            body: ResourceBody {
                insts: vec![],
                res: (
                    HeapVal::Empty,
                    Val::Literal(crate::vmir::Literal::Bool(true)),
                ),
            },
        })
    }

    fn method_calling(res: MemberId) -> Declaration {
        let call = ResourceCall {
            resource: res,
            args: vec![],
        };
        let inst: Inst = Inst {
            pc: PathConds::default(),
            heap: None,
            kind: InstKind::Heap(HeapInst::Inhale {
                bind: crate::vmir::Bind::Fresh,
                base: HeapVal::Empty,
                call,
                perm: crate::vmir::PermVal::write(),
            }),
        };
        Declaration::Method(Method {
            name: lasso::Spur::try_from_usize(0).unwrap(),
            blocks: vec![Block {
                cube: PathConds::default(),
                preds: Preds::Entry,
                join: vec![],
                body: vec![inst],
                h_out: HeapVal::Empty,
            }]
            .into(),
            entry: BlockId(0),
        })
    }

    fn program(names: &[&str], decls: Vec<Declaration>) -> Program {
        let mut interner = Rodeo::new();
        let mut name_ids = names.iter().map(|n| interner.get_or_intern(n));
        let mut final_decls = decls;
        for d in final_decls.iter_mut() {
            if let Some(n) = name_ids.next() {
                match d {
                    Declaration::Resource(r) => r.name = n,
                    Declaration::Method(m) => m.name = n,
                    Declaration::Function(f) => f.name = n,
                    Declaration::Adt(a) => a.name = n,
                    Declaration::Domain(do_) => do_.name = n,
                    Declaration::Axiom(a) => a.name = Some(n),
                }
            }
        }
        Program {
            decls: TiVec::from(final_decls),
            interner,
            groups: Rodeo::new(),
        }
    }

    #[test]
    fn circular_resources_rejected() {
        // A requires B, B requires A.
        let a = MemberId(0);
        let b = MemberId(1);
        let prog = program(
            &["A", "B"],
            vec![resource_requiring(Some(b)), resource_requiring(Some(a))],
        );
        match analyze(prog) {
            Err(AnalysisError::CircularDependency(names)) => {
                assert!(names.contains(&"A".to_string()));
                assert!(names.contains(&"B".to_string()));
            }
            other => panic!("expected circular dependency, got {other:?}"),
        }
    }

    #[test]
    fn chain_then_method_last() {
        // A requires B, B requires C; method M calls A.
        let a = MemberId(0);
        let b = MemberId(1);
        let c = MemberId(2);
        let prog = program(
            &["A", "B", "C", "M"],
            vec![
                resource_requiring(Some(b)),
                resource_requiring(Some(c)),
                resource_requiring(None),
                method_calling(a),
            ],
        );
        let analyzed = analyze(prog).expect("acyclic");
        let m = MemberId(3);
        // Linear chain → unique dependency-first SCC order (all singletons).
        let order: Vec<MemberId> = analyzed.scc_order.iter().map(|scc| scc[0]).collect();
        assert_eq!(order, vec![c, b, a, m]);
        // Stored graph carries the dependency edges.
        assert!(analyzed.dep_graph.contains_edge(c, b));
        assert!(analyzed.dep_graph.contains_edge(b, a));
        assert!(analyzed.dep_graph.contains_edge(a, m));
        // None of these are recursive.
        assert!(analyzed.recursive_scc(a).is_none());
    }

    #[test]
    fn self_recursive_function_accepted() {
        // A function whose body calls itself: a self-looping singleton SCC —
        // permitted (limited-function encoding), grouped as a recursive SCC.
        let f = MemberId(0);
        let prog = program(&["f"], vec![function_calling(f)]);
        let analyzed = analyze(prog).expect("function recursion is accepted");
        assert!(analyzed.dep_graph.contains_edge(f, f));
        let scc = analyzed.recursive_scc(f).expect("f is recursive");
        assert_eq!(scc, HashSet::from_iter([f]));
    }

    #[test]
    fn mutually_recursive_functions_accepted() {
        // f calls g, g calls f: one two-member function SCC — permitted.
        let f = MemberId(0);
        let g = MemberId(1);
        let prog = program(&["f", "g"], vec![function_calling(g), function_calling(f)]);
        let analyzed = analyze(prog).expect("mutual function recursion is accepted");
        let scc = analyzed.recursive_scc(f).expect("f is recursive");
        assert_eq!(scc, HashSet::from_iter([f, g]));
        assert_eq!(analyzed.recursive_scc(g), Some(HashSet::from_iter([f, g])));
    }

    #[test]
    fn mutually_recursive_resources_rejected() {
        // A cyclic SCC containing a non-function (here, resources) is still
        // rejected — only function-only cycles are allowed.
        let a = MemberId(0);
        let b = MemberId(1);
        let prog = program(
            &["A", "B"],
            vec![resource_requiring(Some(b)), resource_requiring(Some(a))],
        );
        assert!(matches!(
            analyze(prog),
            Err(AnalysisError::CircularDependency(_))
        ));
    }

    #[test]
    fn independent_nodes_all_scheduled() {
        // Three independent resources: all scheduled, order unconstrained.
        let prog = program(
            &["A", "B", "C"],
            vec![
                resource_requiring(None),
                resource_requiring(None),
                resource_requiring(None),
            ],
        );
        let analyzed = analyze(prog).expect("acyclic");
        let scheduled: HashSet<MemberId> = analyzed.dep_graph.nodes().collect();
        assert_eq!(
            scheduled,
            HashSet::from_iter([MemberId(0), MemberId(1), MemberId(2)])
        );
    }
}
