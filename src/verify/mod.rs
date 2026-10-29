use crate::vmir;

// Declared first: `#[macro_use]` scoping is textual, so every module below
// needs this one above it to see `expr!`.
#[macro_use]
mod expr;

mod analysis;
mod cert;
mod context;
mod declaration;
mod error;
mod func_registry;
mod heap;
mod interval;
pub mod lang;
mod quant;
mod rewrite;
mod stats;
#[cfg(test)]
pub(crate) mod test_support;
mod types;
mod viz;

pub use error::VerifyError;
pub use stats::VerifyStats;

/// Result for one verification unit (method or resource): its name and whether
/// verification succeeded.
pub type VerifyResult = (String, Result<(), VerifyError>);

/// Verify an already-analyzed program in dependency order. Returns one entry
/// per verification unit. Resources are verified self-contained (well-formed
/// side conditions) ahead of the methods that use them; methods are then
/// verified, reusing the resources' established proofs.
pub fn verify(analyzed: &vmir::AnalyzedProgram) -> Vec<VerifyResult> {
    verify_with_stats(analyzed).0
}

/// Like [`verify`], but also returns per-member wall-clock verify times
/// (name + elapsed, 1:1 with the result rows) and the aggregated
/// [`VerifyStats`] (e-graph cost metrics) for the whole run — the latter used
/// by the performance regression tests.
pub fn verify_with_stats(
    analyzed: &vmir::AnalyzedProgram,
) -> (
    Vec<VerifyResult>,
    Vec<(String, std::time::Duration)>,
    VerifyStats,
) {
    stats::reset_stats();
    let program = &analyzed.program;
    let mut results = Vec::new();
    let mut member_times: Vec<(String, std::time::Duration)> = Vec::new();
    // The dependency graph's SCCs in dependency-first order (from `analyze`; may
    // contain recursive function groups). Refine: all function/resource groups
    // before any method group (valid — nothing ever depends on a method).
    // Method groups are always singletons, so this never splits a recursive SCC.
    // Verifying methods last guarantees every function certificate exists by the
    // time the assertion-bearing methods assume the axioms.
    let (early, methods): (Vec<_>, Vec<_>) = analyzed.scc_order.iter().cloned().partition(|g| {
        !g.iter()
            .any(|&id| matches!(program.decls[id], vmir::Declaration::Method(_)))
    });
    let groups = early.into_iter().chain(methods);
    // Resources are verified before the methods that use them (dependency
    // order), so each resource's proof certificate is cached and grafted at
    // call sites rather than re-walking the body.
    let mut certs: crate::dhash::HashMap<vmir::MemberId, cert::ResourceDefinition> =
        crate::dhash::HashMap::default();
    // Verified function bodies, cached in dependency order (callees before
    // callers). Each unit's `assume_axioms` installs one lazy unfold rule per
    // entry here (see `rewrite::function_rule`), which installs the
    // definitional equality `f(args) == body` the moment a `FuncApp(f, ..)`
    // occurrence is seen during that unit's own saturation.
    let mut fn_certs: crate::dhash::HashMap<
        vmir::MemberId,
        std::sync::Arc<cert::FunctionDefinition>,
    > = crate::dhash::HashMap::default();
    // Shared function-id registry: one per run so ADT/builtin ids stay
    // consistent across certificate grafts. Threaded `&mut` into each unit.
    let mut alloc = func_registry::FuncRegistry::new(program);
    for group in groups {
        // A recursion cycle (multi-member SCC, or a self-looping singleton) is a
        // batch of functions verified together: each with its in-SCC calls routed
        // to limited twins (uninterpreted during the batch), and all their
        // certificates inserted into `fn_certs` only after the whole batch is
        // verified — so no member sees another's (or its own) unfold rule while
        // being verified, which is what makes recursion terminate.
        let recursive = group.len() > 1 || analyzed.dep_graph.contains_edge(group[0], group[0]);
        if recursive {
            let scc: crate::dhash::HashSet<vmir::MemberId> = group.iter().copied().collect();
            let mut batch_defs = Vec::new();
            for &id in &group {
                let name = program.name(id).to_string();
                let vmir::Declaration::Function(f) = &program.decls[id] else {
                    unreachable!("analyze guarantees a recursive SCC contains only functions");
                };
                let start = std::time::Instant::now();
                let outcome = match declaration::verify_function(
                    program,
                    &name,
                    id,
                    f,
                    &certs,
                    &fn_certs,
                    Some(&scc),
                    &mut alloc,
                ) {
                    Ok(None) => None,
                    Ok(Some(cert)) => {
                        batch_defs.push((id, cert));
                        Some(Ok(()))
                    }
                    Err(e) => Some(Err(e)),
                };
                let elapsed = start.elapsed();
                if let Some(outcome) = outcome {
                    member_times.push((name.clone(), elapsed));
                    results.push((name, outcome));
                }
            }
            for (id, cert) in batch_defs {
                fn_certs.insert(id, cert);
            }
            continue;
        }
        let id = group[0];
        let name = program.name(id).to_string();
        let start = std::time::Instant::now();
        let outcome = match &program.decls[id] {
            vmir::Declaration::Resource(r) => {
                match declaration::verify_resource(program, &name, r, &certs, &fn_certs, &mut alloc)
                {
                    Ok(cert) => {
                        certs.insert(id, cert);
                        Some(Ok(()))
                    }
                    Err(e) => Some(Err(e)),
                }
            }
            vmir::Declaration::Function(f) => {
                match declaration::verify_function(
                    program, &name, id, f, &certs, &fn_certs, None, &mut alloc,
                ) {
                    Ok(None) => None,
                    Ok(Some(cert)) => {
                        fn_certs.insert(id, cert);
                        // An abstract function's synthesized post axiom is not
                        // a verification — no result row for it.
                        f.body.is_some().then_some(Ok(()))
                    }
                    Err(e) => Some(Err(e)),
                }
            }
            vmir::Declaration::Method(m) => Some(declaration::verify_method(
                program, &name, m, &certs, &fn_certs, &mut alloc,
            )),
            _ => None,
        };
        let elapsed = start.elapsed();
        if let Some(outcome) = outcome {
            member_times.push((name.clone(), elapsed));
            results.push((name, outcome));
        }
    }
    let stats = stats::take_stats();
    (results, member_times, stats)
}
