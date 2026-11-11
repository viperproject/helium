use crate::vmir::display::VmirDisplay;
use crate::vmir::{Inst, MemberId, TyParams, Type, Val};
use lasso::Spur;
use std::fmt::{self, Display, Formatter};

/// A **pure, heap-free** function. Its value is a plain uninterpreted
/// application in the e-graph — no context heap. A Silver function's contracts
/// are *not* stored here: they are separate boolean functions (`f#requires`,
/// `f#ensures`) recorded in the frontend `contracts` map and stitched as pure
/// `assume`/`assert` at definition and call sites.
///
/// A **heap-dependent** function (one whose `requires` grants permission) is
/// still this same declaration: its `f#requires` is a self-framed `Resource`,
/// and the function takes that resource's snapshot as an ordinary trailing
/// parameter (`Type::Snap(req_id)`). Call sites build the snapshot with
/// a frame-only `exhale` of its `#requires`; the body reconstructs its precondition heap with
/// `HeapInst::a bound inhale` and reads it via `Deref`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Function {
    pub name: Spur,
    /// Generic over this many types (`Generic(0..n)` in `params` / `ret`). Only a
    /// domain function is generic: over the domain parameters its signature
    /// mentions, renumbered `0..n` in order of first occurrence. A call's
    /// [`FunctionCall::type_args`] instantiates them.
    pub ty_params: TyParams,
    pub params: Params,
    pub ret: Type,
    /// The function's definition, when it has a body. `None` ⟹ abstract /
    /// uninterpreted. A function with a body and contracts assumes `f#requires`
    /// at entry and asserts `f#ensures` at exit (via their calls); a boolean
    /// contract function (`f#requires` / `f#ensures`) stores the lowered
    /// pre/postcondition here.
    pub body: Option<FunctionBody>,
    /// Contract link: the precondition, applied to this function's params. Also
    /// set on the generated `f#ensures` decl itself (over its leading params),
    /// so the verifier knows which pre-token guards facts exported from its body.
    pub requires: Option<Requires>,
    /// Contract link: the postcondition member (`f#ensures`, a boolean
    /// `Function`) applied to this function's params plus
    /// [`ContractArg::Result`] (plus the trailing snapshot param when
    /// heap-dependent). `Result` is only expressible here — a precondition
    /// cannot mention the result.
    pub ensures: Option<ContractCall<ContractArg>>,
}

/// A function's precondition link. The two variants are the two function
/// flavours: a heap-free function's precondition is an ordinary boolean
/// function, a heap-dependent one's is a self-framed resource whose snapshot the
/// function takes as a parameter. Keeping them apart here means the invariant
/// "heap-dependent ⟺ `#requires` is a `Resource` ⟺ there is a trailing
/// `Type::Snap` param" holds by construction rather than by three conventions
/// agreeing, and the verifier never has to probe the callee's declaration kind
/// or hunt for the snapshot's param index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Requires {
    /// Heap-free: a boolean `Function` (`f#requires`) applied to the params. It
    /// is *defined* (it has a body), so it doubles as the pre-token guarding the
    /// function's exported facts.
    Pure(ContractCall<Val>),
    /// Heap-dependent: a self-framed `Resource` (footprint + bool) applied to
    /// `args`, plus `snap` — the trailing snapshot parameter that call sites
    /// build with a frame-only `exhale`, and the body opens with a bound `inhale`.
    /// There is no boolean requires-function, so the verifier guards this
    /// function's facts with an uninterpreted pre-token over `args ++ [snap]`,
    /// released where a `Snap`'s implicit precondition check passes.
    Framed {
        resource: MemberId,
        args: Vec<Val>,
        snap: Val,
    },
}

impl Requires {
    /// The precondition member — a boolean `Function` or a self-framed
    /// `Resource` (used for dependency edges, where the flavour is irrelevant).
    pub fn member(&self) -> MemberId {
        match self {
            Requires::Pure(c) => c.member,
            Requires::Framed { resource, .. } => *resource,
        }
    }
}

/// An argument of a contract link: a value over the owning function's params,
/// or — postcondition links only — the function's own result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContractArg {
    Val(Val),
    Result,
}

/// A contract link: `member` applied to explicit `args` (over the owning
/// function's params; `A = ContractArg` additionally admits `Result`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContractCall<A> {
    pub member: MemberId,
    pub args: Vec<A>,
}

/// A pure function body: a stream of pure/heap instructions plus the result
/// `Val`. Mirrors [`crate::vmir::ResourceBody`] but returns a single value with
/// no heap delta (a function produces a value, not a heap change).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionBody {
    pub insts: Vec<Inst>,
    pub res: Val,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Params(Vec<Type>);

impl From<Vec<Type>> for Params {
    fn from(v: Vec<Type>) -> Self {
        Self(v)
    }
}

impl FromIterator<Type> for Params {
    fn from_iter<I: IntoIterator<Item = Type>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl Params {
    pub fn iter(&self) -> std::slice::Iter<'_, Type> {
        self.0.iter()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A function invocation.
///
/// A (possibly generic) function application. `type_args` records the result-type
/// instantiation for the verifier's `FuncApp` payload (empty for a fully-concrete
/// result). Always pure and heap-free: a heap-dependent function receives its
/// precondition snapshot (yielded by the frame-only `exhale` at the call site) as an
/// ordinary trailing argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionCall {
    pub function: MemberId,
    pub type_args: Vec<Type>,
    pub args: Args,
    /// **Export** this call's callee precondition token: when the body containing
    /// this call is replayed at a call site, release the callee's `%pre(args)`
    /// there under this call's path condition (Silicon's
    /// `bodyPreconditionPropagation`). Rendered as a leading `export`.
    ///
    /// Set by the translator, which knows why it emitted the call, rather than
    /// re-derived by the verifier from the call's shape. It is `true` for a
    /// genuine value-position use of a Silver `function` — the one whose result
    /// flows onward, so whose application really is materialized at the caller
    /// and really does need its token. A **contract** definition
    /// (`#requires` / `#ensures`) is no exception: it is an ordinary function and
    /// a call in its body is an ordinary value position, or the fact the contract
    /// carries is stranded at the client. It is `false` for:
    ///
    /// - an **obligation** call — `g#requires(gargs)` before a call, the entry
    ///   `assume f#requires(..)`, the exit `assert f#ensures(..)` — which feeds no
    ///   result, so no enclosing body has to re-mint it when replayed;
    /// - a field or predicate `@addr` application and a domain function, which are
    ///   not Silver `function`s at all: no body recipe, no unfold rule, nothing a
    ///   token could activate;
    /// - every call in a **method body**, which nothing ever replays — there is no
    ///   later site for the token to be re-released at, and the verifier reads
    ///   this field only while building a recipe (`ctx.recipe.is_some()`, which a
    ///   method walk never sets).
    ///
    /// The verifier still mints the token *at* the call while checking this body
    /// itself; that is a different role of the same token (presence here vs.
    /// re-release there) and this flag does not govern it.
    pub export: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Args(Vec<Val>);

impl From<Vec<Val>> for Args {
    fn from(v: Vec<Val>) -> Self {
        Self(v)
    }
}

impl Args {
    pub fn iter(&self) -> std::slice::Iter<'_, Val> {
        self.0.iter()
    }
}

impl Display for VmirDisplay<'_, &'_ Params> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, param) in self.item.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "e{i}: {}", self.with(param))?;
        }
        write!(f, ")")
    }
}

impl<'a> Display for VmirDisplay<'a, &'a Function> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let name = self.interner.resolve(&self.item.name);
        let params = &self.item.params;
        let ret = &self.item.ret;
        write!(
            f,
            "function {name}{}{} -> {}",
            self.item.ty_params,
            self.with(params),
            self.with(ret)
        )?;
        if let Some(rq) = &self.item.requires {
            write!(f, "\n  requires {}", self.with(rq))?;
        }
        if let Some(en) = &self.item.ensures {
            write!(f, "\n  ensures {}", self.with(en))?;
        }
        // Heap-free: body heaps count from `h0`.
        match &self.item.body {
            None => Ok(()),
            Some(body) => {
                writeln!(f, " {{")?;
                write!(
                    f,
                    "{}",
                    self.with((self.item.params.0.len(), 0usize, 0usize, &body.insts[..]))
                )?;
                writeln!(f, "  result: {}", body.res)?;
                write!(f, "}}")
            }
        }
    }
}

impl Display for ContractArg {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ContractArg::Val(v) => write!(f, "{v}"),
            ContractArg::Result => write!(f, "result"),
        }
    }
}

impl Display for VmirDisplay<'_, &'_ Requires> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.item {
            Requires::Pure(c) => write!(f, "{}", self.with(c)),
            // The snapshot rides after the resource args, as it does in the
            // pre-token the verifier builds from this link.
            Requires::Framed {
                resource,
                args,
                snap,
            } => {
                write!(f, "{}(", self.member(*resource))?;
                for a in args {
                    write!(f, "{a}, ")?;
                }
                write!(f, "{snap})")
            }
        }
    }
}

impl<A: Display> Display for VmirDisplay<'_, &'_ ContractCall<A>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.member(self.item.member))?;
        for (i, a) in self.item.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{a}")?;
        }
        write!(f, ")")
    }
}

impl Display for Args {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, arg) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{arg}")?;
        }
        write!(f, ")")
    }
}

impl Display for VmirDisplay<'_, &'_ FunctionCall> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let function = self.member(self.item.function);
        let args = &self.item.args;
        if self.item.export {
            write!(f, "export ")?;
        }
        write!(f, "{function}")?;
        // A generic call shows its full type-argument instantiation in angle
        // brackets (`[..]` is reserved for heaps / addr groups).
        if !self.item.type_args.is_empty() {
            write!(f, "<")?;
            for (i, t) in self.item.type_args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.with(t))?;
            }
            write!(f, ">")?;
        }
        write!(f, "{args}")
    }
}
