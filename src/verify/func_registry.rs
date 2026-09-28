//! Verifier-side ADT id allocator.
//!
//! ADT operations (`PureInst::{AdtCons,AdtProj,AdtTag}` and predicate snapshots)
//! are interpreted as `Symbolic::FuncApp`s over **verifier-minted** function ids,
//! distinct from any VMIR declaration id — so no synthetic `@tag`/`@dtor`/
//! constructor declarations exist. Ids are allocated **lazily on first use** of a
//! **concept** `(adt-head)`; minting one mints its whole concept (tag + every
//! constructor + every projection) and appends its reduction rules.
//!
//! The e-graph is **polymorphic**, not monomorphized: a generic op gets **one** id
//! per concept, and its ground type arguments ride in the `FuncApp` operator
//! identity (the discriminant, `Symbolic::FuncApp(FuncId, Box<[Type]>, _)`), not as
//! children. Distinctness across instantiations (`Box[Int]` vs `Box[Bool]`) comes
//! from the differing discriminant, so the reductions are instantiation-agnostic.
//!
//! The allocator is owned by `verify::verify` and threaded `&mut` through each
//! (sequential) verification unit, so a concept gets the **same** id wherever it
//! appears — required for a resource certificate's grafted nodes to
//! congruence-match the caller.

use crate::dhash::HashMap;

use crate::verify::analysis::ConstFold;
use crate::verify::lang::{FuncId, Symbolic};
use crate::verify::rewrite::{inj_rule, proj_rule, tag_rule};
use crate::vmir::{Declaration, MemberId, Program, Type};

type Rule = egg::Rewrite<Symbolic, ConstFold>;

/// Lazily allocates and names the verifier ids for polymorphic ADT
/// constructors / projections / tags, and accumulates their reduction rules.
pub struct FuncRegistry {
    /// Next func id to mint (starts past every real declaration id, so a minted
    /// id never collides with a plain function reusing its declaration index).
    next: usize,
    /// Keyed by **concept**, not by type instantiation: one id per `(head,
    /// variant)` / `(head, variant, field)` / `head`. The e-graph is polymorphic —
    /// the ground type args ride in the `FuncApp` discriminant, so a single id
    /// serves every instantiation (distinctness comes from the differing
    /// discriminant, not from per-instance ids or type e-classes).
    cons: HashMap<(MemberId, usize), FuncId>,
    proj: HashMap<(MemberId, usize, usize), FuncId>,
    tag: HashMap<MemberId, FuncId>,
    /// Limited-function twin id per recursive function (see
    /// [`FuncRegistry::limited`]): an uninterpreted `f'` with no unfold rule,
    /// used to break recursive unfolding.
    limited: HashMap<MemberId, FuncId>,
    /// Precondition-token id keyed on the **function member**, over the
    /// function's own args `fargs`. Minted for every function flavour
    /// identically: it is the `forall`-style presence trigger that gates the
    /// definitional body-unfold (`rewrite::function_rule`), and its *truth*,
    /// released at the call under that call's pc, is what lets the function's
    /// facts fire. The one and only precondition token.
    fn_pre_token: HashMap<MemberId, FuncId>,
    /// Display names for minted ids (which are outside the interner).
    names: HashMap<FuncId, String>,
    rules: Vec<Rule>,
    /// Per ADT *head* (an `Adt` decl id, or a predicate's own Resource id for its
    /// snapshot ADT — see `Type::Snap`), its per-variant field counts — the shape
    /// needed to mint an instance.
    shapes: HashMap<MemberId, Vec<usize>>,
    /// Base display name per ADT head.
    head_names: HashMap<MemberId, String>,
    /// Per ADT head, the source constructor name of each variant (`None` for a
    /// synthetic variant — e.g. a snapshot's sole constructor). Drives minted-id
    /// names: `Adt::Ctor` when named, `Adt#i` when anonymous.
    variant_names: HashMap<MemberId, Vec<Option<String>>>,
    /// Every minted constructor id, mapped to the ADT head it belongs to. Read by
    /// the `ConstFold` analysis to decide constructor distinctness (two different
    /// constructors of one head, at one instantiation, sharing an e-class ⇒ that
    /// class is contradictory). It must be complete before any e-graph exists,
    /// which is why [`FuncRegistry::new`] mints every head eagerly rather than on
    /// first use.
    ctor_head: HashMap<FuncId, MemberId>,
    /// The `forall`s **reached so far**, compiled on demand (see
    /// [`crate::verify::quant::intern_forall`]). It lives here because interning a
    /// recipe needs the `FuncId` minting this registry owns. Shared with the single
    /// instantiation rule, which only reads: a recipe is added on the eval walk,
    /// never from inside a rule, and egg sees the resulting e-node on the next
    /// iteration either way.
    quant_table: std::sync::Arc<std::sync::RwLock<crate::verify::quant::RecipeTable>>,
}

// Builtin reserved operator ids (starting from the top of the usize space).
pub const BUILTIN_OPTION_SOME: FuncId = FuncId(usize::MAX - 1);
pub const BUILTIN_OPTION_NONE: FuncId = FuncId(usize::MAX - 2);
pub const BUILTIN_OPTION_VALUE: FuncId = FuncId(usize::MAX - 3);
pub const BUILTIN_OPTION_TAG: FuncId = FuncId(usize::MAX - 4);
/// Synthetic ADT head for the builtin `Option` (which has no `Adt` declaration).
pub const BUILTIN_OPTION_HEAD: MemberId = MemberId(usize::MAX);

impl FuncRegistry {
    /// Build an allocator for `program`: records the shape of every ADT head
    /// (ADT declarations and predicate snapshots, keyed by the predicate's own
    /// Resource id) so instances can be minted on demand. Mints nothing yet.
    pub fn new(program: &Program) -> Self {
        let mut shapes = HashMap::default();
        let mut head_names = HashMap::default();
        let mut variant_names: HashMap<MemberId, Vec<Option<String>>> = HashMap::default();
        let vname = |adt: &crate::vmir::Adt| -> Vec<Option<String>> {
            adt.variants
                .iter()
                .map(|v| v.name.map(|n| program.interner.resolve(&n).to_string()))
                .collect()
        };

        for (id, decl) in program.decls.iter_enumerated() {
            if let Declaration::Adt(adt) = decl {
                shapes.insert(
                    id,
                    adt.variants.iter().map(|v| v.field_types.len()).collect(),
                );
                head_names.insert(id, program.name(id).to_string());
                variant_names.insert(id, vname(adt));
            }
        }
        for (id, decl) in program.decls.iter_enumerated() {
            if let Declaration::Resource(r) = decl
                && let Some(adt) = r.derive_snapshot()
            {
                shapes.insert(
                    id,
                    adt.variants.iter().map(|v| v.field_types.len()).collect(),
                );
                head_names.insert(id, format!("{}@snap", program.name(id)));
                variant_names.insert(id, vname(&adt));
            }
        }

        let mut names = HashMap::default();
        names.insert(BUILTIN_OPTION_SOME, "Option::Some".to_string());
        names.insert(BUILTIN_OPTION_NONE, "Option::None".to_string());
        names.insert(BUILTIN_OPTION_VALUE, "Option::Some.0".to_string());
        names.insert(BUILTIN_OPTION_TAG, "Option@tag".to_string());

        let mut rules = Vec::new();
        rules.push(proj_rule(BUILTIN_OPTION_VALUE, BUILTIN_OPTION_SOME, 0));
        rules.push(inj_rule(BUILTIN_OPTION_SOME));
        let mut option_tags = HashMap::default();
        option_tags.insert(BUILTIN_OPTION_SOME, 0);
        option_tags.insert(BUILTIN_OPTION_NONE, 1);
        rules.push(tag_rule(BUILTIN_OPTION_TAG, option_tags));
        let mut ctor_head = HashMap::default();
        ctor_head.insert(BUILTIN_OPTION_SOME, BUILTIN_OPTION_HEAD);
        ctor_head.insert(BUILTIN_OPTION_NONE, BUILTIN_OPTION_HEAD);

        let mut registry = FuncRegistry {
            next: program.decls.len(),
            cons: HashMap::default(),
            proj: HashMap::default(),
            tag: HashMap::default(),
            limited: HashMap::default(),
            fn_pre_token: HashMap::default(),
            names,
            rules,
            shapes,
            head_names,
            variant_names,
            ctor_head,
            quant_table: Default::default(),
        };
        // Mint every head up front. Lazily minting on first use would leave
        // `ctor_head` incomplete for any `ConstFold` built before that use, and
        // the analysis is handed an immutable snapshot of the table.
        let heads: Vec<MemberId> = registry.shapes.keys().copied().collect();
        for head in heads {
            registry.ensure(head);
        }
        registry
    }

    /// The program's compiled `forall`s, filled in **as bodies are walked** —
    /// a recipe exists only once the quantifier that needs it has been reached.
    /// Cheap to clone (`Arc`); the single instantiation rule holds the same handle
    /// and only ever reads through it.
    pub(crate) fn quant_table(
        &self,
    ) -> &std::sync::Arc<std::sync::RwLock<crate::verify::quant::RecipeTable>> {
        &self.quant_table
    }

    /// An empty allocator (no ADT heads). For tests / programs without ADTs.
    #[cfg(test)]
    pub fn empty() -> Self {
        let mut names = HashMap::default();
        names.insert(BUILTIN_OPTION_SOME, "Option::Some".to_string());
        names.insert(BUILTIN_OPTION_NONE, "Option::None".to_string());
        names.insert(BUILTIN_OPTION_VALUE, "Option::Some.0".to_string());
        names.insert(BUILTIN_OPTION_TAG, "Option@tag".to_string());

        let mut rules = Vec::new();
        rules.push(proj_rule(BUILTIN_OPTION_VALUE, BUILTIN_OPTION_SOME, 0));
        rules.push(inj_rule(BUILTIN_OPTION_SOME));
        let mut option_tags = HashMap::default();
        option_tags.insert(BUILTIN_OPTION_SOME, 0);
        option_tags.insert(BUILTIN_OPTION_NONE, 1);
        rules.push(tag_rule(BUILTIN_OPTION_TAG, option_tags));
        let mut ctor_head = HashMap::default();
        ctor_head.insert(BUILTIN_OPTION_SOME, BUILTIN_OPTION_HEAD);
        ctor_head.insert(BUILTIN_OPTION_NONE, BUILTIN_OPTION_HEAD);

        FuncRegistry {
            next: 0,
            cons: HashMap::default(),
            proj: HashMap::default(),
            tag: HashMap::default(),
            limited: HashMap::default(),
            fn_pre_token: HashMap::default(),
            names,
            rules,
            quant_table: Default::default(),
            shapes: HashMap::default(),
            head_names: HashMap::default(),
            variant_names: HashMap::default(),
            ctor_head,
        }
    }

    /// Constructor id for variant `variant` of `adt` (minting the concept on first
    /// use). Polymorphic — one id for every instantiation; the ground type args
    /// ride in the `FuncApp` discriminant, never key the id.
    pub fn cons(&mut self, adt: MemberId, variant: usize) -> FuncId {
        self.ensure(adt);
        self.cons[&(adt, variant)]
    }

    /// Field-`field` projection id of variant `variant` of `adt`.
    pub fn proj(&mut self, adt: MemberId, variant: usize, field: usize) -> FuncId {
        self.ensure(adt);
        self.proj[&(adt, variant, field)]
    }

    /// Discriminator-tag id of `adt`.
    pub fn tag(&mut self, adt: MemberId) -> FuncId {
        self.ensure(adt);
        self.tag[&adt]
    }

    /// The limited-function twin id `f'` of recursive function `func` (minting it
    /// on first use). `f'` is **uninterpreted** — no unfold rule is ever
    /// registered for it — so a recursive call routed through `f'` never unfolds
    /// further, bounding saturation. A full `f`'s unfold rule additionally frames
    /// `f(x) == f'(x)` so a materialized `f(x)` value flows to its twin. `name` is
    /// the function's source name (the registry holds no interner), used only for
    /// the display label `name#lim`.
    pub fn limited(&mut self, func: MemberId, name: &str) -> FuncId {
        if let Some(&id) = self.limited.get(&func) {
            return id;
        }
        let id = self.mint(format!("{name}#lim"));
        self.limited.insert(func, id);
        id
    }

    /// The uniform **function** precondition token `f%pre` (minting it on first
    /// use), keyed on the function member and applied to the function's own args
    /// `fargs`. Uninterpreted — no unfold rule — it is only ever *added* (present)
    /// at a genuine value-position call line; its presence is the trigger that
    /// lets [`crate::verify::rewrite::function_rule`] unfold `f(fargs)==body`.
    /// Identical for heap-free, heap-dep, and precondition-free functions.
    pub fn fn_pre_token(&mut self, func: MemberId, name: &str) -> FuncId {
        if let Some(&id) = self.fn_pre_token.get(&func) {
            return id;
        }
        let id = self.mint(format!("{name}%pre"));
        self.fn_pre_token.insert(func, id);
        id
    }

    // ---- Builtin `Option` resolution -------------------------------------
    // `Option[T]` is a builtin parametric type (`vmir::Type::Option`). These are
    // the dedicated way to request its monomorphic instance, rather than open-
    // coding `cons`/`proj`. (Future `Seq`/`Set` follow the same shape.)

    /// `Option[elem]` as a vmir type (the builtin parametric `Type::Option`).
    pub fn option_type(&self, elem: Type) -> Type {
        Type::Option(Box::new(elem))
    }

    /// Constructor id of `Some` (variant 0) of `Option`.
    pub fn option_some(&mut self) -> FuncId {
        BUILTIN_OPTION_SOME
    }

    /// Constructor id of `None` (variant 1) of `Option`.
    pub fn option_none(&mut self) -> FuncId {
        BUILTIN_OPTION_NONE
    }

    /// Projection id recovering the `Some` payload of `Option`.
    pub fn option_value(&mut self) -> FuncId {
        BUILTIN_OPTION_VALUE
    }

    /// The reduction rules minted so far, to inject into a context's runner.
    /// An immutable snapshot of the constructor → ADT-head table, for the
    /// `ConstFold` analysis. Complete: `new` mints every head eagerly.
    pub fn ctor_table(&self) -> std::sync::Arc<HashMap<FuncId, MemberId>> {
        std::sync::Arc::new(self.ctor_head.clone())
    }

    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Display name for a minted id (`None` if `f` is not allocator-minted).
    pub fn name(&self, f: FuncId) -> Option<&str> {
        self.names.get(&f).map(String::as_str)
    }

    /// Mint the **concept** `adt` (tag + all constructors + all projections +
    /// reduction rules) if not already present. One concept covers every
    /// instantiation; the ground type args live in the `FuncApp` discriminant, so
    /// the reductions are instantiation-agnostic.
    fn ensure(&mut self, adt: MemberId) {
        if self.tag.contains_key(&adt) {
            return;
        }
        let counts = self
            .shapes
            .get(&adt)
            .unwrap_or_else(|| panic!("unknown ADT head {}", adt.0))
            .clone();
        let label = self.label(adt);

        let tag_id = self.mint(format!("{label}@tag"));
        self.tag.insert(adt, tag_id);

        let mut ctor_tags = HashMap::default();
        for (variant, &fields) in counts.iter().enumerate() {
            // `Adt::Ctor` for a named constructor, `Adt::#i` for an anonymous one.
            let cons_label = match self.variant_name(adt, variant) {
                Some(name) => format!("{label}::{name}"),
                None => format!("{label}::#{variant}"),
            };
            let cons_id = self.mint(cons_label.clone());
            self.cons.insert((adt, variant), cons_id);
            self.ctor_head.insert(cons_id, adt);
            ctor_tags.insert(cons_id, variant);
            if fields > 0 {
                self.rules.push(inj_rule(cons_id));
            }
            for field in 0..fields {
                let proj_id = self.mint(format!("{cons_label}.{field}"));
                self.proj.insert((adt, variant, field), proj_id);
                self.rules.push(proj_rule(proj_id, cons_id, field));
            }
        }
        self.rules.push(tag_rule(tag_id, ctor_tags));
    }

    /// The source constructor name of `adt`'s `variant`, if any.
    fn variant_name(&self, adt: MemberId, variant: usize) -> Option<&str> {
        self.variant_names
            .get(&adt)
            .and_then(|v| v.get(variant))
            .and_then(|n| n.as_deref())
    }

    fn mint(&mut self, name: String) -> FuncId {
        let id = FuncId(self.next);
        self.next += 1;
        self.names.insert(id, name);
        id
    }

    /// A readable label for the head `adt`, e.g. `Option` or `Box`. One id now
    /// serves every instantiation, so the label carries no type arguments.
    fn label(&self, adt: MemberId) -> String {
        self.head_names
            .get(&adt)
            .cloned()
            .unwrap_or_else(|| format!("d{}", adt.0))
    }
}

/// Derives the verifier `FuncId` for a plain, declaration-backed function.
pub fn func_id_for_member(member: MemberId) -> FuncId {
    FuncId(usize::from(member))
}
