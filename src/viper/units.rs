//! Grouping a parsed program into **units** — the granularity at which the
//! pipeline reports and rejects.
//!
//! A unit is one thing a user wrote: a method, a function, a predicate, a
//! field, a macro, a whole `domain` (its functions and axioms included), a
//! whole `adt` (its constructors included), an `import`. The parser splits some
//! of those across several `Declaration`s — a `domain` becomes a `Domain` plus
//! one `DomainElement` per member — and a rejection has to take the group with
//! it, since half a domain is not a domain.
//!
//! Each unit records what it **provides** (the global names it introduces) and
//! what it **uses** (every identifier mentioned anywhere inside it, an
//! over-approximation: a local shadowing a global name is counted as a use of
//! the global). That is enough to answer the one question the pipeline asks:
//! once a unit is rejected, which other units cannot be verified either?

use crate::dhash::{HashMap, HashSet};

use crate::viper::parsed::ast::*;
use crate::viper::walk::{AstWalkable, AstWalker};

/// One reportable declaration group, with its dependency data.
pub struct Unit {
    /// The name this unit is reported under (`m` for `method m`, `D` for
    /// `domain D`, `import <file>` for an import).
    pub name: String,
    /// Indices into `Program.0` of the declarations that make up this unit.
    pub decls: Vec<usize>,
    /// Global names this unit introduces.
    pub provides: Vec<String>,
    /// Every identifier mentioned inside this unit.
    pub uses: HashSet<String>,
}

/// Every unit of a parsed program, in source order.
pub struct Units(pub Vec<Unit>);

impl Units {
    /// Split `program` into units.
    pub fn of(program: &Program) -> Self {
        let mut units: Vec<Unit> = Vec::new();
        // A `domain`/`adt` and its members are consecutive declarations; the
        // members are appended to the group the header opened.
        let mut open: HashMap<String, usize> = HashMap::default();
        let mut last_adt: Option<usize> = None;

        for (idx, decl) in program.0.iter().enumerate() {
            let uses = idents_of(decl);
            match decl {
                Declaration::DomainElement(de) => {
                    let owner = ident_name(&de.domain);
                    match open.get(&owner) {
                        Some(&u) => {
                            let unit = &mut units[u];
                            unit.decls.push(idx);
                            unit.provides.extend(provided_names(decl));
                            unit.uses.extend(uses);
                        }
                        // A member whose header we never saw: report it alone
                        // rather than dropping it.
                        None => units.push(Unit {
                            name: owner,
                            decls: vec![idx],
                            provides: provided_names(decl),
                            uses,
                        }),
                    }
                }
                Declaration::AdtConstructor(_) => match last_adt {
                    Some(u) => {
                        let unit = &mut units[u];
                        unit.decls.push(idx);
                        unit.provides.extend(provided_names(decl));
                        unit.uses.extend(uses);
                    }
                    None => units.push(Unit {
                        name: unit_name(decl),
                        decls: vec![idx],
                        provides: provided_names(decl),
                        uses,
                    }),
                },
                _ => {
                    let name = unit_name(decl);
                    if matches!(decl, Declaration::Domain(_)) {
                        open.insert(name.clone(), units.len());
                    }
                    if matches!(decl, Declaration::Adt(_)) {
                        last_adt = Some(units.len());
                    }
                    units.push(Unit {
                        name,
                        decls: vec![idx],
                        provides: provided_names(decl),
                        uses,
                    });
                }
            }
        }
        Units(units)
    }

    /// Close `rejected` (unit indices) under "uses a name provided by a
    /// rejected unit". Returns the units dragged down, each paired with the
    /// rejected unit it depends on — the one named in its report line.
    ///
    /// Transitive: a unit poisoned in one round provides names in the next, so
    /// a caller of a caller of a rejected method is reported too.
    pub fn dependents_of(&self, rejected: &HashSet<usize>) -> Vec<(usize, String)> {
        let mut poisoned: HashSet<usize> = rejected.clone();
        let mut out: Vec<(usize, String)> = Vec::new();
        loop {
            // Names currently unavailable, each mapped to the unit that lost it.
            let mut lost: HashMap<&str, &str> = HashMap::default();
            for &u in &poisoned {
                for name in &self.0[u].provides {
                    lost.insert(name.as_str(), self.0[u].name.as_str());
                }
            }
            let mut added = false;
            for (idx, unit) in self.0.iter().enumerate() {
                if poisoned.contains(&idx) {
                    continue;
                }
                if let Some((_, culprit)) = unit
                    .uses
                    .iter()
                    .filter_map(|n| lost.get_key_value(n.as_str()))
                    .next()
                {
                    out.push((idx, (*culprit).to_string()));
                    added = true;
                }
            }
            if !added {
                return out;
            }
            poisoned.extend(out.iter().map(|(idx, _)| *idx));
        }
    }

    /// The index of the unit reported under `name`, if any. Used to attach a
    /// later-stage (typecheck, translate) error to the unit it came from.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.0.iter().position(|u| u.name == name)
    }
}

/// The name a declaration is reported under.
pub fn unit_name(decl: &Declaration) -> String {
    match decl {
        Declaration::Import(i) => format!("import {}", i.path),
        Declaration::Define(d) => ident_name(&d.name.0),
        Declaration::Domain(d) => ident_name(&d.name.0),
        Declaration::DomainElement(de) => ident_name(&de.domain),
        Declaration::Field(f) => ident_name(&f.0.idn.0),
        Declaration::Function(f) => ident_name(&f.signature.name.0),
        Declaration::Predicate(p) => ident_name(&p.signature.name.0),
        Declaration::Method(m) => ident_name(&m.signature.name.0),
        Declaration::Adt(a) => ident_name(&a.name.0),
        Declaration::AdtConstructor(c) => ident_name(&c.signature.name.0),
    }
}

/// The name a declaration is reported under, resolving interned identifiers —
/// the form the passes that run after interning need. Agrees with
/// [`unit_name`] on the same declaration.
pub fn unit_name_interned(decl: &Declaration, interner: &crate::viper::Interner) -> String {
    let ident = |i: &Ident| match i {
        Ident::Raw(s) => s.clone(),
        Ident::Interned(spur) => interner.resolve(spur).to_string(),
    };
    match decl {
        Declaration::Import(i) => format!("import {}", i.path),
        Declaration::Define(d) => ident(&d.name.0),
        Declaration::Domain(d) => ident(&d.name.0),
        Declaration::DomainElement(de) => ident(&de.domain),
        Declaration::Field(f) => ident(&f.0.idn.0),
        Declaration::Function(f) => ident(&f.signature.name.0),
        Declaration::Predicate(p) => ident(&p.signature.name.0),
        Declaration::Method(m) => ident(&m.signature.name.0),
        Declaration::Adt(a) => ident(&a.name.0),
        Declaration::AdtConstructor(c) => ident(&c.signature.name.0),
    }
}

/// The global names a declaration introduces. An `adt` provides its own name,
/// its constructors and its destructors (variant field names), because a use of
/// any of them must fall with the `adt`.
fn provided_names(decl: &Declaration) -> Vec<String> {
    match decl {
        Declaration::Adt(a) => std::iter::once(ident_name(&a.name.0))
            .chain(a.variants.iter().flat_map(|v| {
                std::iter::once(ident_name(&v.name.0)).chain(v.fields.iter().filter_map(
                    |f| match f {
                        ArgOrType::Arg(a) => Some(ident_name(&a.idn.0)),
                        ArgOrType::Type(_) => None,
                    },
                ))
            }))
            .collect(),
        Declaration::DomainElement(de) => match &de.kind {
            DomainElementKind::Function(df) => vec![ident_name(&df.signature.name.0)],
            DomainElementKind::Axiom(_) => Vec::new(),
        },
        // An import provides nothing here — its file is never read, which is
        // precisely why it is unsupported.
        Declaration::Import(_) => Vec::new(),
        other => vec![unit_name(other)],
    }
}

fn ident_name(i: &Ident) -> String {
    match i {
        Ident::Raw(s) => s.clone(),
        // Units are built before interning, so this is unreachable in the
        // pipeline; a stable placeholder keeps it total.
        Ident::Interned(spur) => format!("{spur:?}"),
    }
}

/// Every identifier mentioned inside one declaration.
fn idents_of(decl: &Declaration) -> HashSet<String> {
    struct Collect(HashSet<String>);
    impl<'a> AstWalker<'a> for Collect {
        fn walk_ident(&mut self, i: &'a Ident) {
            if let Ident::Raw(s) = i {
                self.0.insert(s.clone());
            }
        }
    }
    let mut c = Collect(HashSet::default());
    decl.walk(&mut c);
    c.0
}
