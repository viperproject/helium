use crate::viper::parsed::ast::*;

peg::parser! {
    pub grammar viper_parser() for str {
        rule _ = quiet! { ___ __ ** ___ ___ }

        rule white_space() = quiet! { " " / "\t" / "\n" / "\r\n" } / expected!("whitespace")
        rule ___ = white_space()*

        rule __ = "//" (! "\n" [_])* / "/*" (! "*/" [_])* "*/"

        rule start_char() -> &'input str
            = $(['A'..='Z' | 'a'..='z'| '$' | '_' ])

        rule char() -> &'input str
            = $(['A'..='Z' | 'a'..='z'| '$' | '_' | '\'' | '0'..='9' ])

        rule reserved()
            = "havoc" / "true" / "false"
            / "Set" / "Seq" / "Map" / "Multiset" / "Int" / "Bool" / "Perm" / "Ref" / "Rational"
            / "forperm" / "let" / "in" / "if" / "else" / "elseif" / "while" / "do"
            / "assert" / "assume" / "havoc"
            / "return" / "continue" / "skip"
            / "forall" / "exists"
            / "inhale" / "exhale" / "unfold" / "fold" / "acc"
            / "none" / "wildcard" / "write" / "epsilon"
            / "requires" / "ensures" / "returns" / "decreases" / "result"

        rule ident() -> Ident
            = quiet! { !(reserved() white_space()) n:$(start_char() char()*) { Ident::Raw(n.to_string())} }
            / expected!("identifier")

        rule idndecl() -> IdnDecl = n:ident() { IdnDecl(n) }

        rule label() -> Ident
            = quiet! {  n:$(start_char() char()*) { Ident::Raw(n.to_string())} }
            / expected!("label")

        rule kw<R>(r: rule<R>) -> () = r() !char()

        rule integer() -> num::BigInt = s:$("-"? ['0'..='9']+) {? s.parse().or(Err("invalid integer")) }

        rule comma() = _ "," _

        rule annotation() = quiet! { "@" annotation_ident() annotation_args() }  / expected!("annotation")

        rule annotation_ident() = ident() ++ "."

        rule annotation_args() = "(" _ string_lit() ** comma() _ ")"

        // TODO: insert annotations
        rule annotated<R>(r: rule<R>) -> R = annotation() ** _ _ r:r() { r }

        /// Types

        rule type_() -> Type
            =
              "Int" { Type::Int }
            / "Bool" { Type::Bool }
            / "Perm" { Type::Real }
            / "Ref" { Type::Ref }
            / "Rational"   { Type::Real }
            // The collection types are parsed but not implemented (expect Seq and Set): they are kept
            // as ordinary domain types under their own name so that
            // `scan_unsupported` can name the construct and reject only the
            // declaration that mentions one, instead of failing the whole file
            // with a parse error.
            / "Seq" _ "[" _ ty:type_() _ "]" { Type::Seq(Box::new(ty)) }
            / "Set" _ "[" _ ty:type_() _ "]" { Type::Set(Box::new(ty)) }
            / "Multiset" _ "[" _ ty:type_() _ "]" { Type::Domain(Ident::Raw("Multiset".to_string()), vec![ty]) }
            / "Map" _ "[" _ a:type_() _ "," _ b:type_() _ "]" { Type::Domain(Ident::Raw("Map".to_string()), vec![a, b]) }
            / type_constr()

        rule type_constr() -> Type = nm:ident() _ tys:("[" _ tys:(type_() ** comma()) _ "]" { tys })?
            {   let tys = tys.unwrap_or_default();
                Type::Domain(nm, tys)
            }

        rule formal_arg() -> IdnDeclTyped = idn:idndecl() _ ":" _ ty:type_() { IdnDeclTyped { idn, ty } }

        rule decl_named_formal_arg() -> ArgOrType = a:formal_arg()  { ArgOrType::Arg(a) }

        rule decl_formal_arg() -> ArgOrType = decl_named_formal_arg() / t:type_() { ArgOrType::Type(t) }
        /// Accessors
        rule res_access() -> ResAccess
            = e:acc_exp() { ResAccess::Exp(e) }
            / e:suffix_exp() { ResAccess::Loc(Exp::impure(e)) }

        rule acc_exp() -> AccExp
            = "acc" _ "(" _ loc:suffix_exp() _ perm:("," _ e:exp() { e })? _ ")" {
                AccExp { loc: Exp::impure(loc), perm: perm.unwrap_or_else(ExpKind::write) }
            }

        /// A `fold`/`unfold` target: either `acc(P(args), perm)` or a bare
        /// predicate instance `P(args)` (full permission).
        rule fold_target() -> AccExp
            = e:acc_exp() { e }
            / loc:suffix_exp() { AccExp { loc: Exp::impure(loc), perm: ExpKind::write() } }

        rule trigger() -> Trigger = "{" _ es:(exp() ** comma()) _ "}" { Trigger { exp: es } }


        /// Expressions

        /// A sequence literal: `Seq[T]()`, `Seq()`, `Seq(e1, ..., en)`, or
        /// `Seq[T](e1, ..., en)`. `kw` keeps `Sequence(..)` a function call.
        rule seq_exp() -> ExpKind
            = kw(<"Seq">) _ ty:("[" _ ty:type_() _ "]" { ty })? _ "(" _ es:(exp() ** comma()) _ ")"
                { ExpKind::SeqLit(ty, es) }

        /// A set literal, written like a sequence literal. `kw` keeps
        /// `Settings(..)` a function call.
        rule set_exp() -> ExpKind
            = kw(<"Set">) _ ty:("[" _ ty:type_() _ "]" { ty })? _ "(" _ es:(exp() ** comma()) _ ")"
                { ExpKind::SetLit(ty, es) }

        /// The other collection literals. Parsed into an `Unsupported` marker
        /// rather than a real expression: the construct is named, the
        /// declaration around it is rejected, and the rest of the file still runs.
        rule collection_exp() -> ExpKind
            = n:$("Multiset" / "Map") _ ("[" _ type_() ** comma() _ "]")? _ "(" _ collection_elems() _ ")"
                { ExpKind::Unsupported(collection_name(n)) }

        rule collection_elems() = (e:exp() _ (":=" _ exp())? {}) ** comma()

        /// `|s|` — the length of a sequence, the cardinality of a set.
        rule cardinality_exp() -> ExpKind = "|" _ e:exp() _ "|" { ExpKind::UnOp(UnOp::Cardinality, e) }

        /// `[a..b)` — the sequence of the integers from `a` (inclusive) to `b`
        /// (exclusive).
        rule range_exp() -> ExpKind = "[" _ a:exp() _ ".." _ b:exp() _ ")" { ExpKind::BinOp(BinOp::Range, a, b) }

        // rule set_constructor_exp() -> ExpKind
        //     = "Set" _ "[" _ ty:type_() _ "]" _ "(" _ ")" { ExpKind::Ascribe(Box::new(ExpKind::Call(Call { kind: None, name: Ident::set(), args: Vec::new() })), Type::Domain(Ident::set(), vec![ty])) }
        //     / "Set" _ "(" _ es:(exp() ** comma()) _ ")" { ExpKind::Call(Call { kind: None, name: Ident::set(), args: es }) }
        //     / "Multiset" _ "[" _ ty:type_() _ "]" _ "(" _ ")"{ ExpKind::Ascribe(Box::new(ExpKind::Call(Call { kind: None, name: Ident::multiset(), args: Vec::new() })), Type::Domain(Ident::multiset(), vec![ty])) }
        //     / "Multiset" _ "(" _ es:(exp() ** comma()) _ ")" { ExpKind::Call(Call { kind: None, name: Ident::multiset(), args: es }) }
        //
        // rule map_constructor_exp() -> ExpKind
        //     = "Map" _ "[" _ a:type_() _ "," _ b:type_() _ "]" _ "(" _ ")" { ExpKind::Ascribe(Box::new(ExpKind::Call(Call { kind: None, name: Ident::map(), args: Vec::new() })), Type::Domain(Ident::map(), vec![a, b])) }
        //     / "Map" _ "(" _ es:((l:exp() _ ":=" _ r:exp() { (l, r)}) ** comma()) _ ")" {
        //         es.into_iter().fold(ExpKind::Call(Call { kind: None, name: Ident::map(), args: Vec::new() }), |acc, (l, r)| {
        //             ExpKind::Index(Box::new(acc), IndexOp::Assign(l, r))
        //         })
        //     }

        rule forperm_exp() -> ExpKind = "forperm" _ args:(formal_arg() ++ comma()) _ "[" _ res:res_access() _ "]" _ "::" _ exp:exp()
            { ExpKind::ForPerm(args, res, exp) }

        rule let_in_exp() -> ExpKind = "let" _ id:idndecl() _ "==" _ "(" _ e:exp() _ ")" _ "in" _ body:exp()
            { ExpKind::LetIn(id, e, body) }

        rule func_app() -> ExpKind = id:ident() (" ")* "(" _ es:(exp() ** comma()) _ ")" { ExpKind::Call(Call { kind: None, name: id, args: es }) }

        rule atom() -> ExpKind
            = kw(<"true">) { ExpKind::Const(ConstKind::Bool(true)) }
            / kw(<"false">) { ExpKind::Const(ConstKind::Bool(false)) }
            / i:integer() { ExpKind::Const(ConstKind::Int(i)) }
            / kw(<"null">) { ExpKind::Const(ConstKind::Null) }
            / kw(<"result">) { ExpKind::Result }
            / "(" _ e:exp_kind() _ ty:(":" _ ty:type_() { ty })? _ ")" { match ty {
                Some(ty) => ExpKind::Ascribe(Exp::unknown(e), ty),
                None => e
                }
            }
            / kw(<"old">) _ i:("[" _ i:ident() _ "]" {i})? _ "(" _ e:exp() _ ")" { ExpKind::Old(i, e) }
            // / "[" _ i:ident() _ "]" _ "(" _ e:exp() _ ")" { ExpKind::At(i, Box::new(e)) }
            / kw(<"lhs">) _ "(" _ exp() _ ")" { ExpKind::Unsupported("lhs") }
            / kw(<"none">) { ExpKind::Const(ConstKind::Real(num::BigInt::from(0).into())) }
            / kw(<"write">) { ExpKind::Const(ConstKind::Real(num::BigInt::from(1).into())) }
            / kw(<"epsilon">) { ExpKind::Const(ConstKind::Epsilon) }
            / kw(<"wildcard">) { ExpKind::Const(ConstKind::Wildcard) }
            / kw(<"perm">) _ "(" _ l:exp() _ ")" { ExpKind::UnOp(UnOp::Perm, l) }
            / "[" _ e:exp() _ "," _ f:exp() _ "]" { ExpKind::BinOp(BinOp::InhaleExhale, e, f)}

            / kw(<"unfolding">) _ acc:fold_target() _ "in" _ e:exp() { ExpKind::HeapUpdate(HeapUpdateOp::Unfold, acc, e) }
            / kw(<"folding">) _ fold_target() _ "in" _ exp() { ExpKind::Unsupported("folding") }
            / kw(<"applying">) _ "(" _ exp() _ ")" _ "in" _ exp() { ExpKind::Unsupported("applying") }
            / kw(<"packaging">) _ "(" _ exp() _ ")" _ "in" _ exp() { ExpKind::Unsupported("packaging") }
            / kw(<"forall">) _ args:(formal_arg() ++ comma()) _ "::" _ triggers:(trigger()**_) _ e:exp() { ExpKind::Quantifier(QuantifierKind::Forall, args, triggers, e) }
            / kw(<"exists">) _ args:(formal_arg() ++ comma()) _ "::" _ triggers:(trigger()**_) _ e:exp() { ExpKind::Quantifier(QuantifierKind::Exists, args, triggers, e) }

            / seq_exp()
            / set_exp()
            / collection_exp()
            / cardinality_exp()
            / range_exp()
            / let_in_exp()
            / forperm_exp()
            / a:acc_exp() { ExpKind::Acc(a) }
            / func_app()
            / i:ident() { ExpKind::Ident(i) }



        rule full_exp() -> ExpKind = precedence! {
            x:@ z:(_ "?" _ z:exp() _ ":" _ {z}) y:(@) { ExpKind::Ternary(Exp::unknown(x), z, Exp::unknown(y)) }
            --
            x:@ (_ "<==>" _) y:(@) { ExpKind::BinOp(BinOp::Iff, Exp::unknown(x), Exp::unknown(y)) }
            --
            x:@ (_ "==>" _) y:(@) { ExpKind::BinOp(BinOp::Implies, Exp::unknown(x), Exp::unknown(y)) }
            --
            x:@ (_ "--*" _) y:(@) { ExpKind::MagicWand(Exp::unknown(x), Exp::unknown(y)) }
            --
            x:@ (_ "||" _) y:(@) { ExpKind::BinOp(BinOp::Or, Exp::unknown(x), Exp::unknown(y)) }
            --
            x:@ (_ "&&" _) y:(@) { ExpKind::BinOp(BinOp::And, Exp::unknown(x), Exp::unknown(y)) }
            --
            x:@ (_ "!=" _) y:(@) { ExpKind::BinOp(BinOp::Neq, Exp::unknown(x), Exp::unknown(y)) }
            x:@ (_ "==" _) y:(@) { ExpKind::BinOp(BinOp::Eq, Exp::unknown(x), Exp::unknown(y)) }
            --
            x:@ (_ "<=" _) y:(@) { ExpKind::BinOp(BinOp::Le, Exp::unknown(x), Exp::unknown(y)) }
            x:@ (_ ">=" _) y:(@) { ExpKind::BinOp(BinOp::Ge, Exp::unknown(x), Exp::unknown(y)) }
            x:@ (_ ">" _) y:(@) {  ExpKind::BinOp(BinOp::Gt, Exp::unknown(x), Exp::unknown(y)) }
            x:@ (_ "<" _) y:(@) { ExpKind::BinOp(BinOp::Lt, Exp::unknown(x), Exp::unknown(y)) }
            x:@ (_ "in" &(white_space() / "(") _) y:(@) { ExpKind::BinOp(BinOp::In, Exp::unknown(x), Exp::unknown(y))}
            --
            x:(@) (_ "-" _) y:@ { ExpKind::BinOp(BinOp::Minus, Exp::unknown(x), Exp::unknown(y)) }
            x:(@) (_ "+" _) y:@ { ExpKind::BinOp(BinOp::Plus, Exp::unknown(x), Exp::unknown(y)) }
            x:(@) (_ "++" _) y:@ { ExpKind::BinOp(BinOp::Concat, Exp::unknown(x), Exp::unknown(y)) }
            x:(@) (_ "union" white_space() _) y:@ { ExpKind::BinOp(BinOp::Union, Exp::unknown(x), Exp::unknown(y)) }
            x:(@) (_ "setminus" white_space() _) y:@ { ExpKind::BinOp(BinOp::SetMinus, Exp::unknown(x), Exp::unknown(y))}
            x:(@) (_ "intersection" white_space() _) y:@ { ExpKind::BinOp(BinOp::Intersection, Exp::unknown(x), Exp::unknown(y))}
            x:(@) (_ "subset" white_space() _) y:@ { ExpKind::BinOp(BinOp::Subset, Exp::unknown(x), Exp::unknown(y))}
            --
            x:(@) (_ "*" _) y:@ { ExpKind::BinOp(BinOp::Mult, Exp::unknown(x), Exp::unknown(y)) }
            x:(@) (_ "/" _) y:@ { ExpKind::BinOp(BinOp::Div, Exp::unknown(x), Exp::unknown(y)) }
            x:(@) (_ "\\" _) y:@ { ExpKind::BinOp(BinOp::IntDiv, Exp::unknown(x), Exp::unknown(y)) }
            x:(@) (_ "%" _) y:@ { ExpKind::BinOp(BinOp::Mod, Exp::unknown(x), Exp::unknown(y)) }
            "-" _ x:@ { ExpKind::UnOp(UnOp::Neg, Exp::unknown(x)) }
            "!" _ x:@ { ExpKind::UnOp(UnOp::Not, Exp::unknown(x)) }
            --
            x:@ i:(_ "." i:ident() {i}) { ExpKind::Field(Exp::unknown(x), i) }
            x:@ _ "[" _ s:seq_op() _ "]" _  { ExpKind::Index(Exp::unknown(x), s) }
            --
            a:atom() {a}
        }

        rule exp_kind() -> ExpKind = annotated(<full_exp()>)

        pub(super) rule exp() -> Exp = e:exp_kind() { Exp::unknown(e) }

        rule suffix_exp() -> ExpKind = a:atom() _ suff:(("." id:ident() { Ok(id) } / "[" _ e:exp() _ "]" { Err(e) }) ** _)
            {
                let mut res = a;
                for s in suff {
                    match s {
                        Ok(id) => res = ExpKind::Field(Exp::unknown(res), id),
                        Err(e) => res = ExpKind::Index(Exp::unknown(res), IndexOp::Index(e))
                    }
                }
                res
            }

        rule seq_op() -> IndexOp
            = ".." _ e:exp() { IndexOp::UpperBound(e) }
            / e:exp() _ ".." _ f:exp()? { match f {
                Some(f) => IndexOp::Range(e, f),
                None => IndexOp::LowerBound(e)
            } }
            / e:exp() _ ":=" _ f:exp() { IndexOp::Assign(e, f) }
            / e:exp() { IndexOp::Index(e) }

        /// Statements

        rule block() -> StmtBlock = "{" _ s:(s:annotated(<statement()>) opt_semi() { s})* "}" { Block(s) }

        rule block_exp() -> ExpBlock = "{" _ e:exp() _ "}" { Block(e) }

        rule statement() -> Statement
            = kw(<"assert">) _ e:exp() { Statement::Assert(e)}
            / kw(<"refute">) _ e:exp() { Statement::Refute(e)}
            / kw(<"assume">) _ e:exp() { Statement::Assume(e)}
            / kw(<"inhale">) _ e:exp() { Statement::Inhale(e)}
            / kw(<"exhale">) _ e:exp() { Statement::Exhale(e)}
            / kw(<"fold">) _ e:fold_target() { Statement::Fold(e)}
            / kw(<"unfold">) _ e:fold_target() { Statement::Unfold(e)}
            / kw(<"goto">) _ id:label() { Statement::Goto(id)}
            / kw(<"label">) _ id:label() _ invs:(invariant() ** _) { Statement::Label(IdnDecl(id), invs)}
            / kw(<"var">) _ args:(formal_arg() ** comma()) _ e:(":=" _ e:assign_rhs() {e})? { Statement::Var(args, e)}
            / while_statement()
            / if_statement()
            // Statements we do not implement. Each is parsed into an
            // `Unsupported` marker so the declaration containing it is reported
            // by name and the rest of the file still verifies.
            / kw(<"package">) _ exp() _ block()? { Statement::Unsupported("package") }
            / kw(<"apply">) _ exp() { Statement::Unsupported("apply") }
            / kw(<"quasihavocall">) _ (formal_arg() ++ comma()) _ "::" _ exp() { Statement::Unsupported("quasihavocall") }
            / kw(<"quasihavoc">) _ exp() { Statement::Unsupported("quasihavoc") }
            / kw(<"fresh">) _ (ident() ++ comma()) { Statement::Unsupported("fresh") }
            / constraining_block() { Statement::Unsupported("constraining") }
            // `a, b := e1, e2`: Viper binds several targets only from a method
            // call, so this is reported as unsupported rather than as a parse
            // error that names the comma.
            / multi_assign_stmt()
            / assign_stmt()
            / b:block() { Statement::Block(b) }


        rule semi() = _ ";" _
        rule opt_semi() = _ ";"? _

        rule semied<R>(r: rule<R>) -> R  = r:r() opt_semi() { r }
        rule tupled<R>(r: rule<R>) -> Vec<R> = "(" _ res:(r() ** comma()) _ ")" { res }
        rule bracketed<R>(r: rule<R>) -> Vec<R> = "[" _ res:(r() ** comma()) _ "]" { res }
        rule braced<R>(r: rule<R>) -> Vec<R> = "{" _ res:(r() ** comma()) _ "}" { res }

        rule while_statement() -> Statement = "while" _ "(" _ cond:exp() _ ")" _ spec:semied(<while_spec_item()>)* _ block:block()
            {
                let mut invs = Vec::new();
                let mut decs = Vec::new();
                for item in spec {
                    match item {
                        Ok(inv) => invs.push(inv),
                        Err(dec) => decs.push(dec),
                    }
                }
                Statement::While(cond, invs, decs, block)
            }

        rule while_spec_item() -> Result<Invariant, Decreases> = i:invariant() { Ok(i) } / d:decreases_raw() { Err(d) }

        rule invariant() -> Invariant = "invariant" _ e:exp() { Invariant(e) }

        rule if_statement() -> Statement = "if" _ "(" _ cond:exp() _ ")" _ then:block() _ elsifs:(elsif_block()** _) _ else_:("else" _ else_:block() { else_})? {
            let mut elsifs = [(cond, then)].into_iter().chain(elsifs).rev();
            let (cond, then) = elsifs.next().unwrap();
            elsifs.fold(Statement::If(cond, then, else_), |acc, (cond, then)| Statement::If(cond, then, Some(Block(vec![acc]))))
        }

        rule elsif_block() -> (Exp, StmtBlock) =
            "elseif" _ "(" _ exp:exp() _ ")" _ block:block() { (exp, block)}

        rule multi_assign_stmt() -> Statement =
            (assign_target() ++ comma()) _ ":=" _ e:(exp() ++ comma())
            {? if e.len() > 1 {
                Ok(Statement::Unsupported("multi-target assignment from an expression"))
            } else {
                Err("a single right-hand side")
            } }

        rule assign_stmt() -> Statement = tgts:(tgts:(assign_target() ++ comma()) _ ":=" { tgts })? _ rhs:assign_rhs()
            { Statement::Assign(tgts.unwrap_or_default(), rhs) }

        rule assign_target() -> AssignLhs = e:suffix_exp() {? match e {
            ExpKind::Ident(idn) => Ok(AssignLhs::Ident(idn)),
            ExpKind::Field(base, idn) => Ok(AssignLhs::Field(base, idn)),
            _ => Err("assignment lhs must be identifier or field access"),
        }}

        rule assign_rhs() -> AssignRhs =
              "new" _ "(" _ "*" _ ")" { AssignRhs::New(StarOrNames::Star) }
            / "new" _ "(" _ args:(ident() ** comma()) _ ")" { AssignRhs::New(StarOrNames::Names(args))}
            / e:exp() { match *e.kind {
                ExpKind::Call(Call { name, args, .. }) => AssignRhs::Call(Call { kind: None, name, args }),
                _ => AssignRhs::Exp(e)
            }}

        rule constraining_block() -> () = "constraining" _ "(" _ exp() ++ comma() _ ")" _ block()

        rule expression_or_block() -> ExpOrBlock = exp:exp()  { ExpOrBlock::Exp(exp) } / block:block() { ExpOrBlock::Block(block) }

        /// Declarations

        pub rule vpr_program() -> Program = _ decls:annotated(<d:single_decl() { vec![d] } / multi_decl()>) ** opt_semi() _
            { Program(decls.into_iter().flatten().collect()) }

        rule single_decl() -> Declaration
            = i:import() { Declaration::Import(i) }
            / d:define() { Declaration::Define(d) }
            / f:function() { Declaration::Function(f) }
            / p:predicate() { Declaration::Predicate(p) }
            / m:method() { Declaration::Method(m) }

        rule import() -> Import = "import" _ r:("<" _ r:relative_path() _ ">" { (r, false) } / "\"" _ r:relative_path() _ "\"" { (r, true) })
            { Import { path: r.0, local: r.1 } }

        rule define() -> Define = "define" _ nm:idndecl() _ ids:(tupled(<idndecl()>))? _ body:expression_or_block()
            { Define { name: nm, args: ids.unwrap_or_default(), body } }

        rule multi_decl() -> Vec<Declaration> =
            domain() / field() / adt()

        rule domain_params() -> Vec<IdnDecl> = params:(bracketed(<idndecl()>))? { params.unwrap_or_default() }

        rule domain() -> Vec<Declaration> =
            "domain" _
            name:idndecl() _
            params:domain_params() _
            interp:domain_interpretation()? _
            "{" _ elements:(annotated(<domain_element()>) ** _) _ "}"
        {
            [Declaration::Domain(Domain { name: name.clone(), params, interpretation: interp.unwrap_or_default() })].into_iter().chain(
                elements.into_iter().map(|kind| Declaration::DomainElement(DomainElement { domain: name.0.clone(), kind }))
            ).collect()
        }

        rule domain_interpretation() -> Vec<(Ident, String)> = "interpretation" _ rs:tupled(<interpretation_elt()>) { rs }

        rule interpretation_elt() -> (Ident, String) = i:ident() _ ":" _ s:string_lit() { (i, s)}

        rule domain_element() -> DomainElementKind = f:domain_function() { DomainElementKind::Function(f)} / a:axiom() { DomainElementKind::Axiom(a) }

        rule domain_function() -> DomainFunction = u:"unique"? _ sig:domain_function_signature() _ interp:func_interpretation()?
            { DomainFunction { unique: u.is_some(), signature: sig, interpretation: interp } }

        rule domain_function_signature() -> Signature = "function" _ id:idndecl() _ "(" _ args:(decl_formal_arg() ** comma()) _ ")" _ ":" _ ret:type_()
            { Signature { name: id, args, ret: vec!(ArgOrType::Type(ret)) } }

        rule func_interpretation() -> String = "interpretation" _ s:string_lit() { s }

        rule field() -> Vec<Declaration> = "field" _ fields:((f:formal_arg() {
            Declaration::Field(Field(f))
        }) ** comma()) { fields }

        rule function() -> Function = sig:function_signature() _ cont:contract()  _ body:block_exp()?
            { Function { signature: sig, contract: cont, body } }

        rule function_signature() -> Signature = "function" _ id:idndecl() _ "(" _ args:(decl_named_formal_arg() ** comma()) _ ")" _ ":" _ ret:type_()
            { Signature { name: id, args, ret: vec!(ArgOrType::Type(ret)) } }

        rule predicate() -> Predicate = "predicate" _ id:idndecl() _ args:tupled(<decl_named_formal_arg()>) _ exp:block_exp()?
            { Predicate { signature: Signature { name: id, args, ret: Vec::new() }, body: exp } }

        rule formal_returns()  -> Vec<ArgOrType> = "returns" _ rets:tupled(<decl_named_formal_arg()>) { rets }

        rule method() -> Method = "method" _ id:idndecl() _ args:tupled(<decl_named_formal_arg()>) _ ret:formal_returns()? _ cont:contract() _ body:block()?
            { Method { signature: Signature { name: id, args, ret: ret.unwrap_or_default() }, contract: cont, body } }

        rule adt() -> Vec<Declaration> = "adt" _ name:idndecl() _ params:domain_params() _ vars:adt_variants() _ derives:derives()?
            {
                let variants = vars.iter().map(|v| Variant { name: v.signature.name.clone(), fields: v.signature.args.clone() }).collect();
                let adt = Adt { name: name.clone(), params, variants, derives: derives.map(|s| vec![s]).unwrap_or_default() };
                let identity = adt.identity();
                [Declaration::Adt(adt)].into_iter().chain(vars.into_iter().map(|mut v| { v.signature.ret = vec![ArgOrType::Type(identity.clone())]; Declaration::AdtConstructor(v) })).collect()
            }

        rule adt_variant() -> AdtConstructor = name:idndecl() _ fields:tupled(<formal_arg()>) _
                { AdtConstructor { signature: Signature { name, args: fields.into_iter().map(ArgOrType::Arg).collect(), ret: vec![] } } }

        rule adt_variants() -> Vec<AdtConstructor> = "{" _ vars:adt_variant()* _ "}" { vars }

        rule derives() -> String = "derives" _ "{" _ str:$((!"}" [_])*) _ "}" { str.to_string() }

        rule relative_path() -> String = s:$(['~' | '.']? (['/']? ['.' | 'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '\\' | '-' | ' '])+) { s.to_string() }

        rule string_lit() -> String = str:$("\"" _ (!"\"" [_])* _ "\"") { str.to_string() }

        rule axiom() -> Axiom = "axiom" _ name :idndecl()? _ exp:block_exp()
            { Axiom {name, exp } }

        rule precondition() -> Exp = "requires" _ e:exp() { e }

        rule postcondition() -> Exp = "ensures" _ e:exp() { e }

        rule decreases_raw() -> Decreases = "decreases" _ d:decreases_kind()? _ e:("if" _ e:exp() { e })? { Decreases { kind: d, guard: e } }

        rule decreases_kind() -> DecreasesKind = "*" { DecreasesKind::Star } / "_" { DecreasesKind::Underscore } / e:(exp() ** comma()) { DecreasesKind::Exp(e) }

        rule contract() -> Contract =
            pres:(p:(e:precondition() { Ok(e) } / d:decreases_raw() { Err(d) }) opt_semi() {p})*
            _ posts:(p:(e:postcondition() { Ok(e) } / d:decreases_raw() { Err(d) }) opt_semi() {p})*
            {
                let mut precondition = Vec::new();
                let mut postcondition = Vec::new();
                let mut decreases = Vec::new();
                for p in pres {
                    match p {
                        Ok(e) => precondition.push(e),
                        Err(d) => decreases.push(d),
                    }
                }
                for p in posts {
                    match p {
                        Ok(e) => postcondition.push(e),
                        Err(d) => decreases.push(d),
                    }
                }
                Contract { precondition, postcondition, decreases }
            }


    }
}

/// The construct name reported for a collection literal, as a `&'static str`
/// (the grammar action only has a borrowed slice of the input).
fn collection_name(n: &str) -> &'static str {
    match n {
        "Multiset" => "Multiset",
        _ => "Map",
    }
}

#[test]
fn precedence_test() {
    let exp = viper_parser::exp("!r.b").unwrap();
    assert_eq!(
        exp,
        Exp::unknown(ExpKind::UnOp(
            UnOp::Not,
            Exp::unknown(ExpKind::Field(
                Exp::unknown(ExpKind::Ident(Ident::Raw("r".to_string()))),
                Ident::Raw("b".to_string())
            ))
        ))
    );
}

#[cfg(test)]
fn raw_ident(name: &str) -> Exp {
    Exp::unknown(ExpKind::Ident(Ident::Raw(name.to_string())))
}

#[cfg(test)]
fn int_lit(n: i64) -> Exp {
    Exp::unknown(ExpKind::Const(ConstKind::Int(n.into())))
}

#[test]
fn seq_literals() {
    let parse = |src| *viper_parser::exp(src).unwrap().kind;
    assert_eq!(
        parse("Seq[Int]()"),
        ExpKind::SeqLit(Some(Type::Int), vec![])
    );
    assert_eq!(parse("Seq()"), ExpKind::SeqLit(None, vec![]));
    assert_eq!(
        parse("Seq(1, x)"),
        ExpKind::SeqLit(None, vec![int_lit(1), raw_ident("x")])
    );
    assert_eq!(
        parse("Seq[Seq[Bool]](s)"),
        ExpKind::SeqLit(Some(Type::Seq(Box::new(Type::Bool))), vec![raw_ident("s")])
    );
    // `Seq` must stand alone: a longer name is an ordinary function call.
    assert!(matches!(parse("Sequence(1)"), ExpKind::Call(_)));
}

#[test]
fn seq_length_and_range() {
    let parse = |src| *viper_parser::exp(src).unwrap().kind;
    assert_eq!(
        parse("|s|"),
        ExpKind::UnOp(UnOp::Cardinality, raw_ident("s"))
    );
    assert_eq!(
        parse("[0..n)"),
        ExpKind::BinOp(BinOp::Range, int_lit(0), raw_ident("n"))
    );
    // `|s ++ t| == |s| + |t|` keeps each `|..|` around its own operand.
    let ExpKind::BinOp(BinOp::Eq, lhs, rhs) = parse("|s ++ t| == |s| + |t|") else {
        panic!("expected an equality");
    };
    assert!(matches!(&*lhs.kind, ExpKind::UnOp(UnOp::Cardinality, inner)
        if matches!(&*inner.kind, ExpKind::BinOp(BinOp::Concat, ..))));
    assert!(matches!(&*rhs.kind, ExpKind::BinOp(BinOp::Plus, ..)));
}

#[test]
fn set_literals_and_types() {
    let parse = |src| *viper_parser::exp(src).unwrap().kind;
    assert_eq!(
        parse("Set[Int]()"),
        ExpKind::SetLit(Some(Type::Int), vec![])
    );
    assert_eq!(parse("Set()"), ExpKind::SetLit(None, vec![]));
    assert_eq!(
        parse("Set(1, x)"),
        ExpKind::SetLit(None, vec![int_lit(1), raw_ident("x")])
    );
    assert_eq!(
        parse("Set[Seq[Bool]](s)"),
        ExpKind::SetLit(Some(Type::Seq(Box::new(Type::Bool))), vec![raw_ident("s")])
    );
    assert!(matches!(parse("Settings(1)"), ExpKind::Call(_)));
    assert_eq!(
        parse("Set[Set[Int]]()"),
        ExpKind::SetLit(Some(Type::Set(Box::new(Type::Int))), vec![])
    );
    // Multiset stays an unsupported construct.
    assert_eq!(parse("Multiset(1)"), ExpKind::Unsupported("Multiset"));
}

#[test]
fn set_operators() {
    let parse = |src| *viper_parser::exp(src).unwrap().kind;
    for (src, op) in [
        ("a union b", BinOp::Union),
        ("a intersection b", BinOp::Intersection),
        ("a setminus b", BinOp::SetMinus),
        ("a subset b", BinOp::Subset),
        ("x in a", BinOp::In),
    ] {
        let ExpKind::BinOp(got, ..) = parse(src) else {
            panic!("`{src}` is not a binary operation");
        };
        assert_eq!(got, op, "{src}");
    }
    // As in Silver, the set operators bind like `+`: tighter than `in` and `==`.
    let ExpKind::BinOp(BinOp::In, _, rhs) = parse("x in a union b") else {
        panic!("expected `in` at the root");
    };
    assert!(matches!(&*rhs.kind, ExpKind::BinOp(BinOp::Union, ..)));
}
