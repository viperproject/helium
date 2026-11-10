//! The types of a verification unit (`verify::unit_types`): which ground types are
//! collected, and which are deliberately not.

use super::*;
use crate::verify::unit_types::{UnitTypes, function_types, method_types, resource_types};
use crate::vmir::Type;

fn dom(program: &vmir::Program, name: &str, args: Vec<Type>) -> Type {
    Type::Domain(program.id(name).expect(name), args.into())
}

fn of_method(program: &vmir::Program, name: &str) -> UnitTypes {
    let vmir::Declaration::Method(m) = &program.decls[program.id(name).expect(name)] else {
        panic!("`{name}` is not a method")
    };
    method_types(program, m)
}

#[test]
fn own_values_and_type_arguments() {
    // Rule 1 and 5: parameters, locals, and the type arguments of an application.
    let program = lower(
        r#"
domain Box[T] {
  function box(x: T): Box[T]
  function unbox(b: Box[T]): T
}
method m(a: Box[Int], b: Bool) {
  var c: Box[Bool] := box(b)
}
"#,
    );
    let tys = of_method(&program, "m");
    assert!(
        tys.contains(&dom(&program, "Box", vec![Type::Int])),
        "param"
    );
    assert!(
        tys.contains(&dom(&program, "Box", vec![Type::Bool])),
        "local"
    );
    assert!(tys.contains(&Type::Bool));
    // Mentioned only as a type argument, never as the type of a value.
    assert!(!tys.contains(&Type::Real));
}

#[test]
fn nested_types_do_not_count_until_unwrapped() {
    let program = lower(
        r#"
domain Box[T] { function box(x: T): Box[T] }
adt Opt[T] {
  None()
  Some(value: T)
}
method outer(x: Opt[Box[Int]]) { }
method unwrapped(x: Opt[Box[Int]]) {
  var y: Box[Int] := x.value
}
"#,
    );
    let boxed = dom(&program, "Box", vec![Type::Int]);
    let opt = dom(&program, "Opt", vec![boxed.clone()]);
    let outer = of_method(&program, "outer");
    assert!(outer.contains(&opt));
    assert!(
        !outer.contains(&boxed),
        "`Box<Int>` is only nested in `Opt`"
    );
    let unwrapped = of_method(&program, "unwrapped");
    assert!(unwrapped.contains(&opt) && unwrapped.contains(&boxed));
}

#[test]
fn callee_body_counts_one_call_away_but_not_two() {
    // Rule 4: a function called directly is unfolded, so its body's values are
    // terms of the caller's e-graph. The function *it* calls is not unfolded.
    let program = lower(
        r#"
domain A { function a(): A }
domain B { function b(): B }
function two_away(): Bool { b() == b() }
function one_away(): Bool { two_away() && a() == a() }
method m() { assert one_away() }
"#,
    );
    let tys = of_method(&program, "m");
    assert!(tys.contains(&dom(&program, "A", vec![])), "one call away");
    assert!(!tys.contains(&dom(&program, "B", vec![])), "two calls away");
}

#[test]
fn callee_contract_counts() {
    // Rule 3: a callee's `#requires` is exhaled at the call.
    let program = lower(
        r#"
domain C { function c(): C }
method callee() requires c() == c()
method m() { callee() }
"#,
    );
    assert!(of_method(&program, "m").contains(&dom(&program, "C", vec![])));
}

#[test]
fn own_contract_and_function_unit() {
    // Rule 2 for a function: the definitions its `requires` / `ensures` link to.
    let program = lower(
        r#"
domain C { function c(): C }
domain D { function d(): D }
function f(x: Int): Int
  requires c() == c()
  ensures d() == d()
{ x }
"#,
    );
    let vmir::Declaration::Function(f) = &program.decls[program.id("f").unwrap()] else {
        panic!("`f` is not a function")
    };
    let tys = function_types(&program, f);
    assert!(tys.contains(&dom(&program, "C", vec![])), "requires");
    assert!(tys.contains(&dom(&program, "D", vec![])), "ensures");
    assert!(tys.contains(&Type::Int));
}

#[test]
fn predicate_unit_and_generic_domain_function() {
    let program = lower(
        r#"
domain Box[T] { function box(x: T): Box[T] }
field f: Int
predicate P(x: Ref, b: Box[Int]) { acc(x.f) }
"#,
    );
    let vmir::Declaration::Resource(r) = &program.decls[program.id("P").unwrap()] else {
        panic!("`P` is not a resource")
    };
    assert!(resource_types(&program, r).contains(&dom(&program, "Box", vec![Type::Int])));
    // A generic domain function has no values, and its signature is not ground.
    let vmir::Declaration::Function(box_fn) = &program.decls[program.id("box").unwrap()] else {
        panic!("`box` is not a function")
    };
    assert!(function_types(&program, box_fn).is_empty());
}
