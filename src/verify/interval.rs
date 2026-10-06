//! Integer intervals: the numeric abstract domain of the e-class analysis.
//!
//! An [`Interval`] over-approximates the values an `Int` e-class can take. The
//! analysis (`verify::analysis`) computes one per class from its nodes
//! (literals, `+ - * \ %`, `ite` hulls), meets it on union, and narrows it by
//! comparisons that become decided; a comparison the intervals decide folds.
//!
//! Bounds are `i128`, unbounded on a `None` side. Every operation rounds
//! **outward** where a result leaves the representable range (an overflowing
//! upper bound becomes unbounded, an overflowing lower bound saturates at the
//! largest representable one), so the interval stays a sound over-approximation
//! of a [`BigInt`] value whatever its size; only precision is lost.

use num::{BigInt, ToPrimitive};

/// The integers in `[lo, hi]`, unbounded on a `None` side. Never empty: an
/// operation whose result would be empty returns `None` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Interval {
    pub lo: Option<i128>,
    pub hi: Option<i128>,
}

/// An extended integer, for the arithmetic: the bounds of an [`Interval`] with
/// their infinities made explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Ext {
    NegInf,
    Fin(i128),
    PosInf,
}

impl Ext {
    fn lo(b: Option<i128>) -> Ext {
        b.map_or(Ext::NegInf, Ext::Fin)
    }

    fn hi(b: Option<i128>) -> Ext {
        b.map_or(Ext::PosInf, Ext::Fin)
    }

    /// The sign of the value: `-1`, `0` or `1`.
    fn signum(self) -> i8 {
        match self {
            Ext::NegInf => -1,
            Ext::Fin(v) => v.signum() as i8,
            Ext::PosInf => 1,
        }
    }

    /// The infinity of the given sign; an overflowing finite result is replaced
    /// by it, which is outward for whichever bound it ends up in (see
    /// [`Ext::as_lo`] / [`Ext::as_hi`]).
    fn inf(sign: i8) -> Ext {
        if sign < 0 { Ext::NegInf } else { Ext::PosInf }
    }

    fn add(self, o: Ext) -> Ext {
        match (self, o) {
            (Ext::Fin(a), Ext::Fin(b)) => a
                .checked_add(b)
                .map_or_else(|| Ext::inf(a.signum() as i8), Ext::Fin),
            // `-inf + +inf` cannot arise: a lower bound is never `+inf` and an
            // upper bound never `-inf`, and sums only combine like bounds.
            (Ext::NegInf, _) | (_, Ext::NegInf) => Ext::NegInf,
            _ => Ext::PosInf,
        }
    }

    fn neg(self) -> Ext {
        match self {
            Ext::NegInf => Ext::PosInf,
            Ext::PosInf => Ext::NegInf,
            // `-i128::MIN` overflows to just above `i128::MAX`.
            Ext::Fin(v) => v.checked_neg().map_or(Ext::PosInf, Ext::Fin),
        }
    }

    /// Product, with `0 * inf = 0`: an interval bound by `0` on one side
    /// multiplied by an unbounded one is still bounded by `0` there.
    fn mul(self, o: Ext) -> Ext {
        let sign = self.signum() * o.signum();
        match (self, o) {
            _ if sign == 0 => Ext::Fin(0),
            (Ext::Fin(a), Ext::Fin(b)) => a.checked_mul(b).map_or_else(|| Ext::inf(sign), Ext::Fin),
            _ => Ext::inf(sign),
        }
    }

    /// As a lower bound: `-inf` is unbounded; `+inf` only arises from a finite
    /// result that overflowed upward, whose value lies above `i128::MAX`.
    fn as_lo(self) -> Option<i128> {
        match self {
            Ext::NegInf => None,
            Ext::Fin(v) => Some(v),
            Ext::PosInf => Some(i128::MAX),
        }
    }

    /// As an upper bound; the mirror of [`Ext::as_lo`].
    fn as_hi(self) -> Option<i128> {
        match self {
            Ext::PosInf => None,
            Ext::Fin(v) => Some(v),
            Ext::NegInf => Some(i128::MIN),
        }
    }
}

impl Interval {
    pub const TOP: Interval = Interval { lo: None, hi: None };

    pub fn point(v: i128) -> Self {
        Interval {
            lo: Some(v),
            hi: Some(v),
        }
    }

    /// `[lo, hi]`, or `None` when it is empty.
    pub fn new(lo: Option<i128>, hi: Option<i128>) -> Option<Self> {
        match (lo, hi) {
            (Some(l), Some(h)) if l > h => None,
            _ => Some(Interval { lo, hi }),
        }
    }

    /// The smallest interval holding the integer `n`: the point itself when it
    /// is representable, else the outward half-line beyond the representable
    /// range.
    pub fn of(n: &BigInt) -> Self {
        match n.to_i128() {
            Some(v) => Interval::point(v),
            None if n.sign() == num::bigint::Sign::Minus => Interval {
                lo: None,
                hi: Some(i128::MIN),
            },
            None => Interval {
                lo: Some(i128::MAX),
                hi: None,
            },
        }
    }

    pub fn is_top(&self) -> bool {
        *self == Interval::TOP
    }

    /// The single value, when this is a point.
    pub fn as_point(&self) -> Option<i128> {
        match (self.lo, self.hi) {
            (Some(l), Some(h)) if l == h => Some(l),
            _ => None,
        }
    }

    /// Whether the integer `n` lies in this interval. Exact for any `n`.
    pub fn contains(&self, n: &BigInt) -> bool {
        self.lo.is_none_or(|l| *n >= BigInt::from(l))
            && self.hi.is_none_or(|h| *n <= BigInt::from(h))
    }

    /// The common part, `None` when the two are disjoint.
    pub fn meet(&self, o: &Self) -> Option<Self> {
        let lo = match (self.lo, o.lo) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        let hi = match (self.hi, o.hi) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        Interval::new(lo, hi)
    }

    /// The smallest interval holding both (the join of the lattice).
    pub fn hull(&self, o: &Self) -> Self {
        Interval {
            lo: self.lo.zip(o.lo).map(|(a, b)| a.min(b)),
            hi: self.hi.zip(o.hi).map(|(a, b)| a.max(b)),
        }
    }

    pub fn add(&self, o: &Self) -> Self {
        Interval {
            lo: Ext::lo(self.lo).add(Ext::lo(o.lo)).as_lo(),
            hi: Ext::hi(self.hi).add(Ext::hi(o.hi)).as_hi(),
        }
    }

    pub fn sub(&self, o: &Self) -> Self {
        Interval {
            lo: Ext::lo(self.lo).add(Ext::hi(o.hi).neg()).as_lo(),
            hi: Ext::hi(self.hi).add(Ext::lo(o.lo).neg()).as_hi(),
        }
    }

    pub fn mul(&self, o: &Self) -> Self {
        let (a, b) = (Ext::lo(self.lo), Ext::hi(self.hi));
        let (c, d) = (Ext::lo(o.lo), Ext::hi(o.hi));
        let corners = [a.mul(c), a.mul(d), b.mul(c), b.mul(d)];
        Interval {
            lo: corners.iter().min().unwrap().as_lo(),
            hi: corners.iter().max().unwrap().as_hi(),
        }
    }

    /// The smallest absolute value in the interval, and the largest (`None`
    /// when unbounded).
    fn abs_range(&self) -> (Ext, Ext) {
        let (lo, hi) = (Ext::lo(self.lo), Ext::hi(self.hi));
        let (alo, ahi) = (lo.neg().max(lo), hi.neg().max(hi));
        if lo <= Ext::Fin(0) && Ext::Fin(0) <= hi {
            (Ext::Fin(0), alo.max(ahi))
        } else {
            (alo.min(ahi), alo.max(ahi))
        }
    }

    /// Viper's `x % m` (SMT-LIB `mod`, Euclidean: the remainder lies in
    /// `[0, |m|)`), or `None` when `m` may be zero: the result is then
    /// unspecified, not bounded.
    ///
    /// Where `x` already lies in `[0, |m|)` for every `m`, the remainder is `x`
    /// itself; a non-negative `x` bounds the remainder from above.
    pub fn euclid_mod(&self, m: &Self) -> Option<Self> {
        let (min_abs, max_abs) = m.abs_range();
        if min_abs == Ext::Fin(0) {
            return None;
        }
        if self.mod_is_identity(m) {
            return Some(*self);
        }
        let nonneg = self.lo.is_some_and(|l| l >= 0);
        let mut hi = max_abs.add(Ext::Fin(-1)).as_hi();
        if nonneg {
            hi = hi.map_or(self.hi, |h| Some(self.hi.map_or(h, |x| x.min(h))));
        }
        Some(Interval { lo: Some(0), hi })
    }

    /// Whether `x % m` is `x` itself for every `x` in `self` and `m` in `m`:
    /// `x` lies in `[0, |m|)` for the smallest `|m|`.
    pub fn mod_is_identity(&self, m: &Self) -> bool {
        let (min_abs, _) = m.abs_range();
        self.lo.is_some_and(|l| l >= 0) && Ext::hi(self.hi) < min_abs
    }

    /// Viper's `x \ m` (SMT-LIB `div`, Euclidean) for a literal divisor `m`, or
    /// `None` when `m` is not a non-zero point. With `m > 0` the quotient is
    /// `floor(x / m)`, monotone in `x`; with `m < 0` it is `-(x \ |m|)`.
    pub fn euclid_div(&self, m: &Self) -> Option<Self> {
        let k = m.as_point().filter(|k| *k != 0)?;
        let Some(abs) = k.checked_abs() else {
            // `|i128::MIN|`: every representable `x` divides to `0` or `-1`
            // (or `1` for `x = i128::MIN`, which `[-1, 1]` still holds).
            return Some(Interval {
                lo: Some(-1),
                hi: Some(1),
            });
        };
        let floor = |b: Option<i128>| b.map(|v| v.div_euclid(abs));
        let q = Interval {
            lo: floor(self.lo),
            hi: floor(self.hi),
        };
        Some(if k > 0 {
            q
        } else {
            Interval {
                lo: Ext::hi(q.hi).neg().as_lo(),
                hi: Ext::lo(q.lo).neg().as_hi(),
            }
        })
    }

    /// The standard narrowing operator: `new` (within `self`) refines an
    /// unbounded side of `self`, and a bounded side stays. A single value is
    /// taken as is — a class becomes a point once.
    pub fn narrow(&self, new: &Self) -> Self {
        if new.as_point().is_some() {
            return *new;
        }
        Interval {
            lo: self.lo.or(new.lo),
            hi: self.hi.or(new.hi),
        }
    }

    /// `self < o`, when the intervals decide it.
    pub fn lt(&self, o: &Self) -> Option<bool> {
        if let (Some(h), Some(l)) = (self.hi, o.lo)
            && h < l
        {
            return Some(true);
        }
        if let (Some(l), Some(h)) = (self.lo, o.hi)
            && l >= h
        {
            return Some(false);
        }
        None
    }

    /// The part of `self` consistent with `self < o` (`holds`) or `self >= o`
    /// (`!holds`), given `o`'s interval. `None` when nothing is.
    pub fn below(&self, o: &Self, holds: bool) -> Option<Self> {
        if holds {
            // self <= o.hi - 1
            let hi = Ext::hi(o.hi).add(Ext::Fin(-1)).as_hi();
            self.meet(&Interval { lo: None, hi })
        } else {
            self.meet(&Interval { lo: o.lo, hi: None })
        }
    }

    /// The part of `self` consistent with `o < self` (`holds`) or `o >= self`
    /// (`!holds`).
    pub fn above(&self, o: &Self, holds: bool) -> Option<Self> {
        if holds {
            let lo = Ext::lo(o.lo).add(Ext::Fin(1)).as_lo();
            self.meet(&Interval { lo, hi: None })
        } else {
            self.meet(&Interval { lo: None, hi: o.hi })
        }
    }

    /// `self` without the value `k`, as far as an interval can say so: only an
    /// endpoint can be cut off. `None` when `self` is exactly `{k}`.
    pub fn without(&self, k: i128) -> Option<Self> {
        if self.as_point() == Some(k) {
            return None;
        }
        // An endpoint `k` with more values beyond it moves by one; that cannot
        // overflow, except for a lower bound of `i128::MAX` with no upper bound,
        // which already stands for everything from there up and stays.
        let mut out = *self;
        if self.lo == Some(k) {
            out.lo = Some(k.checked_add(1).unwrap_or(k));
        }
        if self.hi == Some(k) {
            out.hi = Some(k.checked_sub(1).unwrap_or(k));
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::analysis::eval_binary;
    use crate::vmir::{BinOp, Literal};

    /// Every interval with bounds in `-3..=3` or unbounded.
    fn intervals() -> Vec<Interval> {
        let bounds: Vec<Option<i128>> = std::iter::once(None).chain((-3..=3).map(Some)).collect();
        let mut out = Vec::new();
        for &lo in &bounds {
            for &hi in &bounds {
                out.extend(Interval::new(lo, hi));
            }
        }
        out
    }

    /// The members of `iv` in `-8..=8`: an unbounded side is sampled past every
    /// finite bound in use.
    fn members(iv: &Interval) -> Vec<i128> {
        (-8..=8)
            .filter(|v| iv.contains(&BigInt::from(*v)))
            .collect()
    }

    /// The result of each operation holds every value the operation takes on
    /// members of its operands, folded exactly as the analysis folds literals
    /// (Euclidean `%` and `\`).
    #[test]
    fn arithmetic_over_approximates() {
        let ivs = intervals();
        for a in &ivs {
            for b in &ivs {
                for x in members(a) {
                    for y in members(b) {
                        let (bx, by) = (BigInt::from(x), BigInt::from(y));
                        assert!(a.add(b).contains(&(&bx + &by)), "{a:?} + {b:?} at {x}, {y}");
                        assert!(a.sub(b).contains(&(&bx - &by)), "{a:?} - {b:?} at {x}, {y}");
                        assert!(a.mul(b).contains(&(&bx * &by)), "{a:?} * {b:?} at {x}, {y}");
                        assert!(a.hull(b).contains(&bx) && a.hull(b).contains(&by));
                        let (lx, ly) = (Literal::Int(bx.clone()), Literal::Int(by.clone()));
                        if let Some(Literal::Int(r)) = eval_binary(BinOp::Mod, &lx, &ly)
                            && let Some(m) = a.euclid_mod(b)
                        {
                            assert!(m.contains(&r), "{a:?} % {b:?} at {x}, {y}");
                        }
                        if let Some(Literal::Int(r)) = eval_binary(BinOp::Mod, &lx, &ly)
                            && a.mod_is_identity(b)
                        {
                            assert_eq!(r, bx, "{a:?} % {b:?} is not the identity at {x}, {y}");
                        }
                        if let Some(Literal::Int(q)) = eval_binary(BinOp::DivI, &lx, &ly)
                            && let Some(d) = a.euclid_div(b)
                        {
                            assert!(d.contains(&q), "{a:?} \\ {b:?} at {x}, {y}");
                        }
                    }
                }
            }
        }
    }

    /// A divisor range holding `0` bounds nothing: the remainder is unspecified
    /// there.
    #[test]
    fn zero_divisor_bounds_nothing() {
        let m = Interval::new(Some(0), Some(3)).unwrap();
        assert_eq!(Interval::TOP.euclid_mod(&m), None);
        assert!(!Interval::point(1).mod_is_identity(&m));
        assert_eq!(Interval::TOP.euclid_div(&Interval::point(0)), None);
    }

    /// Comparisons decide only what holds for every pair of members, and each
    /// narrowing keeps every member consistent with what it narrows by.
    #[test]
    fn comparisons_and_narrowing_are_sound() {
        let ivs = intervals();
        for a in &ivs {
            for b in &ivs {
                for x in members(a) {
                    for y in members(b) {
                        if let Some(d) = a.lt(b) {
                            assert_eq!(d, x < y, "{a:?} < {b:?} at {x}, {y}");
                        }
                        let (bx, by) = (BigInt::from(x), BigInt::from(y));
                        let lt = x < y;
                        assert!(a.below(b, lt).is_some_and(|n| n.contains(&bx)));
                        assert!(b.above(a, lt).is_some_and(|n| n.contains(&by)));
                        if x == y {
                            assert!(a.meet(b).is_some_and(|m| m.contains(&bx)));
                        }
                    }
                }
                for k in -4..=4 {
                    for x in members(a).into_iter().filter(|x| *x != k) {
                        assert!(a.without(k).is_some_and(|n| n.contains(&BigInt::from(x))));
                    }
                }
                if let Some(m) = a.meet(b) {
                    let n = a.narrow(&m);
                    assert!(members(&m).iter().all(|v| n.contains(&BigInt::from(*v))));
                    assert!(members(&n).iter().all(|v| a.contains(&BigInt::from(*v))));
                }
            }
        }
    }

    /// Results beyond `i128` round outward, never inward.
    #[test]
    fn overflow_rounds_outward() {
        let max = Interval::point(i128::MAX);
        assert!(
            max.add(&Interval::point(1))
                .contains(&(BigInt::from(i128::MAX) + 1))
        );
        let neg = Interval::point(0).sub(&Interval::point(i128::MIN));
        assert!(neg.contains(&-BigInt::from(i128::MIN)));
        let sq = BigInt::from(i128::MAX) * BigInt::from(i128::MAX);
        assert!(max.mul(&max).contains(&sq));
        let low = BigInt::from(i128::MIN) * BigInt::from(i128::MAX);
        assert!(Interval::point(i128::MIN).mul(&max).contains(&low));
        let huge = BigInt::from(2).pow(200);
        assert!(Interval::of(&huge).contains(&huge));
        assert!(Interval::of(&-&huge).contains(&-&huge));
        assert_eq!(
            Interval::of(&huge).lt(&Interval::point(i128::MAX)),
            Some(false)
        );
        assert_eq!(Interval::point(i128::MAX).lt(&Interval::of(&huge)), None);
    }
}
