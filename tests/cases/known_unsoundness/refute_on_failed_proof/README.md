# A `refute` passes because its proof was cut short

`plans/verifier-soundness-interfaces.md` §4.4 and open question §9.1: for `refute`,
incompleteness is unsoundness. Source: #55.

`refute A` fails when `A` is proven. The verifier lets it pass whenever the attempt to prove
`A` fails, including when the attempt was stopped by a termination bound (node, iteration or
growth limit) before it could finish. A "gave up" is read as a counter-model.

`refute_proven_by_the_bounded_pass.vpr`: a cubic matching loop (`grow`) beside a 90-deep
chain of function definitions. The loop fills the saturation bound before the 90 unfoldings
finish, so `c90(y) == y` is not proven and `refute c90(y) == y` passes. Silicon reports
"Refute holds in all cases". #55 rejects it by proving it in a generation-bounded first
pass, which is one way the proof can finish, not a fix of the class.

**What a structural fix looks like.** A proof attempt returns `Option<Proof>` plus why it
stopped, and `refute` cannot treat "stopped by a bound" as "not provable". Either it is
reported as unsupported/unknown in that case, or a `refute` passes only on evidence (a
saturated graph without the goal). See §4.4.

This case takes about 70 s in a release build; it is the slowest file of the debug suite.
