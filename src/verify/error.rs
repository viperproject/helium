//! The verifier's error type.

#[derive(Debug)]
pub enum VerifyError {
    AssertionFailed,
    /// A `refute` whose expression turned out to be provable (so the refutation
    /// fails).
    RefuteFailed,
    InsufficientPermission,
    /// An instruction's side condition (e.g. `acc` permission ≥ 0, division
    /// divisor ≠ 0) could not be discharged. Carries a human-readable
    /// description. See [`inst_obligations`](crate::verify::declaration).
    SideCondition(&'static str),
    /// Encountered a method-only heap extension (e.g. `Assign`) in a body
    /// the verifier doesn't yet handle structurally. Reserved for
    /// not-yet-implemented variants.
    Unimplemented(&'static str),
    /// A call/fold/unfold targets a resource that produced no certificate —
    /// i.e. the resource itself failed its well-formedness verification.
    DependencyFailed,
    /// Verification failed while executing a specific instruction. `at`/`total`
    /// give the failing instruction's index and the body's instruction count —
    /// a progress marker, so runs can be compared on how far a failing member
    /// got, not just pass/fail.
    AtInst {
        inst: String,
        at: usize,
        total: usize,
        source: Box<VerifyError>,
    },
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AssertionFailed => write!(f, "assertion failed"),
            Self::RefuteFailed => write!(f, "refuted expression is actually provable"),
            Self::InsufficientPermission => write!(f, "insufficient permission"),
            Self::SideCondition(what) => write!(f, "side condition may not hold: {what}"),
            Self::Unimplemented(what) => write!(f, "unimplemented: {what}"),
            Self::DependencyFailed => {
                write!(f, "depends on a resource that failed to verify")
            }
            Self::AtInst {
                inst,
                at,
                total,
                source,
            } => write!(f, "{source}\n    instruction ({at}/{total}): {inst}"),
        }
    }
}

impl VerifyError {
    /// Wrap this error with the instruction text it occurred at, plus the
    /// instruction's index and the body's total (idempotent — an
    /// already-wrapped error keeps its innermost source).
    pub(crate) fn with_inst(self, inst: String, at: usize, total: usize) -> Self {
        match self {
            Self::AtInst { .. } => self,
            _ => Self::AtInst {
                inst,
                at,
                total,
                source: Box::new(self),
            },
        }
    }

    /// Whether the failure rests on a proof rather than on a failure to prove:
    /// a `refute` whose expression was proven. Knowing more facts cannot undo
    /// it.
    pub(crate) fn rests_on_proof(&self) -> bool {
        matches!(self.root_cause(), Self::RefuteFailed)
    }

    /// The innermost error, peeling any `AtInst` wrapping.
    pub(crate) fn root_cause(&self) -> &VerifyError {
        match self {
            Self::AtInst { source, .. } => source.root_cause(),
            _ => self,
        }
    }
}
