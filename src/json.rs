//! A minimal JSON value and writer, for the machine-readable output of the
//! command-line tools (`verify --json`, `verify --viper-metrics`).
//!
//! Hand-rolled rather than `serde_json` so the verifier crate does not gain a
//! dependency for what is a few dozen lines of output code. Only writing is
//! supported; the consumers (the `bench` runner) parse with a real library.

use std::fmt::{self, Write};

/// A JSON value. Objects keep insertion order, so output is stable.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    UInt(u64),
    /// Non-finite values are written as `null`: JSON has no NaN or infinity.
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// An object from `(key, value)` pairs.
    pub fn obj<K: Into<String>>(fields: impl IntoIterator<Item = (K, Json)>) -> Self {
        Json::Obj(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    pub fn str(s: impl Into<String>) -> Self {
        Json::Str(s.into())
    }

    /// `Some(v)` as `v`, `None` as `null`.
    pub fn opt<T>(v: Option<T>, f: impl FnOnce(T) -> Json) -> Self {
        v.map_or(Json::Null, f)
    }
}

impl From<u64> for Json {
    fn from(v: u64) -> Self {
        Json::UInt(v)
    }
}

impl From<usize> for Json {
    fn from(v: usize) -> Self {
        Json::UInt(v as u64)
    }
}

impl From<f64> for Json {
    fn from(v: f64) -> Self {
        Json::Float(v)
    }
}

impl From<u32> for Json {
    fn from(v: u32) -> Self {
        Json::UInt(v.into())
    }
}

/// `Some(v)` as `v`, `None` as `null`.
impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(v: Option<T>) -> Self {
        v.map_or(Json::Null, Into::into)
    }
}

impl From<bool> for Json {
    fn from(v: bool) -> Self {
        Json::Bool(v)
    }
}

impl From<&str> for Json {
    fn from(v: &str) -> Self {
        Json::Str(v.to_string())
    }
}

impl From<String> for Json {
    fn from(v: String) -> Self {
        Json::Str(v)
    }
}

fn write_str(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    f.write_char('"')?;
    for c in s.chars() {
        match c {
            '"' => f.write_str("\\\"")?,
            '\\' => f.write_str("\\\\")?,
            '\n' => f.write_str("\\n")?,
            '\r' => f.write_str("\\r")?,
            '\t' => f.write_str("\\t")?,
            c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
            c => f.write_char(c)?,
        }
    }
    f.write_char('"')
}

/// Compact output, no whitespace.
impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Json::Null => f.write_str("null"),
            Json::Bool(b) => write!(f, "{b}"),
            Json::UInt(n) => write!(f, "{n}"),
            // `{:?}` is the shortest representation that round-trips, and every
            // form it produces (`1.0`, `1e-7`) is valid JSON.
            Json::Float(x) if x.is_finite() => write!(f, "{x:?}"),
            Json::Float(_) => f.write_str("null"),
            Json::Str(s) => write_str(f, s),
            Json::Arr(items) => {
                f.write_char('[')?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_char(',')?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_char(']')
            }
            Json::Obj(fields) => {
                f.write_char('{')?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_char(',')?;
                    }
                    write_str(f, k)?;
                    f.write_char(':')?;
                    write!(f, "{v}")?;
                }
                f.write_char('}')
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Json;

    #[test]
    fn writes_nested_values_and_escapes() {
        let v = Json::obj([
            ("a", Json::from(1u64)),
            ("b", Json::from(0.5)),
            ("c", Json::from("q\"\\\n\u{1}")),
            (
                "d",
                Json::Arr(vec![Json::Null, Json::Bool(true), Json::Float(f64::NAN)]),
            ),
        ]);
        assert_eq!(
            v.to_string(),
            r#"{"a":1,"b":0.5,"c":"q\"\\\n\u0001","d":[null,true,null]}"#
        );
    }
}
