//! Reading one `verify` run: `--json` output, or — for builds older than
//! `--json` — a best-effort parse of the text rows and the `[TIMING]`,
//! `[VERIFY-BREAKDOWN]` and `[STATS]` blocks on stderr.
//!
//! Old builds treat unknown flags as the file name and keep the last one, so
//! `verify --json --breakdown file.vpr` runs on any version: a new build
//! answers in JSON, an old one in text.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// What one `verify` process reported.
#[derive(Debug, Clone, Default)]
pub struct HeliumRun {
    /// `(member, status)` rows in output order. Status is the printed tag:
    /// `OK`, `FAIL`, `UNSUPPORTED`, `ERROR` or `SKIP`.
    pub results: Vec<(String, String)>,
    /// A file-level error (parse failure, IO): no rows at all.
    pub error: Option<String>,
    /// Seconds per phase, and the pipeline total.
    pub phases: BTreeMap<String, f64>,
    pub total: Option<f64>,
    pub member_times: BTreeMap<String, f64>,
    /// The full `VerifyStats`, as JSON.
    pub stats: Option<Value>,
    pub peak_rss_mb: Option<f64>,
    /// Whether this came from `--json` (false: the text fallback).
    pub json: bool,
}

pub fn parse(stdout: &str, stderr: &str) -> Option<HeliumRun> {
    let trimmed = stdout.trim_start();
    if trimmed.starts_with('{') {
        if let Some(line) = trimmed.lines().next() {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                return Some(from_json(&v));
            }
        }
    }
    parse_text(stdout, stderr)
}

fn from_json(v: &Value) -> HeliumRun {
    let num_map = |v: Option<&Value>| -> BTreeMap<String, f64> {
        v.and_then(Value::as_object)
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
                    .collect()
            })
            .unwrap_or_default()
    };
    HeliumRun {
        results: v["results"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| {
                        Some((
                            r["name"].as_str()?.to_string(),
                            r["status"].as_str()?.to_string(),
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default(),
        error: v["error"].as_str().map(String::from),
        phases: num_map(v.get("phases")),
        total: v["total"].as_f64(),
        member_times: num_map(v.get("member_times")),
        stats: v.get("stats").filter(|s| s.is_object()).cloned(),
        peak_rss_mb: v["peak_rss_mb"].as_f64(),
        json: true,
    }
}

/// `12.345ms`, `1.2s`, `850.1µs`, `3ns` (Rust's `Debug` for `Duration`) in
/// seconds.
fn parse_duration(s: &str) -> Option<f64> {
    let s = s.trim();
    let (num, scale) = if let Some(n) = s.strip_suffix("ms") {
        (n, 1e-3)
    } else if let Some(n) = s.strip_suffix("µs").or_else(|| s.strip_suffix("us")) {
        (n, 1e-6)
    } else if let Some(n) = s.strip_suffix("ns") {
        (n, 1e-9)
    } else if let Some(n) = s.strip_suffix('s') {
        (n, 1.0)
    } else {
        return None;
    };
    num.trim().parse::<f64>().ok().map(|x| x * scale)
}

fn parse_text(stdout: &str, stderr: &str) -> Option<HeliumRun> {
    let mut run = HeliumRun::default();
    for line in stdout.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix('[') else {
            continue;
        };
        let Some((tag, rest)) = rest.split_once(']') else {
            continue;
        };
        if !matches!(tag, "OK" | "FAIL" | "UNSUPPORTED" | "ERROR" | "SKIP") {
            continue;
        }
        let rest = rest.trim();
        let name = rest.split_once(": ").map_or(rest, |(n, _)| n);
        run.results.push((name.to_string(), tag.to_string()));
    }

    let mut section = "";
    for line in stderr.lines() {
        if let Some(e) = line.strip_prefix("[PIPELINE-ERROR] ") {
            run.error = Some(e.to_string());
            continue;
        }
        if line.starts_with("[TIMING]") {
            section = "timing";
            continue;
        }
        if line.starts_with("[VERIFY-BREAKDOWN]") {
            section = "breakdown";
            continue;
        }
        if let Some(stats) = line.strip_prefix("[STATS] ") {
            run.stats = Some(parse_stats_debug(stats));
            section = "";
            continue;
        }
        if line.starts_with('[') {
            section = "";
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(name), Some(dur)) = (parts.next(), parts.next()) else {
            continue;
        };
        let Some(secs) = parse_duration(dur) else {
            continue;
        };
        match section {
            "timing" if name == "total" => run.total = Some(secs),
            "timing" => {
                run.phases.insert(name.to_string(), secs);
            }
            "breakdown" => {
                run.member_times.insert(name.to_string(), secs);
            }
            _ => {}
        }
    }
    (!run.results.is_empty() || run.error.is_some() || run.total.is_some()).then_some(run)
}

/// The integer fields of `VerifyStats { a: 1, per_rule: {"r": 2}, .. }` (its
/// `Debug` rendering). Timing wrappers and anything else are dropped.
fn parse_stats_debug(s: &str) -> Value {
    let mut out = Map::new();
    let body = s
        .trim()
        .strip_prefix("VerifyStats {")
        .and_then(|b| b.strip_suffix('}'))
        .unwrap_or(s);
    // Split on top-level commas only.
    let mut depth = 0i32;
    let mut in_str = false;
    let mut fields = Vec::new();
    let mut cur = String::new();
    for c in body.chars() {
        match c {
            '"' => in_str = !in_str,
            '{' | '(' | '[' if !in_str => depth += 1,
            '}' | ')' | ']' if !in_str => depth -= 1,
            ',' if depth == 0 && !in_str => {
                fields.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    fields.push(cur);
    for field in fields {
        let Some((k, v)) = field.split_once(':') else {
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        if let Ok(n) = v.parse::<u64>() {
            out.insert(k.to_string(), Value::from(n));
        } else if k == "per_rule" {
            let inner = v.trim_start_matches('{').trim_end_matches('}');
            let mut rules = Map::new();
            for entry in inner.split(", ") {
                if let Some((r, n)) = entry.rsplit_once(": ") {
                    if let Ok(n) = n.trim().parse::<u64>() {
                        rules.insert(r.trim().trim_matches('"').to_string(), Value::from(n));
                    }
                }
            }
            out.insert("per_rule".into(), Value::Object(rules));
        }
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_output() {
        let out = r#"{"schema":1,"ok":false,"error":null,"results":[{"name":"m_a","status":"OK","detail":null},{"name":"m_b","status":"FAIL","detail":"x"}],"phases":{"parse":0.5,"verify":1.5},"total":2.25,"member_times":{"m_a":0.25},"stats":{"prove_probe":3},"peak_rss_mb":12.5}"#;
        let r = parse(out, "").unwrap();
        assert!(r.json);
        assert_eq!(
            r.results,
            [("m_a".into(), "OK".into()), ("m_b".into(), "FAIL".into())]
        );
        assert_eq!(r.total, Some(2.25));
        assert_eq!(r.phases["verify"], 1.5);
        assert_eq!(r.member_times["m_a"], 0.25);
        assert_eq!(r.stats.unwrap()["prove_probe"], 3);
        assert_eq!(r.peak_rss_mb, Some(12.5));
    }

    #[test]
    fn parses_old_text_output() {
        let stdout = "  [OK] m_a\n  [FAIL] m_b: postcondition might not hold: x\n  [SKIP] m_c: depends on `m_b`\n";
        let stderr = "[TIMING]\n  parse          1.500ms\n  verify       2.000s\n  total        2.100s\n\
[VERIFY-BREAKDOWN] (slowest first)\n  m_b           1.900s\n  m_a         100.000µs\n\
[STATS] VerifyStats { saturations: 3, reduces: 0, per_rule: {\"a-b\": 4, \"c\": 5}, timing: TimingTrend(Timing { search: 0.1 }), prove_probe: 7 }\n";
        let r = parse(stdout, stderr).unwrap();
        assert!(!r.json);
        assert_eq!(r.results.len(), 3);
        assert_eq!(r.results[1], ("m_b".into(), "FAIL".into()));
        assert_eq!(r.phases["parse"], 0.0015);
        assert_eq!(r.total, Some(2.1));
        assert!((r.member_times["m_a"] - 1e-4).abs() < 1e-12);
        let s = r.stats.unwrap();
        assert_eq!(s["saturations"], 3);
        assert_eq!(s["prove_probe"], 7);
        assert_eq!(s["per_rule"]["a-b"], 4);
        assert!(s.get("timing").is_none());
    }

    #[test]
    fn pipeline_error_in_text() {
        let r = parse("", "[PIPELINE-ERROR] parse: bad\n").unwrap();
        assert_eq!(r.error.as_deref(), Some("parse: bad"));
    }
}
