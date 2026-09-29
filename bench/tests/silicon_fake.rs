//! The Silicon column end to end, against a stand-in `java` that prints what
//! Silicon prints: measuring, per-member verdicts and the cache round trip.

use std::path::PathBuf;
use std::time::Duration;

use bench::silicon::{self, Cache, Silicon};

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!("bench-silicon-fake-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A script standing in for `java -jar silicon.jar file.vpr`.
fn fake_java(dir: &std::path::Path) -> PathBuf {
    let lines = [
        "Silicon 1.1-SNAPSHOT (abc1234@main)",
        "Silicon found 1 error in 0.25s:",
        "  [0] Assert might fail. Assertion false might not hold. (f.vpr@5.3)",
    ];
    #[cfg(windows)]
    {
        let p = dir.join("java.cmd");
        let body: String = lines.iter().map(|l| format!("@echo {l}\r\n")).collect();
        std::fs::write(&p, body).unwrap();
        p
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join("java");
        let body: String = lines.iter().map(|l| format!("echo '{l}'\n")).collect();
        std::fs::write(&p, format!("#!/bin/sh\n{body}")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }
}

#[test]
fn measures_attributes_errors_and_caches() {
    let dir = scratch();
    let jar = dir.join("silicon.jar");
    std::fs::write(&jar, b"not really a jar").unwrap();
    let vpr = dir.join("f.vpr");
    std::fs::write(
        &vpr,
        "field f: Int\n\nmethod m_a()\n{\n  assert false\n}\n\nmethod m_b()\n{\n}\n",
    )
    .unwrap();

    let mut sil = Silicon::new(fake_java(&dir), jar, vec!["-Xss128m".into()], vec![]).unwrap();
    let r = silicon::measure(
        &mut sil,
        &vpr,
        "vprhash",
        1,
        3,
        Duration::from_secs(60),
        &dir.join("work"),
    )
    .unwrap();

    assert_eq!(
        sil.version.as_deref(),
        Some("Silicon 1.1-SNAPSHOT (abc1234@main)")
    );
    assert!(
        r.silicon
            .starts_with("Silicon 1.1-SNAPSHOT (abc1234@main)@sha256:")
    );
    assert_eq!(r.verified, Some(false));
    assert_eq!(r.wall.runs.len(), 3);
    assert_eq!(r.verify.median, Some(0.25));
    assert!(r.failed_members.contains("m_a"));
    assert!(!r.failed_members.contains("m_b"));

    let cache_path = dir.join("cache.json");
    let mut cache = Cache::load(&cache_path).unwrap();
    cache
        .entries
        .insert(Cache::key("vprhash", &sil.jar_sha256), r);
    cache.save(&cache_path).unwrap();
    let back = Cache::load(&cache_path).unwrap();
    let hit = &back.entries[&Cache::key("vprhash", &sil.jar_sha256)];
    assert_eq!(hit.errors[0].member.as_deref(), Some("m_a"));
    let _ = std::fs::remove_dir_all(&dir);
}
