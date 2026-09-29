//! This process's peak resident memory, for `verify --json`.
//!
//! Read from the OS rather than tracked by an allocator, so it covers
//! everything (egg's e-graphs, the parser, Z3 if it is ever loaded) and costs
//! nothing until asked. No dependency: `/proc` on Linux, one `kernel32` call on
//! Windows, `None` elsewhere. The `bench` runner measures child processes from
//! the outside as well, which works on any platform and for any build.

/// Peak resident set size in bytes, if this platform exposes it.
pub fn peak_rss_bytes() -> Option<u64> {
    imp::peak_rss_bytes()
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn peak_rss_bytes() -> Option<u64> {
        // `VmHWM:   123456 kB` — the high-water mark of the resident set.
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
}

#[cfg(windows)]
mod imp {
    /// `PROCESS_MEMORY_COUNTERS` from `psapi.h`.
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    pub fn peak_rss_bytes() -> Option<u64> {
        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            ..Default::default()
        };
        // SAFETY: `GetCurrentProcess` returns a pseudo-handle that needs no
        // closing, and `counters` is a correctly sized, writable struct whose
        // size is passed alongside it.
        let ok =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        (ok != 0).then_some(counters.peak_working_set_size as u64)
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod imp {
    pub fn peak_rss_bytes() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(any(target_os = "linux", windows))]
    fn reports_a_plausible_peak() {
        let peak = super::peak_rss_bytes().expect("supported platform");
        // A running test binary is at least a few hundred KiB and far below 1 TiB.
        assert!(peak > 100 * 1024 && peak < 1 << 40, "{peak}");
    }
}
