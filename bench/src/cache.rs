//! Results that depend only on a file's content and a tool's version, not on
//! the commit being measured, cached across runs in a JSON file.
//!
//! Silicon's result depends only on the `.vpr` and the jar, and rustc's time
//! only on the `.rs`, the compiler and its arguments; neither changes with
//! Helium. Each is measured once per input and reused until the input or the
//! tool changes. The key format belongs to the user of the cache.

use std::collections::BTreeMap;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Cache<T> {
    pub schema: u32,
    pub entries: BTreeMap<String, T>,
}

impl<T> Default for Cache<T> {
    fn default() -> Self {
        Cache {
            schema: 1,
            entries: BTreeMap::new(),
        }
    }
}

impl<T: Serialize + DeserializeOwned> Cache<T> {
    /// Read the cache at `path`; a missing file is an empty cache.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(std::io::Error::other),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Cache::default()),
            Err(e) => Err(e),
        }
    }

    /// Write the cache through a temporary file, so a crash mid-write never
    /// leaves a truncated cache behind.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(tmp, path)
    }
}
