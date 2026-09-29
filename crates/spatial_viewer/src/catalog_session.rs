//! Remembers which catalog roots were loaded last so the next launch can
//! restore them instead of opening a default catalog.

use std::fs;
use std::io;
use std::path::PathBuf;

const SESSION_DIRECTORY_NAME: &str = "spatial_viewer";
const LAST_ROOTS_FILE_NAME: &str = "last_catalog_roots.txt";

/// Per-user config file holding the last loaded catalog roots, one per line.
/// `None` when the platform exposes no per-user config location.
pub fn last_catalog_roots_path() -> Option<PathBuf> {
    user_state_directory().map(|directory| directory.join(LAST_ROOTS_FILE_NAME))
}

/// Per-user directory holding everything the viewer persists between runs
/// (last catalog roots, benchmark reports). `None` when the platform exposes
/// no per-user config location.
pub fn user_state_directory() -> Option<PathBuf> {
    dirs::config_local_dir().map(|directory| directory.join(SESSION_DIRECTORY_NAME))
}

/// Roots loaded in the previous session; empty when nothing was saved yet
/// or the file cannot be read.
pub fn load_last_catalog_roots() -> Vec<String> {
    let Some(path) = last_catalog_roots_path() else {
        return Vec::new();
    };
    match fs::read_to_string(&path) {
        Ok(contents) => parse_roots(&contents),
        Err(_) => Vec::new(),
    }
}

/// Persists `roots` as the catalog to restore on the next launch.
pub fn save_last_catalog_roots(roots: &[String]) -> io::Result<()> {
    let path = last_catalog_roots_path().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "no per-user config directory available",
        )
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serialize_roots(roots))
}

fn parse_roots(contents: &str) -> Vec<String> {
    let mut roots: Vec<String> = Vec::new();
    for line in contents.lines() {
        let root = line.trim();
        if root.is_empty() || roots.iter().any(|existing| existing == root) {
            continue;
        }
        roots.push(root.to_owned());
    }
    roots
}

fn serialize_roots(roots: &[String]) -> String {
    let mut contents = String::new();
    for root in roots {
        contents.push_str(root);
        contents.push('\n');
    }
    contents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_round_trip_one_per_line() {
        let roots = vec!["E:/one".to_owned(), "E:/two words".to_owned()];
        assert_eq!(parse_roots(&serialize_roots(&roots)), roots);
    }

    #[test]
    fn parse_skips_blank_and_duplicate_lines() {
        let parsed = parse_roots("E:/one\n\n  E:/one \nE:/two\n");
        assert_eq!(parsed, vec!["E:/one".to_owned(), "E:/two".to_owned()]);
    }
}
