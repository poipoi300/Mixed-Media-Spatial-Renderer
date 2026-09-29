//! Finding the media files a shape arranges.

use std::path::{Path, PathBuf};

const IMAGE_EXTENSIONS: [&str; 6] = ["png", "jpg", "jpeg", "bmp", "webp", "gif"];
const VIDEO_EXTENSIONS: [&str; 6] = ["mp4", "webm", "mov", "mkv", "m4v", "avi"];
/// Bounds a walk over an unexpectedly deep or large tree, so a mistyped root
/// cannot hang the server.
const MAX_FILES: usize = 200_000;
const MAX_DEPTH: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaFile {
    pub path: PathBuf,
    pub is_video: bool,
}

/// Every image and video under `roots`, in a stable order.
///
/// Sorting matters: the layout assigns files to slots by position, so an
/// unsorted walk would reshuffle the scene on every reload and make the
/// Randomize button's seed meaningless.
pub fn collect_media(roots: &[String]) -> Vec<MediaFile> {
    let mut files = Vec::new();
    for root in roots {
        walk(Path::new(root), 0, &mut files);
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    files.dedup();
    files
}

fn walk(directory: &Path, depth: usize, files: &mut Vec<MediaFile>) {
    if depth > MAX_DEPTH || files.len() >= MAX_FILES {
        return;
    }
    if directory.is_file() {
        push_media(directory, files);
        return;
    }
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        if files.len() >= MAX_FILES {
            return;
        }
        let path = entry.path();
        match entry.file_type() {
            // Symlinks are not followed: a link back up the tree would
            // otherwise walk forever despite the depth bound.
            Ok(file_type) if file_type.is_dir() => walk(&path, depth + 1, files),
            Ok(file_type) if file_type.is_file() => push_media(&path, files),
            _ => {}
        }
    }
}

fn push_media(path: &Path, files: &mut Vec<MediaFile>) {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return;
    };
    let extension = extension.to_ascii_lowercase();
    let is_video = VIDEO_EXTENSIONS.contains(&extension.as_str());
    if !is_video && !IMAGE_EXTENSIONS.contains(&extension.as_str()) {
        return;
    }
    files.push(MediaFile {
        path: path.to_path_buf(),
        is_video,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_root() -> PathBuf {
        let root = std::env::temp_dir().join(format!("shape_api_media_{}", std::process::id()));
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        for name in ["b.png", "a.JPG", "notes.txt", "clip.mp4"] {
            std::fs::write(root.join(name), b"x").unwrap();
        }
        std::fs::write(nested.join("c.webp"), b"x").unwrap();
        root
    }

    #[test]
    fn collects_media_recursively_in_a_stable_order_and_skips_other_files() {
        let root = fixture_root();

        let files = collect_media(&[root.to_string_lossy().into_owned()]);
        let names: Vec<String> = files
            .iter()
            .map(|file| {
                file.path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        assert_eq!(names, vec!["a.JPG", "b.png", "clip.mp4", "c.webp"]);
        assert!(
            files
                .iter()
                .find(|f| f.path.ends_with("clip.mp4"))
                .unwrap()
                .is_video
        );
        assert!(
            !files
                .iter()
                .find(|f| f.path.ends_with("b.png"))
                .unwrap()
                .is_video
        );
        // A second walk must agree, or a reload would reshuffle the scene.
        assert_eq!(collect_media(&[root.to_string_lossy().into_owned()]), files);

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_missing_root_yields_nothing_rather_than_failing() {
        assert!(collect_media(&["Z:/definitely/not/here".to_owned()]).is_empty());
        assert!(collect_media(&[]).is_empty());
    }
}
