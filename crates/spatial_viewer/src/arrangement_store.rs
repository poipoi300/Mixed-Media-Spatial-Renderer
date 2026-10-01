//! Arrangements kept between sessions, one per view: a catalog's roots and
//! the server's control values together name a view, and whatever the user
//! arranged in it comes back when it is loaded again.
//!
//! Images are kept by path, which outlasts the ids a server hands out for
//! one load, and loose positions relative to the layout's origin, where a
//! new load starts. Changes are saved in the background shortly after they
//! happen, and once more on exit.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use spatial_api::{ControlValues, ProjectionPage};
use spatial_viewer_ui::{FolderControls, FolderDisplay, NavigationSettings, ViewSettings};

use crate::catalog_load::{CatalogLoadTask, InitialPlayerPlacement};
use crate::catalog_session::user_state_directory;
use crate::folders::{ArrangedContents, FolderKey, ManualArrangement, ManualPlacement};
use crate::manual_spacing::SelectionState;
use crate::media_settings::{read_state_file, SettingsWriter};
use crate::{ExplorerScene, FlyCamera};

const STORE_FILE_NAME: &str = "arrangements.json";
/// How long after the first unsaved change the arrangements are written.
const SAVE_DELAY: Duration = Duration::from_secs(1);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
enum SavedFolderKey {
    Group(String),
    Manual(u64),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
enum SavedPlacement {
    /// Relative to the layout's origin.
    Loose([f32; 3]),
    InFolder {
        folder: SavedFolderKey,
        offset: [f32; 3],
    },
}

/// One view's arrangement as the file holds it. Missing fields load empty,
/// so a file written by an older viewer still loads.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
struct SavedArrangement {
    /// Where images stand, by path.
    images: BTreeMap<String, SavedPlacement>,
    made_folders: Vec<u64>,
    folders: Vec<(SavedFolderKey, SavedPlacement)>,
    deleted_groups: Vec<String>,
    names: Vec<(SavedFolderKey, String)>,
    tag_overrides: Vec<(SavedFolderKey, bool)>,
    folders_made: u64,
}

impl SavedArrangement {
    fn is_empty(&self) -> bool {
        self.images.is_empty()
            && self.made_folders.is_empty()
            && self.folders.is_empty()
            && self.deleted_groups.is_empty()
            && self.names.is_empty()
            && self.tag_overrides.is_empty()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct StoreFile {
    /// Whether folders show their tags, unless one says otherwise.
    show_folder_tags: bool,
    /// Whether open folders show the icon that closes them.
    show_folder_close_icons: bool,
    /// Whether folders show their cubes' walls.
    show_folder_backgrounds: bool,
    /// By view key; see [`ArrangementStore::view_key`].
    views: BTreeMap<String, SavedArrangement>,
    /// Where the camera was left in each view, by view key.
    cameras: BTreeMap<String, SavedCamera>,
}

impl Default for StoreFile {
    fn default() -> Self {
        Self::new(FolderDisplay::default(), BTreeMap::new(), BTreeMap::new())
    }
}

impl StoreFile {
    fn new(
        display: FolderDisplay,
        views: BTreeMap<String, SavedArrangement>,
        cameras: BTreeMap<String, SavedCamera>,
    ) -> Self {
        Self {
            show_folder_tags: display.tags,
            show_folder_close_icons: display.close_icons,
            show_folder_backgrounds: display.backgrounds,
            views,
            cameras,
        }
    }

    fn display(&self) -> FolderDisplay {
        FolderDisplay {
            tags: self.show_folder_tags,
            close_icons: self.show_folder_close_icons,
            backgrounds: self.show_folder_backgrounds,
        }
    }
}

/// Where the camera was left in one view: its position relative to the
/// layout's origin (where a load starts), where it looked, and how fast it
/// flew.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub(crate) struct SavedCamera {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub base_speed: f32,
}

fn saved_key(key: &FolderKey) -> SavedFolderKey {
    match key {
        FolderKey::Group(key) => SavedFolderKey::Group(key.to_string()),
        FolderKey::Manual(number) => SavedFolderKey::Manual(*number),
    }
}

fn live_key(key: &SavedFolderKey) -> FolderKey {
    match key {
        SavedFolderKey::Group(key) => FolderKey::group(key),
        SavedFolderKey::Manual(number) => FolderKey::Manual(*number),
    }
}

fn saved_placement(placement: &ManualPlacement, origin: Vec3) -> SavedPlacement {
    match placement {
        ManualPlacement::Loose(position) => SavedPlacement::Loose((*position - origin).to_array()),
        ManualPlacement::InFolder { folder, offset } => SavedPlacement::InFolder {
            folder: saved_key(folder),
            offset: offset.to_array(),
        },
    }
}

fn live_placement(placement: &SavedPlacement, origin: Vec3) -> ManualPlacement {
    match placement {
        SavedPlacement::Loose(position) => {
            ManualPlacement::Loose(Vec3::from_array(*position) + origin)
        }
        SavedPlacement::InFolder { folder, offset } => ManualPlacement::InFolder {
            folder: live_key(folder),
            offset: Vec3::from_array(*offset),
        },
    }
}

/// `contents` as the file keeps it: images by path, found in `projection`,
/// and loose positions relative to `origin`.
fn save_contents(
    contents: &ArrangedContents,
    folders_made: u64,
    projection: &ProjectionPage,
    origin: Vec3,
) -> SavedArrangement {
    let paths: HashMap<usize, &str> = projection
        .points
        .iter()
        .map(|point| (point.image_id, point.path.as_str()))
        .collect();
    let mut images: BTreeMap<String, SavedPlacement> = contents
        .saved_placements
        .iter()
        .map(|(path, placement)| (path.to_string(), saved_placement(placement, origin)))
        .collect();
    for (image_id, placement) in contents.placements.iter() {
        if let Some(path) = paths.get(image_id) {
            images.insert((*path).to_owned(), saved_placement(placement, origin));
        }
    }
    let mut folders: Vec<(SavedFolderKey, SavedPlacement)> = contents
        .folder_placements
        .iter()
        .map(|(key, placement)| (saved_key(key), saved_placement(placement, origin)))
        .collect();
    folders.sort_by(|left, right| format!("{:?}", left.0).cmp(&format!("{:?}", right.0)));
    let mut deleted_groups: Vec<String> = contents
        .deleted_groups
        .iter()
        .filter_map(|key| match key {
            FolderKey::Group(key) => Some(key.to_string()),
            FolderKey::Manual(_) => None,
        })
        .collect();
    deleted_groups.sort();
    let mut names: Vec<(SavedFolderKey, String)> = contents
        .names
        .iter()
        .map(|(key, name)| (saved_key(key), name.clone()))
        .collect();
    names.sort_by(|left, right| format!("{:?}", left.0).cmp(&format!("{:?}", right.0)));
    let mut tag_overrides: Vec<(SavedFolderKey, bool)> = contents
        .tag_overrides
        .iter()
        .map(|(key, shown)| (saved_key(key), *shown))
        .collect();
    tag_overrides.sort_by(|left, right| format!("{:?}", left.0).cmp(&format!("{:?}", right.0)));
    SavedArrangement {
        images,
        made_folders: contents
            .made_folders
            .iter()
            .filter_map(|key| match key {
                FolderKey::Manual(number) => Some(*number),
                FolderKey::Group(_) => None,
            })
            .collect(),
        folders,
        deleted_groups,
        names,
        tag_overrides,
        folders_made,
    }
}

/// What the file kept, as a new view starts it: every image placement by
/// path, and loose positions relative to `origin`.
fn live_contents(saved: &SavedArrangement, origin: Vec3) -> ArrangedContents {
    ArrangedContents {
        placements: Arc::default(),
        saved_placements: Arc::new(
            saved
                .images
                .iter()
                .map(|(path, placement)| {
                    (Arc::from(path.as_str()), live_placement(placement, origin))
                })
                .collect(),
        ),
        made_folders: Arc::new(
            saved
                .made_folders
                .iter()
                .map(|&number| FolderKey::Manual(number))
                .collect(),
        ),
        folder_placements: Arc::new(
            saved
                .folders
                .iter()
                .map(|(key, placement)| (live_key(key), live_placement(placement, origin)))
                .collect(),
        ),
        deleted_groups: Arc::new(
            saved
                .deleted_groups
                .iter()
                .map(|key| FolderKey::group(key))
                .collect::<HashSet<_>>(),
        ),
        names: Arc::new(
            saved
                .names
                .iter()
                .map(|(key, name)| (live_key(key), name.clone()))
                .collect(),
        ),
        tag_overrides: Arc::new(
            saved
                .tag_overrides
                .iter()
                .map(|(key, shown)| (live_key(key), *shown))
                .collect(),
        ),
    }
}

/// Every view's arrangement, which view the live one belongs to, and
/// which parts of every folder show.
#[derive(Resource)]
pub(crate) struct ArrangementStore {
    folder_display: FolderDisplay,
    views: BTreeMap<String, SavedArrangement>,
    cameras: BTreeMap<String, SavedCamera>,
    /// Where the arrangements are saved; `None` keeps them in memory only.
    file: Option<PathBuf>,
    /// The view the live arrangement belongs to.
    current_view: Option<String>,
    /// The live arrangement's revision last kept in `views`.
    kept_revision: Option<u64>,
    unsaved_since: Option<Instant>,
    writer: SettingsWriter,
}

impl ArrangementStore {
    /// The saved arrangements, or none when there are none. A file that
    /// cannot be read is moved aside rather than overwritten.
    pub(crate) fn load() -> Self {
        let file = user_state_directory().map(|directory| directory.join(STORE_FILE_NAME));
        let (saved, file) = match file {
            Some(file) => match read_state_file::<StoreFile>(&file, "arrangements") {
                Ok(saved) => (saved.unwrap_or_default(), Some(file)),
                Err(()) => (StoreFile::default(), None),
            },
            None => (StoreFile::default(), None),
        };
        Self::from_file(saved, file)
    }

    /// Arrangements kept in memory only, for runs (benchmarks, the
    /// performance harness) that must neither follow nor change the user's.
    pub(crate) fn in_memory() -> Self {
        Self::from_file(StoreFile::default(), None)
    }

    fn from_file(saved: StoreFile, file: Option<PathBuf>) -> Self {
        Self {
            folder_display: saved.display(),
            views: saved.views,
            cameras: saved.cameras,
            file,
            current_view: None,
            kept_revision: None,
            unsaved_since: None,
            writer: SettingsWriter::new("arrangements"),
        }
    }

    /// Where the camera was left in the current view, if it was kept.
    pub(crate) fn current_camera(&self) -> Option<SavedCamera> {
        self.cameras.get(self.current_view.as_ref()?).copied()
    }

    /// Keeps where the camera stands in the current view.
    fn remember_camera(&mut self, camera: SavedCamera) {
        let Some(view) = &self.current_view else {
            return;
        };
        if self.cameras.get(view) != Some(&camera) {
            self.cameras.insert(view.clone(), camera);
            self.unsaved_since.get_or_insert_with(Instant::now);
        }
    }

    pub(crate) fn folder_display(&self) -> FolderDisplay {
        self.folder_display
    }

    fn set_folder_display(&mut self, display: FolderDisplay) {
        if self.folder_display != display {
            self.folder_display = display;
            self.unsaved_since.get_or_insert_with(Instant::now);
        }
    }

    /// Names the view `roots` shown under `values` give.
    pub(crate) fn view_key(roots: &[String], values: &ControlValues) -> String {
        let mut roots = roots.to_vec();
        roots.sort();
        serde_json::to_string(&(roots, values)).expect("roots and control values encode")
    }

    /// Keeps the live arrangement for the view it belongs to, then starts
    /// `view` arranged as it was kept. `projection` and `origin` are the
    /// scene's as the live arrangement was laid out in it; the new view
    /// starts at the origin.
    pub(crate) fn switch_view(
        &mut self,
        view: String,
        arrangement: &mut ManualArrangement,
        projection: &ProjectionPage,
        origin: Vec3,
    ) {
        self.keep(arrangement, projection, origin);
        let saved = self.views.get(&view).cloned().unwrap_or_default();
        arrangement.start_view(live_contents(&saved, Vec3::ZERO), saved.folders_made);
        self.current_view = Some(view);
        self.kept_revision = Some(arrangement.revision());
    }

    /// Keeps the live arrangement for its view, when it changed since it was
    /// last kept.
    fn keep(&mut self, arrangement: &ManualArrangement, projection: &ProjectionPage, origin: Vec3) {
        let Some(view) = self.current_view.clone() else {
            return;
        };
        if self.kept_revision == Some(arrangement.revision()) {
            return;
        }
        self.kept_revision = Some(arrangement.revision());
        let saved = save_contents(
            &arrangement.snapshot().contents,
            arrangement.folders_made(),
            projection,
            origin,
        );
        let changed = if saved.is_empty() {
            self.views.remove(&view).is_some()
        } else {
            self.views.get(&view) != Some(&saved) && {
                self.views.insert(view, saved);
                true
            }
        };
        if changed {
            self.unsaved_since.get_or_insert_with(Instant::now);
        }
    }

    fn save(&mut self, blocking: bool) {
        if self.writer.take_failure() {
            self.unsaved_since.get_or_insert_with(Instant::now);
        }
        if self.unsaved_since.take().is_none() {
            return;
        }
        let Some(file) = self.file.clone() else {
            return;
        };
        let store = StoreFile::new(
            self.folder_display,
            self.views.clone(),
            self.cameras.clone(),
        );
        match serde_json::to_string_pretty(&store) {
            Ok(contents) => self.writer.write(file, contents, blocking),
            Err(error) => eprintln!("Failed to encode arrangements: {error}"),
        }
    }
}

/// Keeps where the camera rests in the current view while the setting says
/// so: once it has held still for a frame, so flying costs nothing. Waits
/// while a load is yet to place the camera in the view.
pub(crate) fn remember_camera_pose(
    view_settings: Res<ViewSettings>,
    placement: Res<InitialPlayerPlacement>,
    scene: Res<ExplorerScene>,
    navigation: Res<NavigationSettings>,
    camera: Query<(&Transform, &FlyCamera)>,
    mut store: ResMut<ArrangementStore>,
    mut last_frame: Local<Option<SavedCamera>>,
) {
    let Ok((transform, camera)) = camera.get_single() else {
        return;
    };
    if !view_settings.remember_camera || !placement.settled() || scene.projection.points.is_empty()
    {
        *last_frame = None;
        return;
    }
    let pose = SavedCamera {
        position: (transform.translation - scene.folders.origin).to_array(),
        yaw: camera.yaw,
        pitch: camera.pitch,
        base_speed: navigation.base_speed,
    };
    if *last_frame == Some(pose) {
        store.remember_camera(pose);
    }
    *last_frame = Some(pose);
}

/// Keeps the live arrangement whenever no drag is moving it, and which
/// parts of every folder show, and saves them shortly after they change.
pub(crate) fn keep_arrangements(
    mut store: ResMut<ArrangementStore>,
    task: Res<CatalogLoadTask>,
    scene: Res<ExplorerScene>,
    selection: Res<SelectionState>,
    folder_controls: Res<FolderControls>,
) {
    store.set_folder_display(folder_controls.display);
    if !selection.pressing() {
        store.keep(&task.arrangement(), &scene.projection, scene.folders.origin);
    }
    if store
        .unsaved_since
        .is_some_and(|since| since.elapsed() >= SAVE_DELAY)
    {
        store.save(false);
    }
}

/// Keeps and writes the live arrangement before the app closes.
pub(crate) fn keep_arrangements_on_exit(
    mut exit: EventReader<AppExit>,
    mut store: ResMut<ArrangementStore>,
    task: Res<CatalogLoadTask>,
    scene: Res<ExplorerScene>,
) {
    if exit.read().count() > 0 {
        store.keep(&task.arrangement(), &scene.projection, scene.folders.origin);
        store.save(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatial_api::ProjectionPoint;

    fn projection() -> ProjectionPage {
        ProjectionPage {
            points: (0..3)
                .map(|image_id| ProjectionPoint {
                    image_id,
                    path: format!("image_{image_id}.png"),
                    position: [0.0; 3],
                    group: None,
                    width: None,
                    height: None,
                    media_type: "image".to_owned(),
                    duration_seconds: None,
                    coordinate_labels: [None, None, None],
                })
                .collect(),
            ..default()
        }
    }

    #[test]
    fn a_view_comes_back_as_it_was_left_and_another_starts_empty() {
        let mut store = ArrangementStore::in_memory();
        let mut arrangement = ManualArrangement::default();
        let projection = projection();
        let origin = Vec3::new(10.0, 0.0, 0.0);
        store.switch_view("first".to_owned(), &mut arrangement, &projection, origin);
        let folder = arrangement.make_folder(ManualPlacement::Loose(Vec3::new(14.0, 1.0, 0.0)));
        arrangement.place(
            1,
            ManualPlacement::InFolder {
                folder: folder.clone(),
                offset: Vec3::X,
            },
        );
        arrangement.place(2, ManualPlacement::Loose(Vec3::new(12.0, 0.0, 0.0)));
        arrangement.rename_folder(&folder, "Kept");

        store.switch_view("second".to_owned(), &mut arrangement, &projection, origin);
        assert_eq!(arrangement.snapshot().contents, ArrangedContents::default());

        store.switch_view("first".to_owned(), &mut arrangement, &projection, origin);
        let contents = arrangement.snapshot().contents;
        // Relative to the new view's origin, where loads start.
        assert_eq!(
            contents.folder_placements.get(&folder),
            Some(&ManualPlacement::Loose(Vec3::new(4.0, 1.0, 0.0)))
        );
        assert_eq!(
            contents.saved_placements.get("image_2.png"),
            Some(&ManualPlacement::Loose(Vec3::new(2.0, 0.0, 0.0)))
        );
        assert_eq!(
            contents.saved_placements.get("image_1.png"),
            Some(&ManualPlacement::InFolder {
                folder: folder.clone(),
                offset: Vec3::X,
            })
        );
        assert_eq!(
            contents.names.get(&folder).map(String::as_str),
            Some("Kept")
        );
        // A folder made next never reuses a kept one's number.
        assert_ne!(
            arrangement.make_folder(ManualPlacement::Loose(Vec3::ZERO)),
            folder
        );
    }

    #[test]
    fn the_file_round_trips() {
        let saved = SavedArrangement {
            images: BTreeMap::from([("a.png".to_owned(), SavedPlacement::Loose([1.0, 2.0, 3.0]))]),
            made_folders: vec![0],
            folders: vec![(
                SavedFolderKey::Manual(0),
                SavedPlacement::InFolder {
                    folder: SavedFolderKey::Group("\u{1f}=2024".to_owned()),
                    offset: [1.0, 0.0, 0.0],
                },
            )],
            deleted_groups: vec!["=x".to_owned()],
            names: vec![(SavedFolderKey::Manual(0), "Mine".to_owned())],
            tag_overrides: vec![(SavedFolderKey::Manual(0), false)],
            folders_made: 1,
        };
        let display = FolderDisplay {
            tags: false,
            close_icons: true,
            backgrounds: false,
        };
        let camera = SavedCamera {
            position: [1.0, 2.0, -3.0],
            yaw: 0.5,
            pitch: -0.25,
            base_speed: 4.0,
        };
        let file = StoreFile::new(
            display,
            BTreeMap::from([("view".to_owned(), saved.clone())]),
            BTreeMap::from([("view".to_owned(), camera)]),
        );
        let text = serde_json::to_string(&file).expect("encodes");
        let read: StoreFile = serde_json::from_str(&text).expect("decodes");
        assert_eq!(read.views.get("view"), Some(&saved));
        assert_eq!(read.cameras.get("view"), Some(&camera));
        assert_eq!(read.display(), display);
        // A file from before the settings existed shows everything.
        let older: StoreFile = serde_json::from_str(r#"{"views": {}}"#).expect("decodes");
        assert_eq!(older.display(), FolderDisplay::default());
    }

    #[test]
    fn view_keys_ignore_root_order() {
        let values = ControlValues::new();
        assert_eq!(
            ArrangementStore::view_key(&["b".to_owned(), "a".to_owned()], &values),
            ArrangementStore::view_key(&["a".to_owned(), "b".to_owned()], &values)
        );
    }
}
