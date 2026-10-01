//! Folders: each group of images sharing a coordinate shows as one cube
//! until the user opens it.
//!
//! A server names each point's group ([`spatial_api::PointGroup`]); a group
//! of two or more images is a folder, and the user can make more of their
//! own ([`FolderCommand`]). Closed, a folder takes one cube of the layout
//! and shows a few of its images, small, inside a grey cube; open, it takes
//! its whole block and its cube turns see-through. The viewer lays the
//! server's groups out itself ([`lay_out_points`]) on a grid of touching
//! slots sized to what each one shows, so opening a folder pushes its
//! neighbours aside; a folder the user made stands where they made it.
//!
//! An image belongs to the folder it was last dropped in
//! ([`ManualArrangement`]): dragged out of every folder it stays where it
//! was put, and dropped into a folder's cube it joins that folder, whichever
//! one it came from ([`FolderScene::drop_placements`]).
//!
//! A press on a folder toggles it ([`FolderViewState::toggle`]). The change
//! re-derives the scene off-thread through the catalog load task, arriving
//! like a snapshot that moves points and adds or drops the images a folder
//! reveals or hides. Everything it moves springs to its new place
//! ([`BillboardMotion`]); images a closing folder hides fly back into it and
//! vanish ([`RetiringBillboard`]); images an opening folder reveals fly out
//! of it as their textures arrive ([`launch_revealed_images`]), and their
//! placeholders wait for its cube to grow ([`WithheldPlaceholders`]). The
//! folder the user toggled stays where it is and its neighbours move
//! instead. Each of these animations can be switched off, and all run at
//! the pause menu's animation scale ([`AnimationSettings`]); switched off,
//! things are simply where they end up.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::asset::embedded_asset;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, Face, RenderPipelineDescriptor, ShaderRef, SpecializedMeshPipelineError,
};
use bevy::window::PrimaryWindow;
use spatial_api::ProjectionPage;
use spatial_geometry::{block_extent, SlotGrid};
use spatial_viewer_ui::{
    Action, Animation, AnimationSettings, BillboardControls, BillboardFacingSettings, ControlInput,
    FolderControls, FolderRequest, PauseMenuState, RenderResolutionSettings, TextEntry,
    TextEntryTarget, UiInputCapture,
};

use crate::arrangement_history::{step_arrangement, ArrangementHistory};
use crate::catalog_load::CatalogLoadTask;
use crate::folder_labels::{FolderHandleKind, FolderHandles};
use crate::image_loading::{
    billboard_rotation, BillboardPoint, BillboardSurfaceAssets, BillboardWorldSize,
    ImageLoadingState, MediaBillboard,
};
use crate::manual_spacing::{cursor_world_ray, nearest_billboard_hit, SelectionState};
use crate::point_cloud::PointCloud;
use crate::{ExplorerScene, FlyCamera};

/// Names a folder across snapshots and relayouts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum FolderKey {
    /// A group the server named; see [`spatial_api::PointGroup::key`].
    Group(Arc<str>),
    /// A folder the user made, numbered in the order they made them.
    Manual(u64),
}

impl FolderKey {
    pub(crate) fn group(key: &str) -> Self {
        Self::Group(Arc::from(key))
    }

    /// Seeds the folder's preview draw.
    fn seed(&self) -> u64 {
        match self {
            Self::Group(key) => fnv1a(key.as_bytes()),
            Self::Manual(number) => fnv1a(&number.to_le_bytes()),
        }
    }
}

/// Most images a closed folder shows.
const MAX_FOLDER_PREVIEWS: usize = 4;
/// A preview's size relative to a full billboard.
pub(crate) const FOLDER_PREVIEW_SCALE: f32 = 0.4;
/// How far preview centers sit from the folder center along each axis, in
/// cubes. Previews sit on the corners of a tetrahedron, so every viewing
/// angle sees them spread apart rather than stacked.
const FOLDER_PREVIEW_REACH: f32 = 0.16;
const TETRAHEDRON_CORNERS: [Vec3; 4] = [
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(1.0, -1.0, -1.0),
    Vec3::new(-1.0, 1.0, -1.0),
    Vec3::new(-1.0, -1.0, 1.0),
];
/// Mixed into every folder's preview draw, so the pick is random-looking
/// but the same every time the folder closes.
const FOLDER_PREVIEW_SEED: u64 = 0x5eed_f01d_e125_0b0e;
/// Side of a closed folder's cube, in cubes: a little under one, so
/// neighbouring closed folders read as separate boxes.
const CLOSED_FOLDER_SIDE: f32 = 0.92;
/// Room an open folder keeps around its images on every side, in cubes.
const OPEN_FOLDER_PADDING: f32 = 0.25;
/// How far in front of the camera a folder made in empty space stands when
/// no image near the pointer lends it a depth, in cubes.
const NEW_FOLDER_DEFAULT_DEPTH: f32 = 4.0;

const CLOSED_FOLDER_OPACITY: f32 = 0.8;
const OPEN_FOLDER_OPACITY: f32 = 0.35;
/// Shell opacities are quantized to this many shared materials, so a
/// thousand folders share a handful of materials however they animate.
const FOLDER_OPACITY_LEVELS: usize = 24;
/// How quickly a shell's opacity approaches its target, per second.
const FOLDER_OPACITY_RATE: f32 = 9.0;
const FOLDER_SHELL_COLOR: [f32; 3] = [0.6, 0.62, 0.66];
/// A selected folder's cube: the selection accent, see-through.
const SELECTED_FOLDER_COLOR: Color = Color::srgba(1.0, 0.86, 0.24, 0.45);
/// The cube of the folder a drag would drop into.
const DROP_TARGET_FOLDER_COLOR: Color = Color::srgba(0.30, 0.78, 1.0, 0.5);

/// A hovered image grows by this factor.
const HOVER_GROWTH: f32 = 1.15;
/// A hovered closed folder's cube grows by this factor: just short of
/// filling its cell, so it never overlaps its neighbours.
const FOLDER_HOVER_GROWTH: f32 = 1.08;
/// Opening a folder fans its images out from the center: each waits this
/// long per cube of distance from it, up to [`MAX_STAGGER_SECONDS`].
const STAGGER_SECONDS_PER_CUBE: f32 = 0.05;
const MAX_STAGGER_SECONDS: f32 = 0.3;
/// Images of a folder opened this recently fly out of it when their
/// textures arrive; later arrivals simply appear.
const LAUNCH_WINDOW_SECONDS: f32 = 3.0;
/// Size an image flying out of a folder starts at, relative to full size.
const LAUNCH_SCALE: f32 = 0.1;
/// How long an image a closing folder hides takes to fly back into it.
const RETIRE_SECONDS: f32 = 0.45;
/// How long the placeholders of an opening folder's images wait for its
/// cube to grow around them, before their own stagger.
const PLACEHOLDER_REVEAL_SECONDS: f32 = 0.45;

/// A damped spring. Under-damped, so motion overshoots a little and
/// settles: the folders should feel springy, not mechanical.
#[derive(Clone, Copy)]
struct Spring {
    /// Natural frequency, radians per second.
    frequency: f32,
    damping_ratio: f32,
}

const POSITION_SPRING: Spring = Spring {
    frequency: 13.0,
    damping_ratio: 0.62,
};
const SCALE_SPRING: Spring = Spring {
    frequency: 17.0,
    damping_ratio: 0.42,
};
const SHELL_SPRING: Spring = Spring {
    frequency: 11.0,
    damping_ratio: 0.5,
};
/// Frames longer than this advance the springs by only this much, so a
/// hitch cannot fling anything past its target. It is also the longest
/// single step a spring takes, which keeps it stable at any animation scale.
const MAX_SPRING_STEP_SECONDS: f32 = 1.0 / 30.0;

impl Spring {
    /// One semi-implicit Euler step toward `target`.
    fn step(self, value: Vec3, velocity: Vec3, target: Vec3, seconds: f32) -> (Vec3, Vec3) {
        let acceleration = (target - value) * self.frequency * self.frequency
            - velocity * 2.0 * self.damping_ratio * self.frequency;
        let velocity = velocity + acceleration * seconds;
        (value + velocity * seconds, velocity)
    }

    /// Advances `seconds` of animation toward `target`, in steps no longer
    /// than [`MAX_SPRING_STEP_SECONDS`].
    fn advance(
        self,
        mut value: Vec3,
        mut velocity: Vec3,
        target: Vec3,
        seconds: f32,
    ) -> (Vec3, Vec3) {
        let steps = (seconds / MAX_SPRING_STEP_SECONDS).ceil().max(1.0);
        let step = seconds / steps;
        for _ in 0..steps as u32 {
            (value, velocity) = self.step(value, velocity, target, step);
        }
        (value, velocity)
    }

    fn advance_scalar(self, value: f32, velocity: f32, target: f32, seconds: f32) -> (f32, f32) {
        let (value, velocity) = self.advance(
            Vec3::splat(value),
            Vec3::splat(velocity),
            Vec3::splat(target),
            seconds,
        );
        (value.x, velocity.x)
    }
}

/// How far this frame advances animations: at most a hitch's worth of real
/// time, at the animation scale.
fn animation_step_seconds(real_time: &Time<Real>, animations: &AnimationSettings) -> f32 {
    animations.animation_seconds(real_time.delta_secs().min(MAX_SPRING_STEP_SECONDS))
}

/// What the viewer's own folder state adds to a projection to lay it out.
#[derive(Clone)]
pub(crate) struct FolderLayoutInput {
    /// World size of one cube.
    pub cube_size: f32,
    pub open: Arc<HashSet<FolderKey>>,
    /// World offset of the server's groups; see [`FolderScene::origin`].
    pub origin: Vec3,
    /// The folder the user just changed and where it stood, so the layout
    /// can move everything else instead of it. Its images fan out in a
    /// stagger.
    pub anchor: Option<(FolderKey, Vec3)>,
    /// The [`FolderViewState`] revision this input was taken at.
    pub revision: u64,
    pub arrangement: ArrangementSnapshot,
}

impl FolderLayoutInput {
    /// The same input with the arrangement as it stands now, for a load
    /// thread that lays out many snapshots from one input.
    pub(crate) fn with_arrangement(&self, arrangement: &ManualArrangement) -> Self {
        Self {
            arrangement: arrangement.snapshot(),
            ..self.clone()
        }
    }
}

/// Where the user put an image or a folder by hand.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ManualPlacement {
    /// Out of every folder, at this world position.
    Loose(Vec3),
    /// In this folder, this far from its center, in cubes.
    InFolder { folder: FolderKey, offset: Vec3 },
}

/// What the user arranged by hand in one view: where they put images and
/// folders, the folders they made, deleted and named.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ArrangedContents {
    /// Where images stand, by image id.
    pub(crate) placements: Arc<HashMap<usize, ManualPlacement>>,
    /// Where images stand, by path: placements kept from an earlier
    /// session, for images a snapshot may not hold yet. A placement by id
    /// wins over one by path.
    pub(crate) saved_placements: Arc<HashMap<Arc<str>, ManualPlacement>>,
    /// The folders the user made, in the order they made them. Each has a
    /// placement in `folder_placements`.
    pub(crate) made_folders: Arc<Vec<FolderKey>>,
    /// Where folders stand: every folder the user made, and every server
    /// folder they moved out of its slot.
    pub(crate) folder_placements: Arc<HashMap<FolderKey, ManualPlacement>>,
    /// Server folders the user deleted; their groups show as plain blocks.
    pub(crate) deleted_groups: Arc<HashSet<FolderKey>>,
    /// Names the user gave folders.
    pub(crate) names: Arc<HashMap<FolderKey, String>>,
    /// Folders whose tag shows, or hides, whatever the setting for every
    /// folder says.
    pub(crate) tag_overrides: Arc<HashMap<FolderKey, bool>>,
}

impl ArrangedContents {
    /// Where the image with this id and path stands, if the user put it
    /// somewhere.
    fn placement_of(&self, image_id: usize, path: &str) -> Option<&ManualPlacement> {
        self.placements
            .get(&image_id)
            .or_else(|| self.saved_placements.get(path))
    }
}

/// What the user arranged by hand in the current view. Shared with the load
/// thread, so each snapshot it lays out already carries it and the main
/// thread re-derives one only when the arrangement changed while it was in
/// flight.
#[derive(Debug, Default)]
pub(crate) struct ManualArrangement {
    contents: ArrangedContents,
    /// Folders made so far, never lowered: a folder key stays unique across
    /// views and undos.
    folders_made: u64,
    revision: u64,
    /// Counts the views this arrangement has started, so the undo history
    /// knows a new one began.
    view: u64,
}

impl ManualArrangement {
    pub(crate) fn place(&mut self, image_id: usize, placement: ManualPlacement) {
        Arc::make_mut(&mut self.contents.placements).insert(image_id, placement);
        self.revision += 1;
    }

    pub(crate) fn placement(&self, image_id: usize) -> Option<&ManualPlacement> {
        self.contents.placements.get(&image_id)
    }

    pub(crate) fn place_folder(&mut self, key: FolderKey, placement: ManualPlacement) {
        Arc::make_mut(&mut self.contents.folder_placements).insert(key, placement);
        self.revision += 1;
    }

    pub(crate) fn folder_placement(&self, key: &FolderKey) -> Option<&ManualPlacement> {
        self.contents.folder_placements.get(key)
    }

    /// Makes an empty folder standing at `placement` and returns its key.
    pub(crate) fn make_folder(&mut self, placement: ManualPlacement) -> FolderKey {
        let key = FolderKey::Manual(self.folders_made);
        self.folders_made += 1;
        Arc::make_mut(&mut self.contents.made_folders).push(key.clone());
        self.place_folder(key.clone(), placement);
        key
    }

    /// Names a folder, or gives it back its own name when `name` is blank.
    pub(crate) fn rename_folder(&mut self, key: &FolderKey, name: &str) {
        let name = name.trim();
        let names = Arc::make_mut(&mut self.contents.names);
        let changed = if name.is_empty() {
            names.remove(key).is_some()
        } else {
            names.insert(key.clone(), name.to_owned()).as_deref() != Some(name)
        };
        if changed {
            self.revision += 1;
        }
    }

    /// Shows or hides a folder's tag whatever the setting for every folder
    /// says, or (`None`) lets that setting decide again.
    pub(crate) fn set_tag_shown(&mut self, key: &FolderKey, shown: Option<bool>) {
        let overrides = Arc::make_mut(&mut self.contents.tag_overrides);
        let changed = match shown {
            Some(shown) => overrides.insert(key.clone(), shown) != Some(shown),
            None => overrides.remove(key).is_some(),
        };
        if changed {
            self.revision += 1;
        }
    }

    /// Forgets a folder. What it held must have been placed elsewhere first.
    pub(crate) fn delete_folder(&mut self, key: &FolderKey) {
        Arc::make_mut(&mut self.contents.folder_placements).remove(key);
        Arc::make_mut(&mut self.contents.names).remove(key);
        Arc::make_mut(&mut self.contents.tag_overrides).remove(key);
        match key {
            FolderKey::Manual(_) => {
                Arc::make_mut(&mut self.contents.made_folders).retain(|made| made != key);
            }
            FolderKey::Group(_) => {
                Arc::make_mut(&mut self.contents.deleted_groups).insert(key.clone());
            }
        }
        self.revision += 1;
    }

    /// Starts a new view arranged as `contents`, whose made folders are
    /// numbered below `folders_made`.
    pub(crate) fn start_view(&mut self, contents: ArrangedContents, folders_made: u64) {
        self.contents = contents;
        self.folders_made = self.folders_made.max(folders_made);
        self.view += 1;
        self.revision += 1;
    }

    /// Forgets everything arranged by hand in this view, as a new revision
    /// (one undo step), so the view lays out as the server places it.
    pub(crate) fn clear(&mut self) {
        self.contents = ArrangedContents::default();
        self.revision += 1;
    }

    /// Puts the arrangement back as `snapshot` had it, as a new revision.
    pub(crate) fn restore(&mut self, snapshot: &ArrangementSnapshot) {
        self.contents = snapshot.contents.clone();
        self.revision += 1;
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn view(&self) -> u64 {
        self.view
    }

    pub(crate) fn folders_made(&self) -> u64 {
        self.folders_made
    }

    pub(crate) fn snapshot(&self) -> ArrangementSnapshot {
        ArrangementSnapshot {
            contents: self.contents.clone(),
            revision: self.revision,
        }
    }
}

/// A [`ManualArrangement`] as it stood at one revision.
#[derive(Debug, Clone, Default)]
pub(crate) struct ArrangementSnapshot {
    pub(crate) contents: ArrangedContents,
    pub revision: u64,
}

/// A folder as laid out.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Folder {
    pub key: FolderKey,
    pub open: bool,
    /// Every folder around it is open, so it shows.
    pub visible: bool,
    /// The folder it is nested in, as an index into [`FolderScene::folders`].
    pub parent: Option<usize>,
    /// World center of the folder's cube when it shows.
    pub home: Vec3,
    /// World center its cube is drawn at: its home while it shows, or else
    /// the center of the folder it has shrunk into.
    pub center: Vec3,
    /// World size its cube is drawn at; nothing while it does not show.
    pub size: Vec3,
    /// Every image in the folder, shown or not, with where it stands open,
    /// relative to `home`, in cubes.
    pub members: Vec<(usize, Vec3)>,
    /// The images a closed folder shows.
    pub previews: Vec<usize>,
    /// Images in it at any depth.
    pub image_count: usize,
    /// For each axis, the value every image in it at any depth shares, if
    /// they share one.
    pub shared_coordinates: [Option<Arc<str>>; 3],
    /// The name the user gave it.
    pub custom_name: Option<Arc<str>>,
    /// Whether its tag shows whatever the setting for every folder says.
    pub tag_override: Option<bool>,
    /// Where the folder's contents stand in it.
    cells: FolderCells,
}

/// A folder's contents on its grid of cells.
///
/// Everything in a folder, image or folder, takes one cell, and the cells
/// are packed like the scene's own slots: each row, column and layer is as
/// wide as the widest thing in it, so an open folder inside pushes the rest
/// aside instead of covering it. A placement in a folder names a cell (see
/// [`ManualPlacement::InFolder`]), and the packing turns it into a place.
#[derive(Debug, Clone, Default, PartialEq)]
struct FolderCells {
    /// What takes each cell.
    contents: Vec<(Occupant, IVec3)>,
    /// The lowest cell, which packs into rank zero.
    lowest: IVec3,
    grid: SlotGrid,
    /// The middle of the packed slots, which stands at the folder's home.
    middle: Vec3,
    /// The axes the contents spread along, which new contents fill.
    spread: BVec3,
}

impl FolderCells {
    /// Packs `contents` (what takes each wanted cell, and how large it
    /// stands) into cells of their own and places them. Two things wanting
    /// one cell keep it in order; the later one takes the nearest free cell.
    fn pack(wanted: Vec<(Occupant, IVec3, Vec3)>) -> (Self, Vec<Vec3>) {
        let (low, high) = wanted
            .iter()
            .fold((IVec3::MAX, IVec3::MIN), |(low, high), &(_, cell, _)| {
                (low.min(cell), high.max(cell))
            });
        let spread = if wanted.is_empty() {
            BVec3::new(true, true, false)
        } else {
            let spread = high.cmpgt(low);
            if spread.any() {
                spread
            } else {
                BVec3::new(true, true, false)
            }
        };
        let mut taken = HashSet::new();
        let placed: Vec<(Occupant, IVec3, Vec3)> = wanted
            .into_iter()
            .map(|(occupant, cell, extent)| {
                let cell = free_cell(&taken, cell, spread);
                taken.insert(cell);
                (occupant, cell, extent)
            })
            .collect();
        let lowest = placed
            .iter()
            .fold(IVec3::MAX, |low, &(_, cell, _)| low.min(cell));
        let rank = |cell: IVec3| (cell - lowest).as_uvec3().to_array();
        let grid = SlotGrid::pack(placed.iter().map(|&(_, cell, extent)| (rank(cell), extent)));
        let (low_edge, high_edge) = grid.span();
        let middle = (low_edge + high_edge) * 0.5;
        let offsets = placed
            .iter()
            .map(|&(_, cell, _)| grid.center(rank(cell)) - middle)
            .collect();
        let cells = Self {
            contents: placed
                .into_iter()
                .map(|(occupant, cell, _)| (occupant, cell))
                .collect(),
            lowest,
            grid,
            middle,
            spread,
        };
        (cells, offsets)
    }

    /// Size of the packed contents, in cubes; one cube when there are none.
    fn extent(&self) -> Vec3 {
        if self.contents.is_empty() {
            return Vec3::ONE;
        }
        let (low, high) = self.grid.span();
        high - low
    }

    /// The cell holding the place `offset` cubes from the folder's home.
    fn cell_at(&self, offset: Vec3) -> IVec3 {
        if self.contents.is_empty() {
            return offset.round().as_ivec3();
        }
        let packed = offset + self.middle;
        let rank = IVec3::from_array(std::array::from_fn(|axis| {
            self.grid.rank_at(axis, packed[axis]) as i32
        }));
        self.lowest + rank
    }
}

impl Folder {
    /// The folder's name: the one the user gave it, or else the coordinates
    /// everything in it shares, or else the order the user made it in.
    pub(crate) fn name(&self) -> String {
        if let Some(name) = &self.custom_name {
            return name.to_string();
        }
        let shared: Vec<&str> = self
            .shared_coordinates
            .iter()
            .flatten()
            .map(|value| &**value)
            .collect();
        match (&self.key, shared.is_empty()) {
            (FolderKey::Manual(number), true) => format!("Folder {}", number + 1),
            (FolderKey::Group(_), true) => "Folder".to_owned(),
            (_, false) => shared.join(" / "),
        }
    }
}

/// Every folder of the laid-out scene.
#[derive(Debug, Clone, Default)]
pub(crate) struct FolderScene {
    pub folders: Vec<Folder>,
    by_key: HashMap<FolderKey, usize>,
    /// Folder index of every image in a folder, shown or not.
    folder_of: HashMap<usize, usize>,
    /// World offset of the server's groups. Toggling a folder shifts it so
    /// that folder stays put; every group is placed relative to it.
    pub origin: Vec3,
    /// The [`FolderViewState`] revision this layout reflects.
    pub revision: u64,
    /// The folder changed by the change this layout reflects, whose images
    /// fan out in a stagger.
    toggled: Option<usize>,
}

/// The cells of each folder taken so far during one drop, by folder index,
/// so things dropped together in one folder never share a cell.
pub(crate) type TakenCells = HashMap<usize, HashSet<IVec3>>;

/// What is being dropped: its placement must not count the cells it held
/// itself as taken, and a folder never lands inside itself.
#[derive(Clone, Copy)]
enum Dropped<'a> {
    Image(usize),
    Folder(usize),
    /// A new folder that takes these images in.
    NewFolder(&'a [usize]),
    /// What deleted folders held, moving up into the cells they free.
    ContentsOf(&'a [usize]),
}

/// Something in a folder's cell.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Occupant {
    Image(usize),
    Folder(usize),
}

impl Occupant {
    fn leaves_with(self, dropped: Dropped) -> bool {
        match (self, dropped) {
            (Self::Image(image), Dropped::Image(dropped)) => image == dropped,
            (Self::Image(image), Dropped::NewFolder(images)) => images.contains(&image),
            (Self::Folder(folder), Dropped::Folder(dropped)) => folder == dropped,
            (Self::Folder(folder), Dropped::ContentsOf(deleted)) => deleted.contains(&folder),
            _ => false,
        }
    }
}

impl FolderScene {
    pub(crate) fn get(&self, key: &FolderKey) -> Option<&Folder> {
        self.index_of(key).map(|index| &self.folders[index])
    }

    pub(crate) fn index_of(&self, key: &FolderKey) -> Option<usize> {
        self.by_key.get(key).copied()
    }

    pub(crate) fn folder_of(&self, image_id: usize) -> Option<&Folder> {
        self.folder_of
            .get(&image_id)
            .map(|&index| &self.folders[index])
    }

    pub(crate) fn open_count(&self) -> usize {
        self.folders.iter().filter(|folder| folder.open).count()
    }

    /// `index` and every folder it holds at any depth.
    pub(crate) fn subtree(&self, index: usize) -> Vec<usize> {
        (0..self.folders.len())
            .filter(|&candidate| self.is_within(candidate, index))
            .collect()
    }

    /// Whether folder `index` is `ancestor` or nested in it at any depth.
    pub(crate) fn is_within(&self, index: usize, ancestor: usize) -> bool {
        self.ancestry(index).any(|folder| folder == ancestor)
    }

    /// The keys of `index` and each folder around it, outward.
    pub(crate) fn ancestry_keys(&self, index: usize) -> impl Iterator<Item = &FolderKey> + '_ {
        self.ancestry(index).map(|folder| &self.folders[folder].key)
    }

    /// `index` and then each folder around it, outward.
    fn ancestry(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(Some(index), |&folder| self.folders[folder].parent)
    }

    /// Whether the image is in folder `ancestor` at any depth.
    pub(crate) fn image_within(&self, image_id: usize, ancestor: usize) -> bool {
        self.folder_of
            .get(&image_id)
            .is_some_and(|&folder| self.is_within(folder, ancestor))
    }

    /// Moves a folder, everything nested in it and their images by `delta`,
    /// returning the images it moved.
    pub(crate) fn shift_subtree(&mut self, index: usize, delta: Vec3) -> Vec<usize> {
        let subtree = self.subtree(index);
        let mut images = Vec::new();
        for folder in subtree {
            let folder = &mut self.folders[folder];
            folder.home += delta;
            folder.center += delta;
            images.extend(folder.members.iter().map(|&(image_id, _)| image_id));
        }
        images
    }

    fn is_closed_preview(&self, image_id: usize) -> Option<usize> {
        let &index = self.folder_of.get(&image_id)?;
        let folder = &self.folders[index];
        (!folder.open && folder.previews.contains(&image_id)).then_some(index)
    }

    /// How long a moved image waits before it starts moving: images of the
    /// folder just toggled leave in order of distance from its center.
    pub(crate) fn stagger_seconds(&self, image_id: usize, position: Vec3, cube_size: f32) -> f32 {
        let Some(toggled) = self.toggled else {
            return 0.0;
        };
        if !self.image_within(image_id, toggled) {
            return 0.0;
        }
        stagger_from(self.folders[toggled].center, position, cube_size)
    }

    /// The folder a press along the ray is on, given the nearest picture
    /// the ray hits (image id and distance).
    ///
    /// A closed folder is a solid cube: pressing it presses it, but one of
    /// its previews takes a press on itself, as any picture does. An open
    /// folder takes no press: whatever lies inside or
    /// behind it does, and its own handles close it or stand in for it (see
    /// [`crate::folder_labels`]).
    pub(crate) fn pressed_folder(
        &self,
        ray_origin: Vec3,
        ray_direction: Vec3,
        picture_hit: Option<(usize, f32)>,
    ) -> Option<&FolderKey> {
        // A preview is inside its own cube, which would otherwise always be
        // entered first.
        let preview_folder = picture_hit.and_then(|(image_id, _)| self.is_closed_preview(image_id));
        let mut nearest: Option<(f32, Option<usize>)> =
            picture_hit.map(|(_, distance)| (distance, None));
        for (index, folder) in self.folders.iter().enumerate() {
            if !folder.visible || folder.open || preview_folder == Some(index) {
                continue;
            }
            let Some((enter, _)) =
                ray_box_hit(ray_origin, ray_direction, folder.center, folder.size * 0.5)
            else {
                continue;
            };
            if nearest.is_none_or(|(distance, _)| enter < distance) {
                nearest = Some((enter, Some(index)));
            }
        }
        let index = nearest?.1?;
        Some(&self.folders[index].key)
    }

    /// Where images the user dropped belong now: each in the smallest shown
    /// folder whose cube holds the place it was dropped at, or loose there.
    ///
    /// In a folder, an image takes the cell under the place it was dropped
    /// at, or the free cell nearest it. A closed folder's cube shows none of
    /// its cells, so an image dropped on one takes the free cell nearest the
    /// folder's middle.
    pub(crate) fn drop_placements(
        &self,
        drops: impl IntoIterator<Item = (usize, Vec3)>,
        cube_size: f32,
        taken_cells: &mut TakenCells,
    ) -> Vec<(usize, ManualPlacement)> {
        drops
            .into_iter()
            .map(|(image_id, position)| {
                let placement =
                    self.drop_placement(Dropped::Image(image_id), position, cube_size, taken_cells);
                (image_id, placement)
            })
            .collect()
    }

    /// Where folder `index` belongs dropped at `position`: like an image,
    /// but never inside itself.
    pub(crate) fn folder_drop_placement(
        &self,
        index: usize,
        position: Vec3,
        cube_size: f32,
        taken_cells: &mut TakenCells,
    ) -> ManualPlacement {
        self.drop_placement(Dropped::Folder(index), position, cube_size, taken_cells)
    }

    /// Where a folder made at `position`, taking `images` in, stands: in the
    /// folder there, if there is one, like anything dropped there.
    pub(crate) fn new_folder_placement(
        &self,
        position: Vec3,
        images: &[usize],
        cube_size: f32,
    ) -> ManualPlacement {
        self.drop_placement(
            Dropped::NewFolder(images),
            position,
            cube_size,
            &mut TakenCells::new(),
        )
    }

    fn drop_placement(
        &self,
        dropped: Dropped,
        position: Vec3,
        cube_size: f32,
        taken_cells: &mut TakenCells,
    ) -> ManualPlacement {
        match self.folder_holding(position, dropped) {
            Some(index) => self.place_in(index, dropped, position, cube_size, taken_cells),
            None => ManualPlacement::Loose(position),
        }
    }

    /// Puts what is dropped at `position` in folder `index`.
    fn place_in(
        &self,
        index: usize,
        dropped: Dropped,
        position: Vec3,
        cube_size: f32,
        taken_cells: &mut TakenCells,
    ) -> ManualPlacement {
        let folder = &self.folders[index];
        let taken = taken_cells.entry(index).or_insert_with(|| {
            folder
                .cells
                .contents
                .iter()
                .filter(|(occupant, _)| !occupant.leaves_with(dropped))
                .map(|&(_, cell)| cell)
                .collect()
        });
        // A closed folder's cube is smaller than one cell, so anything in it
        // is over its middle cell.
        let wanted = folder.cells.cell_at((position - folder.home) / cube_size);
        let cell = free_cell(taken, wanted, folder.cells.spread);
        taken.insert(cell);
        ManualPlacement::InFolder {
            folder: folder.key.clone(),
            offset: cell.as_vec3(),
        }
    }

    /// The folder an image dropped at `position` would join.
    pub(crate) fn image_drop_target(&self, image_id: usize, position: Vec3) -> Option<&FolderKey> {
        self.folder_holding(position, Dropped::Image(image_id))
            .map(|index| &self.folders[index].key)
    }

    /// The folder folder `index` dropped at `position` would join.
    pub(crate) fn folder_drop_target(&self, index: usize, position: Vec3) -> Option<&FolderKey> {
        self.folder_holding(position, Dropped::Folder(index))
            .map(|folder| &self.folders[folder].key)
    }

    /// The smallest shown folder whose cube holds `position`, other than
    /// the dropped folder and those nested in it.
    fn folder_holding(&self, position: Vec3, dropped: Dropped) -> Option<usize> {
        self.folders
            .iter()
            .enumerate()
            .filter(|&(index, folder)| {
                folder.visible
                    && !matches!(dropped, Dropped::Folder(moving) if self.is_within(index, moving))
                    && (position - folder.center)
                        .abs()
                        .cmple(folder.size * 0.5)
                        .all()
            })
            .min_by(|(_, left), (_, right)| {
                left.size
                    .element_product()
                    .total_cmp(&right.size.element_product())
            })
            .map(|(index, _)| index)
    }
}

/// How long an image `position` away from `center` waits to move when its
/// folder opens or closes.
fn stagger_from(center: Vec3, position: Vec3, cube_size: f32) -> f32 {
    let cubes = position.distance(center) / cube_size.max(f32::EPSILON);
    (cubes * STAGGER_SECONDS_PER_CUBE).min(MAX_STAGGER_SECONDS)
}

/// The free cell nearest `around` in the plane (or line) of `axes` through
/// it: nearest by ring first, then in a straight line, then in a fixed axis
/// order, so a folder fills up from where things are put outward.
fn free_cell(taken: &HashSet<IVec3>, around: IVec3, axes: BVec3) -> IVec3 {
    (0..)
        .find_map(|ring: i32| {
            let reach = |along: bool| if along { ring } else { 0 };
            let (x_reach, y_reach, z_reach) = (reach(axes.x), reach(axes.y), reach(axes.z));
            (-x_reach..=x_reach)
                .flat_map(|x| (-y_reach..=y_reach).map(move |y| (x, y)))
                .flat_map(|(x, y)| (-z_reach..=z_reach).map(move |z| IVec3::new(x, y, z)))
                .filter(|step| {
                    step.abs().max_element() == ring && !taken.contains(&(around + *step))
                })
                .min_by_key(|step| (step.length_squared(), step.z, step.y, step.x))
                .map(|step| around + step)
        })
        .expect("finitely many cells are taken, so some ring has a free one")
}

/// Distances along the ray where it enters and leaves the axis-aligned box,
/// or `None` when it misses the box or starts inside it.
fn ray_box_hit(
    ray_origin: Vec3,
    ray_direction: Vec3,
    center: Vec3,
    half_size: Vec3,
) -> Option<(f32, f32)> {
    let inverse = ray_direction.recip();
    let near_corner = (center - half_size - ray_origin) * inverse;
    let far_corner = (center + half_size - ray_origin) * inverse;
    let enter = near_corner.min(far_corner).max_element();
    let exit = near_corner.max(far_corner).min_element();
    (enter > 0.0 && enter <= exit).then_some((enter, exit))
}

/// A projection laid out under the viewer's folder state.
pub(crate) struct LaidOutProjection {
    /// Only the images the scene shows: a closed folder's hidden images are
    /// not points at all, so nothing loads or draws them.
    pub points: Vec<BillboardPoint>,
    pub folders: FolderScene,
}

struct GroupMembers<'a> {
    key: &'a str,
    index: [u32; 3],
    /// Indices into the projection's points.
    points: Vec<usize>,
}

/// Where one point of the projection goes, decided before the groups are
/// packed.
enum Assignment {
    /// Where the server put it: it is in no group.
    Server,
    /// Where the user left it, out of every folder.
    Loose(Vec3),
    /// In the slot of a group that is not a folder, this far from its
    /// center, in cubes.
    Slot { group: usize, offset: Vec3 },
    /// In a folder, in this cell of it.
    Folder { folder: usize, cell: IVec3 },
}

/// Where a folder stands, before the groups are packed.
#[derive(Clone, Copy)]
enum FolderSpot {
    /// The slot of the server group at this index.
    Slot(usize),
    /// Where the user put it, in world space.
    Fixed(Vec3),
    /// In this cell of another folder.
    Nested { parent: usize, cell: IVec3 },
}

/// A folder while the scene is laid out.
struct PlannedFolder {
    key: FolderKey,
    /// The server group the folder is, if it is one.
    group: Option<usize>,
    spot: FolderSpot,
    open: bool,
    /// Indices into the projection's points, and the cells they want.
    members: Vec<(usize, IVec3)>,
}

impl PlannedFolder {
    fn parent(&self) -> Option<usize> {
        match self.spot {
            FolderSpot::Nested { parent, .. } => Some(parent),
            FolderSpot::Slot(_) | FolderSpot::Fixed(_) => None,
        }
    }
}

/// The cell an offset in cubes falls in: a server's offsets put an even
/// block's cells half a cube off its center, and this puts each of them in
/// a cell of its own either way.
fn cell_of(offset: Vec3) -> IVec3 {
    (offset + 0.5).floor().as_ivec3()
}

/// Places every point of `projection`: grouped points on the slot grid, each
/// folder closed or open as `input` says, ungrouped points where the server
/// put them, and the user's own placements over all of these.
///
/// Every server group of two or more images is a folder until the user
/// deletes it, whatever they have since moved in or out of it, so an
/// emptied folder stays there to be filled again. Folders nest: a folder
/// shows only while every folder around it is open, and its contents are
/// packed on cells sized to what each shows (see [`FolderCells`]).
pub(crate) fn lay_out_points(
    projection: &ProjectionPage,
    input: &FolderLayoutInput,
) -> LaidOutProjection {
    let cube_size = input.cube_size;
    let arrangement = &input.arrangement.contents;
    let mut group_by_key: HashMap<&str, usize> = HashMap::new();
    let mut groups: Vec<GroupMembers> = Vec::new();
    for (point_index, point) in projection.points.iter().enumerate() {
        let Some(group) = &point.group else {
            continue;
        };
        let group_index = *group_by_key.entry(&group.key).or_insert_with(|| {
            groups.push(GroupMembers {
                key: &group.key,
                index: group.index,
                points: Vec::new(),
            });
            groups.len() - 1
        });
        groups[group_index].points.push(point_index);
    }

    let mut planned: Vec<PlannedFolder> = Vec::new();
    let mut folder_of_group: Vec<Option<usize>> = vec![None; groups.len()];
    for (group_index, group) in groups.iter().enumerate() {
        let key = FolderKey::group(group.key);
        if group.points.len() >= 2 && !arrangement.deleted_groups.contains(&key) {
            folder_of_group[group_index] = Some(planned.len());
            planned.push(PlannedFolder {
                key,
                group: Some(group_index),
                spot: FolderSpot::Slot(group_index),
                open: false,
                members: Vec::new(),
            });
        }
    }
    for key in arrangement.made_folders.iter() {
        planned.push(PlannedFolder {
            key: key.clone(),
            group: None,
            // Every folder the user made has a placement; this is replaced
            // below.
            spot: FolderSpot::Fixed(input.origin),
            open: false,
            members: Vec::new(),
        });
    }
    let folder_index: HashMap<FolderKey, usize> = planned
        .iter()
        .enumerate()
        .map(|(index, folder)| (folder.key.clone(), index))
        .collect();
    for folder in &mut planned {
        folder.open = input.open.contains(&folder.key);
        folder.spot = match arrangement.folder_placements.get(&folder.key) {
            Some(ManualPlacement::Loose(position)) => FolderSpot::Fixed(*position),
            Some(ManualPlacement::InFolder {
                folder: parent,
                offset,
            }) => match folder_index.get(parent) {
                Some(&parent) => FolderSpot::Nested {
                    parent,
                    cell: cell_of(*offset),
                },
                None => folder.spot,
            },
            None => folder.spot,
        };
    }
    // A folder nested in itself, however deeply, stands on its own instead.
    for index in 0..planned.len() {
        let mut current = planned[index].parent();
        for _ in 0..planned.len() {
            match current {
                Some(ancestor) if ancestor == index => {
                    planned[index].spot = match planned[index].group {
                        Some(group) => FolderSpot::Slot(group),
                        None => FolderSpot::Fixed(input.origin),
                    };
                    break;
                }
                Some(ancestor) => current = planned[ancestor].parent(),
                None => break,
            }
        }
    }

    let server_offset = |point_index: usize| {
        let group = projection.points[point_index]
            .group
            .as_ref()
            .expect("only grouped points are collected into groups");
        Vec3::from_array(group.offset)
    };
    let assignments: Vec<Assignment> = projection
        .points
        .iter()
        .enumerate()
        .map(|(point_index, point)| {
            // A placement in a folder the scene no longer has falls back to
            // where the server put the image.
            let manual = match arrangement.placement_of(point.image_id, &point.path) {
                Some(ManualPlacement::Loose(position)) => Some(Assignment::Loose(*position)),
                Some(ManualPlacement::InFolder { folder, offset }) => {
                    folder_index.get(folder).map(|&folder| Assignment::Folder {
                        folder,
                        cell: cell_of(*offset),
                    })
                }
                None => None,
            };
            manual.unwrap_or_else(|| {
                let Some(group) = &point.group else {
                    return Assignment::Server;
                };
                let group = group_by_key[group.key.as_str()];
                let offset = server_offset(point_index);
                match folder_of_group[group] {
                    Some(folder) => Assignment::Folder {
                        folder,
                        cell: cell_of(offset),
                    },
                    None => Assignment::Slot { group, offset },
                }
            })
        })
        .collect();
    for (point_index, assignment) in assignments.iter().enumerate() {
        if let Assignment::Folder { folder, cell } = assignment {
            planned[*folder].members.push((point_index, *cell));
        }
    }

    // Innermost folders first, so each folder packs around the folders it
    // holds.
    let depth: Vec<usize> = (0..planned.len())
        .map(|index| {
            std::iter::successors(planned[index].parent(), |&parent| planned[parent].parent())
                .count()
        })
        .collect();
    let mut innermost_first: Vec<usize> = (0..planned.len()).collect();
    innermost_first.sort_by_key(|&index| std::cmp::Reverse(depth[index]));
    let mut children: Vec<Vec<(usize, IVec3)>> = vec![Vec::new(); planned.len()];
    for (index, folder) in planned.iter().enumerate() {
        if let FolderSpot::Nested { parent, cell } = folder.spot {
            children[parent].push((index, cell));
        }
    }
    // What each folder's label reports, from everything in it at any depth.
    let mut image_count = vec![0usize; planned.len()];
    let mut shared_coordinates: Vec<Option<[Option<Arc<str>>; 3]>> = vec![None; planned.len()];
    let share = |shared: &mut Option<[Option<Arc<str>>; 3]>, labels: &[Option<Arc<str>>; 3]| {
        *shared = Some(match shared.take() {
            None => labels.clone(),
            Some(current) => std::array::from_fn(|axis| {
                current[axis]
                    .clone()
                    .filter(|value| labels[axis].as_deref() == Some(&**value))
            }),
        });
    };
    let mut cells = vec![FolderCells::default(); planned.len()];
    // Where each image and each nested folder stands in its folder open,
    // relative to the folder's home, in cubes.
    let mut member_offsets: Vec<Vec<Vec3>> = vec![Vec::new(); planned.len()];
    let mut offset_in_parent = vec![Vec3::ZERO; planned.len()];
    let mut footprint = vec![Vec3::ONE; planned.len()];
    let mut open_extent = vec![Vec3::ONE; planned.len()];
    for &index in &innermost_first {
        // Folders first: a folder the user put in a cell keeps it.
        let wanted: Vec<(Occupant, IVec3, Vec3)> = children[index]
            .iter()
            .map(|&(child, cell)| (Occupant::Folder(child), cell, footprint[child]))
            .chain(planned[index].members.iter().map(|&(point, cell)| {
                (
                    Occupant::Image(projection.points[point].image_id),
                    cell,
                    Vec3::ONE,
                )
            }))
            .collect();
        let (packed, offsets) = FolderCells::pack(wanted);
        let child_count = children[index].len();
        for (&(child, _), &offset) in children[index].iter().zip(&offsets) {
            offset_in_parent[child] = offset;
        }
        member_offsets[index] = offsets[child_count..].to_vec();
        image_count[index] = planned[index].members.len()
            + children[index]
                .iter()
                .map(|&(child, _)| image_count[child])
                .sum::<usize>();
        let mut shared = None;
        for &(point, _) in &planned[index].members {
            let labels: [Option<Arc<str>>; 3] = projection.points[point]
                .coordinate_labels
                .clone()
                .map(|label| label.map(Arc::from));
            share(&mut shared, &labels);
        }
        for &(child, _) in &children[index] {
            if let Some(child_shared) = shared_coordinates[child].clone() {
                share(&mut shared, &child_shared);
            }
        }
        shared_coordinates[index] = shared;
        open_extent[index] = packed.extent() + Vec3::splat(2.0 * OPEN_FOLDER_PADDING);
        if planned[index].open {
            footprint[index] = open_extent[index];
        }
        cells[index] = packed;
    }

    // A folder's slot fits what it shows, and a folder moved out of its
    // slot takes none; any other group's slot fits its block.
    let grid = SlotGrid::pack(
        groups
            .iter()
            .enumerate()
            .filter_map(|(group_index, group)| {
                let footprint = match folder_of_group[group_index] {
                    Some(folder) => match planned[folder].spot {
                        FolderSpot::Slot(_) => footprint[folder],
                        FolderSpot::Fixed(_) | FolderSpot::Nested { .. } => return None,
                    },
                    None => block_extent(group.points.iter().map(|&point| server_offset(point))),
                };
                Some((group.index, footprint))
            }),
    );

    // Outermost folders first, so each nested folder is placed in its
    // parent. Folders under a server slot are placed relative to the
    // layout's origin until it is known.
    let mut relative_home = vec![Vec3::ZERO; planned.len()];
    let mut on_grid = vec![false; planned.len()];
    for &index in innermost_first.iter().rev() {
        (relative_home[index], on_grid[index]) = match planned[index].spot {
            FolderSpot::Slot(group) => (grid.center(groups[group].index) * cube_size, true),
            FolderSpot::Fixed(position) => (position, false),
            FolderSpot::Nested { parent, .. } => (
                relative_home[parent] + offset_in_parent[index] * cube_size,
                on_grid[parent],
            ),
        };
    }
    // The folder the user just changed stays where it was, if moving the
    // whole grid can keep it there.
    let origin = input
        .anchor
        .as_ref()
        .and_then(|(key, world_home)| {
            let &index = folder_index.get(key)?;
            on_grid[index].then(|| *world_home - relative_home[index])
        })
        .unwrap_or(input.origin);
    let home: Vec<Vec3> = relative_home
        .iter()
        .zip(&on_grid)
        .map(|(&home, &on_grid)| if on_grid { home + origin } else { home })
        .collect();
    let slot_center = |group: usize| grid.center(groups[group].index) * cube_size + origin;
    let mut visible = vec![true; planned.len()];
    let mut center = vec![Vec3::ZERO; planned.len()];
    for &index in innermost_first.iter().rev() {
        (visible[index], center[index]) = match planned[index].parent() {
            None => (true, home[index]),
            Some(parent) => {
                let shows = visible[parent] && planned[parent].open;
                (shows, if shows { home[index] } else { center[parent] })
            }
        };
    }

    let mut placements: Vec<Option<(Vec3, f32)>> = projection
        .points
        .iter()
        .zip(&assignments)
        .map(|(point, assignment)| match assignment {
            Assignment::Server => {
                Some((Vec3::from_array(point.position) * cube_size + origin, 1.0))
            }
            Assignment::Loose(position) => Some((*position, 1.0)),
            Assignment::Slot { group, offset } => {
                Some((slot_center(*group) + *offset * cube_size, 1.0))
            }
            // Placed with their folder below, if it shows them at all.
            Assignment::Folder { .. } => None,
        })
        .collect();
    let mut folders = FolderScene {
        origin,
        revision: input.revision,
        by_key: folder_index,
        ..default()
    };
    for (index, (folder, cells)) in planned.into_iter().zip(cells).enumerate() {
        let members: Vec<(usize, Vec3)> = folder
            .members
            .iter()
            .zip(&member_offsets[index])
            .map(|(&(point, _), &offset)| (projection.points[point].image_id, offset))
            .collect();
        let previews = choose_previews(
            &folder.key,
            members.iter().map(|&(image_id, _)| image_id).collect(),
        );
        if visible[index] && folder.open {
            for (&(point, _), &offset) in folder.members.iter().zip(&member_offsets[index]) {
                placements[point] = Some((home[index] + offset * cube_size, 1.0));
            }
        } else if visible[index] {
            for (slot, image_id) in previews.iter().enumerate() {
                let (point, _) = folder
                    .members
                    .iter()
                    .find(|&&(point, _)| projection.points[point].image_id == *image_id)
                    .expect("previews are drawn from the folder's members");
                placements[*point] = Some((
                    home[index] + preview_offset(slot, previews.len()) * cube_size,
                    FOLDER_PREVIEW_SCALE,
                ));
            }
        }
        folders
            .folder_of
            .extend(members.iter().map(|&(image_id, _)| (image_id, index)));
        let parent = folder.parent();
        let custom_name = arrangement
            .names
            .get(&folder.key)
            .map(|name| Arc::from(name.as_str()));
        let tag_override = arrangement.tag_overrides.get(&folder.key).copied();
        folders.folders.push(Folder {
            size: if !visible[index] {
                Vec3::ZERO
            } else if folder.open {
                open_extent[index] * cube_size
            } else {
                Vec3::splat(CLOSED_FOLDER_SIDE * cube_size)
            },
            key: folder.key,
            open: folder.open,
            visible: visible[index],
            parent,
            home: home[index],
            center: center[index],
            image_count: image_count[index],
            shared_coordinates: shared_coordinates[index].take().unwrap_or_default(),
            custom_name,
            tag_override,
            members,
            previews,
            cells,
        });
    }
    folders.toggled = input
        .anchor
        .as_ref()
        .and_then(|(key, _)| folders.by_key.get(key).copied());

    let points = projection
        .points
        .iter()
        .zip(placements)
        .filter_map(|(point, placement)| {
            let (position, scale) = placement?;
            Some(BillboardPoint {
                image_id: point.image_id,
                path: Arc::from(point.path.as_str()),
                position,
                scale,
                is_video: point.media_type == "video",
                duration_seconds: point.duration_seconds,
                source_size: point
                    .width
                    .zip(point.height)
                    .map(|(width, height)| UVec2::new(width, height)),
                coordinate_labels: point
                    .coordinate_labels
                    .clone()
                    .map(|label| label.map(Arc::from)),
            })
        })
        .collect();
    LaidOutProjection { points, folders }
}

/// Up to [`MAX_FOLDER_PREVIEWS`] of a folder's images, drawn at random but
/// seeded by the folder's key, so a folder shows the same images every time.
fn choose_previews(key: &FolderKey, mut image_ids: Vec<usize>) -> Vec<usize> {
    image_ids.sort_unstable();
    let count = image_ids.len().min(MAX_FOLDER_PREVIEWS);
    let mut state = FOLDER_PREVIEW_SEED ^ key.seed();
    for slot in 0..count {
        let remaining = (image_ids.len() - slot) as u64;
        let pick = slot + (splitmix64(&mut state) % remaining) as usize;
        image_ids.swap(slot, pick);
    }
    image_ids.truncate(count);
    image_ids
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

/// Where preview `slot` of `count` sits relative to the folder center, in
/// cubes: the first `count` corners of a tetrahedron, recentered.
fn preview_offset(slot: usize, count: usize) -> Vec3 {
    let corners = &TETRAHEDRON_CORNERS[..count];
    let mean = corners.iter().sum::<Vec3>() / count as f32;
    (corners[slot] - mean) * FOLDER_PREVIEW_REACH
}

/// Which folders are open, as the user left them. The scene catches up
/// through [`apply_folder_changes`] whenever `revision` moves past the
/// revision it was laid out at.
#[derive(Resource, Default)]
pub(crate) struct FolderViewState {
    open: HashSet<FolderKey>,
    revision: u64,
    /// The folder toggled since the last layout was started.
    anchor: Option<FolderKey>,
    /// Real-time second each open folder was opened at.
    opened_at: HashMap<FolderKey, f32>,
    /// The next layout starts from the layout's own origin, as a newly
    /// loaded catalog's does, rather than keeping where folder toggles have
    /// shifted it.
    restart_origin: bool,
}

impl FolderViewState {
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn is_open(&self, key: &FolderKey) -> bool {
        self.open.contains(key)
    }

    pub(crate) fn toggle(&mut self, key: &FolderKey, now_seconds: f32) {
        if self.open.remove(key) {
            self.opened_at.remove(key);
        } else {
            self.open.insert(key.clone());
            self.opened_at.insert(key.clone(), now_seconds);
        }
        self.anchor = Some(key.clone());
        self.revision += 1;
    }

    pub(crate) fn open_all<'a>(
        &mut self,
        keys: impl IntoIterator<Item = &'a FolderKey>,
        now_seconds: f32,
    ) {
        for key in keys {
            if self.open.insert(key.clone()) {
                self.opened_at.insert(key.clone(), now_seconds);
            }
        }
        self.anchor = None;
        self.revision += 1;
    }

    fn close_all(&mut self) {
        self.open.clear();
        self.opened_at.clear();
        self.anchor = None;
        self.revision += 1;
    }

    /// Closes everything for a newly loaded catalog.
    pub(crate) fn reset(&mut self) {
        self.close_all();
    }

    /// Closes everything and lays the scene out from its own origin, as it
    /// was when the catalog first opened.
    pub(crate) fn reset_to_first_open(&mut self) {
        self.close_all();
        self.restart_origin = true;
    }

    /// Lays the scene out again for a change in what the folders hold,
    /// keeping `anchor` (when given) where it stands.
    pub(crate) fn relayout(&mut self, anchor: Option<FolderKey>) {
        if anchor.is_some() {
            self.anchor = anchor;
        }
        self.revision += 1;
    }

    /// The layout input for the current state, placed relative to
    /// `current`. Takes the anchor: only the first layout after a change
    /// keeps the changed folder in place, later ones keep the origin.
    pub(crate) fn layout_input(
        &mut self,
        cube_size: f32,
        current: &FolderScene,
        arrangement: &ManualArrangement,
    ) -> FolderLayoutInput {
        let anchor = self
            .anchor
            .take()
            .and_then(|key| Some((key.clone(), current.get(&key)?.home)));
        let origin = if std::mem::take(&mut self.restart_origin) {
            FolderScene::default().origin
        } else {
            current.origin
        };
        FolderLayoutInput {
            cube_size,
            open: Arc::new(self.open.clone()),
            origin,
            anchor,
            revision: self.revision,
            arrangement: arrangement.snapshot(),
        }
    }

    fn recently_opened(&self, key: &FolderKey, now_seconds: f32) -> bool {
        self.opened_at
            .get(key)
            .is_some_and(|opened| now_seconds - opened <= LAUNCH_WINDOW_SECONDS)
    }
}

/// Applies the folders pill's buttons and publishes what it shows.
pub(crate) fn apply_folder_requests(
    real_time: Res<Time<Real>>,
    scene: Res<ExplorerScene>,
    mut controls: ResMut<FolderControls>,
    mut folder_view: ResMut<FolderViewState>,
) {
    match controls.take_request() {
        Some(FolderRequest::OpenAll) => folder_view.open_all(
            scene.folders.folders.iter().map(|folder| &folder.key),
            real_time.elapsed_secs(),
        ),
        Some(FolderRequest::CloseAll) => folder_view.close_all(),
        None => {}
    }
    let folder_count = scene.folders.folders.len();
    let open_count = scene.folders.open_count();
    if controls.folder_count != folder_count || controls.open_count != open_count {
        controls.folder_count = folder_count;
        controls.open_count = open_count;
    }
}

/// Takes each settled change to the arrangement as an undo step: one per
/// drag, however many frames it moved things for.
pub(crate) fn record_arrangement_history(
    selection: Res<SelectionState>,
    task: Res<CatalogLoadTask>,
    mut history: ResMut<ArrangementHistory>,
) {
    if !selection.pressing() {
        history.record(&task.arrangement());
    }
}

/// A folder the user asked for from a right-click menu.
#[derive(Event, Clone, Debug, PartialEq)]
pub(crate) enum FolderCommand {
    /// A new folder, closed, holding this image, or the whole selection
    /// when the image is part of it. It stands among them, in the folder
    /// they were in.
    AddToNew {
        image_id: usize,
    },
    /// A new, empty folder standing here, in the folder there if any.
    Create {
        center: Vec3,
    },
    /// Selects the folder, alone, for the translate gizmo to move.
    Move {
        folder: FolderKey,
    },
    /// Deletes the folder. What it held moves up to the folder around it,
    /// or out of every folder, where it stood with the folder open.
    Delete {
        folder: FolderKey,
    },
    /// Shows or hides the folder's tag whatever the setting for every folder
    /// says, or (`None`) lets that setting decide again.
    SetTagShown {
        folder: FolderKey,
        shown: Option<bool>,
    },
    /// Opens the rename prompt on the folder.
    Rename {
        folder: FolderKey,
    },
    /// Undoes the last change to how things are arranged.
    Undo,
    Redo,
}

/// The folder the rename prompt is naming.
#[derive(Resource, Default)]
pub(crate) struct FolderRename(Option<FolderKey>);

/// Names the folder the rename prompt was naming once its name is
/// submitted, and forgets it once the prompt is closed.
pub(crate) fn apply_folder_rename(
    mut rename: ResMut<FolderRename>,
    mut text_entry: ResMut<TextEntry>,
    task: Res<CatalogLoadTask>,
    mut folder_view: ResMut<FolderViewState>,
) {
    let Some(folder) = rename.0.clone() else {
        return;
    };
    if let Some(name) = text_entry.take_submitted(TextEntryTarget::FolderName) {
        task.arrangement().rename_folder(&folder, &name);
        folder_view.relayout(None);
    }
    if text_entry.editing() != Some(TextEntryTarget::FolderName) {
        rename.0 = None;
    }
}

/// Carries out what the right-click menus asked of folders, and the delete
/// key on the selected folders.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_folder_commands(
    mut folder_commands: EventReader<FolderCommand>,
    input: ControlInput,
    pause_menu: Res<PauseMenuState>,
    mut selection: ResMut<SelectionState>,
    mut rename: ResMut<FolderRename>,
    mut text_entry: ResMut<TextEntry>,
    mut history: ResMut<ArrangementHistory>,
    loading: Res<ImageLoadingState>,
    scene: Res<ExplorerScene>,
    task: Res<CatalogLoadTask>,
    mut folder_view: ResMut<FolderViewState>,
) {
    let cube_size = scene.billboard_world_size;
    let mut deleted: Vec<usize> = Vec::new();
    for command in folder_commands.read() {
        match command {
            FolderCommand::AddToNew { image_id } => {
                let image_ids: Vec<usize> = if selection.is_selected(*image_id) {
                    selection.selected().collect()
                } else {
                    vec![*image_id]
                };
                let images: Vec<(usize, Vec3)> = image_ids
                    .into_iter()
                    .filter_map(|image_id| {
                        Some((image_id, loading.loaded_point(image_id)?.position))
                    })
                    .collect();
                if images.is_empty() {
                    continue;
                }
                // The folder stands among its images, and opened shows each
                // where it was.
                let center = images.iter().map(|&(_, position)| position).sum::<Vec3>()
                    / images.len() as f32;
                let mut arrangement = task.arrangement();
                let image_ids: Vec<usize> = images.iter().map(|&(image_id, _)| image_id).collect();
                let placement = scene
                    .folders
                    .new_folder_placement(center, &image_ids, cube_size);
                let folder = arrangement.make_folder(placement);
                for &(image_id, position) in &images {
                    arrangement.place(
                        image_id,
                        ManualPlacement::InFolder {
                            folder: folder.clone(),
                            offset: (position - center) / cube_size,
                        },
                    );
                }
                selection.deselect(images.iter().map(|&(image_id, _)| image_id));
                folder_view.relayout(Some(folder));
            }
            FolderCommand::Create { center } => {
                let placement = scene.folders.new_folder_placement(*center, &[], cube_size);
                let folder = task.arrangement().make_folder(placement);
                folder_view.relayout(Some(folder));
            }
            FolderCommand::Move { folder } => selection.select_only_folder(folder.clone()),
            FolderCommand::Undo | FolderCommand::Redo => step_arrangement(
                &mut history,
                &mut task.arrangement(),
                &mut folder_view,
                *command == FolderCommand::Undo,
            ),
            FolderCommand::SetTagShown { folder, shown } => {
                task.arrangement().set_tag_shown(folder, *shown);
                folder_view.relayout(None);
            }
            FolderCommand::Rename { folder } => {
                let Some(shown) = scene.folders.get(folder) else {
                    continue;
                };
                text_entry.begin(TextEntryTarget::FolderName, &shown.name());
                rename.0 = Some(folder.clone());
            }
            // Deleted below, with every other folder deleted this frame, so
            // what each held moves past all of them.
            FolderCommand::Delete { folder } => deleted.extend(scene.folders.index_of(folder)),
        }
    }
    if !pause_menu.paused && input.just_pressed(Action::DeleteFolder) {
        deleted.extend(
            selection
                .selected_folders()
                .filter_map(|key| scene.folders.index_of(key)),
        );
    }
    deleted.sort_unstable();
    deleted.dedup();
    if deleted.is_empty() {
        return;
    }
    let survivor = delete_folders(&scene.folders, &deleted, cube_size, &task);
    for &index in &deleted {
        let key = &scene.folders.folders[index].key;
        selection.deselect_folder(key);
        if rename.0.as_ref() == Some(key) {
            rename.0 = None;
            text_entry.cancel();
        }
    }
    folder_view.relayout(survivor);
}

/// Deletes the folders at `deleted` and returns the folder the first one
/// was in, if that survives, for the relayout to keep in place.
fn delete_folders(
    folders: &FolderScene,
    deleted: &[usize],
    cube_size: f32,
    task: &CatalogLoadTask,
) -> Option<FolderKey> {
    let mut arrangement = task.arrangement();
    delete_folders_from(folders, deleted, cube_size, &mut arrangement);
    let first = *deleted.first()?;
    let survivor = surviving_parent(folders, first, deleted)?;
    Some(folders.folders[survivor].key.clone())
}

/// The nearest folder around `index` that is not being deleted.
fn surviving_parent(folders: &FolderScene, index: usize, deleted: &[usize]) -> Option<usize> {
    folders
        .ancestry(index)
        .skip(1)
        .find(|ancestor| !deleted.contains(ancestor))
}

/// Deletes the folders at `deleted`, moving what they hold up to the nearest
/// folder around them that is not being deleted: into the cells nearest
/// where each thing stood with its folder open, or out of every folder,
/// right there. A server folder still in its slot leaves its own images to
/// that slot instead.
fn delete_folders_from(
    folders: &FolderScene,
    deleted: &[usize],
    cube_size: f32,
    arrangement: &mut ManualArrangement,
) {
    let mut taken_cells = TakenCells::new();
    for &index in deleted {
        let folder = &folders.folders[index];
        let target = surviving_parent(folders, index, deleted);
        let mut up_a_level = |position: Vec3| match target {
            Some(target) => folders.place_in(
                target,
                Dropped::ContentsOf(deleted),
                position,
                cube_size,
                &mut taken_cells,
            ),
            None => ManualPlacement::Loose(position),
        };
        let in_its_slot = matches!(folder.key, FolderKey::Group(_))
            && arrangement.folder_placement(&folder.key).is_none();
        for &(image_id, offset) in &folder.members {
            if in_its_slot && arrangement.placement(image_id).is_none() {
                continue;
            }
            arrangement.place(image_id, up_a_level(folder.home + offset * cube_size));
        }
        for (child_index, child) in folders.folders.iter().enumerate() {
            if child.parent == Some(index) && !deleted.contains(&child_index) {
                arrangement.place_folder(child.key.clone(), up_a_level(child.home));
            }
        }
    }
    for &index in deleted {
        arrangement.delete_folder(&folders.folders[index].key);
    }
}

/// Where a folder made in empty space stands: along the pointer's ray, as
/// deep as the image nearest the ray, so it lands among what the user is
/// looking at.
pub(crate) fn new_folder_center(
    ray_origin: Vec3,
    ray_direction: Vec3,
    image_positions: impl IntoIterator<Item = Vec3>,
    cube_size: f32,
) -> Vec3 {
    let depth = image_positions
        .into_iter()
        .filter_map(|position| {
            let along = (position - ray_origin).dot(ray_direction);
            let off_ray = (position - ray_origin - ray_direction * along).length();
            (along > 0.0).then_some((off_ray / along, along))
        })
        .min_by(|(left, _), (right, _)| left.total_cmp(right))
        .map_or(NEW_FOLDER_DEFAULT_DEPTH * cube_size, |(_, along)| along);
    ray_origin + ray_direction * depth
}

/// Settles where the images and folders the user just dropped belong
/// ([`FolderScene::drop_placements`], [`FolderScene::folder_drop_placement`])
/// and lays the scene out again when that changed any folder, keeping the
/// folder something joined in place, or else the one it left.
///
/// `folders` are indices into the scene's folders, each with the place it
/// was dropped at.
pub(crate) fn settle_drops(
    images: impl IntoIterator<Item = (usize, Vec3)>,
    folders: impl IntoIterator<Item = (usize, Vec3)>,
    scene: &ExplorerScene,
    arrangement: &mut ManualArrangement,
    folder_view: &mut FolderViewState,
) {
    let cube_size = scene.billboard_world_size;
    let mut joined = None;
    let mut left = None;
    let mut moved_folder = false;
    let mut taken_cells = TakenCells::new();
    let mut note = |placement: &ManualPlacement, from: Option<usize>| {
        if let ManualPlacement::InFolder { folder, .. } = placement {
            joined.get_or_insert_with(|| folder.clone());
        }
        if let Some(from) = from {
            left.get_or_insert_with(|| scene.folders.folders[from].key.clone());
        }
    };
    for (index, position) in folders {
        let placement =
            scene
                .folders
                .folder_drop_placement(index, position, cube_size, &mut taken_cells);
        note(&placement, scene.folders.folders[index].parent);
        arrangement.place_folder(scene.folders.folders[index].key.clone(), placement);
        moved_folder = true;
    }
    let placements = scene
        .folders
        .drop_placements(images, cube_size, &mut taken_cells);
    for (image_id, placement) in placements {
        note(&placement, scene.folders.folder_of.get(&image_id).copied());
        arrangement.place(image_id, placement);
    }
    // Images moved from loose to loose need no relayout: they are already
    // where they were dropped.
    if moved_folder || joined.is_some() || left.is_some() {
        folder_view.relayout(joined.or(left));
    }
}

/// Springs a billboard toward where its loaded record says it belongs, at
/// the record's scale, and its [`BillboardGrowth`] toward the hover's.
/// Removed once the billboard settles, so a still scene pays nothing for it.
#[derive(Component, Default)]
pub(crate) struct BillboardMotion {
    velocity: Vec3,
    /// Of the scale the layout gives the billboard, apart from its growth.
    base_scale_velocity: f32,
    growth_velocity: f32,
    /// Seconds of animation left before the billboard starts moving.
    delay: f32,
}

/// How much larger than the size its layout gives it a billboard is drawn
/// right now: its hover growth, part way there while that animates. Its
/// scale is always its layout's times this, so the two animate apart.
#[derive(Component, Clone, Copy)]
pub(crate) struct BillboardGrowth(f32);

impl Default for BillboardGrowth {
    fn default() -> Self {
        Self(1.0)
    }
}

impl BillboardGrowth {
    pub(crate) fn factor(self) -> f32 {
        self.0
    }
}

impl BillboardMotion {
    pub(crate) fn delayed(delay: f32) -> Self {
        Self { delay, ..default() }
    }

    /// Restarts a billboard's motion toward a moved target, keeping any
    /// velocity it already has so a retarget mid-flight stays smooth.
    pub(crate) fn retarget(
        commands: &mut Commands,
        entity: Entity,
        motion: Option<Mut<Self>>,
        delay: f32,
    ) {
        match motion {
            Some(mut motion) => motion.delay = motion.delay.max(delay),
            None => {
                commands.entity(entity).insert(Self::delayed(delay));
            }
        }
    }
}

/// What the pointer is on, as a press there would take it: an image
/// (previews in a closed folder included), or else a closed folder's cube or
/// tag. Either grows a little while things grow on hover.
#[derive(Resource, Default)]
pub(crate) struct BillboardHover {
    image_id: Option<usize>,
    folder: Option<FolderKey>,
    grows: bool,
}

impl BillboardHover {
    pub(crate) fn image_id(&self) -> Option<usize> {
        self.image_id
    }

    /// How much larger than its own size the folder's cube is drawn: grown
    /// while it is under the pointer and things grow on hover.
    pub(crate) fn folder_growth(&self, key: &FolderKey) -> f32 {
        if self.grows && self.folder.as_ref() == Some(key) {
            FOLDER_HOVER_GROWTH
        } else {
            1.0
        }
    }

    /// How much larger than its own size the image is drawn: grown while it
    /// is under the pointer and images grow on hover.
    pub(crate) fn growth(&self, image_id: usize) -> f32 {
        if self.grows && self.image_id == Some(image_id) {
            HOVER_GROWTH
        } else {
            1.0
        }
    }
}

/// Moves and sizes billboards toward their layout and hover growth: each
/// springs while its animation plays ([`Animation::Rearranging`] and
/// [`Animation::HoverGrowth`]) and is simply there while it does not.
#[allow(clippy::too_many_arguments)]
pub(crate) fn animate_billboards(
    mut commands: Commands,
    real_time: Res<Time<Real>>,
    animations: Res<AnimationSettings>,
    loading: Res<ImageLoadingState>,
    hover: Res<BillboardHover>,
    facing: Res<BillboardFacingSettings>,
    cube_size: Res<BillboardWorldSize>,
    camera: Query<&Transform, (With<FlyCamera>, Without<MediaBillboard>)>,
    mut billboards: Query<(
        Entity,
        &MediaBillboard,
        &mut Transform,
        &mut BillboardMotion,
        &mut BillboardGrowth,
    )>,
) {
    let seconds = animation_step_seconds(&real_time, &animations);
    let rearranges = animations.plays(Animation::Rearranging);
    let grows_gradually = animations.plays(Animation::HoverGrowth);
    let camera = camera.get_single().ok();
    let settle_distance = cube_size.0 * 1e-3;
    for (entity, billboard, mut transform, mut motion, mut growth) in &mut billboards {
        let Some(point) = loading.loaded_point(billboard.image_id) else {
            commands.entity(entity).remove::<BillboardMotion>();
            continue;
        };
        let target_growth = hover.growth(billboard.image_id);
        let (next_growth, growth_velocity) = if grows_gradually {
            SCALE_SPRING.advance_scalar(growth.0, motion.growth_velocity, target_growth, seconds)
        } else {
            (target_growth, 0.0)
        };
        let base_scale = transform.scale.x / growth.0;
        let waiting = rearranges && motion.delay > 0.0;
        if waiting {
            motion.delay -= seconds;
        }
        let (position, velocity, base_scale, base_scale_velocity) = if waiting {
            (
                transform.translation,
                motion.velocity,
                base_scale,
                motion.base_scale_velocity,
            )
        } else if rearranges {
            let (position, velocity) = POSITION_SPRING.advance(
                transform.translation,
                motion.velocity,
                point.position,
                seconds,
            );
            let (base_scale, base_scale_velocity) = SCALE_SPRING.advance_scalar(
                base_scale,
                motion.base_scale_velocity,
                point.scale,
                seconds,
            );
            (position, velocity, base_scale, base_scale_velocity)
        } else {
            (point.position, Vec3::ZERO, point.scale, 0.0)
        };
        let settled = !waiting
            && position.distance(point.position) < settle_distance
            && velocity.length() < settle_distance
            && (base_scale - point.scale).abs() < 1e-3
            && base_scale_velocity.abs() < 1e-3
            && (next_growth - target_growth).abs() < 1e-3
            && growth_velocity.abs() < 1e-3;
        let (position, base_scale, next_growth) = if settled {
            commands.entity(entity).remove::<BillboardMotion>();
            (point.position, point.scale, target_growth)
        } else {
            motion.velocity = velocity;
            motion.base_scale_velocity = base_scale_velocity;
            motion.growth_velocity = growth_velocity;
            (position, base_scale.max(0.01), next_growth.max(0.01))
        };
        growth.0 = next_growth;
        transform.scale = Vec3::splat(base_scale * next_growth);
        if position != transform.translation {
            transform.translation = position;
            if let Some(camera) = camera {
                face_camera(&mut transform, camera, &facing);
            }
        }
    }
}

/// Re-faces a billboard the facing system would miss: it only works while
/// the camera moves, and these billboards move while the camera holds still.
fn face_camera(transform: &mut Transform, camera: &Transform, facing: &BillboardFacingSettings) {
    let to_camera = camera.translation - transform.translation;
    if to_camera.length_squared() > f32::EPSILON {
        transform.rotation = billboard_rotation(
            to_camera.normalize(),
            camera.rotation.mul_vec3(Vec3::Z),
            camera.rotation.mul_vec3(Vec3::Y),
            facing.axis,
        );
    }
}

/// Sends an image an opening folder reveals flying out of its own folder's
/// center when its texture arrives, instead of simply appearing, while
/// [`Animation::FolderSlide`] plays.
pub(crate) fn launch_revealed_images(
    mut commands: Commands,
    real_time: Res<Time<Real>>,
    animations: Res<AnimationSettings>,
    scene: Res<ExplorerScene>,
    folder_view: Res<FolderViewState>,
    mut spawned: Query<(Entity, &MediaBillboard, &mut Transform), Added<MediaBillboard>>,
) {
    if !animations.plays(Animation::FolderSlide) {
        return;
    }
    let now_seconds = real_time.elapsed_secs();
    let folders = &scene.folders;
    for (entity, billboard, mut transform) in &mut spawned {
        let Some(&index) = folders.folder_of.get(&billboard.image_id) else {
            continue;
        };
        let folder = &folders.folders[index];
        // Opening the folder or any folder around it reveals the image.
        let revealed = std::iter::successors(Some(index), |&folder| folders.folders[folder].parent)
            .any(|folder| folder_view.recently_opened(&folders.folders[folder].key, now_seconds));
        if !folder.open || !folder.visible || !revealed {
            continue;
        }
        transform.translation = folder.center;
        transform.scale = Vec3::splat(LAUNCH_SCALE);
        commands.entity(entity).insert(BillboardMotion::default());
    }
}

/// An image a closing folder hid: no longer a billboard, it flies back into
/// the folder, shrinking, and is gone when it arrives.
#[derive(Component)]
pub(crate) struct RetiringBillboard {
    target: Vec3,
    velocity: Vec3,
    start_scale: f32,
    remaining_seconds: f32,
    surface_assets: BillboardSurfaceAssets,
}

impl RetiringBillboard {
    pub(crate) fn new(
        target: Vec3,
        start_scale: f32,
        surface_assets: BillboardSurfaceAssets,
    ) -> Self {
        Self {
            target,
            velocity: Vec3::ZERO,
            start_scale,
            remaining_seconds: RETIRE_SECONDS,
            surface_assets,
        }
    }
}

/// Flies retiring images into their folders, or removes them at once
/// while [`Animation::FolderSlide`] does not play.
pub(crate) fn animate_retiring_billboards(
    mut commands: Commands,
    real_time: Res<Time<Real>>,
    animations: Res<AnimationSettings>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut retiring: Query<(Entity, &mut Transform, &mut RetiringBillboard)>,
) {
    let seconds = animation_step_seconds(&real_time, &animations);
    let closes_gradually = animations.plays(Animation::FolderSlide);
    for (entity, mut transform, mut billboard) in &mut retiring {
        billboard.remaining_seconds -= seconds;
        if !closes_gradually || billboard.remaining_seconds <= 0.0 {
            billboard.surface_assets.remove(&mut images, &mut materials);
            commands.entity(entity).despawn_recursive();
            continue;
        }
        let (position, velocity) = POSITION_SPRING.advance(
            transform.translation,
            billboard.velocity,
            billboard.target,
            seconds,
        );
        billboard.velocity = velocity;
        transform.translation = position;
        let remaining = billboard.remaining_seconds / RETIRE_SECONDS;
        transform.scale = Vec3::splat((billboard.start_scale * remaining).max(0.01));
    }
}

/// Finds what the pointer is on, the way a press resolves it, and starts
/// the images whose growth changed toward their new size (cubes follow
/// [`BillboardHover::folder_growth`] by themselves).
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_billboard_hover(
    mut commands: Commands,
    mut hover: ResMut<BillboardHover>,
    scene: Res<ExplorerScene>,
    handles: Res<FolderHandles>,
    loading: Res<ImageLoadingState>,
    billboard_controls: Res<BillboardControls>,
    pause_menu: Res<PauseMenuState>,
    ui_capture: Res<UiInputCapture>,
    selection: Res<SelectionState>,
    render_resolution: Res<RenderResolutionSettings>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<FlyCamera>>,
    billboards: Query<(&MediaBillboard, &GlobalTransform, &Visibility)>,
    motions: Query<(), With<BillboardMotion>>,
) {
    let (hovered, hovered_folder) = (|| {
        if pause_menu.paused || ui_capture.blocks_world_clicks() || selection.drag_active() {
            return None;
        }
        let window = window.get_single().ok()?;
        let (camera, camera_pose) = camera.get_single().ok()?;
        let (ray_origin, ray_direction) =
            cursor_world_ray(window, camera, camera_pose, &render_resolution)?;
        let picture_hit = nearest_billboard_hit(
            ray_origin,
            ray_direction,
            scene.billboard_world_size,
            &billboards,
        );
        // As a press: a handle in front of every picture stands in for its
        // folder, a tag for the folder itself and a close icon for nothing
        // that grows.
        let handle = handles
            .hit(ray_origin, ray_direction)
            .filter(|handle| picture_hit.is_none_or(|hit| handle.distance < hit.distance));
        let on_handle = handle.is_some();
        let folder = match handle {
            Some(handle) => (handle.kind == FolderHandleKind::Label).then_some(handle.key),
            None => scene
                .folders
                .pressed_folder(
                    ray_origin,
                    ray_direction,
                    picture_hit.map(|hit| (hit.image_id, hit.distance)),
                )
                .cloned(),
        };
        let closed_folder =
            folder.filter(|key| scene.folders.get(key).is_some_and(|folder| !folder.open));
        let on_folder = on_handle || closed_folder.is_some();
        let image_id = picture_hit.filter(|_| !on_folder).map(|hit| hit.image_id);
        Some((image_id, closed_folder))
    })()
    .unwrap_or_default();
    let grows = billboard_controls.grow_on_hover;
    if hovered_folder != hover.folder {
        hover.folder = hovered_folder;
    }
    if hovered == hover.image_id && grows == hover.grows {
        return;
    }
    for image_id in [hover.image_id, hovered].into_iter().flatten() {
        if let Some(entity) = loading.loaded_entity(image_id) {
            if motions.get(entity).is_err() {
                commands.entity(entity).insert(BillboardMotion::default());
            }
        }
    }
    hover.image_id = hovered;
    hover.grows = grows;
}

/// Re-lays out the scene when the folder state has moved past the revision
/// it was laid out at.
///
/// The relayout re-derives the current projection on a background thread
/// and arrives through the catalog load task. While a load is in flight nothing starts: the load's snapshots are
/// re-derived on arrival instead, and this catches up once the task is free.
pub(crate) fn apply_folder_changes(
    mut folder_view: ResMut<FolderViewState>,
    scene: Res<ExplorerScene>,
    mut task: ResMut<CatalogLoadTask>,
) {
    if folder_view.revision() == scene.folders.revision {
        return;
    }
    if !task.is_idle() {
        return;
    }
    let input = folder_view.layout_input(
        scene.billboard_world_size,
        &scene.folders,
        &task.arrangement(),
    );
    task.start_relayout(scene.projection.clone(), input);
}

/// The translucent cube of one folder; see [`FolderWallMaterial`].
#[derive(Component)]
pub(crate) struct FolderShell {
    key: FolderKey,
    /// The size its layout gives it; it is drawn `growth` times as large.
    size: Vec3,
    size_velocity: Vec3,
    /// Its hover growth, part way there while that animates, so hover and
    /// layout animate apart as a billboard's do ([`BillboardGrowth`]).
    growth: f32,
    growth_velocity: f32,
    velocity: Vec3,
    opacity: f32,
    opacity_level: usize,
    /// Drawn as a backdrop; see [`FolderWallMaterial`].
    open: bool,
    accent: ShellAccent,
}

/// What a folder's cube is colored for, over its own gray.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShellAccent {
    None,
    /// What a drag would drop into.
    DropTarget,
    Selected,
}

/// The folders what is being dragged would drop into, were it released now.
#[derive(Resource, Default)]
pub(crate) struct DropTargets(pub(crate) HashSet<FolderKey>);

impl FolderShell {
    pub(crate) fn key(&self) -> &FolderKey {
        &self.key
    }

    /// Puts the cube at `center` at once, for a folder the user drags: a
    /// spring would trail behind the images the drag moves directly.
    pub(crate) fn place_at(&mut self, transform: &mut Transform, center: Vec3) {
        transform.translation = center;
        self.velocity = Vec3::ZERO;
    }
}

/// Registers the folder wall material and its shader.
pub(crate) struct FolderWallPlugin;

impl Plugin for FolderWallPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "folder_wall.wgsl");
        app.add_plugins(MaterialPlugin::<FolderWallMaterial>::default());
    }
}

/// The walls of a folder's cube, drawn only from the inside: the faces
/// whose outside looks away from the viewer, so no wall stands between the
/// viewer and what the folder holds.
///
/// An open folder's walls are a backdrop: they sit at the far plane, behind
/// everything else in the scene, so no open folder the view looks through or
/// stands in ever hides anything. A closed folder's walls are an ordinary
/// see-through box.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
#[bind_group_data(FolderWallKey)]
pub(crate) struct FolderWallMaterial {
    #[uniform(0)]
    color: LinearRgba,
    backdrop: bool,
    /// Added to the view depth the scene's blended meshes are sorted by.
    sort_bias: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FolderWallKey {
    backdrop: bool,
}

impl From<&FolderWallMaterial> for FolderWallKey {
    fn from(material: &FolderWallMaterial) -> Self {
        Self {
            backdrop: material.backdrop,
        }
    }
}

impl Material for FolderWallMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://spatial_viewer/folder_wall.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn depth_bias(&self) -> f32 {
        self.sort_bias
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = Some(Face::Front);
        if key.bind_group_data.backdrop {
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("BACKDROP".into());
            }
        }
        Ok(())
    }
}

/// Blended meshes draw back to front by their origin alone, which the
/// images inside a folder share with its cube: this much sort bias, in
/// world units, draws a backdrop before everything else blended.
const BACKDROP_SORT_BIAS: f32 = -1.0e5;

/// One cube mesh and a ladder of materials every folder shell shares.
#[derive(Resource)]
pub(crate) struct FolderShellAssets {
    mesh: Handle<Mesh>,
    /// By opacity level, for closed folders and for open ones.
    closed: Vec<Handle<FolderWallMaterial>>,
    open: Vec<Handle<FolderWallMaterial>>,
    /// A selected folder's cube, closed and open.
    selected_closed: Handle<FolderWallMaterial>,
    selected_open: Handle<FolderWallMaterial>,
    /// The cube of what a drag would drop into, closed and open.
    target_closed: Handle<FolderWallMaterial>,
    target_open: Handle<FolderWallMaterial>,
}

impl FolderShellAssets {
    pub(crate) fn new(
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<FolderWallMaterial>,
        cube_size: f32,
    ) -> Self {
        let mesh = meshes.add(Cuboid::from_length(1.0));
        let [red, green, blue] = FOLDER_SHELL_COLOR;
        let mut material = |color: Color, backdrop: bool| {
            materials.add(FolderWallMaterial {
                color: color.to_linear(),
                backdrop,
                // A closed cube draws its walls just before the previews
                // inside it.
                sort_bias: if backdrop {
                    BACKDROP_SORT_BIAS
                } else {
                    -cube_size
                },
            })
        };
        let mut ladder = |backdrop: bool| -> Vec<_> {
            (0..FOLDER_OPACITY_LEVELS)
                .map(|level| {
                    material(
                        Color::srgba(red, green, blue, opacity_of_level(level)),
                        backdrop,
                    )
                })
                .collect()
        };
        let closed = ladder(false);
        let open = ladder(true);
        Self {
            mesh,
            closed,
            open,
            selected_closed: material(SELECTED_FOLDER_COLOR, false),
            selected_open: material(SELECTED_FOLDER_COLOR, true),
            target_closed: material(DROP_TARGET_FOLDER_COLOR, false),
            target_open: material(DROP_TARGET_FOLDER_COLOR, true),
        }
    }

    fn material(
        &self,
        level: usize,
        open: bool,
        accent: ShellAccent,
    ) -> Handle<FolderWallMaterial> {
        match (accent, open) {
            (ShellAccent::Selected, true) => self.selected_open.clone(),
            (ShellAccent::Selected, false) => self.selected_closed.clone(),
            (ShellAccent::DropTarget, true) => self.target_open.clone(),
            (ShellAccent::DropTarget, false) => self.target_closed.clone(),
            (ShellAccent::None, true) => self.open[level].clone(),
            (ShellAccent::None, false) => self.closed[level].clone(),
        }
    }
}

fn opacity_of_level(level: usize) -> f32 {
    let fraction = level as f32 / (FOLDER_OPACITY_LEVELS - 1) as f32;
    OPEN_FOLDER_OPACITY + (CLOSED_FOLDER_OPACITY - OPEN_FOLDER_OPACITY) * fraction
}

fn level_of_opacity(opacity: f32) -> usize {
    let fraction = (opacity - OPEN_FOLDER_OPACITY) / (CLOSED_FOLDER_OPACITY - OPEN_FOLDER_OPACITY);
    (fraction.clamp(0.0, 1.0) * (FOLDER_OPACITY_LEVELS - 1) as f32).round() as usize
}

fn folder_opacity(folder: &Folder) -> f32 {
    if folder.open {
        OPEN_FOLDER_OPACITY
    } else {
        CLOSED_FOLDER_OPACITY
    }
}

/// Spawns a shell for every folder the scene gained and despawns the shells
/// of folders it lost, whenever the scene was laid out anew.
pub(crate) fn sync_folder_shells(
    mut commands: Commands,
    scene: Res<ExplorerScene>,
    assets: Res<FolderShellAssets>,
    shells: Query<(Entity, &FolderShell)>,
    mut synced_generation: Local<Option<u64>>,
) {
    if *synced_generation == Some(scene.folder_generation) {
        return;
    }
    *synced_generation = Some(scene.folder_generation);
    let mut existing = HashSet::new();
    for (entity, shell) in &shells {
        if scene.folders.get(&shell.key).is_some() {
            existing.insert(shell.key.clone());
        } else {
            commands.entity(entity).despawn_recursive();
        }
    }
    for folder in &scene.folders.folders {
        if existing.contains(&folder.key) {
            continue;
        }
        let opacity = folder_opacity(folder);
        let level = level_of_opacity(opacity);
        commands.spawn((
            Mesh3d(assets.mesh.clone()),
            MeshMaterial3d(assets.material(level, folder.open, ShellAccent::None)),
            Transform::from_translation(folder.center).with_scale(folder.size),
            FolderShell {
                key: folder.key.clone(),
                size: folder.size,
                size_velocity: Vec3::ZERO,
                growth: 1.0,
                growth_velocity: 0.0,
                velocity: Vec3::ZERO,
                opacity,
                opacity_level: level,
                open: folder.open,
                accent: ShellAccent::None,
            },
            Name::new("folder"),
        ));
    }
}

/// Springs every shell toward its folder's place, size and opacity, or puts
/// it there while [`Animation::FolderCubes`] does not play, and toward its
/// hover growth while [`Animation::HoverGrowth`] plays. Without backgrounds
/// a shell shows only while it is accented (selected, or a drop target);
/// it springs all the same, so it shows in place when it is.
#[allow(clippy::too_many_arguments)]
pub(crate) fn animate_folder_shells(
    real_time: Res<Time<Real>>,
    animations: Res<AnimationSettings>,
    hover: Res<BillboardHover>,
    scene: Res<ExplorerScene>,
    selection: Res<SelectionState>,
    drop_targets: Res<DropTargets>,
    folder_controls: Res<FolderControls>,
    assets: Res<FolderShellAssets>,
    mut shells: Query<(
        &mut Transform,
        &mut FolderShell,
        &mut MeshMaterial3d<FolderWallMaterial>,
        &mut Visibility,
    )>,
) {
    let seconds = animation_step_seconds(&real_time, &animations);
    let animates = animations.plays(Animation::FolderCubes);
    let grows_gradually = animations.plays(Animation::HoverGrowth);
    let settle_distance = scene.billboard_world_size * 1e-3;
    for (mut transform, mut shell, mut material, mut visibility) in &mut shells {
        let Some(folder) = scene.folders.get(&shell.key) else {
            continue;
        };
        let target_growth = hover.folder_growth(&shell.key);
        let target_opacity = folder_opacity(folder);
        let accent = if selection.is_folder_selected(&shell.key) {
            ShellAccent::Selected
        } else if drop_targets.0.contains(&shell.key) {
            ShellAccent::DropTarget
        } else {
            ShellAccent::None
        };
        visibility.set_if_neq(
            if folder_controls.display.backgrounds || accent != ShellAccent::None {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            },
        );
        let at_rest = transform.translation == folder.center
            && shell.size == folder.size
            && shell.growth == target_growth
            && shell.opacity == target_opacity
            && shell.accent == accent
            && shell.open == folder.open;
        if at_rest {
            continue;
        }
        let ((position, velocity), (size, size_velocity)) = if animates {
            (
                SHELL_SPRING.advance(
                    transform.translation,
                    shell.velocity,
                    folder.center,
                    seconds,
                ),
                SHELL_SPRING.advance(shell.size, shell.size_velocity, folder.size, seconds),
            )
        } else {
            ((folder.center, Vec3::ZERO), (folder.size, Vec3::ZERO))
        };
        let settled = position.distance(folder.center) < settle_distance
            && velocity.length() < settle_distance
            && size.distance(folder.size) < settle_distance
            && size_velocity.length() < settle_distance;
        (shell.velocity, shell.size_velocity) = if settled {
            (Vec3::ZERO, Vec3::ZERO)
        } else {
            (velocity, size_velocity)
        };
        shell.size = if settled {
            folder.size
        } else {
            size.max(Vec3::splat(settle_distance))
        };
        transform.translation = if settled { folder.center } else { position };
        let (growth, growth_velocity) = if grows_gradually {
            SCALE_SPRING.advance_scalar(shell.growth, shell.growth_velocity, target_growth, seconds)
        } else {
            (target_growth, 0.0)
        };
        (shell.growth, shell.growth_velocity) =
            if (growth - target_growth).abs() < 1e-3 && growth_velocity.abs() < 1e-3 {
                (target_growth, 0.0)
            } else {
                (growth, growth_velocity)
            };
        transform.scale = shell.size * shell.growth;

        let approach = if animates {
            1.0 - (-FOLDER_OPACITY_RATE * seconds).exp()
        } else {
            1.0
        };
        shell.opacity += (target_opacity - shell.opacity) * approach;
        if (shell.opacity - target_opacity).abs() < 2e-3 {
            shell.opacity = target_opacity;
        }
        let level = level_of_opacity(shell.opacity);
        if level != shell.opacity_level || accent != shell.accent || folder.open != shell.open {
            shell.opacity_level = level;
            shell.accent = accent;
            shell.open = folder.open;
            material.0 = assets.material(level, folder.open, accent);
        }
    }
}

/// Placeholders held back while an opening folder's cube grows around
/// them, with the seconds each still waits. The point cloud is rebuilt
/// with every layout, so the ones still waiting are held back again then.
#[derive(Resource, Default)]
pub(crate) struct WithheldPlaceholders {
    waiting: HashMap<usize, f32>,
}

/// Holds back the placeholders of the images each newly opened folder
/// reveals (opened itself, or through a folder around it), whenever the
/// scene was laid out anew: each appears once the cube has grown, in a
/// ripple outward from its center. Only while
/// [`Animation::FolderSlide`] plays.
pub(crate) fn withhold_revealed_placeholders(
    animations: Res<AnimationSettings>,
    scene: Res<ExplorerScene>,
    mut cloud: ResMut<PointCloud>,
    mut withheld: ResMut<WithheldPlaceholders>,
    mut synced_generation: Local<Option<u64>>,
    mut was_showing: Local<HashMap<FolderKey, bool>>,
) {
    if *synced_generation == Some(scene.folder_generation) {
        return;
    }
    *synced_generation = Some(scene.folder_generation);
    let cube_size = scene.billboard_world_size;
    let ripples = animations.plays(Animation::FolderSlide);
    for point in scene.image_points.iter().filter(|_| ripples) {
        let Some(folder) = scene.folders.folder_of(point.image_id) else {
            continue;
        };
        // A folder the previous layout did not have is new, not opening.
        if !shows_images(folder) || was_showing.get(&folder.key) != Some(&false) {
            continue;
        }
        let wait =
            PLACEHOLDER_REVEAL_SECONDS + stagger_from(folder.center, point.position, cube_size);
        withheld.waiting.insert(point.image_id, wait);
    }
    *was_showing = scene
        .folders
        .folders
        .iter()
        .map(|folder| (folder.key.clone(), shows_images(folder)))
        .collect();
    for &image_id in withheld.waiting.keys() {
        cloud.set_point_withheld(image_id, true);
    }
}

/// A folder shows its images at full size: it is open, and so is every
/// folder around it.
fn shows_images(folder: &Folder) -> bool {
    folder.visible && folder.open
}

/// Lets each withheld placeholder appear once its wait is over, and all of
/// them once [`Animation::FolderSlide`] stops playing.
pub(crate) fn release_withheld_placeholders(
    real_time: Res<Time<Real>>,
    animations: Res<AnimationSettings>,
    mut cloud: ResMut<PointCloud>,
    mut withheld: ResMut<WithheldPlaceholders>,
) {
    if withheld.waiting.is_empty() {
        return;
    }
    let seconds = if animations.plays(Animation::FolderSlide) {
        animations.animation_seconds(real_time.delta_secs())
    } else {
        f32::INFINITY
    };
    withheld.waiting.retain(|&image_id, wait| {
        *wait -= seconds;
        let waiting = *wait > 0.0;
        if !waiting {
            cloud.set_point_withheld(image_id, false);
        }
        waiting
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatial_api::{PointGroup, ProjectionPoint};

    const CUBE: f32 = 2.0;

    fn point(image_id: usize, key: &str, index: [u32; 3], offset: [f32; 3]) -> ProjectionPoint {
        ProjectionPoint {
            image_id,
            path: format!("{image_id}.png"),
            position: [0.0; 3],
            group: Some(PointGroup {
                key: key.to_owned(),
                index,
                offset,
            }),
            width: None,
            height: None,
            media_type: "image".to_owned(),
            duration_seconds: None,
            coordinate_labels: [None, None, None],
        }
    }

    /// A folder of five in slot 0 (a 3x2 block), then a single image in
    /// slot 1.
    fn projection() -> ProjectionPage {
        let mut points: Vec<ProjectionPoint> = [
            [-1.0, -0.5, 0.0],
            [0.0, -0.5, 0.0],
            [1.0, -0.5, 0.0],
            [-1.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(image_id, offset)| point(image_id, "folder", [0, 0, 0], offset))
        .collect();
        points.push(point(9, "single", [1, 0, 0], [0.0; 3]));
        ProjectionPage {
            points,
            ..default()
        }
    }

    fn input(open: &[&str]) -> FolderLayoutInput {
        FolderLayoutInput {
            cube_size: CUBE,
            open: Arc::new(open.iter().map(|&name| key(name)).collect()),
            origin: Vec3::ZERO,
            anchor: None,
            revision: 0,
            arrangement: ArrangementSnapshot::default(),
        }
    }

    fn key(name: &str) -> FolderKey {
        FolderKey::group(name)
    }

    fn arranged(open: &[&str], arrangement: &ManualArrangement) -> FolderLayoutInput {
        FolderLayoutInput {
            arrangement: arrangement.snapshot(),
            ..input(open)
        }
    }

    fn position_of(laid_out: &LaidOutProjection, image_id: usize) -> Option<Vec3> {
        laid_out
            .points
            .iter()
            .find(|point| point.image_id == image_id)
            .map(|point| point.position)
    }

    #[test]
    fn a_closed_folder_takes_one_cube_and_shows_only_its_previews() {
        let laid_out = lay_out_points(&projection(), &input(&[]));
        let folder = laid_out.folders.get(&key("folder")).unwrap();
        assert!(!folder.open);
        assert_eq!(folder.previews.len(), MAX_FOLDER_PREVIEWS);
        assert_eq!(laid_out.points.len(), MAX_FOLDER_PREVIEWS + 1);
        for point in &laid_out.points {
            if point.image_id == 9 {
                assert_eq!(point.position, Vec3::new(CUBE, 0.0, 0.0));
                assert_eq!(point.scale, 1.0);
            } else {
                assert!(folder.previews.contains(&point.image_id));
                assert_eq!(point.scale, FOLDER_PREVIEW_SCALE);
                assert!(point.position.distance(folder.center) < CUBE * 0.5);
            }
        }
        // Only a folder, not a single image, is a folder.
        assert!(laid_out.folders.get(&key("single")).is_none());
    }

    #[test]
    fn an_open_folder_shows_its_block_and_pushes_neighbours_aside() {
        let laid_out = lay_out_points(&projection(), &input(&["folder"]));
        assert_eq!(laid_out.points.len(), 6);
        assert_eq!(
            position_of(&laid_out, 2),
            Some(Vec3::new(CUBE, -0.5 * CUBE, 0.0))
        );
        // The block is 3 wide plus padding on both sides; the next slot
        // starts at its edge.
        let slot_width = 3.0 + 2.0 * OPEN_FOLDER_PADDING;
        let single = position_of(&laid_out, 9).unwrap();
        assert_eq!(single.x, (slot_width * 0.5 + 0.5) * CUBE);
        let folder = laid_out.folders.get(&key("folder")).unwrap();
        assert_eq!(folder.size.x, slot_width * CUBE);
    }

    #[test]
    fn toggling_keeps_the_toggled_folder_in_place() {
        let mut page = projection();
        page.points.extend([
            point(20, "far", [2, 0, 0], [-0.5, 0.0, 0.0]),
            point(21, "far", [2, 0, 0], [0.5, 0.0, 0.0]),
        ]);
        let closed = lay_out_points(&page, &input(&[]));
        let far_center = closed.folders.get(&key("far")).unwrap().center;
        let opened = lay_out_points(
            &page,
            &FolderLayoutInput {
                anchor: Some((key("far"), far_center)),
                ..input(&["far"])
            },
        );
        assert_eq!(opened.folders.get(&key("far")).unwrap().center, far_center);
        // The open block is 2 wide plus padding; the folder before it moves
        // aside by the half of that growth on its side.
        let growth = 2.0 + 2.0 * OPEN_FOLDER_PADDING - 1.0;
        let before =
            |laid_out: &LaidOutProjection| laid_out.folders.get(&key("folder")).unwrap().center;
        assert_eq!(before(&opened).x, before(&closed).x - growth * 0.5 * CUBE);
    }

    #[test]
    fn springs_settle_at_the_fastest_animation_scale() {
        // A quarter of the usual duration advances four times as far per
        // frame, past the step a spring stays stable at.
        let frame_seconds = MAX_SPRING_STEP_SECONDS * 4.0;
        for spring in [POSITION_SPRING, SCALE_SPRING, SHELL_SPRING] {
            let (mut value, mut velocity) = (Vec3::ZERO, Vec3::ZERO);
            for _ in 0..60 {
                (value, velocity) = spring.advance(value, velocity, Vec3::ONE, frame_seconds);
            }
            assert!((value - Vec3::ONE).length() < 1e-3, "{value}");
            assert!(velocity.length() < 1e-3, "{velocity}");
        }
    }

    #[test]
    fn previews_are_the_same_every_time_and_differ_between_folders() {
        let ids: Vec<usize> = (0..40).collect();
        assert_eq!(
            choose_previews(&key("a"), ids.clone()),
            choose_previews(&key("a"), ids.clone())
        );
        assert_ne!(
            choose_previews(&key("a"), ids.clone()),
            choose_previews(&key("b"), ids.clone())
        );
        let mut shuffled = ids.clone();
        shuffled.reverse();
        assert_eq!(
            choose_previews(&key("a"), shuffled),
            choose_previews(&key("a"), ids)
        );
    }

    #[test]
    fn previews_center_on_the_folder() {
        for count in 2..=MAX_FOLDER_PREVIEWS {
            let sum: Vec3 = (0..count).map(|slot| preview_offset(slot, count)).sum();
            assert!(sum.length() < 1e-5);
        }
    }

    #[test]
    fn a_dragged_image_leaves_its_closed_folder() {
        let mut arrangement = ManualArrangement::default();
        arrangement.place(0, ManualPlacement::Loose(Vec3::splat(50.0)));
        arrangement.place(1, ManualPlacement::Loose(Vec3::splat(60.0)));
        let laid_out = lay_out_points(&projection(), &arranged(&[], &arrangement));
        assert_eq!(position_of(&laid_out, 0), Some(Vec3::splat(50.0)));
        assert_eq!(position_of(&laid_out, 1), Some(Vec3::splat(60.0)));
        let folder = laid_out.folders.get(&key("folder")).unwrap();
        assert_eq!(folder.members.len(), 3);
        assert!(laid_out.folders.folder_of(0).is_none());
    }

    #[test]
    fn an_image_dropped_into_a_folder_joins_it_and_an_emptied_folder_stays() {
        let mut page = projection();
        page.points.extend([
            point(20, "pair", [2, 0, 0], [-0.5, 0.0, 0.0]),
            point(21, "pair", [2, 0, 0], [0.5, 0.0, 0.0]),
        ]);
        let mut arrangement = ManualArrangement::default();
        arrangement.place(20, ManualPlacement::Loose(Vec3::splat(80.0)));
        arrangement.place(21, ManualPlacement::Loose(Vec3::splat(90.0)));
        let open = lay_out_points(&page, &arranged(&["folder"], &arrangement));
        // Emptied, the pair is still a folder to drop images into.
        let pair = open.folders.get(&key("pair")).unwrap();
        assert!(pair.members.is_empty());

        // Dropped inside the open folder over image 4's cell, an image
        // takes the free cell nearest it; dropped on the closed pair, it
        // takes the pair's middle cell.
        let folder = open.folders.get(&key("folder")).unwrap();
        let inside = folder.center + Vec3::new(0.4, 0.1, 0.0) * CUBE;
        let placements = open.folders.drop_placements(
            [(20, inside), (21, pair.center)],
            CUBE,
            &mut TakenCells::new(),
        );
        assert_eq!(
            placements,
            vec![
                (
                    20,
                    ManualPlacement::InFolder {
                        folder: key("folder"),
                        offset: Vec3::new(1.0, 1.0, 0.0),
                    },
                ),
                (
                    21,
                    ManualPlacement::InFolder {
                        folder: key("pair"),
                        offset: Vec3::ZERO,
                    },
                ),
            ]
        );
        for (image_id, placement) in placements {
            arrangement.place(image_id, placement);
        }
        let laid_out = lay_out_points(&page, &arranged(&["folder"], &arrangement));
        assert_eq!(
            laid_out.folders.folder_of(20).map(|folder| &folder.key),
            Some(&key("folder"))
        );
        // The free corner of the folder's 3x2 block.
        let folder = laid_out.folders.get(&key("folder")).unwrap();
        assert_eq!(
            position_of(&laid_out, 20),
            Some(folder.center + Vec3::new(1.0, 0.5, 0.0) * CUBE)
        );
        assert_eq!(
            laid_out.folders.get(&key("pair")).unwrap().previews,
            vec![21]
        );
    }

    #[test]
    fn things_dropped_together_on_a_closed_folder_take_separate_cells() {
        let mut arrangement = ManualArrangement::default();
        let loose = arrangement.make_folder(ManualPlacement::Loose(Vec3::splat(60.0)));
        let laid_out = lay_out_points(&projection(), &arranged(&[], &arrangement));
        let folders = &laid_out.folders;
        let target = folders.get(&key("folder")).unwrap().center;
        let mut taken = TakenCells::new();
        let folder = folders.folder_drop_placement(
            folders.index_of(&loose).unwrap(),
            target,
            CUBE,
            &mut taken,
        );
        let image = folders.drop_placements([(9, target)], CUBE, &mut taken);
        let offset_of = |placement: &ManualPlacement| match placement {
            ManualPlacement::InFolder { offset, .. } => *offset,
            ManualPlacement::Loose(_) => panic!("dropped on a folder"),
        };
        assert_ne!(offset_of(&folder), offset_of(&image[0].1));
    }

    #[test]
    fn a_drop_outside_every_folder_is_loose() {
        let laid_out = lay_out_points(&projection(), &input(&[]));
        let far = Vec3::splat(100.0);
        assert_eq!(
            laid_out
                .folders
                .drop_placements([(0, far)], CUBE, &mut TakenCells::new()),
            vec![(0, ManualPlacement::Loose(far))]
        );
    }

    #[test]
    fn a_folder_fills_from_where_things_are_put_in_its_own_plane() {
        let taken = HashSet::from([IVec3::ZERO]);
        let row = BVec3::new(true, false, false);
        assert_eq!(free_cell(&taken, IVec3::ZERO, row), IVec3::new(-1, 0, 0));
        let taken = HashSet::from([IVec3::ZERO, IVec3::X, IVec3::NEG_X]);
        assert_eq!(free_cell(&taken, IVec3::ZERO, row), IVec3::new(-2, 0, 0));
        let plane = BVec3::new(true, true, false);
        assert_eq!(free_cell(&taken, IVec3::ZERO, plane), IVec3::new(0, -1, 0));
        assert_eq!(
            free_cell(&taken, IVec3::new(5, 5, 0), plane),
            IVec3::new(5, 5, 0)
        );
    }

    #[test]
    fn a_folder_the_user_made_stands_where_they_made_it() {
        let mut arrangement = ManualArrangement::default();
        let center = Vec3::new(30.0, 0.0, 0.0);
        let made = arrangement.make_folder(ManualPlacement::Loose(center));
        let empty = arrangement.make_folder(ManualPlacement::Loose(Vec3::new(-30.0, 0.0, 0.0)));
        arrangement.place(
            9,
            ManualPlacement::InFolder {
                folder: made.clone(),
                offset: Vec3::new(1.0, 0.0, 0.0),
            },
        );
        let closed = lay_out_points(&projection(), &arranged(&[], &arrangement));
        let folder = closed.folders.get(&made).unwrap();
        assert_eq!(folder.center, center);
        assert_eq!(folder.previews, vec![9]);
        assert_eq!(position_of(&closed, 9), Some(center));
        assert!(closed.folders.get(&empty).unwrap().members.is_empty());

        let open = lay_out_points(
            &projection(),
            &FolderLayoutInput {
                open: Arc::new(HashSet::from([made.clone()])),
                ..arranged(&[], &arrangement)
            },
        );
        // Alone in it, the image stands in the middle.
        assert_eq!(position_of(&open, 9), Some(center));
        // Opening it moves nothing of the server's.
        assert_eq!(
            open.folders.get(&key("folder")).unwrap().center,
            closed.folders.get(&key("folder")).unwrap().center
        );
    }

    #[test]
    fn pressing_a_closed_folder_opens_it_and_an_open_one_takes_no_press() {
        let closed = lay_out_points(&projection(), &input(&[]));
        let folder = closed.folders.get(&key("folder")).unwrap();
        let origin = folder.center + Vec3::new(0.0, 0.0, 10.0 * CUBE);
        let toward = Vec3::NEG_Z;
        assert_eq!(
            closed.folders.pressed_folder(origin, toward, None).cloned(),
            Some(key("folder"))
        );
        // A picture in front of the folder takes the press.
        assert_eq!(
            closed
                .folders
                .pressed_folder(origin, toward, Some((9, 1.0))),
            None
        );
        // So does one of its previews, inside it, though its cube is
        // entered first; any other picture that far is behind the cube.
        let (enter, exit) = ray_box_hit(origin, toward, folder.center, folder.size * 0.5).unwrap();
        let inside = (enter + exit) * 0.5;
        let preview = folder.previews[0];
        assert_eq!(
            closed
                .folders
                .pressed_folder(origin, toward, Some((preview, inside))),
            None
        );
        assert_eq!(
            closed
                .folders
                .pressed_folder(origin, toward, Some((9, inside)))
                .cloned(),
            Some(key("folder"))
        );

        let open = lay_out_points(&projection(), &input(&["folder"]));
        let folder = open.folders.get(&key("folder")).unwrap();
        let origin = folder.center + Vec3::new(0.0, 0.0, 10.0 * CUBE);
        let (enter, exit) = ray_box_hit(origin, toward, folder.center, folder.size * 0.5).unwrap();
        let inside = (enter + exit) * 0.5;
        assert_eq!(
            open.folders
                .pressed_folder(origin, toward, Some((4, inside))),
            None
        );
        assert_eq!(open.folders.pressed_folder(origin, toward, None), None);
    }

    #[test]
    fn whatever_an_open_folder_lets_through_takes_the_press() {
        let mut page = projection();
        page.points.extend([
            point(20, "behind", [0, 0, 1], [-0.5, 0.0, 0.0]),
            point(21, "behind", [0, 0, 1], [0.5, 0.0, 0.0]),
        ]);
        let laid_out = lay_out_points(&page, &input(&["folder"]));
        let folders = &laid_out.folders;
        let open = folders.get(&key("folder")).unwrap();
        let behind = folders.get(&key("behind")).unwrap();
        // Looking down the Z axis through the open folder at the closed one.
        let origin = Vec3::new(
            behind.center.x,
            behind.center.y,
            open.center.z - 10.0 * CUBE,
        );
        let toward = Vec3::Z;
        assert_eq!(
            folders.pressed_folder(origin, toward, None).cloned(),
            Some(key("behind"))
        );
        // A picture far behind the open folder takes the press too.
        let alone = lay_out_points(&projection(), &input(&["folder"]));
        let origin = alone.folders.get(&key("folder")).unwrap().center - 10.0 * CUBE * toward;
        assert_eq!(
            alone
                .folders
                .pressed_folder(origin, toward, Some((30, 1e4))),
            None
        );
    }

    #[test]
    fn a_nested_folder_shows_only_while_its_parent_is_open() {
        let mut arrangement = ManualArrangement::default();
        let nested = arrangement.make_folder(ManualPlacement::InFolder {
            folder: key("folder"),
            offset: Vec3::new(2.0, 0.0, 0.0),
        });
        arrangement.place(
            9,
            ManualPlacement::InFolder {
                folder: nested.clone(),
                offset: Vec3::ZERO,
            },
        );

        let closed = lay_out_points(&projection(), &arranged(&[], &arrangement));
        let parent = closed.folders.get(&key("folder")).unwrap();
        let hidden = closed.folders.get(&nested).unwrap();
        assert!(!hidden.visible);
        assert_eq!(hidden.size, Vec3::ZERO);
        assert_eq!(hidden.center, parent.center);
        assert_eq!(position_of(&closed, 9), None);

        let open = lay_out_points(&projection(), &arranged(&["folder"], &arrangement));
        let parent = open.folders.get(&key("folder")).unwrap();
        let shown = open.folders.get(&nested).unwrap();
        assert!(shown.visible);
        // Its cell starts a fourth column after the parent's 3x2 block,
        // which the parent recenters on.
        assert_eq!(shown.center, parent.home + Vec3::new(1.5, -0.5, 0.0) * CUBE);
        assert_eq!(position_of(&open, 9), Some(shown.center));
        assert_eq!(parent.size.x, (4.0 + 2.0 * OPEN_FOLDER_PADDING) * CUBE);

        // Open, the nested folder widens its column and row, pushing the
        // parent's images aside instead of covering them.
        let open_input = FolderLayoutInput {
            open: Arc::new(HashSet::from([key("folder"), nested.clone()])),
            ..arranged(&[], &arrangement)
        };
        let both_open = lay_out_points(&projection(), &open_input);
        let parent = both_open.folders.get(&key("folder")).unwrap();
        let shown = both_open.folders.get(&nested).unwrap();
        let nested_extent = 1.0 + 2.0 * OPEN_FOLDER_PADDING;
        assert_eq!(
            parent.size.x,
            (3.0 + nested_extent + 2.0 * OPEN_FOLDER_PADDING) * CUBE
        );
        for image_id in 0..5 {
            let position = position_of(&both_open, image_id).unwrap();
            let inside = (position - shown.center)
                .abs()
                .cmplt(shown.size * 0.5)
                .all();
            assert!(!inside, "image {image_id} stands inside the nested folder");
        }
    }

    #[test]
    fn opening_a_nested_folder_keeps_it_in_place() {
        let mut arrangement = ManualArrangement::default();
        let nested = arrangement.make_folder(ManualPlacement::InFolder {
            folder: key("folder"),
            offset: Vec3::new(2.0, 0.0, 0.0),
        });
        let closed = lay_out_points(&projection(), &arranged(&["folder"], &arrangement));
        let before = closed.folders.get(&nested).unwrap().home;
        let opened = lay_out_points(
            &projection(),
            &FolderLayoutInput {
                open: Arc::new(HashSet::from([key("folder"), nested.clone()])),
                anchor: Some((nested.clone(), before)),
                ..arranged(&[], &arrangement)
            },
        );
        assert_eq!(opened.folders.get(&nested).unwrap().home, before);
    }

    #[test]
    fn a_closed_folder_nested_in_an_open_one_takes_a_press_through_it() {
        let mut arrangement = ManualArrangement::default();
        let nested = arrangement.make_folder(ManualPlacement::InFolder {
            folder: key("folder"),
            offset: Vec3::new(2.0, 0.0, 0.0),
        });
        let laid_out = lay_out_points(&projection(), &arranged(&["folder"], &arrangement));
        let folders = &laid_out.folders;
        let target = folders.get(&nested).unwrap().center;
        let origin = target - Vec3::Z * 10.0 * CUBE;
        assert_eq!(
            folders.pressed_folder(origin, Vec3::Z, None).cloned(),
            Some(nested)
        );
    }

    #[test]
    fn a_folder_is_named_for_what_everything_in_it_shares() {
        let mut page = projection();
        for point in &mut page.points {
            point.coordinate_labels = [Some("2024".to_owned()), None, None];
        }
        let laid_out = lay_out_points(&page, &input(&[]));
        let folder = laid_out.folders.get(&key("folder")).unwrap();
        assert_eq!(folder.image_count, 5);
        assert_eq!(folder.name(), "2024");
        let mut arrangement = ManualArrangement::default();
        arrangement.rename_folder(&key("folder"), "  Holidays ");
        let renamed = lay_out_points(&page, &arranged(&[], &arrangement));
        assert_eq!(
            renamed.folders.get(&key("folder")).unwrap().name(),
            "Holidays"
        );
        arrangement.rename_folder(&key("folder"), "");
        let restored = lay_out_points(&page, &arranged(&[], &arrangement));
        assert_eq!(restored.folders.get(&key("folder")).unwrap().name(), "2024");
    }

    #[test]
    fn a_folder_keeps_its_own_tag_setting_until_it_is_let_go() {
        let mut arrangement = ManualArrangement::default();
        arrangement.set_tag_shown(&key("folder"), Some(false));
        let hidden = lay_out_points(&projection(), &arranged(&[], &arrangement));
        assert_eq!(
            hidden.folders.get(&key("folder")).unwrap().tag_override,
            Some(false)
        );
        arrangement.set_tag_shown(&key("folder"), None);
        let following = lay_out_points(&projection(), &arranged(&[], &arrangement));
        assert_eq!(
            following.folders.get(&key("folder")).unwrap().tag_override,
            None
        );
    }

    #[test]
    fn a_folder_never_drops_into_itself_and_a_loop_stands_on_its_own() {
        let mut arrangement = ManualArrangement::default();
        let outer = arrangement.make_folder(ManualPlacement::Loose(Vec3::splat(40.0)));
        let inner = arrangement.make_folder(ManualPlacement::InFolder {
            folder: outer.clone(),
            offset: Vec3::ZERO,
        });
        let open: Vec<FolderKey> = vec![outer.clone(), inner.clone()];
        let laid_out = lay_out_points(
            &projection(),
            &FolderLayoutInput {
                open: Arc::new(open.iter().cloned().collect()),
                ..arranged(&[], &arrangement)
            },
        );
        let folders = &laid_out.folders;
        let outer_index = folders.index_of(&outer).unwrap();
        let inner_center = folders.get(&inner).unwrap().center;
        // Dropped on its own child, the outer folder stays out of it.
        assert_eq!(
            folders.folder_drop_placement(outer_index, inner_center, CUBE, &mut TakenCells::new()),
            ManualPlacement::Loose(inner_center)
        );

        // A loop the arrangement somehow holds breaks rather than hangs.
        arrangement.place_folder(
            outer.clone(),
            ManualPlacement::InFolder {
                folder: inner.clone(),
                offset: Vec3::ZERO,
            },
        );
        let looped = lay_out_points(&projection(), &arranged(&[], &arrangement));
        assert!(looped.folders.get(&outer).is_some());
        assert!(looped.folders.get(&inner).is_some());
    }

    #[test]
    fn deleting_a_folder_moves_what_it_held_up_a_level() {
        let mut arrangement = ManualArrangement::default();
        let outer = arrangement.make_folder(ManualPlacement::Loose(Vec3::splat(40.0)));
        let inner = arrangement.make_folder(ManualPlacement::InFolder {
            folder: outer.clone(),
            offset: Vec3::new(1.0, 0.0, 0.0),
        });
        arrangement.place(
            9,
            ManualPlacement::InFolder {
                folder: inner.clone(),
                offset: Vec3::new(0.0, 1.0, 0.0),
            },
        );
        let laid_out = lay_out_points(&projection(), &arranged(&[], &arrangement));
        let folders = &laid_out.folders;
        delete_folders_from(
            folders,
            &[folders.index_of(&inner).unwrap()],
            CUBE,
            &mut arrangement,
        );
        assert_eq!(
            arrangement.placement(9),
            // The cell the deleted folder held.
            Some(&ManualPlacement::InFolder {
                folder: outer.clone(),
                offset: Vec3::new(1.0, 0.0, 0.0),
            })
        );
        let relaid = lay_out_points(&projection(), &arranged(&[], &arrangement));
        assert!(relaid.folders.get(&inner).is_none());
        assert_eq!(
            relaid.folders.folder_of(9).map(|folder| &folder.key),
            Some(&outer)
        );

        // A server folder in its slot leaves its own images to the slot.
        delete_folders_from(
            &relaid.folders,
            &[relaid.folders.index_of(&key("folder")).unwrap()],
            CUBE,
            &mut arrangement,
        );
        let dissolved = lay_out_points(&projection(), &arranged(&[], &arrangement));
        assert!(dissolved.folders.get(&key("folder")).is_none());
        assert_eq!(
            position_of(&dissolved, 0),
            Some(Vec3::new(-CUBE, -0.5 * CUBE, 0.0))
        );
        assert_eq!(dissolved.points.len(), 6);
    }

    #[test]
    fn deleting_a_folder_with_its_nested_folder_moves_everything_past_both() {
        let mut arrangement = ManualArrangement::default();
        let outer = arrangement.make_folder(ManualPlacement::Loose(Vec3::splat(40.0)));
        let middle = arrangement.make_folder(ManualPlacement::InFolder {
            folder: outer.clone(),
            offset: Vec3::new(1.0, 0.0, 0.0),
        });
        let inner = arrangement.make_folder(ManualPlacement::InFolder {
            folder: middle.clone(),
            offset: Vec3::ZERO,
        });
        arrangement.place(
            9,
            ManualPlacement::InFolder {
                folder: inner.clone(),
                offset: Vec3::ZERO,
            },
        );
        let laid_out = lay_out_points(&projection(), &arranged(&[], &arrangement));
        let folders = &laid_out.folders;
        let deleted = [
            folders.index_of(&middle).unwrap(),
            folders.index_of(&inner).unwrap(),
        ];
        delete_folders_from(folders, &deleted, CUBE, &mut arrangement);
        let relaid = lay_out_points(&projection(), &arranged(&[], &arrangement));
        assert!(relaid.folders.get(&middle).is_none());
        assert!(relaid.folders.get(&inner).is_none());
        assert_eq!(
            relaid.folders.folder_of(9).map(|folder| &folder.key),
            Some(&outer)
        );
    }

    #[test]
    fn ungrouped_points_keep_their_server_positions() {
        let mut page = projection();
        let mut ungrouped = point(30, "", [0; 3], [0.0; 3]);
        ungrouped.group = None;
        ungrouped.position = [4.0, 5.0, 6.0];
        page.points.push(ungrouped);
        let laid_out = lay_out_points(&page, &input(&[]));
        assert_eq!(position_of(&laid_out, 30), Some(Vec3::new(8.0, 10.0, 12.0)));
    }

    #[test]
    fn resetting_to_first_open_forgets_the_arrangement_and_the_shifted_origin() {
        let mut arrangement = ManualArrangement::default();
        arrangement.place(1, ManualPlacement::Loose(Vec3::ONE));
        let folder = arrangement.make_folder(ManualPlacement::Loose(Vec3::X));
        arrangement.rename_folder(&folder, "Mine");
        let before = arrangement.revision();
        arrangement.clear();
        assert!(arrangement.revision() > before);
        assert_eq!(arrangement.snapshot().contents, ArrangedContents::default());
        // Folders made later still get keys of their own.
        assert_eq!(
            arrangement.make_folder(ManualPlacement::Loose(Vec3::Y)),
            FolderKey::Manual(1)
        );

        let shifted = FolderScene {
            origin: Vec3::new(5.0, 0.0, 0.0),
            ..default()
        };
        let mut folder_view = FolderViewState::default();
        folder_view.toggle(&folder, 0.0);
        folder_view.reset_to_first_open();
        let input = folder_view.layout_input(1.0, &shifted, &arrangement);
        assert_eq!(input.origin, Vec3::ZERO);
        assert!(input.open.is_empty());
        // Only the first layout after the reset starts over.
        let next = folder_view.layout_input(1.0, &shifted, &arrangement);
        assert_eq!(next.origin, shifted.origin);
    }
}
