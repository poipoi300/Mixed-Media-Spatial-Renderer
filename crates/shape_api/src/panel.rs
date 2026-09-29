//! The control panel this server publishes, and the snapshot it answers with.
//!
//! Nothing here knows about dimensions or axes: a shape API's controls are a
//! shape, a radius, a jitter toggle, a seed and a randomize button. That the
//! viewer renders them at all is the point — it renders whatever a server
//! describes.

use std::hash::{DefaultHasher, Hash, Hasher};

use generation_api::{
    ControlOption, ControlPanel, ControlValue, ControlValues, ControlWidget, StatLine,
};
use serde::Serialize;

use crate::media::{collect_media, MediaFile};
use crate::shapes::{seed_from_text, Fill, Layout, Rng, Shape};

pub const SHAPE_CONTROL: &str = "shape";
/// Only published while a shape with a fill choice is selected.
pub const FILL_CONTROL: &str = "fill";
pub const RADIUS_CONTROL: &str = "radius";
pub const JITTER_CONTROL: &str = "jitter";
pub const SEED_CONTROL: &str = "seed";
pub const RANDOMIZE_CONTROL: &str = "randomize";

const DEFAULT_RADIUS: f64 = 24.0;
const MIN_RADIUS: f64 = 2.0;
const MAX_RADIUS: f64 = 200.0;
const RADIUS_STEP: f64 = 1.0;

/// A point as the viewer's `ProjectionPoint` expects it.
#[derive(Debug, Serialize)]
pub struct ProjectionPoint {
    pub image_id: usize,
    pub path: String,
    pub position: [f32; 3],
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub media_type: &'static str,
    pub duration_seconds: Option<f32>,
    pub coordinate_labels: [Option<String>; 3],
}

#[derive(Debug, Serialize)]
pub struct ProjectionPage {
    pub axis_labels: [Option<String>; 3],
    pub coordinate_spacing: f32,
    pub duplicate_spacing: f32,
    pub sprite_world_height: f32,
    pub offset: usize,
    pub limit: usize,
    pub total: usize,
    pub points: Vec<ProjectionPoint>,
}

#[derive(Debug, Serialize)]
pub struct CatalogSnapshot {
    pub panel: ControlPanel,
    pub projection: ProjectionPage,
}

/// The controls a request submitted, resolved against this server's rules.
pub struct Controls {
    pub shape: Shape,
    pub fill: Fill,
    pub radius: f64,
    pub jitter: bool,
    pub seed_text: String,
    /// Bumped each time Randomize is pressed, so repeated presses keep
    /// producing new arrangements while the same seed still reproduces one.
    pub shuffle_round: u64,
    pub error: Option<String>,
}

impl Controls {
    /// Reads the submitted values, reporting anything it could not honor
    /// rather than silently substituting a default.
    pub fn from_values(values: &ControlValues, activated: Option<&str>) -> Self {
        let mut error = None;

        let shape_text = values
            .get(SHAPE_CONTROL)
            .map(ControlValue::as_text)
            .unwrap_or_else(|| Shape::Sphere.id().to_owned());
        let shape = Shape::from_id(&shape_text).unwrap_or_else(|| {
            error = Some(format!("Unknown shape '{shape_text}'."));
            Shape::Sphere
        });

        // Validated whatever the shape, so a bad value from `--control` is
        // reported rather than silently carried until the sphere is chosen.
        let fill_text = values
            .get(FILL_CONTROL)
            .map(ControlValue::as_text)
            .unwrap_or_else(|| Fill::Surface.id().to_owned());
        let fill = Fill::from_id(&fill_text).unwrap_or_else(|| {
            error.get_or_insert_with(|| format!("Unknown fill '{fill_text}'."));
            Fill::Surface
        });

        let radius = values
            .get(RADIUS_CONTROL)
            .and_then(ControlValue::as_number)
            .unwrap_or(DEFAULT_RADIUS)
            .clamp(MIN_RADIUS, MAX_RADIUS);

        let jitter = values
            .get(JITTER_CONTROL)
            .map(ControlValue::as_bool)
            .unwrap_or(false);

        let seed_text = values
            .get(SEED_CONTROL)
            .map(ControlValue::as_text)
            .unwrap_or_else(|| "1".to_owned());

        Self {
            shape,
            fill,
            radius,
            jitter,
            seed_text,
            shuffle_round: 0,
            error,
        }
        .with_randomize(activated)
    }

    fn with_randomize(mut self, activated: Option<&str>) -> Self {
        if activated == Some(RANDOMIZE_CONTROL) {
            self.shuffle_round = 1;
        }
        self
    }

    pub fn seed(&self) -> u64 {
        seed_from_text(&self.seed_text)
    }
}

/// This server's whole state: which files it found, and how many times
/// Randomize has been pressed. Kept because the button must produce a new
/// arrangement on each press while the seed still reproduces one — the values
/// alone cannot express "again".
#[derive(Default)]
pub struct ServerState {
    pub roots: Vec<String>,
    pub media: Vec<MediaFile>,
    pub shuffle_round: u64,
}

impl ServerState {
    /// Rescans when the roots changed; a reprojection reuses what it has.
    pub fn ensure_media(&mut self, roots: &[String]) {
        if !roots.is_empty() && roots != self.roots.as_slice() {
            self.roots = roots.to_vec();
            self.media = collect_media(roots);
        }
    }
}

/// Builds the snapshot for one request.
pub fn build_snapshot(
    state: &ServerState,
    controls: &Controls,
    coordinate_spacing: f32,
    sprite_world_height: f32,
    limit: usize,
) -> CatalogSnapshot {
    let total = state.media.len();
    let shown = total.min(limit.max(1));

    // Which file lands in which slot is what Randomize changes; the slots
    // themselves are fixed by the shape.
    let mut order: Vec<usize> = (0..total).collect();
    if state.shuffle_round > 0 {
        Rng::new(controls.seed() ^ state.shuffle_round.wrapping_mul(0x9E37_79B9))
            .shuffle(&mut order);
    }

    let layout = Layout {
        shape: controls.shape,
        fill: controls.fill,
        radius: controls.radius as f32,
        jitter: controls.jitter,
        seed: controls.seed(),
    };
    let positions = layout.positions(total);
    let axis_labels = controls.shape.axis_labels();

    let points = order
        .iter()
        .take(shown)
        .enumerate()
        .map(|(slot, &file_index)| {
            let file = &state.media[file_index];
            let position = positions[slot];
            ProjectionPoint {
                // Stable per file, not per slot, so the viewer keeps a
                // decoded texture with its image across a reshuffle.
                image_id: file_index,
                path: file.path.to_string_lossy().into_owned(),
                position,
                width: None,
                height: None,
                media_type: if file.is_video { "video" } else { "image" },
                duration_seconds: None,
                coordinate_labels: position.map(|axis| Some(format!("{axis:.1}"))),
            }
        })
        .collect();

    CatalogSnapshot {
        panel: build_panel(state, controls, shown, total),
        projection: ProjectionPage {
            axis_labels,
            coordinate_spacing,
            duplicate_spacing: 0.0,
            sprite_world_height,
            offset: 0,
            limit,
            total,
            points,
        },
    }
}

fn build_panel(
    state: &ServerState,
    controls: &Controls,
    shown: usize,
    total: usize,
) -> ControlPanel {
    let videos = state.media.iter().filter(|file| file.is_video).count();
    let widgets = panel_widgets(controls, total);
    ControlPanel {
        revision: structural_revision(&widgets),
        title: "Shape".to_owned(),
        summary: format!("{}  {shown} pts", shape_description(controls)),
        stats: vec![
            StatLine {
                label: "shape".to_owned(),
                value: shape_description(controls),
            },
            StatLine {
                label: "shown".to_owned(),
                value: format!("{shown} / {total}"),
            },
            StatLine {
                label: "videos".to_owned(),
                value: videos.to_string(),
            },
            StatLine {
                label: "radius".to_owned(),
                value: format!("{:.0}", controls.radius),
            },
            StatLine {
                label: "shuffles".to_owned(),
                value: state.shuffle_round.to_string(),
            },
        ],
        error: controls.error.clone(),
        widgets,
    }
}

/// "Cube" alone is ambiguous once it has two fills, so name the fill too.
fn shape_description(controls: &Controls) -> String {
    if controls.shape.has_fill() {
        format!(
            "{} ({})",
            controls.shape.label(),
            controls.fill.label().to_lowercase()
        )
    } else {
        controls.shape.label().to_owned()
    }
}

/// The widget set depends on the selected shape: the fill dropdown exists
/// only for shapes with an interior to fill, so it appears and disappears as
/// the shape changes. Sphere and cube share it, so switching between them
/// keeps the chosen fill and needs no widget rebuild.
fn panel_widgets(controls: &Controls, total: usize) -> Vec<ControlWidget> {
    let mut widgets = vec![ControlWidget::Select {
        id: SHAPE_CONTROL.to_owned(),
        label: "Shape".to_owned(),
        detail: Some(controls.shape.detail().to_owned()),
        value: controls.shape.id().to_owned(),
        options: Shape::ALL
            .into_iter()
            .map(|shape| ControlOption {
                value: shape.id().to_owned(),
                label: shape.label().to_owned(),
                detail: Some(shape.detail().to_owned()),
            })
            .collect(),
        disabled: false,
        submits: true,
    }];
    if controls.shape.has_fill() {
        let shape = controls.shape;
        widgets.push(ControlWidget::Select {
            id: FILL_CONTROL.to_owned(),
            label: "Fill".to_owned(),
            detail: shape.fill_detail(controls.fill).map(str::to_owned),
            value: controls.fill.id().to_owned(),
            options: Fill::ALL
                .into_iter()
                .map(|fill| ControlOption {
                    value: fill.id().to_owned(),
                    label: fill.label().to_owned(),
                    detail: shape.fill_detail(fill).map(str::to_owned),
                })
                .collect(),
            disabled: false,
            submits: true,
        });
    }
    widgets.extend([
        ControlWidget::Group {
            id: "layout".to_owned(),
            label: "Layout".to_owned(),
            detail: None,
            collapsible: true,
            children: vec![
                ControlWidget::Slider {
                    id: RADIUS_CONTROL.to_owned(),
                    label: "Radius".to_owned(),
                    detail: None,
                    value: controls.radius,
                    minimum: MIN_RADIUS,
                    maximum: MAX_RADIUS,
                    step: RADIUS_STEP,
                    disabled: false,
                    submits: true,
                },
                ControlWidget::Toggle {
                    id: JITTER_CONTROL.to_owned(),
                    label: "Jitter".to_owned(),
                    detail: Some("loosen the lattice".to_owned()),
                    value: controls.jitter,
                    disabled: false,
                    submits: true,
                },
            ],
        },
        ControlWidget::Group {
            id: "arrangement".to_owned(),
            label: "Arrangement".to_owned(),
            detail: None,
            collapsible: true,
            children: vec![
                ControlWidget::Text {
                    id: SEED_CONTROL.to_owned(),
                    label: "Seed".to_owned(),
                    detail: Some("press Randomize to apply".to_owned()),
                    value: controls.seed_text.clone(),
                    disabled: false,
                    // A form field: editing the seed alone changes
                    // nothing until Randomize submits it.
                    submits: false,
                },
                ControlWidget::Button {
                    id: RANDOMIZE_CONTROL.to_owned(),
                    label: "Randomize".to_owned(),
                    detail: None,
                    // Nothing to shuffle without images.
                    disabled: total == 0,
                    submits: true,
                },
            ],
        },
    ]);
    widgets
}

/// A revision derived from the widget structure: kinds, ids, nesting, and
/// whether a group collapses — what the viewer spawns entities from.
///
/// The viewer rebuilds its widget entities whenever this changes, so it must
/// move exactly when the structure does. Deriving it rather than bumping a
/// counter by hand keeps that true as the fill dropdown comes and goes, and
/// leaves values, labels and options (which the viewer reads live) out of it.
fn structural_revision(widgets: &[ControlWidget]) -> u64 {
    fn hash_widgets(widgets: &[ControlWidget], hasher: &mut DefaultHasher) {
        widgets.len().hash(hasher);
        for widget in widgets {
            std::mem::discriminant(widget).hash(hasher);
            widget.id().hash(hasher);
            if let ControlWidget::Group { collapsible, .. } = widget {
                collapsible.hash(hasher);
            }
            hash_widgets(widget.children(), hasher);
        }
    }
    let mut hasher = DefaultHasher::new();
    hash_widgets(widgets, &mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn state(count: usize) -> ServerState {
        ServerState {
            roots: vec!["fixture".to_owned()],
            media: (0..count)
                .map(|index| MediaFile {
                    path: PathBuf::from(format!("{index}.png")),
                    is_video: index % 5 == 4,
                })
                .collect(),
            shuffle_round: 0,
        }
    }

    fn controls(values: &[(&str, ControlValue)], activated: Option<&str>) -> Controls {
        let values: ControlValues = values
            .iter()
            .map(|(id, value)| ((*id).to_owned(), value.clone()))
            .collect();
        Controls::from_values(&values, activated)
    }

    fn snapshot(state: &ServerState, controls: &Controls) -> CatalogSnapshot {
        build_snapshot(state, controls, 6.0, 4.0, 1000)
    }

    fn widget_ids(panel: &ControlPanel) -> Vec<&str> {
        panel
            .walk()
            .into_iter()
            .map(|(widget, _)| widget.id())
            .collect()
    }

    #[test]
    fn the_panel_offers_a_shape_dropdown_and_a_randomize_button() {
        let snapshot = snapshot(&state(12), &controls(&[], None));

        assert_eq!(
            widget_ids(&snapshot.panel),
            vec![
                "shape",
                "fill",
                "layout",
                "radius",
                "jitter",
                "arrangement",
                "seed",
                "randomize"
            ]
        );
        let shape = snapshot.panel.find("shape").unwrap();
        assert_eq!(
            shape.children().len(),
            0,
            "a select is a leaf; its options are not child widgets"
        );
        assert!(snapshot.panel.find("randomize").unwrap().submits());
        // Editing the seed alone must not reshuffle.
        assert!(!snapshot.panel.find("seed").unwrap().submits());
    }

    #[test]
    fn the_panel_reports_the_values_it_actually_used() {
        let snapshot = snapshot(
            &state(12),
            &controls(
                &[
                    ("shape", ControlValue::Text("cube".to_owned())),
                    ("radius", ControlValue::Number(40.0)),
                    ("jitter", ControlValue::Bool(true)),
                ],
                None,
            ),
        );

        assert_eq!(
            snapshot
                .panel
                .find("shape")
                .unwrap()
                .value()
                .unwrap()
                .as_text(),
            "cube"
        );
        assert_eq!(
            snapshot
                .panel
                .find("radius")
                .unwrap()
                .value()
                .unwrap()
                .as_number(),
            Some(40.0)
        );
        assert!(snapshot
            .panel
            .find("jitter")
            .unwrap()
            .value()
            .unwrap()
            .as_bool());
        assert_eq!(
            snapshot.projection.axis_labels[0].as_deref(),
            Some("Width"),
            "the gizmo follows the shape's own axis names"
        );
    }

    #[test]
    fn an_unknown_shape_is_reported_and_falls_back_to_a_usable_scene() {
        let snapshot = snapshot(
            &state(12),
            &controls(
                &[("shape", ControlValue::Text("dodecahedron".to_owned()))],
                None,
            ),
        );

        assert_eq!(
            snapshot.panel.error.as_deref(),
            Some("Unknown shape 'dodecahedron'.")
        );
        assert_eq!(snapshot.projection.points.len(), 12);
    }

    #[test]
    fn the_fill_dropdown_is_offered_for_the_sphere_and_cube_only() {
        let with_shape = |shape: &str| {
            snapshot(
                &state(12),
                &controls(&[("shape", ControlValue::Text(shape.to_owned()))], None),
            )
        };
        for shape in ["sphere", "cube"] {
            let snapshot = with_shape(shape);
            let fill = snapshot
                .panel
                .find(FILL_CONTROL)
                .unwrap_or_else(|| panic!("{shape} has a fill"));
            let ControlWidget::Select { options, value, .. } = fill else {
                panic!("fill is a dropdown");
            };
            assert_eq!(
                options
                    .iter()
                    .map(|option| option.label.as_str())
                    .collect::<Vec<_>>(),
                vec!["Surface", "Volume"]
            );
            assert_eq!(value, "surface", "{shape} defaults to its surface");
        }
        for shape in ["ring", "spiral"] {
            assert!(
                with_shape(shape).panel.find(FILL_CONTROL).is_none(),
                "a {shape} has no interior to fill"
            );
        }
    }

    #[test]
    fn revision_moves_with_the_widget_structure_and_nothing_else() {
        let sphere = |values: &[(&str, ControlValue)]| {
            snapshot(&state(12), &controls(values, None)).panel.revision
        };
        let surface = sphere(&[]);
        // Values, stats and detail text change; the widgets the viewer spawns
        // do not, so it must not rebuild them.
        assert_eq!(
            surface,
            sphere(&[
                ("fill", ControlValue::Text("volume".to_owned())),
                ("radius", ControlValue::Number(80.0)),
                ("jitter", ControlValue::Bool(true)),
            ])
        );
        // Sphere and cube publish the same widgets, so switching between them
        // must not rebuild either.
        assert_eq!(
            surface,
            sphere(&[("shape", ControlValue::Text("cube".to_owned()))])
        );
        // Dropping the fill dropdown is a structural change and must rebuild;
        // shapes without it must agree, so a round trip is not a new state.
        let ring = sphere(&[("shape", ControlValue::Text("ring".to_owned()))]);
        assert_ne!(surface, ring);
        assert_eq!(
            ring,
            sphere(&[("shape", ControlValue::Text("spiral".to_owned()))])
        );
    }

    #[test]
    fn volume_fill_reaches_the_layout() {
        let volume = snapshot(
            &state(400),
            &controls(&[("fill", ControlValue::Text("volume".to_owned()))], None),
        );
        let interior = volume
            .projection
            .points
            .iter()
            .filter(|point| {
                let [x, y, z] = point.position;
                (x * x + y * y + z * z).sqrt() < DEFAULT_RADIUS as f32 - 0.5
            })
            .count();
        assert!(interior > 0, "a volume sphere has points inside it");
        assert_eq!(
            volume.panel.stats[0].value, "Sphere (volume)",
            "the stats name the fill"
        );

        let grid = snapshot(
            &state(400),
            &controls(
                &[
                    ("shape", ControlValue::Text("cube".to_owned())),
                    ("fill", ControlValue::Text("volume".to_owned())),
                ],
                None,
            ),
        );
        let off_faces = grid
            .projection
            .points
            .iter()
            .filter(|point| {
                point
                    .position
                    .iter()
                    .all(|axis| axis.abs() < DEFAULT_RADIUS as f32 - 0.5)
            })
            .count();
        assert!(off_faces > 0, "a volume cube has points inside it");
        assert_eq!(grid.panel.stats[0].value, "Cube (volume)");
    }

    #[test]
    fn an_unknown_fill_is_reported_and_falls_back_to_the_surface() {
        let snapshot = snapshot(
            &state(12),
            &controls(&[("fill", ControlValue::Text("hollow".to_owned()))], None),
        );

        assert_eq!(
            snapshot.panel.error.as_deref(),
            Some("Unknown fill 'hollow'.")
        );
        assert_eq!(snapshot.projection.points.len(), 12);
    }

    #[test]
    fn radius_is_clamped_to_the_slider_range() {
        let huge = snapshot(
            &state(4),
            &controls(&[("radius", ControlValue::Number(5000.0))], None),
        );
        assert_eq!(
            huge.panel
                .find("radius")
                .unwrap()
                .value()
                .unwrap()
                .as_number(),
            Some(MAX_RADIUS)
        );

        let tiny = snapshot(
            &state(4),
            &controls(&[("radius", ControlValue::Number(-5.0))], None),
        );
        assert_eq!(
            tiny.panel
                .find("radius")
                .unwrap()
                .value()
                .unwrap()
                .as_number(),
            Some(MIN_RADIUS)
        );
    }

    #[test]
    fn randomize_reassigns_images_to_slots_without_moving_the_slots() {
        let mut state = state(40);
        let before = snapshot(&state, &controls(&[], None));

        state.shuffle_round = 1;
        let after = snapshot(&state, &controls(&[], Some(RANDOMIZE_CONTROL)));

        let slot_positions = |snapshot: &CatalogSnapshot| {
            snapshot
                .projection
                .points
                .iter()
                .map(|point| point.position.map(|axis| (axis * 1000.0) as i32))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            slot_positions(&before),
            slot_positions(&after),
            "the shape is unchanged; only which image sits where"
        );
        let image_order = |snapshot: &CatalogSnapshot| {
            snapshot
                .projection
                .points
                .iter()
                .map(|point| point.image_id)
                .collect::<Vec<_>>()
        };
        assert_ne!(image_order(&before), image_order(&after));
        // Every image still appears exactly once.
        let mut shuffled = image_order(&after);
        shuffled.sort_unstable();
        assert_eq!(shuffled, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn the_same_seed_reproduces_an_arrangement() {
        let mut state = state(40);
        state.shuffle_round = 1;
        let seeded = |seed: &str| {
            snapshot(
                &state,
                &controls(
                    &[("seed", ControlValue::Text(seed.to_owned()))],
                    Some(RANDOMIZE_CONTROL),
                ),
            )
            .projection
            .points
            .iter()
            .map(|point| point.image_id)
            .collect::<Vec<_>>()
        };

        assert_eq!(seeded("42"), seeded("42"));
        assert_ne!(seeded("42"), seeded("43"));
    }

    #[test]
    fn an_empty_catalog_still_describes_its_controls() {
        let snapshot = snapshot(&state(0), &controls(&[], None));

        assert_eq!(widget_ids(&snapshot.panel).len(), 8);
        assert!(snapshot.projection.points.is_empty());
        assert!(
            snapshot.panel.find("randomize").unwrap().disabled(),
            "there is nothing to shuffle"
        );
    }

    #[test]
    fn the_limit_bounds_the_points_without_hiding_the_total() {
        let snapshot = build_snapshot(&state(500), &controls(&[], None), 6.0, 4.0, 100);

        assert_eq!(snapshot.projection.points.len(), 100);
        assert_eq!(snapshot.projection.total, 500);
    }

    #[test]
    fn media_is_rescanned_only_when_the_roots_change() {
        let mut state = ServerState::default();
        state.ensure_media(&["Z:/missing".to_owned()]);
        assert_eq!(state.roots, vec!["Z:/missing".to_owned()]);

        state.media = vec![MediaFile {
            path: PathBuf::from("kept.png"),
            is_video: false,
        }];
        // Same roots: a reprojection reuses the scan.
        state.ensure_media(&["Z:/missing".to_owned()]);
        assert_eq!(state.media.len(), 1);
        // No roots (a plain reprojection) also reuses it.
        state.ensure_media(&[]);
        assert_eq!(state.media.len(), 1);
    }
}
