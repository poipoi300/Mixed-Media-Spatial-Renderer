//! Arranging a set of media files into a 3D shape.
//!
//! Layout is a pure function of the controls: the same controls always
//! produce the same coordinates, which is what makes the Randomize button
//! reproducible from its seed rather than merely different each press.

/// Deterministic xorshift64*, so a seed reproduces a layout exactly. Matching
/// the generator style the viewer's performance harness already uses keeps
/// this crate dependency-free.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // A zero state is a fixed point of xorshift, so it is nudged off it.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    /// Uniform in `[-1, 1)`.
    pub fn signed_unit(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    /// Fisher-Yates, so every permutation is equally likely.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let swap = (self.next_u64() % (index as u64 + 1)) as usize;
            items.swap(index, swap);
        }
    }
}

/// Seeds the generator from arbitrary text, so the seed control accepts a
/// word as readily as a number.
pub fn seed_from_text(text: &str) -> u64 {
    if let Ok(value) = text.trim().parse::<u64>() {
        return value;
    }
    // FNV-1a: short, well-distributed for short strings, no dependency.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Sphere,
    Cube,
    Ring,
    Spiral,
}

impl Shape {
    pub const ALL: [Shape; 4] = [Shape::Sphere, Shape::Cube, Shape::Ring, Shape::Spiral];

    pub fn id(self) -> &'static str {
        match self {
            Shape::Sphere => "sphere",
            Shape::Cube => "cube",
            Shape::Ring => "ring",
            Shape::Spiral => "spiral",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Shape::Sphere => "Sphere",
            Shape::Cube => "Cube",
            Shape::Ring => "Ring",
            Shape::Spiral => "Spiral",
        }
    }

    pub fn detail(self) -> &'static str {
        match self {
            Shape::Sphere => "Fibonacci lattice, shell or solid",
            Shape::Cube => "cubic lattice, faces or solid",
            Shape::Ring => "single flat circle",
            Shape::Spiral => "rising helix",
        }
    }

    /// Whether this shape offers the surface/volume choice. A ring and a
    /// spiral are curves, with no interior to fill.
    pub fn has_fill(self) -> bool {
        matches!(self, Shape::Sphere | Shape::Cube)
    }

    /// What a fill means for this shape, or `None` where the shape has no
    /// fill choice.
    pub fn fill_detail(self, fill: Fill) -> Option<&'static str> {
        match (self, fill) {
            (Shape::Sphere, Fill::Surface) => Some("outer shell only"),
            (Shape::Sphere, Fill::Volume) => Some("nested shells filling the ball"),
            (Shape::Cube, Fill::Surface) => Some("the six faces"),
            (Shape::Cube, Fill::Volume) => Some("solid lattice, a grid"),
            (Shape::Ring | Shape::Spiral, _) => None,
        }
    }

    /// What each world axis means for this shape, shown on the viewer's
    /// gizmo. A shape has no dimensions, so these are its own axis names.
    pub fn axis_labels(self) -> [Option<String>; 3] {
        let labels = match self {
            Shape::Sphere => ["Longitude", "Latitude", "Depth"],
            Shape::Cube => ["Width", "Height", "Depth"],
            Shape::Ring => ["Ring X", "Ring Y", "Ring Z"],
            Shape::Spiral => ["Spiral X", "Rise", "Spiral Z"],
        };
        labels.map(|label| Some(label.to_owned()))
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|shape| shape.id() == id)
    }
}

/// Whether a solid shape is laid out as its hollow surface or filled through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    Surface,
    Volume,
}

impl Fill {
    pub const ALL: [Fill; 2] = [Fill::Surface, Fill::Volume];

    pub fn id(self) -> &'static str {
        match self {
            Fill::Surface => "surface",
            Fill::Volume => "volume",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Fill::Surface => "Surface",
            Fill::Volume => "Volume",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|fill| fill.id() == id)
    }
}

/// How the media files are laid out for one set of control values.
pub struct Layout {
    pub shape: Shape,
    /// Ignored by shapes without a fill choice (see [`Shape::has_fill`]).
    pub fill: Fill,
    pub radius: f32,
    pub jitter: bool,
    pub seed: u64,
}

impl Layout {
    /// Positions for `count` items, in the order the caller holds them.
    ///
    /// Solid shapes list their surface first, so the images a `--limit`
    /// keeps when it truncates a catalog are the visible ones.
    ///
    /// Jitter is applied last and scaled to the shape's own size, so it
    /// loosens a lattice without turning it into noise.
    pub fn positions(&self, count: usize) -> Vec<[f32; 3]> {
        if count == 0 {
            return Vec::new();
        }
        let radius = self.radius;
        let mut positions: Vec<[f32; 3]> = match (self.shape, self.fill) {
            (Shape::Sphere, Fill::Surface) => (0..count)
                .map(|index| fibonacci_sphere_point(index, count, radius))
                .collect(),
            (Shape::Sphere, Fill::Volume) => sphere_volume_positions(count, radius),
            (Shape::Cube, fill) => cube_lattice_positions(count, radius, fill),
            (Shape::Ring, _) => (0..count)
                .map(|index| ring_point(index, count, radius))
                .collect(),
            (Shape::Spiral, _) => (0..count)
                .map(|index| spiral_point(index, count, radius))
                .collect(),
        };
        if self.jitter {
            let mut rng = Rng::new(self.seed ^ 0x5DEE_CE66_D1CE);
            let amount = radius * 0.04;
            for position in &mut positions {
                for axis in position.iter_mut() {
                    *axis += rng.signed_unit() * amount;
                }
            }
        }
        positions
    }
}

fn ring_point(index: usize, count: usize, radius: f32) -> [f32; 3] {
    let angle = std::f32::consts::TAU * index as f32 / count as f32;
    [angle.cos() * radius, 0.0, angle.sin() * radius]
}

fn spiral_point(index: usize, count: usize, radius: f32) -> [f32; 3] {
    let turns = 4.0;
    let progress = if count <= 1 {
        0.0
    } else {
        index as f32 / (count - 1) as f32
    };
    let angle = std::f32::consts::TAU * turns * progress;
    [
        angle.cos() * radius,
        (progress - 0.5) * radius * 2.0,
        angle.sin() * radius,
    ]
}

/// Point `index` of `count` on a sphere of `radius`, by Fibonacci lattice:
/// even coverage for any count, unlike a lat/long grid which crowds the poles.
fn fibonacci_sphere_point(index: usize, count: usize, radius: f32) -> [f32; 3] {
    let y = if count <= 1 {
        0.0
    } else {
        1.0 - 2.0 * index as f32 / (count - 1) as f32
    };
    let ring_radius = (1.0 - y * y).max(0.0).sqrt();
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let theta = golden_angle * index as f32;
    [
        theta.cos() * ring_radius * radius,
        y * radius,
        theta.sin() * ring_radius * radius,
    ]
}

/// Fills a ball with concentric Fibonacci shells.
///
/// Shells are spaced about as far apart as neighbours on a shell, and each
/// takes a share of the points proportional to its area, so density is even
/// through the whole volume. The outermost shell sits on the surface, so a
/// Volume sphere is the Surface sphere with its interior filled in.
fn sphere_volume_positions(count: usize, radius: f32) -> Vec<[f32; 3]> {
    // Spacing that fits `count` points into the ball's volume.
    let spacing = radius * (4.0 * std::f32::consts::PI / (3.0 * count as f32)).cbrt();
    let shell_count = ((radius / spacing).round() as usize).clamp(1, count);
    let shell_radii: Vec<f32> = (1..=shell_count)
        .map(|shell| radius * shell as f32 / shell_count as f32)
        .collect();

    // Largest-remainder apportionment by area (r squared), so the counts sum
    // to exactly `count` rather than drifting by rounding.
    let area_total: f32 = shell_radii.iter().map(|r| r * r).sum();
    let quotas: Vec<f32> = shell_radii
        .iter()
        .map(|r| count as f32 * r * r / area_total)
        .collect();
    let mut shell_sizes: Vec<usize> = quotas.iter().map(|quota| quota.floor() as usize).collect();
    let remainder = |shell: usize| quotas[shell] - quotas[shell].floor();
    let mut by_remainder: Vec<usize> = (0..shell_count).collect();
    by_remainder.sort_by(|&left, &right| remainder(right).total_cmp(&remainder(left)));
    let assigned: usize = shell_sizes.iter().sum();
    for &shell in by_remainder.iter().take(count - assigned) {
        shell_sizes[shell] += 1;
    }

    // Outermost first, so the lowest slots, which a limit keeps when the
    // catalog is truncated, land on the visible surface.
    shell_radii
        .iter()
        .zip(&shell_sizes)
        .rev()
        .flat_map(|(&shell_radius, &size)| {
            (0..size).map(move |index| fibonacci_sphere_point(index, size, shell_radius))
        })
        .collect()
}

/// Places `count` items on one cubic lattice spanning `[-radius, radius]` on
/// every axis: its boundary cells for a surface, every cell for a volume.
///
/// Both fills share the lattice, so a Volume cube is the Surface cube with
/// its interior filled in — the same relationship the sphere fills have —
/// and every cell is distinct, so no two images can land on one spot. The
/// lattice is the smallest that fits `count`; its leftover cells are spread
/// evenly across it rather than all left at one end, which would empty the
/// last faces or layers of the cube.
fn cube_lattice_positions(count: usize, radius: f32, fill: Fill) -> Vec<[f32; 3]> {
    let side = cube_lattice_side(count, fill);
    let spacing = if side > 1 {
        radius * 2.0 / (side - 1) as f32
    } else {
        0.0
    };
    let center = (side - 1) as f32 / 2.0;
    // How many cells in from the nearest face; the surface is layer 0.
    let layer = |cell: &[usize; 3]| {
        cell.iter()
            .map(|&coordinate| coordinate.min(side - 1 - coordinate))
            .min()
            .unwrap_or(0)
    };

    let mut cells: Vec<[usize; 3]> = (0..side)
        .flat_map(|x| (0..side).flat_map(move |y| (0..side).map(move |z| [x, y, z])))
        .collect();
    if fill == Fill::Surface {
        cells.retain(|cell| layer(cell) == 0);
    }
    // Outermost layer first, as for the sphere; the sort is stable, so each
    // layer keeps its scan order.
    cells.sort_by_key(layer);

    let available = cells.len();
    (0..count)
        .map(|slot| {
            cells[slot * available / count].map(|coordinate| (coordinate as f32 - center) * spacing)
        })
        .collect()
}

/// Smallest lattice side whose cells, all of them or only the boundary ones,
/// can hold `count` items.
fn cube_lattice_side(count: usize, fill: Fill) -> usize {
    let capacity = |side: usize| match fill {
        Fill::Volume => side.pow(3),
        Fill::Surface => side.pow(3) - side.saturating_sub(2).pow(3),
    };
    let mut side = 1;
    while capacity(side) < count {
        side += 1;
    }
    side
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: Shape) -> Layout {
        Layout {
            shape,
            fill: Fill::Surface,
            radius: 10.0,
            jitter: false,
            seed: 7,
        }
    }

    fn volume(shape: Shape) -> Layout {
        Layout {
            fill: Fill::Volume,
            ..layout(shape)
        }
    }

    fn volume_sphere() -> Layout {
        volume(Shape::Sphere)
    }

    fn on_a_cube_face(position: [f32; 3]) -> bool {
        position.iter().any(|axis| (axis.abs() - 10.0).abs() < 1e-3)
    }

    fn distance(position: [f32; 3]) -> f32 {
        (position[0] * position[0] + position[1] * position[1] + position[2] * position[2]).sqrt()
    }

    #[test]
    fn a_volume_sphere_places_every_point_inside_the_ball() {
        for count in [1, 2, 7, 50, 500, 3000] {
            let positions = volume_sphere().positions(count);
            assert_eq!(positions.len(), count, "count {count}");
            for position in &positions {
                assert!(position.iter().all(|axis| axis.is_finite()));
                assert!(distance(*position) <= 10.0 + 1e-3, "{position:?}");
            }
        }
    }

    #[test]
    fn a_volume_sphere_fills_its_interior_evenly() {
        let positions = volume_sphere().positions(2000);

        // The inner half of the radius holds 1/8 of the volume, so an even
        // fill puts about 1/8 of the points there. A surface sphere puts none.
        let inner = positions
            .iter()
            .filter(|position| distance(**position) < 5.0)
            .count() as f32
            / positions.len() as f32;
        assert!(
            (0.07..0.20).contains(&inner),
            "inner-half share {inner} is not an even fill"
        );
        let surface_inner = layout(Shape::Sphere)
            .positions(2000)
            .iter()
            .filter(|position| distance(**position) < 5.0)
            .count();
        assert_eq!(surface_inner, 0);
    }

    #[test]
    fn a_volume_sphere_leads_with_its_surface() {
        let positions = volume_sphere().positions(2000);
        assert!(
            (distance(positions[0]) - 10.0).abs() < 1e-3,
            "a truncated catalog should keep the outer shell"
        );
        assert!(
            distance(positions[positions.len() - 1]) < 10.0,
            "the last points belong to the innermost shell"
        );
    }

    #[test]
    fn fill_does_not_affect_other_shapes() {
        for shape in Shape::ALL.into_iter().filter(|shape| !shape.has_fill()) {
            assert!(shape.fill_detail(Fill::Volume).is_none());
            assert_eq!(
                volume(shape).positions(40),
                layout(shape).positions(40),
                "{}",
                shape.id()
            );
        }
    }

    #[test]
    fn fills_round_trip_through_their_ids() {
        for fill in Fill::ALL {
            assert_eq!(Fill::from_id(fill.id()), Some(fill));
        }
        assert_eq!(Fill::from_id("hollow"), None);
    }

    #[test]
    fn no_two_images_share_a_position() {
        for shape in Shape::ALL {
            for fill in Fill::ALL {
                for count in [2, 7, 30, 120, 500] {
                    let positions = Layout {
                        fill,
                        ..layout(shape)
                    }
                    .positions(count);
                    let mut keys: Vec<[i64; 3]> = positions
                        .iter()
                        .map(|position| position.map(|axis| (axis * 1000.0).round() as i64))
                        .collect();
                    keys.sort_unstable();
                    keys.dedup();
                    assert_eq!(
                        keys.len(),
                        count,
                        "{} {} with {count} images stacks billboards",
                        shape.id(),
                        fill.id()
                    );
                }
            }
        }
    }

    #[test]
    fn a_cube_reaches_every_face_whatever_the_count() {
        for fill in Fill::ALL {
            for count in [7, 30, 120, 500] {
                let positions = Layout {
                    fill,
                    ..layout(Shape::Cube)
                }
                .positions(count);
                for axis in 0..3 {
                    let lowest = positions
                        .iter()
                        .map(|p| p[axis])
                        .fold(f32::INFINITY, f32::min);
                    let highest = positions
                        .iter()
                        .map(|p| p[axis])
                        .fold(f32::NEG_INFINITY, f32::max);
                    assert!(
                        (lowest + 10.0).abs() < 1e-3 && (highest - 10.0).abs() < 1e-3,
                        "{} cube of {count} leaves axis {axis} short: {lowest}..{highest}",
                        fill.id()
                    );
                }
            }
        }
    }

    #[test]
    fn a_volume_cube_is_a_solid_grid_led_by_its_surface() {
        let positions = volume(Shape::Cube).positions(500);

        // An 8-wide lattice: 216 of its 512 cells are interior.
        let interior = positions.iter().filter(|p| !on_a_cube_face(**p)).count() as f32
            / positions.len() as f32;
        assert!(
            (0.3..0.55).contains(&interior),
            "interior share {interior} is not a filled cube"
        );
        assert!(on_a_cube_face(positions[0]), "the surface comes first");
        assert!(!on_a_cube_face(positions[positions.len() - 1]));
        // Every point sits on the lattice: coordinates step by 20/7.
        let step = 20.0 / 7.0;
        for position in &positions {
            for axis in position {
                let steps = (axis + 10.0) / step;
                assert!(
                    (steps - steps.round()).abs() < 1e-3,
                    "{position:?} is off the lattice"
                );
            }
        }
    }

    #[test]
    fn every_shape_places_every_item_within_its_radius() {
        for shape in Shape::ALL {
            let positions = layout(shape).positions(50);
            assert_eq!(positions.len(), 50, "{}", shape.id());
            for position in positions {
                let distance = (position[0] * position[0]
                    + position[1] * position[1]
                    + position[2] * position[2])
                    .sqrt();
                assert!(
                    distance <= 10.0 * 3.0_f32.sqrt() + 1e-3,
                    "{} placed a point at {distance}",
                    shape.id()
                );
                assert!(position.iter().all(|axis| axis.is_finite()));
            }
        }
    }

    #[test]
    fn a_sphere_puts_points_on_its_shell_not_inside() {
        let positions = layout(Shape::Sphere).positions(200);
        for position in positions {
            let distance =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            assert!((distance - 10.0).abs() < 0.01, "{distance}");
        }
    }

    #[test]
    fn a_cube_leaves_its_interior_empty() {
        let positions = layout(Shape::Cube).positions(120);
        for position in positions {
            let on_a_face = position.iter().any(|axis| (axis.abs() - 10.0).abs() < 0.01);
            assert!(on_a_face, "{position:?} is not on a face");
        }
    }

    #[test]
    fn one_item_and_no_items_are_both_handled() {
        for shape in Shape::ALL {
            assert!(layout(shape).positions(0).is_empty());
            let single = layout(shape).positions(1);
            assert_eq!(single.len(), 1);
            assert!(single[0].iter().all(|axis| axis.is_finite()));
        }
    }

    #[test]
    fn the_same_seed_reproduces_a_shuffle_and_a_different_one_changes_it() {
        let order = |seed: u64| {
            let mut items: Vec<usize> = (0..64).collect();
            Rng::new(seed).shuffle(&mut items);
            items
        };
        assert_eq!(order(42), order(42));
        assert_ne!(order(42), order(43));
        // A shuffle is a permutation, so nothing is lost or duplicated.
        let mut shuffled = order(42);
        shuffled.sort_unstable();
        assert_eq!(shuffled, (0..64).collect::<Vec<_>>());
    }

    #[test]
    fn jitter_perturbs_positions_reproducibly() {
        let jittered = |seed: u64| {
            Layout {
                jitter: true,
                seed,
                ..volume(Shape::Cube)
            }
            .positions(27)
        };
        let plain = volume(Shape::Cube).positions(27);

        assert_ne!(jittered(1), plain);
        assert_eq!(jittered(1), jittered(1));
        assert_ne!(jittered(1), jittered(2));
    }

    #[test]
    fn text_seeds_are_stable_and_numbers_pass_through() {
        assert_eq!(seed_from_text("42"), 42);
        assert_eq!(seed_from_text(" 42 "), 42);
        assert_eq!(seed_from_text("aria"), seed_from_text("aria"));
        assert_ne!(seed_from_text("aria"), seed_from_text("ario"));
    }

    #[test]
    fn shapes_round_trip_through_their_ids() {
        for shape in Shape::ALL {
            assert_eq!(Shape::from_id(shape.id()), Some(shape));
        }
        assert_eq!(Shape::from_id("dodecahedron"), None);
        // Grid became the cube's Volume fill.
        assert_eq!(Shape::from_id("grid"), None);
    }
}
