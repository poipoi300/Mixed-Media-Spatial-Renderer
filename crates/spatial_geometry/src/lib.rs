use bevy_math::Vec3;

/// Groups laid out on a grid of touching slots, in cube units.
///
/// Along each axis every group occupies the slot its rank names, and a slot
/// is as wide as the widest group in it, so the groups sharing a row, column
/// or layer line up and none overlap. Slot 0 is centered on the origin and
/// the rest follow it with no gap.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SlotGrid {
    centers: [Vec<f32>; 3],
    widths: [Vec<f32>; 3],
}

impl SlotGrid {
    /// Packs one footprint per group: its slot ranks and its extent in
    /// cubes. A slot is never narrower than one cube, including a rank no
    /// group occupies, so sparse ranks still keep their order.
    pub fn pack(footprints: impl IntoIterator<Item = ([u32; 3], Vec3)>) -> Self {
        let mut widths: [Vec<f32>; 3] = Default::default();
        for (index, extent) in footprints {
            for (axis, axis_widths) in widths.iter_mut().enumerate() {
                let rank = index[axis] as usize;
                if axis_widths.len() <= rank {
                    axis_widths.resize(rank + 1, 1.0);
                }
                axis_widths[rank] = axis_widths[rank].max(extent[axis]);
            }
        }
        Self {
            centers: widths
                .each_ref()
                .map(|axis_widths| touching_slot_centers(axis_widths)),
            widths,
        }
    }

    /// Center of the slot at `index`, in cube units. Every index must be one
    /// that was packed.
    pub fn center(&self, index: [u32; 3]) -> Vec3 {
        Vec3::from_array(std::array::from_fn(|axis| {
            self.centers[axis][index[axis] as usize]
        }))
    }

    /// The low and high edges of all the slots along each axis, in cube
    /// units; nothing along an axis no group occupies.
    pub fn span(&self) -> (Vec3, Vec3) {
        let edge = |axis: usize, high: bool| {
            let (centers, widths) = (&self.centers[axis], &self.widths[axis]);
            let rank = if high {
                centers.len().checked_sub(1)
            } else {
                Some(0)
            };
            match rank.filter(|&rank| rank < centers.len()) {
                Some(rank) if high => centers[rank] + widths[rank] * 0.5,
                Some(rank) => centers[rank] - widths[rank] * 0.5,
                None => 0.0,
            }
        };
        (
            Vec3::from_array(std::array::from_fn(|axis| edge(axis, false))),
            Vec3::from_array(std::array::from_fn(|axis| edge(axis, true))),
        )
    }

    /// The rank whose slot holds `coordinate` along `axis`, in the frame of
    /// [`Self::center`]. Past either end, ranks carry on one cube wide.
    pub fn rank_at(&self, axis: usize, coordinate: f32) -> i64 {
        let (centers, widths) = (&self.centers[axis], &self.widths[axis]);
        let (Some(&first), Some(&last)) = (centers.first(), centers.last()) else {
            return coordinate.round() as i64;
        };
        let low = first - widths[0] * 0.5;
        if coordinate < low {
            return -((low - coordinate).ceil() as i64);
        }
        if let Some(rank) = centers
            .iter()
            .zip(widths)
            .position(|(center, width)| coordinate < center + width * 0.5)
        {
            return rank as i64;
        }
        let high = last + widths[widths.len() - 1] * 0.5;
        centers.len() as i64 + (coordinate - high).floor() as i64
    }
}

fn touching_slot_centers(widths: &[f32]) -> Vec<f32> {
    let mut centers = Vec::with_capacity(widths.len());
    let mut previous_edge = 0.0;
    for (rank, width) in widths.iter().enumerate() {
        let center = if rank == 0 {
            0.0
        } else {
            previous_edge + width * 0.5
        };
        centers.push(center);
        previous_edge = center + width * 0.5;
    }
    centers
}

/// Extent, in cubes, of a block holding cubes at `offsets` from its center,
/// kept symmetric about the center so the block stays centered on its slot.
/// An empty block is one cube.
pub fn block_extent(offsets: impl IntoIterator<Item = Vec3>) -> Vec3 {
    let reach = offsets
        .into_iter()
        .fold(Vec3::ZERO, |reach, offset| reach.max(offset.abs()));
    reach * 2.0 + Vec3::ONE
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds3 {
    pub minimum: Vec3,
    pub maximum: Vec3,
    pub center: Vec3,
    pub max_extent: f32,
}

pub fn bounds_of(positions: impl IntoIterator<Item = Vec3>) -> Bounds3 {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for position in positions {
        minimum = minimum.min(position);
        maximum = maximum.max(position);
    }

    if !minimum.is_finite() {
        minimum = Vec3::ZERO;
        maximum = Vec3::ZERO;
    }

    let center = (minimum + maximum) * 0.5;
    let max_extent = (maximum.x - minimum.x)
        .max(maximum.y - minimum.y)
        .max(maximum.z - minimum.z)
        .max(1.0);

    Bounds3 {
        minimum,
        maximum,
        center,
        max_extent,
    }
}

/// Per-axis median of `positions`.
///
/// Unlike `Bounds3::center` (the bounding-box midpoint), this stays inside the
/// actual data even when a handful of outliers push the min/max extents far
/// past where most points live (e.g. a high-cardinality axis such as a raw
/// seed value, where a bounding-box center can land in empty space).
pub fn median_position(positions: &[Vec3]) -> Vec3 {
    if positions.is_empty() {
        return Vec3::ZERO;
    }
    Vec3::new(
        median_axis(positions, 0),
        median_axis(positions, 1),
        median_axis(positions, 2),
    )
}

fn median_axis(positions: &[Vec3], axis: usize) -> f32 {
    let mut values: Vec<f32> = positions.iter().map(|position| position[axis]).collect();
    values.sort_by(|left, right| left.total_cmp(right));
    values[values.len() / 2]
}

pub fn estimate_smallest_axis_gap(positions: impl IntoIterator<Item = Vec3>) -> Option<f32> {
    let mut x_values = Vec::new();
    let mut y_values = Vec::new();
    let mut z_values = Vec::new();

    for position in positions {
        x_values.push(position.x);
        y_values.push(position.y);
        z_values.push(position.z);
    }

    [
        smallest_positive_gap(x_values),
        smallest_positive_gap(y_values),
        smallest_positive_gap(z_values),
    ]
    .into_iter()
    .flatten()
    .min_by(|left, right| left.total_cmp(right))
}

fn smallest_positive_gap(mut values: Vec<f32>) -> Option<f32> {
    values.retain(|value| value.is_finite());
    values.sort_by(|left, right| left.total_cmp(right));
    values.dedup_by(|left, right| (*left - *right).abs() <= f32::EPSILON * 8.0);

    values
        .windows(2)
        .map(|window| window[1] - window[0])
        .filter(|gap| *gap > f32::EPSILON * 8.0)
        .min_by(|left, right| left.total_cmp(right))
}

pub fn normalized_or(vector: Vec3, fallback: Vec3) -> Vec3 {
    if vector.length_squared() > f32::EPSILON {
        vector.normalize()
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_touch_and_the_first_is_centered_on_the_origin() {
        let grid = SlotGrid::pack([
            ([0, 0, 0], Vec3::ONE),
            ([1, 0, 0], Vec3::new(3.0, 1.0, 1.0)),
            ([2, 0, 0], Vec3::ONE),
        ]);
        let x = |rank| grid.center([rank, 0, 0]).x;
        assert_eq!(x(0), 0.0);
        // Half of slot 0, then half of the three-cube slot 1.
        assert_eq!(x(1), 2.0);
        assert_eq!(x(2), 4.0);
    }

    #[test]
    fn a_slot_is_as_wide_as_its_widest_group() {
        let grid = SlotGrid::pack([
            ([0, 0, 0], Vec3::ONE),
            ([0, 1, 0], Vec3::new(5.0, 1.0, 1.0)),
            ([1, 0, 0], Vec3::ONE),
        ]);
        // Slot 0 on X holds a one-cube and a five-cube group.
        assert_eq!(grid.center([1, 0, 0]).x, 3.0);
        assert_eq!(grid.center([1, 1, 0]).y, 1.0);
    }

    #[test]
    fn an_unoccupied_rank_keeps_one_cube() {
        let grid = SlotGrid::pack([([0, 0, 0], Vec3::ONE), ([2, 0, 0], Vec3::ONE)]);
        assert_eq!(grid.center([2, 0, 0]).x, 2.0);
    }

    #[test]
    fn the_span_covers_every_slot_and_ranks_map_back_to_slots() {
        let grid = SlotGrid::pack([
            ([0, 0, 0], Vec3::ONE),
            ([1, 0, 0], Vec3::new(3.0, 1.0, 1.0)),
        ]);
        let (low, high) = grid.span();
        assert_eq!(low.x, -0.5);
        assert_eq!(high.x, 3.5);
        assert_eq!(grid.rank_at(0, 0.2), 0);
        assert_eq!(grid.rank_at(0, 0.6), 1);
        assert_eq!(grid.rank_at(0, 3.4), 1);
        assert_eq!(grid.rank_at(0, 3.6), 2);
        assert_eq!(grid.rank_at(0, -0.7), -1);
        assert_eq!(grid.rank_at(0, -1.7), -2);
    }

    #[test]
    fn block_extent_is_symmetric_about_the_center() {
        let extent = block_extent([
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ]);
        assert_eq!(extent, Vec3::new(2.0, 3.0, 1.0));
        assert_eq!(block_extent(std::iter::empty()), Vec3::ONE);
    }

    #[test]
    fn bounds_center_points() {
        let bounds = bounds_of([Vec3::new(0.0, 2.0, 4.0), Vec3::new(10.0, 6.0, 8.0)]);
        assert_eq!(bounds.center, Vec3::new(5.0, 4.0, 6.0));
        assert_eq!(bounds.max_extent, 10.0);
    }

    #[test]
    fn bounds_of_nothing_is_the_origin() {
        let bounds = bounds_of(std::iter::empty());
        assert_eq!(bounds.minimum, Vec3::ZERO);
        assert_eq!(bounds.maximum, Vec3::ZERO);
    }

    #[test]
    fn estimate_smallest_axis_gap_uses_occupied_coordinate_spacing() {
        let gap = estimate_smallest_axis_gap([
            Vec3::ZERO,
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(6.8, 0.0, 0.0),
        ])
        .expect("gap should be detected");
        assert!((gap - 0.8).abs() < 0.0001);
    }

    #[test]
    fn estimate_smallest_axis_gap_returns_none_for_single_point() {
        assert_eq!(estimate_smallest_axis_gap([Vec3::ZERO]), None);
    }
}
