use bevy_math::Vec3;
use generation_api::ProjectionPoint;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds3 {
    pub minimum: Vec3,
    pub maximum: Vec3,
    pub center: Vec3,
    pub max_extent: f32,
}

pub fn api_position_to_viewer(position: [f32; 3]) -> Vec3 {
    Vec3::from_array(position)
}

pub fn projection_bounds(points: &[ProjectionPoint]) -> Bounds3 {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);

    for point in points {
        let position = api_position_to_viewer(point.position);
        minimum = minimum.min(position);
        maximum = maximum.max(position);
    }

    if points.is_empty() {
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

/// Per-axis median position of the loaded points.
///
/// Unlike `Bounds3::center` (the bounding-box midpoint), this stays inside the
/// actual data even when a handful of outliers push the min/max extents far
/// past where most points live (e.g. a high-cardinality axis such as a raw
/// seed value, where a bounding-box center can land in empty space).
pub fn median_position(points: &[ProjectionPoint]) -> Vec3 {
    if points.is_empty() {
        return Vec3::ZERO;
    }
    Vec3::new(
        median_axis(points, 0),
        median_axis(points, 1),
        median_axis(points, 2),
    )
}

fn median_axis(points: &[ProjectionPoint], axis: usize) -> f32 {
    let mut values: Vec<f32> = points.iter().map(|point| point.position[axis]).collect();
    values.sort_by(|left, right| left.total_cmp(right));
    values[values.len() / 2]
}

pub fn estimate_smallest_axis_gap(points: &[ProjectionPoint]) -> Option<f32> {
    let mut x_values = Vec::with_capacity(points.len());
    let mut y_values = Vec::with_capacity(points.len());
    let mut z_values = Vec::with_capacity(points.len());

    for point in points {
        let position = api_position_to_viewer(point.position);
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
    fn projection_bounds_centers_points() {
        let points = vec![
            ProjectionPoint {
                image_id: 0,
                path: "image-0.png".to_owned(),
                position: [0.0, 2.0, 4.0],
                width: None,
                height: None,
                media_type: "image".to_owned(),
                duration_seconds: None,
                coordinate_labels: [None, None, None],
            },
            ProjectionPoint {
                image_id: 1,
                path: "image-1.png".to_owned(),
                position: [10.0, 6.0, 8.0],
                width: None,
                height: None,
                media_type: "image".to_owned(),
                duration_seconds: None,
                coordinate_labels: [None, None, None],
            },
        ];
        let bounds = projection_bounds(&points);
        assert_eq!(bounds.center, Vec3::new(5.0, 4.0, 6.0));
        assert_eq!(bounds.max_extent, 10.0);
    }

    #[test]
    fn estimate_smallest_axis_gap_uses_occupied_coordinate_spacing() {
        let points = vec![
            ProjectionPoint {
                image_id: 0,
                path: "image-0.png".to_owned(),
                position: [0.0, 0.0, 0.0],
                width: None,
                height: None,
                media_type: "image".to_owned(),
                duration_seconds: None,
                coordinate_labels: [None, None, None],
            },
            ProjectionPoint {
                image_id: 1,
                path: "image-1.png".to_owned(),
                position: [6.0, 0.0, 0.0],
                width: None,
                height: None,
                media_type: "image".to_owned(),
                duration_seconds: None,
                coordinate_labels: [None, None, None],
            },
            ProjectionPoint {
                image_id: 2,
                path: "image-2.png".to_owned(),
                position: [6.8, 0.0, 0.0],
                width: None,
                height: None,
                media_type: "image".to_owned(),
                duration_seconds: None,
                coordinate_labels: [None, None, None],
            },
        ];

        let gap = estimate_smallest_axis_gap(&points).expect("gap should be detected");
        assert!((gap - 0.8).abs() < 0.0001);
    }

    #[test]
    fn estimate_smallest_axis_gap_returns_none_for_single_point() {
        let points = vec![ProjectionPoint {
            image_id: 0,
            path: "image-0.png".to_owned(),
            position: [0.0, 0.0, 0.0],
            width: None,
            height: None,
            media_type: "image".to_owned(),
            duration_seconds: None,
            coordinate_labels: [None, None, None],
        }];

        assert_eq!(estimate_smallest_axis_gap(&points), None);
    }
}
