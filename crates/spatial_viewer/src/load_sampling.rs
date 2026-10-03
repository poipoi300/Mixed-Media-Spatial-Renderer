//! Probabilistic selection of which billboard to decode next.
//!
//! The scheduler used to hand each lane (first loads, nearby quality
//! refreshes, distant quality refreshes) its own slice of the worker pool and
//! reserve slots a lane had found work for but could not yet start. Every
//! ration is a guess, the reservations interact, and the failure mode is the
//! worst one available: workers sitting idle while work exists, because the
//! lane holding the slots was not the lane with the candidates.
//!
//! Instead, every candidate — a point that has never been decoded and a
//! resident whose texture no longer matches its distance alike — is scored by
//! the same function and drawn from one weighted sample. There are no lanes
//! and no reservations, so a free worker is always filled if any candidate
//! exists, and the *mix* of work emerges from the scores rather than from a
//! constant someone tuned.
//!
//! The score is a utility: how much showing a billboard is worth, from its
//! distance and from where it sits relative to the view direction, measured
//! from whichever of the current and projected camera position is nearer (so
//! an image being approached scores as though already close). It falls off
//! as a power of distance rather than a Gaussian, so a billboard the camera
//! is looking at keeps a real chance however far away it is, while one
//! behind the camera counts as several times farther than it is.

use bevy::prelude::*;

/// Power the utility falls off with, in distance.
///
/// The number of billboards at a given distance inside the view grows with
/// the square of that distance, so the total weight of everything at one
/// distance falls only when each billboard's weight falls faster than the
/// square. A cube is the smallest whole power that does, which keeps near
/// billboards first without starving far ones the way a Gaussian tail did.
const UTILITY_DISTANCE_EXPONENT: i32 = 4;
/// How many times farther a billboard directly behind the camera counts than
/// one at the same distance straight ahead. Not infinite: turning around must
/// not find an empty cache, and a point beside the camera is one flick of the
/// mouse from being centre-screen.
const UTILITY_BEHIND_STRETCH: f32 = 8.0;

/// What showing the billboard at `offset` from the camera is worth, in
/// `(0, 1]`: 1 straight ahead within one `reference_distance`, then falling
/// with distance, faster the further the billboard sits from the view
/// direction.
pub fn load_utility(offset: Vec3, forward: Vec3, reference_distance: f32) -> f32 {
    let distance = offset.length();
    let alignment = if distance <= f32::EPSILON {
        1.0
    } else {
        (offset / distance).dot(forward)
    };
    load_utility_at(distance, alignment, reference_distance)
}

/// [`load_utility`] from a distance and the cosine of the angle to the view
/// direction. Increasing in `alignment` and decreasing in `distance`, so
/// evaluating it at a region's nearest distance and best alignment bounds
/// every billboard inside that region.
pub fn load_utility_at(distance: f32, alignment: f32, reference_distance: f32) -> f32 {
    let away = (1.0 - alignment.clamp(-1.0, 1.0)) * 0.5;
    let stretch = 1.0 + (UTILITY_BEHIND_STRETCH - 1.0) * away;
    // The stretch applies after the near clamp, so even a billboard
    // touching the camera ranks below one in front of it.
    let widths = (distance / reference_distance.max(1.0)).max(1.0) * stretch;
    widths.powi(-UTILITY_DISTANCE_EXPONENT)
}

/// Half-angle of the cone ahead of the camera that [`cone_probes`] samples.
/// Wider than a typical horizontal field of view, so a small turn does not
/// leave what is now on screen unsampled.
const PROBE_CONE_HALF_ANGLE: f32 = std::f32::consts::FRAC_PI_3;

/// A position drawn ahead of the camera, with the probability density per
/// unit volume it was drawn at.
pub struct ConeProbe {
    pub position: Vec3,
    pub density: f32,
}

/// Draws `count` positions in the cone ahead of `origin`, uniform over
/// directions in the cone and log-uniform in distance between `near` and
/// `far`.
///
/// Log-uniform distance spreads probes as `1/r` per distance, over a shell
/// area growing as `r^2`, so the density per unit volume falls as `1/r^3` —
/// the same falloff as [`load_utility`] straight ahead. Regions are probed
/// in proportion to what they are worth, which is what lets a fixed number
/// of probes reach the whole depth of the view instead of the nearest cells.
pub fn cone_probes(
    origin: Vec3,
    forward: Vec3,
    near: f32,
    far: f32,
    count: usize,
    rng: &mut Rng,
) -> Vec<ConeProbe> {
    let near = near.max(f32::EPSILON);
    if far <= near {
        return Vec::new();
    }
    let (across, up) = forward.any_orthonormal_pair();
    let cos_limit = PROBE_CONE_HALF_ANGLE.cos();
    let log_span = (far / near).ln();
    let solid_angle = std::f32::consts::TAU * (1.0 - cos_limit);
    (0..count)
        .map(|_| {
            let cos_angle = cos_limit + (1.0 - cos_limit) * rng.next_unit();
            let sin_angle = (1.0 - cos_angle * cos_angle).max(0.0).sqrt();
            let around = std::f32::consts::TAU * rng.next_unit();
            let direction =
                forward * cos_angle + (across * around.cos() + up * around.sin()) * sin_angle;
            let distance = near * (log_span * rng.next_unit()).exp();
            ConeProbe {
                position: origin + direction * distance,
                density: 1.0 / (distance.powi(3) * log_span * solid_angle),
            }
        })
        .collect()
}

/// Draws up to `capacity` items without replacement, each with probability
/// proportional to its weight, in a single pass over a stream of unknown
/// length.
///
/// This is the A-Res weighted reservoir algorithm: an item of weight `w`
/// receives key `u^(1/w)` for `u` uniform in `(0, 1)`, and the `k` largest
/// keys are exactly a weighted sample without replacement. One pass matters
/// because candidates are produced by walking a spatial grid — materializing
/// and sorting them all would reintroduce the per-frame cost the sampler
/// exists to avoid.
pub struct WeightedReservoir<T> {
    capacity: usize,
    /// Selected items with their keys, smallest key first so the item at risk
    /// of eviction is always at index 0.
    entries: Vec<(f32, T)>,
    rng: Rng,
    seen: usize,
}

impl<T> WeightedReservoir<T> {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            rng: Rng::new(seed),
            seen: 0,
        }
    }

    /// Offers one candidate. Zero or negative weights are ignored outright,
    /// so a candidate the score function rejected can never be drawn.
    pub fn offer(&mut self, weight: f32, item: T) {
        // NaN and negative weights are both rejected here: a candidate the
        // score function refused must never enter the draw.
        if !weight.is_finite() || weight <= 0.0 || self.capacity == 0 {
            return;
        }
        self.seen += 1;
        let key = self.rng.next_unit().powf(1.0 / weight);
        if self.entries.len() < self.capacity {
            self.insert(key, item);
            return;
        }
        // `entries[0]` holds the smallest key, so this is the only comparison
        // needed to know whether the candidate belongs in the sample.
        if key <= self.entries[0].0 {
            return;
        }
        self.entries.remove(0);
        self.insert(key, item);
    }

    fn insert(&mut self, key: f32, item: T) {
        let position = self
            .entries
            .iter()
            .position(|(existing, _)| *existing > key)
            .unwrap_or(self.entries.len());
        self.entries.insert(position, (key, item));
    }

    /// The drawn sample, highest key first.
    pub fn take(self) -> Vec<T> {
        self.entries
            .into_iter()
            .rev()
            .map(|(_, item)| item)
            .collect()
    }
}

/// Small deterministic PRNG (SplitMix64).
///
/// Seeded per frame from the frame counter so a run is reproducible and a
/// test can assert an exact draw, rather than depending on thread-local
/// entropy that would make selection unrepeatable across runs.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        // A zero state would make SplitMix64 emit a fixed sequence from a
        // degenerate start; the odd constant avoids that without changing
        // the distribution.
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in the open interval `(0, 1)`; never returns exactly 0, whose
    /// `powf` key would collapse every weight to the same value.
    pub(crate) fn next_unit(&mut self) -> f32 {
        let bits = self.next_u64() >> 40; // 24 bits of mantissa
        (bits as f32 + 0.5) / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE: f32 = 10.0;

    #[test]
    fn utility_peaks_at_the_camera_and_decays_with_distance() {
        let forward = Vec3::NEG_Z;
        let at_camera = load_utility(Vec3::ZERO, forward, REFERENCE);
        let near = load_utility(Vec3::NEG_Z * 25.0, forward, REFERENCE);
        let far = load_utility(Vec3::NEG_Z * 200.0, forward, REFERENCE);

        assert_eq!(at_camera, 1.0);
        assert!(near < at_camera);
        assert!(far < near);
        assert!(far > 0.0, "no candidate is ever forbidden outright");
    }

    #[test]
    fn total_weight_per_distance_falls_inside_the_view() {
        // A view-cone shell at distance d holds ~d^2 billboards, so the
        // per-distance total is d^2 times the utility. It must fall with
        // distance, or far billboards would outnumber near ones in the draw.
        let shell_total = |distance: f32| {
            distance * distance * load_utility(Vec3::NEG_Z * distance, Vec3::NEG_Z, REFERENCE)
        };
        let mut previous = shell_total(REFERENCE);
        for step in 2..40 {
            let current = shell_total(REFERENCE * step as f32);
            assert!(current < previous, "shell total rose at {step} widths");
            previous = current;
        }
    }

    #[test]
    fn points_ahead_outweigh_points_beside_and_behind() {
        let forward = Vec3::NEG_Z;
        let ahead = load_utility(Vec3::NEG_Z * 40.0, forward, REFERENCE);
        let beside = load_utility(Vec3::X * 40.0, forward, REFERENCE);
        let behind = load_utility(Vec3::Z * 40.0, forward, REFERENCE);

        assert!(ahead > beside);
        assert!(beside > behind);
        assert!(
            behind > 0.0,
            "turning around must not find an empty cache, so behind is down-weighted, not banned"
        );
        // Behind counts as UTILITY_BEHIND_STRETCH times farther.
        let as_far_ahead = load_utility(
            Vec3::NEG_Z * 40.0 * UTILITY_BEHIND_STRETCH,
            forward,
            REFERENCE,
        );
        assert!((behind - as_far_ahead).abs() <= as_far_ahead * 1e-4);
    }

    #[test]
    fn utility_is_bounded_by_its_value_at_nearest_distance_and_best_alignment() {
        let bound = load_utility_at(30.0, 0.5, REFERENCE);
        for (distance, alignment) in [(30.0, 0.5), (31.0, 0.5), (30.0, 0.2), (60.0, -1.0)] {
            assert!(load_utility_at(distance, alignment, REFERENCE) <= bound);
        }
    }

    #[test]
    fn a_degenerate_scene_scale_still_yields_a_usable_utility() {
        let utility = load_utility(Vec3::NEG_Z * 3.0, Vec3::NEG_Z, 0.0);
        assert!(utility > 0.0 && utility <= 1.0);
    }

    #[test]
    fn reservoir_returns_at_most_its_capacity_and_ignores_zero_weights() {
        let mut reservoir = WeightedReservoir::new(3, 7);
        for index in 0..20 {
            reservoir.offer(1.0, index);
        }
        reservoir.offer(0.0, 999);
        reservoir.offer(-1.0, 998);

        assert_eq!(reservoir.seen, 20, "rejected weights are not counted");
        let drawn = reservoir.take();
        assert_eq!(drawn.len(), 3);
        assert!(!drawn.contains(&999));
        assert!(!drawn.contains(&998));
    }

    #[test]
    fn reservoir_takes_everything_when_offered_fewer_than_capacity() {
        let mut reservoir = WeightedReservoir::new(8, 1);
        for index in 0..3 {
            reservoir.offer(0.5, index);
        }

        let mut drawn = reservoir.take();
        drawn.sort();
        assert_eq!(drawn, vec![0, 1, 2]);
    }

    #[test]
    fn heavier_candidates_are_drawn_far_more_often() {
        // One heavy candidate against many light ones, repeated with
        // different seeds: the heavy one should dominate without ever being
        // guaranteed, which is exactly the property the scheduler relies on.
        let mut heavy_draws = 0;
        let trials = 400;
        for seed in 0..trials {
            let mut reservoir = WeightedReservoir::new(1, seed);
            reservoir.offer(100.0, "heavy");
            for _ in 0..10 {
                reservoir.offer(1.0, "light");
            }
            if reservoir.take() == vec!["heavy"] {
                heavy_draws += 1;
            }
        }

        assert!(
            heavy_draws > trials * 3 / 4,
            "heavy candidate won {heavy_draws}/{trials} draws; expected a large majority"
        );
        assert!(
            heavy_draws < trials,
            "selection must stay probabilistic, not become a maximum"
        );
    }

    #[test]
    fn selection_is_reproducible_for_a_given_seed() {
        let draw = |seed| {
            let mut reservoir = WeightedReservoir::new(4, seed);
            for index in 0..50 {
                reservoir.offer(1.0 + index as f32, index);
            }
            reservoir.take()
        };

        assert_eq!(draw(42), draw(42));
        assert_ne!(draw(42), draw(43));
    }

    #[test]
    fn cone_probes_stay_in_the_cone_and_range_with_the_stated_density() {
        let mut rng = Rng::new(3);
        let forward = Vec3::NEG_Z;
        let probes = cone_probes(Vec3::ZERO, forward, 2.0, 50.0, 2_000, &mut rng);
        assert_eq!(probes.len(), 2_000);
        let cos_limit = PROBE_CONE_HALF_ANGLE.cos();
        for probe in &probes {
            let distance = probe.position.length();
            assert!((2.0..=50.0 + 1e-3).contains(&distance));
            assert!(probe.position.normalize().dot(forward) >= cos_limit - 1e-5);
            let expected = 1.0
                / (distance.powi(3)
                    * (50.0f32 / 2.0).ln()
                    * std::f32::consts::TAU
                    * (1.0 - cos_limit));
            assert!((probe.density - expected).abs() <= expected * 1e-4);
        }
        // Log-uniform: as many probes between 2 and 10 as between 10 and 50.
        let inner = probes
            .iter()
            .filter(|probe| probe.position.length() < 10.0)
            .count();
        assert!((800..1_200).contains(&inner), "{inner} of 2000 inside 10");
    }

    #[test]
    fn cone_probes_need_a_range() {
        let mut rng = Rng::new(3);
        assert!(cone_probes(Vec3::ZERO, Vec3::NEG_Z, 5.0, 5.0, 10, &mut rng).is_empty());
    }

    #[test]
    fn rng_stays_inside_the_open_unit_interval() {
        let mut rng = Rng::new(0);
        for _ in 0..10_000 {
            let value = rng.next_unit();
            assert!(value > 0.0 && value < 1.0, "{value} left (0, 1)");
        }
    }
}
