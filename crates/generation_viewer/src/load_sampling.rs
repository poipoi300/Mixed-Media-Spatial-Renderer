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
//! The score is a Gaussian in the distance from the camera, measured from
//! whichever of the current and projected camera position is nearer (so an
//! image being approached scores as though already close), multiplied by a
//! view-direction factor. Its thin tails are the point: a billboard far
//! outside the view is not forbidden, just drawn with vanishing probability,
//! so no cap is needed to keep the workers pointed at what matters.

use bevy::prelude::*;

/// Standard deviation of the selection Gaussian, in scene reference
/// distances.
///
/// Sized so the full-quality band (`BILLBOARD_FULL_RES_DISTANCE_FACTOR`, 5
/// reference distances) sits at two sigma. A Gaussian only concentrates
/// probability where its tail is genuinely thin: at one sigma per band the
/// distribution is so flat that a handful of distant candidates outweigh the
/// near ones by sheer count, which measured as a near-field share of ~60% —
/// not a priority at all. At two sigma per band the same arrangement gives
/// the near field ~95%, while everything further out keeps a small but real
/// probability instead of being capped away.
const LOAD_VALUE_SIGMA_STEPS: f32 = 2.5;
/// Weight retained by a candidate directly behind the camera, relative to one
/// straight ahead at the same distance. Not zero: turning around must not
/// find an empty cache, and a point beside the camera is one flick of the
/// mouse from being centre-screen.
const LOAD_VALUE_BEHIND_WEIGHT: f32 = 0.12;

/// How much of the camera's view a candidate occupies, as a weight in
/// `(0, 1]`. This is the "p score": one number that folds together distance,
/// where the camera is heading, and where it is looking.
pub fn load_value(position: Vec3, view_position: Vec3, forward: Vec3, sigma: f32) -> f32 {
    let offset = position - view_position;
    let distance = offset.length();
    let sigma = sigma.max(f32::EPSILON);
    let proximity = (-0.5 * (distance / sigma).powi(2)).exp();
    proximity * direction_weight(offset, distance, forward)
}

/// Falls from 1.0 straight ahead to [`LOAD_VALUE_BEHIND_WEIGHT`] straight
/// behind, following the cosine of the angle to the view axis so the
/// transition through the edge of the screen is smooth rather than a cliff.
fn direction_weight(offset: Vec3, distance: f32, forward: Vec3) -> f32 {
    if distance <= f32::EPSILON {
        return 1.0;
    }
    let alignment = (offset / distance).dot(forward).clamp(-1.0, 1.0);
    let ahead = (alignment + 1.0) * 0.5;
    LOAD_VALUE_BEHIND_WEIGHT + (1.0 - LOAD_VALUE_BEHIND_WEIGHT) * ahead
}

/// Selection sigma in world units for the current scene scale.
pub fn load_value_sigma(reference_distance: f32) -> f32 {
    reference_distance.max(1.0) * LOAD_VALUE_SIGMA_STEPS
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
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
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
    fn next_unit(&mut self) -> f32 {
        let bits = self.next_u64() >> 40; // 24 bits of mantissa
        (bits as f32 + 0.5) / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGMA: f32 = 50.0;

    #[test]
    fn value_peaks_at_the_camera_and_decays_with_distance() {
        let forward = Vec3::NEG_Z;
        let at_camera = load_value(Vec3::ZERO, Vec3::ZERO, forward, SIGMA);
        let near = load_value(Vec3::NEG_Z * 25.0, Vec3::ZERO, forward, SIGMA);
        let far = load_value(Vec3::NEG_Z * 200.0, Vec3::ZERO, forward, SIGMA);

        assert!(at_camera > near);
        assert!(near > far);
        assert!(far > 0.0, "no candidate is ever forbidden outright");
        assert!(
            far < near / 1_000.0,
            "the tail must be thin enough that far points are effectively never drawn"
        );
    }

    #[test]
    fn points_ahead_outweigh_points_behind_at_equal_distance() {
        let forward = Vec3::NEG_Z;
        let ahead = load_value(Vec3::NEG_Z * 40.0, Vec3::ZERO, forward, SIGMA);
        let beside = load_value(Vec3::X * 40.0, Vec3::ZERO, forward, SIGMA);
        let behind = load_value(Vec3::Z * 40.0, Vec3::ZERO, forward, SIGMA);

        assert!(ahead > beside);
        assert!(beside > behind);
        assert!(
            behind > 0.0,
            "turning around must not find an empty cache, so behind is down-weighted, not banned"
        );
    }

    #[test]
    fn sigma_tracks_scene_scale() {
        assert_eq!(load_value_sigma(10.0), 10.0 * LOAD_VALUE_SIGMA_STEPS);
        assert_eq!(load_value_sigma(40.0), 40.0 * LOAD_VALUE_SIGMA_STEPS);
        // A degenerate scene scale still yields a usable sigma rather than
        // collapsing the sample to the camera's exact position.
        assert!(load_value_sigma(0.0) > 0.0);
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
    fn rng_stays_inside_the_open_unit_interval() {
        let mut rng = Rng::new(0);
        for _ in 0..10_000 {
            let value = rng.next_unit();
            assert!(value > 0.0 && value < 1.0, "{value} left (0, 1)");
        }
    }
}
