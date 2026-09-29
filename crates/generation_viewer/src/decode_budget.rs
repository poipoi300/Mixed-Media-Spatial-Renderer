use bevy::prelude::*;

/// How fast an unused budget shrinks back toward demand, in decode slots per
/// second. Growth is instant so a burst of demand is never throttled; decay
/// is slow so a brief lull doesn't tear down capacity that's about to be
/// needed again.
const DECODE_BUDGET_DECAY_PER_SECOND: f32 = 0.5;
const MIN_DECODE_BUDGET: usize = 2;

/// Global ceiling on concurrent background image decodes. Video playback
/// streams on threads of its own and draws nothing from it.
#[derive(Resource)]
pub struct DecodeBudget {
    /// Fractional so decay is framerate-independent; truncation gives the
    /// usable limit.
    budget: f32,
    max_budget: usize,
    /// Slots consumed by in-flight work plus grants made this frame.
    used: usize,
}

impl DecodeBudget {
    pub fn new(max_budget: usize) -> Self {
        Self {
            budget: MIN_DECODE_BUDGET as f32,
            max_budget: max_budget.max(MIN_DECODE_BUDGET),
            used: 0,
        }
    }

    /// Adjusts the limit toward `demand` (instant growth, slow decay) and
    /// records how many slots in-flight work already holds. Called once per
    /// frame before any scheduler draws an allowance.
    pub fn update(&mut self, demand: usize, in_flight: usize, delta_seconds: f32) {
        let target = demand.clamp(MIN_DECODE_BUDGET, self.max_budget) as f32;
        if target >= self.budget {
            self.budget = target;
        } else {
            self.budget =
                (self.budget - DECODE_BUDGET_DECAY_PER_SECOND * delta_seconds.max(0.0)).max(target);
        }
        self.used = in_flight;
    }

    pub fn limit(&self) -> usize {
        self.budget as usize
    }

    /// Slots still grantable this frame, so a caller can size a batch to the
    /// budget instead of discovering exhaustion one `take_allowance` at a
    /// time.
    pub fn remaining(&self) -> usize {
        self.limit().saturating_sub(self.used)
    }

    /// Grants up to `want` decode slots from what remains this frame.
    pub fn take_allowance(&mut self, want: usize) -> usize {
        let granted = want.min(self.limit().saturating_sub(self.used));
        self.used += granted;
        granted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_grows_instantly_and_decays_slowly() {
        let mut budget = DecodeBudget::new(20);
        budget.update(10, 0, 0.016);
        assert_eq!(budget.limit(), 10);

        budget.update(2, 0, 1.0);
        assert_eq!(budget.limit(), 9);

        for _ in 0..30 {
            budget.update(2, 0, 1.0);
        }
        assert_eq!(budget.limit(), 2);
    }

    #[test]
    fn demand_is_clamped_to_configured_maximum() {
        let mut budget = DecodeBudget::new(8);
        budget.update(100, 0, 0.016);
        assert_eq!(budget.limit(), 8);
    }

    #[test]
    fn allowance_is_bounded_by_limit_minus_in_flight() {
        let mut budget = DecodeBudget::new(20);
        budget.update(6, 4, 0.016);
        assert_eq!(budget.remaining(), 2);
        assert_eq!(budget.take_allowance(5), 2);
        assert_eq!(budget.remaining(), 0);
        assert_eq!(budget.take_allowance(1), 0);
    }
}
