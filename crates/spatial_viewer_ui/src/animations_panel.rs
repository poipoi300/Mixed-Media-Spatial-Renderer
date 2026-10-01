//! The animations screen of the pause menu: one switch that turns every
//! animation off, a duration scale for all of them, and a switch for each.
//! Only transitions the viewer plays by itself count as animations; the
//! view's own movement, and billboards turning to follow it, are not.

use std::collections::HashSet;

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use crate::{
    menu_panel, set_button_color, set_disabled_button_color, spawn_checkbox_row_with_label,
    spawn_menu_button, spawn_menu_title, spawn_settings_slider, ButtonInteractionQuery,
    UiButtonColorQuery, UiTextQuery, ViewerUiButton, ViewerUiPanel, ViewerUiText,
};

const MIN_DURATION_SCALE: f32 = 0.25;
const MAX_DURATION_SCALE: f32 = 4.0;
/// Slider positions snap the scale to multiples of this.
const DURATION_SCALE_STEP: f32 = 0.05;
/// Opacity of the options "Disable animations" leaves without effect.
const DIMMED_ALPHA: f32 = 0.35;

/// An animation the viewer can play or skip. Skipped, whatever it would
/// have moved is simply where it ends up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Animation {
    /// An image or closed folder under the pointer grows, and shrinks back
    /// when left.
    HoverGrowth,
    /// Images, with their coordinates, slide to where a change of layout
    /// puts them: making room for an opening folder, undo, a drop.
    Rearranging,
    /// Images slide out of an opening folder, fanning out, and back into a
    /// closing one.
    FolderSlide,
    /// A folder's cube grows or shrinks as it opens or closes, and moves
    /// and fades to its new state.
    FolderCubes,
    /// The view flies to what it fits (`F`, search) instead of jumping.
    CameraFlights,
    /// A video's control strip follows the part of the video on screen;
    /// without it, the strip stays put along the video's bottom edge.
    VideoStripFollowing,
}

impl Animation {
    pub const ALL: [Self; 6] = [
        Self::HoverGrowth,
        Self::Rearranging,
        Self::FolderSlide,
        Self::FolderCubes,
        Self::CameraFlights,
        Self::VideoStripFollowing,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::HoverGrowth => "Hover growth",
            Self::Rearranging => "Rearranging images",
            Self::FolderSlide => "Folder slide in/out",
            Self::FolderCubes => "Folders opening and closing",
            Self::CameraFlights => "Camera flights (fit, search)",
            Self::VideoStripFollowing => "Video strip follows the view",
        }
    }

    /// The name it is saved under, which stays the same when labels change.
    pub fn key(self) -> &'static str {
        match self {
            Self::HoverGrowth => "hover_growth",
            Self::Rearranging => "rearranging",
            Self::FolderSlide => "folder_slide",
            Self::FolderCubes => "folder_cubes",
            Self::CameraFlights => "camera_flights",
            Self::VideoStripFollowing => "video_strip_following",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|animation| animation.key() == key)
    }
}

/// Which animations play, and how long they take.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct AnimationSettings {
    /// "Disable animations": none plays, whatever the others say.
    pub all_disabled: bool,
    /// Every animation takes this many times as long: above 1 slower.
    duration_scale: f32,
    turned_off: HashSet<Animation>,
}

impl Default for AnimationSettings {
    fn default() -> Self {
        Self::new(false, 1.0, [])
    }
}

impl AnimationSettings {
    pub fn new(
        all_disabled: bool,
        duration_scale: f32,
        turned_off: impl IntoIterator<Item = Animation>,
    ) -> Self {
        Self {
            all_disabled,
            duration_scale: clamp_duration_scale(duration_scale),
            turned_off: turned_off.into_iter().collect(),
        }
    }

    /// The animation plays: its own switch is on, and animations are not
    /// all disabled.
    pub fn plays(&self, animation: Animation) -> bool {
        !self.all_disabled && self.is_turned_on(animation)
    }

    /// The animation's own switch, which "Disable animations" overrides.
    pub fn is_turned_on(&self, animation: Animation) -> bool {
        !self.turned_off.contains(&animation)
    }

    pub fn toggle(&mut self, animation: Animation) {
        if !self.turned_off.remove(&animation) {
            self.turned_off.insert(animation);
        }
    }

    /// The animations switched off one by one, in [`Animation::ALL`] order.
    pub fn turned_off(&self) -> impl Iterator<Item = Animation> + '_ {
        Animation::ALL
            .into_iter()
            .filter(|animation| self.turned_off.contains(animation))
    }

    pub fn duration_scale(&self) -> f32 {
        self.duration_scale
    }

    /// How far animations advance in `real_seconds`.
    pub fn animation_seconds(&self, real_seconds: f32) -> f32 {
        real_seconds / self.duration_scale
    }

    /// The scale at a slider position: logarithmic, so 1 sits in the middle
    /// and halving takes the same travel as doubling.
    fn set_duration_scale_from_slider(&mut self, position: f32) {
        let range = MAX_DURATION_SCALE / MIN_DURATION_SCALE;
        let scale = MIN_DURATION_SCALE * range.powf(position.clamp(0.0, 1.0));
        self.duration_scale =
            clamp_duration_scale((scale / DURATION_SCALE_STEP).round() * DURATION_SCALE_STEP);
    }

    fn slider_percent(&self) -> f32 {
        let range = MAX_DURATION_SCALE / MIN_DURATION_SCALE;
        (self.duration_scale / MIN_DURATION_SCALE).ln() / range.ln() * 100.0
    }

    fn duration_scale_label(&self) -> String {
        let scale = self.duration_scale;
        let pace = if scale > 1.0 {
            " (slower)"
        } else if scale < 1.0 {
            " (faster)"
        } else {
            ""
        };
        format!("Animation scale {scale:.2}x{pace}")
    }
}

fn clamp_duration_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(MIN_DURATION_SCALE, MAX_DURATION_SCALE)
    } else {
        1.0
    }
}

/// Text that dims while "Disable animations" leaves it without effect.
#[derive(Component, Clone, Copy)]
pub struct DimmedWhileAnimationsDisabled;

/// Every text the animations screen may dim: its [`ViewerUiText`]s and the
/// switches' labels.
type DimmableTextQuery<'w, 's> = Query<
    'w,
    's,
    (&'static mut TextColor, Option<&'static ViewerUiText>),
    Or<(With<ViewerUiText>, With<DimmedWhileAnimationsDisabled>)>,
>;

pub(crate) fn spawn_animations_screen(overlay: &mut ChildBuilder) {
    overlay
        .spawn(menu_panel(ViewerUiPanel::PauseAnimations, 340.0))
        .with_children(|screen| {
            spawn_menu_title(screen, "Animations");
            spawn_checkbox_row_with_label(
                screen,
                ViewerUiButton::ToggleAnimationsDisabled,
                ViewerUiText::AnimationsDisabledMark,
                "Disable animations",
                (),
            );
            spawn_settings_slider(
                screen,
                ViewerUiText::AnimationScaleSummary,
                ViewerUiButton::SetAnimationScaleFromSlider,
                ViewerUiPanel::AnimationScaleFill,
            );
            for animation in Animation::ALL {
                spawn_checkbox_row_with_label(
                    screen,
                    ViewerUiButton::ToggleAnimation(animation),
                    ViewerUiText::AnimationMark(animation),
                    animation.label(),
                    DimmedWhileAnimationsDisabled,
                );
            }
            spawn_menu_button(screen, ViewerUiButton::PauseBack, "Back");
        });
}

/// The switches, and the scale slider while it is held. The options
/// "Disable animations" overrides take no input while it is on.
pub fn handle_animation_buttons(
    mut settings: ResMut<AnimationSettings>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    interaction_query: ButtonInteractionQuery,
    slider_query: Query<(&Interaction, &ViewerUiButton, &RelativeCursorPosition), With<Button>>,
) {
    for (interaction, button) in &interaction_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match *button {
            ViewerUiButton::ToggleAnimationsDisabled => {
                settings.all_disabled = !settings.all_disabled;
            }
            ViewerUiButton::ToggleAnimation(animation) if !settings.all_disabled => {
                settings.toggle(animation);
            }
            _ => {}
        }
    }
    if settings.all_disabled || !mouse_buttons.pressed(MouseButton::Left) {
        return;
    }
    for (interaction, button, cursor_position) in &slider_query {
        if *interaction != Interaction::Pressed
            || !matches!(button, ViewerUiButton::SetAnimationScaleFromSlider)
        {
            continue;
        }
        if let Some(position) = cursor_position.normalized {
            settings.set_duration_scale_from_slider(position.x);
        }
    }
}

pub fn update_animation_text(settings: Res<AnimationSettings>, mut text_query: UiTextQuery) {
    let mark = |checked: bool| if checked { "X" } else { "" }.to_owned();
    for (mut text, text_kind) in &mut text_query {
        match *text_kind {
            ViewerUiText::AnimationsDisabledMark => **text = mark(settings.all_disabled),
            ViewerUiText::AnimationScaleSummary => **text = settings.duration_scale_label(),
            ViewerUiText::AnimationMark(animation) => {
                **text = mark(settings.is_turned_on(animation));
            }
            _ => {}
        }
    }
}

/// Sizes the scale slider's fill and dims what "Disable animations"
/// overrides.
pub fn update_animation_panels(
    settings: Res<AnimationSettings>,
    mut panels: Query<(&ViewerUiPanel, &mut Node, &mut BackgroundColor)>,
    mut texts: DimmableTextQuery,
) {
    let alpha = if settings.all_disabled {
        DIMMED_ALPHA
    } else {
        1.0
    };
    for (panel, mut node, mut color) in &mut panels {
        if let ViewerUiPanel::AnimationScaleFill = panel {
            node.width = Val::Percent(settings.slider_percent());
            color.0.set_alpha(alpha);
        }
    }
    for (mut color, text_kind) in &mut texts {
        // Unmarked texts are the switches' labels.
        let overridden = text_kind.is_none_or(|text_kind| {
            matches!(
                text_kind,
                ViewerUiText::AnimationScaleSummary | ViewerUiText::AnimationMark(_)
            )
        });
        if overridden && color.0.alpha() != alpha {
            color.0.set_alpha(alpha);
        }
    }
}

pub fn update_animation_button_colors(
    settings: Res<AnimationSettings>,
    mut button_query: UiButtonColorQuery,
) {
    for (button, interaction, color) in &mut button_query {
        match *button {
            ViewerUiButton::ToggleAnimationsDisabled => {
                set_button_color(color, settings.all_disabled, interaction);
            }
            ViewerUiButton::ToggleAnimation(_) | ViewerUiButton::SetAnimationScaleFromSlider
                if settings.all_disabled =>
            {
                set_disabled_button_color(color);
            }
            ViewerUiButton::ToggleAnimation(animation) => {
                set_button_color(color, settings.is_turned_on(animation), interaction);
            }
            ViewerUiButton::SetAnimationScaleFromSlider => {
                set_button_color(color, false, interaction);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabling_all_overrides_each_switch_but_keeps_it() {
        let mut settings = AnimationSettings::default();
        settings.toggle(Animation::CameraFlights);
        assert!(!settings.plays(Animation::CameraFlights));
        assert!(settings.plays(Animation::HoverGrowth));

        settings.all_disabled = true;
        assert!(Animation::ALL
            .into_iter()
            .all(|animation| !settings.plays(animation)));
        assert!(settings.is_turned_on(Animation::HoverGrowth));

        settings.all_disabled = false;
        assert!(settings.plays(Animation::HoverGrowth));
        assert_eq!(
            settings.turned_off().collect::<Vec<_>>(),
            [Animation::CameraFlights]
        );
    }

    #[test]
    fn the_slider_is_logarithmic_with_one_in_the_middle() {
        let mut settings = AnimationSettings::default();
        assert!((settings.slider_percent() - 50.0).abs() < 1e-3);
        settings.set_duration_scale_from_slider(0.5);
        assert_eq!(settings.duration_scale(), 1.0);
        settings.set_duration_scale_from_slider(0.0);
        assert_eq!(settings.duration_scale(), MIN_DURATION_SCALE);
        settings.set_duration_scale_from_slider(1.0);
        assert_eq!(settings.duration_scale(), MAX_DURATION_SCALE);
        assert_eq!(settings.animation_seconds(1.0), 0.25);
    }

    #[test]
    fn saved_names_round_trip_and_out_of_range_scales_clamp() {
        for animation in Animation::ALL {
            assert_eq!(Animation::from_key(animation.key()), Some(animation));
        }
        assert_eq!(Animation::from_key("gone"), None);
        assert_eq!(
            AnimationSettings::new(false, 100.0, []).duration_scale(),
            MAX_DURATION_SCALE
        );
        assert_eq!(
            AnimationSettings::new(false, f32::NAN, []).duration_scale(),
            1.0
        );
    }
}
