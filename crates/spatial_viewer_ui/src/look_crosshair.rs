//! The faint crosshair that stands in for the pointer while the view is
//! dragged: the pointer is parked at the center of the view then, so what
//! lies under the crosshair is what the pointer hovers.

use bevy::prelude::*;

const ARM_LENGTH: f32 = 18.0;
const ARM_THICKNESS: f32 = 2.0;
const ARM_COLOR: Color = Color::srgba(1.0, 1.0, 1.0, 0.45);
/// A thin dark edge, so the crosshair still reads over bright images.
const EDGE_COLOR: Color = Color::srgba(0.0, 0.0, 0.0, 0.3);

/// The crosshair's root node, hidden until a view drag shows it.
#[derive(Component)]
pub struct LookCrosshair;

pub fn spawn_look_crosshair(commands: &mut Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            Visibility::Hidden,
            LookCrosshair,
        ))
        .with_children(|root| {
            // Both arms centered on one point: a zero-size anchor at the
            // center of the view, each arm offset by half its size.
            root.spawn(Node {
                width: Val::Px(0.0),
                height: Val::Px(0.0),
                ..default()
            })
            .with_children(|anchor| {
                for (width, height) in [(ARM_LENGTH, ARM_THICKNESS), (ARM_THICKNESS, ARM_LENGTH)] {
                    anchor.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(-width * 0.5),
                            top: Val::Px(-height * 0.5),
                            width: Val::Px(width),
                            height: Val::Px(height),
                            border: UiRect::all(Val::Px(0.5)),
                            ..default()
                        },
                        BackgroundColor(ARM_COLOR),
                        BorderColor(EDGE_COLOR),
                    ));
                }
            });
        });
}
