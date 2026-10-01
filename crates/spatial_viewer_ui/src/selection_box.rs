//! The rectangle a box selection is dragged out as.

use bevy::prelude::*;

const FILL_COLOR: Color = Color::srgba(1.0, 0.86, 0.24, 0.10);
const EDGE_COLOR: Color = Color::srgba(1.0, 0.86, 0.24, 0.85);

/// The box being dragged out, in logical window pixels; set by the viewer.
#[derive(Resource, Default)]
pub struct SelectionBox {
    pub rect: Option<Rect>,
}

#[derive(Component)]
pub struct SelectionBoxNode;

pub fn spawn_selection_box(commands: &mut Commands) {
    commands.spawn((
        Node {
            display: Display::None,
            position_type: PositionType::Absolute,
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(FILL_COLOR),
        BorderColor(EDGE_COLOR),
        SelectionBoxNode,
    ));
}

pub fn update_selection_box(
    selection_box: Res<SelectionBox>,
    mut nodes: Query<&mut Node, With<SelectionBoxNode>>,
) {
    if !selection_box.is_changed() {
        return;
    }
    for mut node in &mut nodes {
        match selection_box.rect {
            Some(rect) => {
                node.display = Display::Flex;
                node.left = Val::Px(rect.min.x);
                node.top = Val::Px(rect.min.y);
                node.width = Val::Px(rect.width());
                node.height = Val::Px(rect.height());
            }
            None => node.display = Display::None,
        }
    }
}
