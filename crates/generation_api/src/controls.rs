//! The control-panel schema: how a server describes the controls and stats
//! the viewer renders in its panel pill.
//!
//! The viewer knows nothing about what a control *means*. A server sends a
//! tree of widgets and a list of stat lines; the viewer draws them, collects
//! values, and sends every value back on each request. That is the whole
//! contract, and it is what lets an API with no concept of "dimensions"
//! drive the same viewer as one built entirely around them.
//!
//! Both sides of the wire share these definitions: the viewer deserializes
//! them and a server (see the `shape_api` crate) serializes them, so the two
//! cannot drift.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A control's current value. Untagged, so the wire form is the bare JSON
/// value (`"sphere"`, `12.5`, `true`) rather than a wrapper object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ControlValue {
    Bool(bool),
    Number(f64),
    Text(String),
}

impl ControlValue {
    /// Text form, used for select values and for display.
    pub fn as_text(&self) -> String {
        match self {
            Self::Bool(value) => value.to_string(),
            Self::Number(value) => format_number(*value),
            Self::Text(value) => value.clone(),
        }
    }

    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            Self::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
            Self::Text(value) => value.parse().ok(),
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(value) => *value,
            Self::Number(value) => *value != 0.0,
            Self::Text(value) => matches!(value.as_str(), "true" | "1" | "yes" | "on"),
        }
    }
}

/// Trailing zeros are dropped so a slider at 12.0 reads "12", not "12.0000".
fn format_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{value:.0}")
    } else {
        let text = format!("{value:.4}");
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    }
}

/// Values keyed by control id. Ordered so a request body serializes
/// identically for identical state, which keeps tests and logs readable.
pub type ControlValues = BTreeMap<String, ControlValue>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlOption {
    pub value: String,
    pub label: String,
    /// Secondary text shown after the label (e.g. "8 coords").
    #[serde(default)]
    pub detail: Option<String>,
}

/// One line of read-only status the panel renders above its controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatLine {
    pub label: String,
    pub value: String,
}

/// One control. `Group` nests; every other kind is a leaf.
///
/// `submits` is how a server declares its apply model: interacting with a
/// submitting widget sends the request immediately, while a non-submitting
/// one only updates the viewer's local values until something else submits.
/// A form is therefore a run of `submits: false` fields plus one submitting
/// button, and apply-on-change is every widget submitting — no extra
/// concept needed for either.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlWidget {
    Group {
        id: String,
        label: String,
        #[serde(default)]
        detail: Option<String>,
        /// Whether the viewer offers a collapse toggle. Collapsed state is
        /// the viewer's own; it never travels back to the server.
        #[serde(default)]
        collapsible: bool,
        #[serde(default)]
        children: Vec<ControlWidget>,
    },
    Select {
        id: String,
        label: String,
        #[serde(default)]
        detail: Option<String>,
        value: String,
        #[serde(default)]
        options: Vec<ControlOption>,
        #[serde(default)]
        disabled: bool,
        #[serde(default = "submits_by_default")]
        submits: bool,
    },
    Button {
        id: String,
        label: String,
        #[serde(default)]
        detail: Option<String>,
        #[serde(default)]
        disabled: bool,
        #[serde(default = "submits_by_default")]
        submits: bool,
    },
    Slider {
        id: String,
        label: String,
        #[serde(default)]
        detail: Option<String>,
        value: f64,
        minimum: f64,
        maximum: f64,
        /// Granularity a drag snaps to; `0` means continuous.
        #[serde(default)]
        step: f64,
        #[serde(default)]
        disabled: bool,
        #[serde(default = "submits_by_default")]
        submits: bool,
    },
    Text {
        id: String,
        label: String,
        #[serde(default)]
        detail: Option<String>,
        value: String,
        #[serde(default)]
        disabled: bool,
        #[serde(default = "submits_by_default")]
        submits: bool,
    },
    Toggle {
        id: String,
        label: String,
        #[serde(default)]
        detail: Option<String>,
        value: bool,
        #[serde(default)]
        disabled: bool,
        #[serde(default = "submits_by_default")]
        submits: bool,
    },
}

/// Apply-on-change, so a minimal server can omit the field entirely.
fn submits_by_default() -> bool {
    true
}

impl ControlWidget {
    pub fn id(&self) -> &str {
        match self {
            Self::Group { id, .. }
            | Self::Select { id, .. }
            | Self::Button { id, .. }
            | Self::Slider { id, .. }
            | Self::Text { id, .. }
            | Self::Toggle { id, .. } => id,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Self::Group { label, .. }
            | Self::Select { label, .. }
            | Self::Button { label, .. }
            | Self::Slider { label, .. }
            | Self::Text { label, .. }
            | Self::Toggle { label, .. } => label,
        }
    }

    pub fn detail(&self) -> Option<&str> {
        match self {
            Self::Group { detail, .. }
            | Self::Select { detail, .. }
            | Self::Button { detail, .. }
            | Self::Slider { detail, .. }
            | Self::Text { detail, .. }
            | Self::Toggle { detail, .. } => detail.as_deref(),
        }
    }

    pub fn disabled(&self) -> bool {
        match self {
            Self::Group { .. } => false,
            Self::Select { disabled, .. }
            | Self::Button { disabled, .. }
            | Self::Slider { disabled, .. }
            | Self::Text { disabled, .. }
            | Self::Toggle { disabled, .. } => *disabled,
        }
    }

    /// Whether interacting with this control sends the request. Groups never
    /// submit: collapsing one is a viewer-side display change.
    pub fn submits(&self) -> bool {
        match self {
            Self::Group { .. } => false,
            Self::Select { submits, .. }
            | Self::Button { submits, .. }
            | Self::Slider { submits, .. }
            | Self::Text { submits, .. }
            | Self::Toggle { submits, .. } => *submits,
        }
    }

    /// The value the server currently holds for this control, or `None` for
    /// the kinds that carry no value (groups and buttons).
    pub fn value(&self) -> Option<ControlValue> {
        match self {
            Self::Group { .. } | Self::Button { .. } => None,
            Self::Select { value, .. } | Self::Text { value, .. } => {
                Some(ControlValue::Text(value.clone()))
            }
            Self::Slider { value, .. } => Some(ControlValue::Number(*value)),
            Self::Toggle { value, .. } => Some(ControlValue::Bool(*value)),
        }
    }

    pub fn children(&self) -> &[ControlWidget] {
        match self {
            Self::Group { children, .. } => children,
            _ => &[],
        }
    }
}

/// Everything the panel pill displays, as the server describes it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ControlPanel {
    /// Bumped by the server only when the widget *structure* changes — a
    /// widget added, removed, reordered, or its kind changed. Values,
    /// labels, stats and errors change freely without it, so the viewer can
    /// key its (comparatively expensive) widget rebuild on this alone.
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub title: String,
    /// One line for the collapsed pill header.
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub stats: Vec<StatLine>,
    /// The server rejecting the submitted values (a bad shape name, two axes
    /// on the same dimension). The scene stays as it was; the panel shows
    /// this. Distinct from a failed request, which ends the stream.
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub widgets: Vec<ControlWidget>,
}

impl ControlPanel {
    /// Depth-first walk over every widget, groups included.
    pub fn walk(&self) -> Vec<(&ControlWidget, usize)> {
        let mut flattened = Vec::new();
        collect_widgets(&self.widgets, 0, &mut flattened);
        flattened
    }

    /// The server's own values for every control that has one, used to seed
    /// the viewer's value map when a panel first arrives.
    pub fn values(&self) -> ControlValues {
        self.walk()
            .into_iter()
            .filter_map(|(widget, _)| widget.value().map(|value| (widget.id().to_owned(), value)))
            .collect()
    }

    pub fn find(&self, control_id: &str) -> Option<&ControlWidget> {
        self.walk()
            .into_iter()
            .find(|(widget, _)| widget.id() == control_id)
            .map(|(widget, _)| widget)
    }
}

fn collect_widgets<'a>(
    widgets: &'a [ControlWidget],
    depth: usize,
    flattened: &mut Vec<(&'a ControlWidget, usize)>,
) {
    for widget in widgets {
        flattened.push((widget, depth));
        collect_widgets(widget.children(), depth + 1, flattened);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel_json() -> &'static str {
        r#"{
            "revision": 2,
            "title": "Shape",
            "summary": "sphere  120 pts",
            "stats": [{"label": "shown", "value": "120"}],
            "widgets": [
                {"kind": "select", "id": "shape", "label": "Shape", "value": "sphere",
                 "options": [{"value": "sphere", "label": "Sphere"}]},
                {"kind": "group", "id": "layout", "label": "Layout", "collapsible": true,
                 "children": [
                    {"kind": "slider", "id": "radius", "label": "Radius",
                     "value": 12.0, "minimum": 1.0, "maximum": 50.0, "step": 0.5},
                    {"kind": "toggle", "id": "jitter", "label": "Jitter", "value": true},
                    {"kind": "text", "id": "seed", "label": "Seed", "value": "42",
                     "submits": false}
                 ]},
                {"kind": "button", "id": "randomize", "label": "Randomize"}
            ]
        }"#
    }

    #[test]
    fn panel_round_trips_through_json() {
        let panel: ControlPanel = serde_json::from_str(panel_json()).unwrap();
        let reencoded = serde_json::to_string(&panel).unwrap();
        let reparsed: ControlPanel = serde_json::from_str(&reencoded).unwrap();
        assert_eq!(panel, reparsed);
    }

    #[test]
    fn walk_visits_nested_widgets_with_their_depth() {
        let panel: ControlPanel = serde_json::from_str(panel_json()).unwrap();
        let walked: Vec<(&str, usize)> = panel
            .walk()
            .into_iter()
            .map(|(widget, depth)| (widget.id(), depth))
            .collect();
        assert_eq!(
            walked,
            vec![
                ("shape", 0),
                ("layout", 0),
                ("radius", 1),
                ("jitter", 1),
                ("seed", 1),
                ("randomize", 0),
            ]
        );
    }

    #[test]
    fn values_cover_every_valued_control_and_skip_the_rest() {
        let panel: ControlPanel = serde_json::from_str(panel_json()).unwrap();
        let values = panel.values();
        assert_eq!(
            values.get("shape"),
            Some(&ControlValue::Text("sphere".to_owned()))
        );
        assert_eq!(values.get("radius"), Some(&ControlValue::Number(12.0)));
        assert_eq!(values.get("jitter"), Some(&ControlValue::Bool(true)));
        // Groups and buttons carry no value.
        assert!(!values.contains_key("layout"));
        assert!(!values.contains_key("randomize"));
    }

    #[test]
    fn submits_defaults_to_apply_on_change_and_is_honored_when_set() {
        let panel: ControlPanel = serde_json::from_str(panel_json()).unwrap();
        assert!(panel.find("shape").unwrap().submits());
        assert!(panel.find("randomize").unwrap().submits());
        // A form field opts out so its edits wait for a submitting control.
        assert!(!panel.find("seed").unwrap().submits());
        // Collapsing a group is a viewer-side display change, never a submit.
        assert!(!panel.find("layout").unwrap().submits());
    }

    #[test]
    fn control_values_keep_their_json_shape() {
        let encoded = serde_json::to_string(&ControlValues::from([
            ("shape".to_owned(), ControlValue::Text("cube".to_owned())),
            ("radius".to_owned(), ControlValue::Number(12.5)),
            ("jitter".to_owned(), ControlValue::Bool(false)),
        ]))
        .unwrap();
        assert_eq!(
            encoded, r#"{"jitter":false,"radius":12.5,"shape":"cube"}"#,
            "values serialize as bare JSON scalars in a deterministic order"
        );
    }

    #[test]
    fn numbers_render_without_trailing_noise() {
        assert_eq!(ControlValue::Number(12.0).as_text(), "12");
        assert_eq!(ControlValue::Number(12.5).as_text(), "12.5");
        assert_eq!(ControlValue::Bool(true).as_text(), "true");
    }
}
