//! The keys and mouse buttons a binding can use, with the names the
//! controls sheet shows and the names they are saved under.

use bevy::prelude::{KeyCode, MouseButton};

/// Every key a binding can use, in the order a capture prefers them when
/// several go down in one frame, with the name the controls sheet shows.
/// Keys outside it (media keys, keys no layout names) cannot be bound, so
/// every bound key can be saved by name and read back.
pub(super) const BINDABLE_KEYS: &[(KeyCode, &str)] = &[
    (KeyCode::KeyA, "A"),
    (KeyCode::KeyB, "B"),
    (KeyCode::KeyC, "C"),
    (KeyCode::KeyD, "D"),
    (KeyCode::KeyE, "E"),
    (KeyCode::KeyF, "F"),
    (KeyCode::KeyG, "G"),
    (KeyCode::KeyH, "H"),
    (KeyCode::KeyI, "I"),
    (KeyCode::KeyJ, "J"),
    (KeyCode::KeyK, "K"),
    (KeyCode::KeyL, "L"),
    (KeyCode::KeyM, "M"),
    (KeyCode::KeyN, "N"),
    (KeyCode::KeyO, "O"),
    (KeyCode::KeyP, "P"),
    (KeyCode::KeyQ, "Q"),
    (KeyCode::KeyR, "R"),
    (KeyCode::KeyS, "S"),
    (KeyCode::KeyT, "T"),
    (KeyCode::KeyU, "U"),
    (KeyCode::KeyV, "V"),
    (KeyCode::KeyW, "W"),
    (KeyCode::KeyX, "X"),
    (KeyCode::KeyY, "Y"),
    (KeyCode::KeyZ, "Z"),
    (KeyCode::Digit0, "0"),
    (KeyCode::Digit1, "1"),
    (KeyCode::Digit2, "2"),
    (KeyCode::Digit3, "3"),
    (KeyCode::Digit4, "4"),
    (KeyCode::Digit5, "5"),
    (KeyCode::Digit6, "6"),
    (KeyCode::Digit7, "7"),
    (KeyCode::Digit8, "8"),
    (KeyCode::Digit9, "9"),
    (KeyCode::F1, "F1"),
    (KeyCode::F2, "F2"),
    (KeyCode::F3, "F3"),
    (KeyCode::F4, "F4"),
    (KeyCode::F5, "F5"),
    (KeyCode::F6, "F6"),
    (KeyCode::F7, "F7"),
    (KeyCode::F8, "F8"),
    (KeyCode::F9, "F9"),
    (KeyCode::F10, "F10"),
    (KeyCode::F11, "F11"),
    (KeyCode::F12, "F12"),
    (KeyCode::ArrowUp, "Up"),
    (KeyCode::ArrowDown, "Down"),
    (KeyCode::ArrowLeft, "Left"),
    (KeyCode::ArrowRight, "Right"),
    (KeyCode::Space, "Space"),
    (KeyCode::Tab, "Tab"),
    (KeyCode::Enter, "Enter"),
    (KeyCode::Escape, "Esc"),
    (KeyCode::Backspace, "Backspace"),
    (KeyCode::Delete, "Delete"),
    (KeyCode::Insert, "Insert"),
    (KeyCode::Home, "Home"),
    (KeyCode::End, "End"),
    (KeyCode::PageUp, "Page Up"),
    (KeyCode::PageDown, "Page Down"),
    (KeyCode::CapsLock, "Caps Lock"),
    (KeyCode::Backquote, "`"),
    (KeyCode::Minus, "-"),
    (KeyCode::Equal, "="),
    (KeyCode::BracketLeft, "["),
    (KeyCode::BracketRight, "]"),
    (KeyCode::Backslash, "\\"),
    (KeyCode::Semicolon, ";"),
    (KeyCode::Quote, "'"),
    (KeyCode::Comma, ","),
    (KeyCode::Period, "."),
    (KeyCode::Slash, "/"),
    (KeyCode::Numpad0, "Num 0"),
    (KeyCode::Numpad1, "Num 1"),
    (KeyCode::Numpad2, "Num 2"),
    (KeyCode::Numpad3, "Num 3"),
    (KeyCode::Numpad4, "Num 4"),
    (KeyCode::Numpad5, "Num 5"),
    (KeyCode::Numpad6, "Num 6"),
    (KeyCode::Numpad7, "Num 7"),
    (KeyCode::Numpad8, "Num 8"),
    (KeyCode::Numpad9, "Num 9"),
    (KeyCode::NumpadAdd, "Num +"),
    (KeyCode::NumpadSubtract, "Num -"),
    (KeyCode::NumpadMultiply, "Num *"),
    (KeyCode::NumpadDivide, "Num /"),
    (KeyCode::NumpadDecimal, "Num ."),
    (KeyCode::NumpadEnter, "Num Enter"),
    (KeyCode::ShiftLeft, "Left Shift"),
    (KeyCode::ShiftRight, "Right Shift"),
    (KeyCode::ControlLeft, "Left Ctrl"),
    (KeyCode::ControlRight, "Right Ctrl"),
    (KeyCode::AltLeft, "Left Alt"),
    (KeyCode::AltRight, "Right Alt"),
];

/// The key that ends a capture without binding anything.
pub(super) const CANCEL_KEY: KeyCode = KeyCode::Escape;
/// The key that empties the slot being captured.
pub(super) const CLEAR_KEY: KeyCode = KeyCode::Backspace;

/// The mouse buttons a binding can use, with the name the controls sheet
/// shows and the name they are saved under.
pub(super) const BINDABLE_BUTTONS: &[(MouseButton, &str, &str)] = &[
    (MouseButton::Left, "Left mouse", "Left"),
    (MouseButton::Right, "Right mouse", "Right"),
    (MouseButton::Middle, "Middle mouse", "Middle"),
    (MouseButton::Back, "Back mouse", "Back"),
    (MouseButton::Forward, "Forward mouse", "Forward"),
];

pub(super) fn key_name(key: KeyCode) -> String {
    BINDABLE_KEYS
        .iter()
        .find(|(bindable, _)| *bindable == key)
        .map_or_else(|| format!("{key:?}"), |(_, name)| (*name).to_owned())
}

/// The name a key is saved under: its variant name, which no keyboard
/// layout changes.
pub(super) fn saved_key_name(key: KeyCode) -> String {
    format!("{key:?}")
}

pub(super) fn key_from_saved_name(name: &str) -> Option<KeyCode> {
    BINDABLE_KEYS
        .iter()
        .map(|&(key, _)| key)
        .find(|&key| saved_key_name(key) == name)
}

pub(super) fn button_name(button: MouseButton) -> String {
    BINDABLE_BUTTONS
        .iter()
        .find(|(bindable, _, _)| *bindable == button)
        .map_or_else(
            || format!("{button:?} mouse"),
            |(_, name, _)| (*name).to_owned(),
        )
}

pub(super) fn saved_button_name(button: MouseButton) -> Option<&'static str> {
    BINDABLE_BUTTONS
        .iter()
        .find(|(bindable, _, _)| *bindable == button)
        .map(|(_, _, saved)| *saved)
}

pub(super) fn button_from_saved_name(name: &str) -> Option<MouseButton> {
    BINDABLE_BUTTONS
        .iter()
        .find(|(_, _, saved)| *saved == name)
        .map(|(button, _, _)| *button)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bindable_key_reads_back_from_its_saved_name() {
        for &(key, name) in BINDABLE_KEYS {
            assert!(!name.is_empty());
            assert_eq!(key_from_saved_name(&saved_key_name(key)), Some(key));
        }
        for &(button, _, saved) in BINDABLE_BUTTONS {
            assert_eq!(button_from_saved_name(saved), Some(button));
        }
    }
}
