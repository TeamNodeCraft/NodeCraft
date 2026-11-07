//! Expression-backed numeric input.
//!
//! The inspector keeps plain `f64` node values; this adapter parses the text the
//! user typed with the SolidCraft-compatible expression language and formats it
//! back to a number. Unit and parameter storage remain in the expression crate,
//! not in the UI widget.

use std::cell::RefCell;
use std::collections::HashMap;

use nodecraft_expr::{eval_with, ExprError, Kind};
use nodecraft_node_contract::NumberKind;

thread_local! {
    /// Expression text keyed by the field's stable egui id. Kept outside egui's
    /// data map so it cannot collide with DragValue's own edit state.
    static EXPRESSIONS: RefCell<HashMap<egui::Id, String>> = RefCell::new(HashMap::new());
}

/// Parse an inspector number, accepting expressions such as `40/2`, `2 in` or
/// `90 deg`. Lengths are stored in millimetres and angles in degrees, matching
/// the node contract.
#[must_use]
pub fn parse_number(text: &str, kind: NumberKind) -> Option<f64> {
    let value = eval_with(text, &|name| {
        Err(ExprError::Expr(format!("unknown parameter `{name}`")))
    })
    .ok()?;
    match kind {
        NumberKind::Length => value.to_kind(Kind::Length).ok(),
        NumberKind::Angle => value.to_kind(Kind::Angle).ok().map(f64::to_degrees),
        NumberKind::Unitless => value.to_kind(Kind::Unitless).ok(),
    }
}

/// The one numeric entry widget: drag to change, type to enter an expression.
///
/// The expression text is kept in an input-only thread-local cache keyed by
/// `id`, so the field continues to show `40/2` after commit. It is dropped when
/// the stored number no longer matches the expression.
pub fn number_drag<'a>(
    value: &'a mut f64,
    kind: NumberKind,
    step: f64,
    decimals: usize,
    id: egui::Id,
) -> egui::DragValue<'a> {
    let mut shown = EXPRESSIONS.with(|cache| cache.borrow().get(&id).cloned());
    if let Some(text) = shown.as_deref()
        && !parse_number(text, kind).is_some_and(|parsed| close_enough(parsed, *value))
    {
        EXPRESSIONS.with(|cache| cache.borrow_mut().remove(&id));
        shown = None;
    }
    let formatter = move |number: f64, _: std::ops::RangeInclusive<usize>| {
        shown.clone().unwrap_or_else(|| format_number(number, decimals))
    };
    egui::DragValue::new(value)
        .speed(step)
        .custom_formatter(formatter)
        .custom_parser(move |text| {
            let parsed = parse_number(text, kind);
            EXPRESSIONS.with(|cache| {
                let mut cache = cache.borrow_mut();
                if parsed.is_some() {
                    cache.insert(id, text.to_owned());
                } else {
                    cache.remove(&id);
                }
            });
            parsed
        })
}

fn close_enough(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1.0e-9 * a.abs().max(b.abs()).max(1.0)
}

/// Format a stored number for the entry field without losing a value the user
/// can round-trip. Trailing zeros are dropped.
#[must_use]
pub fn format_number(number: f64, decimals: usize) -> String {
    let text = format!("{number:.decimals$}");
    let text = match text.split_once('.') {
        Some((whole, fraction)) => {
            let fraction = fraction.trim_end_matches('0');
            if fraction.is_empty() {
                whole.to_owned()
            } else {
                format!("{whole}.{fraction}")
            }
        }
        None => text,
    };
    if text.is_empty() || text == "-0" {
        "0".to_owned()
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_expressions_are_accepted() {
        assert_eq!(parse_number("40/2", NumberKind::Length), Some(20.0));
        assert_eq!(parse_number("10 * 2 + 5", NumberKind::Length), Some(25.0));
        assert_eq!(parse_number("(40 + 2) / 2", NumberKind::Length), Some(21.0));
        assert_eq!(parse_number("sqrt(9)", NumberKind::Length), Some(3.0));
        assert_eq!(parse_number("2/1", NumberKind::Length), Some(2.0));
    }

    #[test]
    fn units_follow_the_field_kind() {
        assert_eq!(parse_number("2 in", NumberKind::Length), Some(50.8));
        assert_eq!(parse_number("90 deg", NumberKind::Angle), Some(90.0));
        assert_eq!(parse_number("1.5707963267948966 rad", NumberKind::Angle), Some(90.0));
        assert_eq!(parse_number("2", NumberKind::Angle), Some(2.0));
        assert_eq!(parse_number("2 mm", NumberKind::Unitless), None);
    }

    #[test]
    fn formatting_keeps_significant_values() {
        assert_eq!(format_number(20.0, 0), "20");
        assert_eq!(format_number(2.5, 1), "2.5");
        assert_eq!(format_number(-0.0, 2), "0");
    }
    #[test]
    fn typing_an_expression_into_the_widget_does_not_panic() {
        use egui::{Event, Pos2, Rect, Vec2};
        let ctx = egui::Context::default();
        let mut value = 0.0;
        let id = egui::Id::new("expression-input-test");
        let mut rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(120.0, 24.0));
        for step in 0..7 {
            let mut input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 120.0))),
                ..Default::default()
            };
            if step == 1 {
                let at = rect.center();
                input.events.push(Event::PointerMoved(at));
                input.events.push(Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() });
                input.events.push(Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() });
            }
            if (2..=5).contains(&step) {
                let text = ["4", "0", "/", "2"][step - 2];
                input.events.push(Event::Text(text.to_owned()));
            }
            if step == 6 {
                input.events.push(Event::Key { key: egui::Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() });
                input.events.push(Event::Key { key: egui::Key::Enter, physical_key: None, pressed: false, repeat: false, modifiers: Default::default() });
            }
            let mut output = ctx.run_ui(input, |ui| {
                let response = ui.add(number_drag(&mut value, NumberKind::Length, 0.1, 2, id));
                rect = response.rect;
            });
            output.textures_delta.clear();
        }
        assert_eq!(value, 20.0);
    }

}
