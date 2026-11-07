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

#[derive(Clone, Default)]
struct FieldState {
    expression: Option<String>,
    /// True after Return or focus loss; false while the user is editing.
    show_number: bool,
    was_focused: bool,
    /// Last drawn rect, so a click can switch to expression text on the same
    /// frame instead of seeding DragValue's editor with the formatted number.
    rect: Option<egui::Rect>,
    /// The expression text differs from the document and the inspector should
    /// persist it in the node-level `expressions` map.
    dirty_expression: bool,
}

thread_local! {
    static FIELDS: RefCell<HashMap<egui::Id, FieldState>> = RefCell::new(HashMap::new());
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
    ui: &egui::Ui,
    value: &'a mut f64,
    kind: NumberKind,
    step: f64,
    decimals: usize,
    id: egui::Id,
    stored_expression: Option<&str>,
) -> egui::DragValue<'a> {
    let widget_id = ui.next_auto_id();
    let focused = ui.memory(|memory| memory.has_focus(widget_id));
    let enter = ui.input(|input| input.key_pressed(egui::Key::Enter));
    let mut state = FIELDS.with(|fields| fields.borrow().get(&id).cloned().unwrap_or_default());
    if !state.was_focused && !state.dirty_expression {
        // The document owns the expression when the field is not being edited.
        state.expression = stored_expression.map(str::to_owned);
        state.show_number = true;
    }
    let clicked_inside = state.rect.is_some_and(|rect| {
        ui.input(|input| {
            input.pointer.primary_pressed()
                && input.pointer.interact_pos().is_some_and(|point| rect.contains(point))
        })
    });
    if clicked_inside || (focused && !state.was_focused) {
        // Entering edit mode always starts from the source expression.
        state.show_number = false;
    } else if state.was_focused && !focused {
        // Leaving the field commits the visible number.
        state.show_number = true;
    }
    if enter && focused {
        // Return commits even if the text editor keeps focus on this platform.
        state.show_number = true;
    }
    state.was_focused = focused;
    if let Some(text) = state.expression.as_deref()
        && !parse_number(text, kind).is_some_and(|parsed| close_enough(parsed, *value))
    {
        if state.expression.is_some() {
            state.dirty_expression = true;
        }
        state.expression = None;
        state.show_number = true;
    }
    let shown = if state.show_number { None } else { state.expression.clone() };
    FIELDS.with(|fields| fields.borrow_mut().insert(id, state));
    if enter && focused {
        // Make the platform's TextEdit leave edit mode after Return so the
        // formatter's number is the text shown next frame.
        ui.memory_mut(|memory| memory.surrender_focus(widget_id));
    }
    let formatter = move |number: f64, _: std::ops::RangeInclusive<usize>| {
        display_text(shown.as_deref(), number, decimals)
    };
    let parser_ctx = ui.ctx().clone();
    egui::DragValue::new(value)
        .speed(step)
        .custom_formatter(formatter)
        .custom_parser(move |text| {
            let parsed = parse_number(text, kind);
            FIELDS.with(|fields| {
                let mut fields = fields.borrow_mut();
                let state = fields.entry(id).or_default();
                if parsed.is_some() {
                    state.expression = Some(text.to_owned());
                    state.dirty_expression = true;
                    // Return commits to the number; normal text updates leave
                    // the current display mode alone (typing is already in the
                    // expression mode, and focus loss has already set the number).
                    if parser_ctx.input(|input| input.key_pressed(egui::Key::Enter)) {
                        state.show_number = true;
                    }
                } else {
                    state.expression = None;
                    state.dirty_expression = true;
                }
            });
            parsed
        })
}

/// Remember where the field was drawn, for the next frame's click detection.
pub fn remember_rect(id: egui::Id, rect: egui::Rect) {
    FIELDS.with(|fields| {
        fields.borrow_mut().entry(id).or_default().rect = Some(rect);
    });
}

/// Take an expression change that the inspector must persist, if any.
pub fn take_expression_change(id: egui::Id) -> Option<Option<String>> {
    FIELDS.with(|fields| {
        let mut fields = fields.borrow_mut();
        let state = fields.get_mut(&id)?;
        if !state.dirty_expression {
            return None;
        }
        state.dirty_expression = false;
        Some(state.expression.clone())
    })
}

/// What the box shows: the source expression, or the evaluated number.
fn display_text(expression: Option<&str>, number: f64, decimals: usize) -> String {
    expression.map_or_else(|| format_number(number, decimals), str::to_owned)
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
    fn the_focus_check_uses_the_drag_values_widget_id() {
        let ctx = egui::Context::default();
        let mut value = 1.0;
        let id = egui::Id::new("expression-input-widget-id");
        let mut expected = None;
        let mut actual = None;
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(300.0, 120.0))),
                ..Default::default()
            },
            |ui| {
                expected = Some(ui.next_auto_id());
                actual = Some(ui.add(number_drag(ui, &mut value, NumberKind::Length, 0.1, 2, id, None)).id);
            },
        );
        output.textures_delta.clear();
        assert_eq!(expected, actual);
    }
    #[test]
    fn return_ends_keyboard_editing() {
        use egui::{Event, Pos2, Rect, Vec2};
        let ctx = egui::Context::default();
        let mut value = 0.0;
        let id = egui::Id::new("return-editing-test");
        let mut widget = None;
        let mut rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(120.0, 24.0));
        for step in 0..8 {
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
            if step == 2 { input.events.push(Event::Text("2".into())); }
            if step == 3 { input.events.push(Event::Text("/".into())); }
            if step == 4 { input.events.push(Event::Text("1".into())); }
            if step == 5 {
                input.events.push(Event::Key { key: egui::Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() });
                input.events.push(Event::Key { key: egui::Key::Enter, physical_key: None, pressed: false, repeat: false, modifiers: Default::default() });
            }
            let mut output = ctx.run_ui(input, |ui| {
                let response = ui.add(number_drag(ui, &mut value, NumberKind::Length, 0.1, 2, id, None));
                rect = response.rect;
                widget = Some(response.id);
            });
            output.textures_delta.clear();
        }
        if let Some(widget) = widget {
            assert!(!ctx.memory(|memory| memory.has_focus(widget)), "Return should leave text editing");
        }
        assert_eq!(value, 2.0);
    }



    #[test]
    fn a_stored_expression_loads_in_number_mode() {
        let ctx = egui::Context::default();
        let mut value = 2.0;
        let id = egui::Id::new("stored-expression-test");
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(300.0, 120.0))),
                ..Default::default()
            },
            |ui| {
                ui.add(number_drag(ui, &mut value, NumberKind::Length, 0.1, 2, id, Some("2/1")));
            },
        );
        output.textures_delta.clear();
        let state = FIELDS.with(|fields| fields.borrow().get(&id).cloned()).expect("field state");
        assert_eq!(state.expression.as_deref(), Some("2/1"));
        assert!(state.show_number, "a loaded field starts by showing the number");
    }
    #[test]
    fn expression_is_restored_when_the_field_is_clicked_again() {
        use egui::{Event, Pos2, Rect, Vec2};
        let ctx = egui::Context::default();
        let mut value = 0.0;
        let id = egui::Id::new("expression-restore-test");
        let mut rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(120.0, 24.0));
        let mut show_after_return = None;
        for step in 0..9 {
            let mut input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 120.0))),
                ..Default::default()
            };
            if step == 1 || step == 7 {
                let at = rect.center();
                input.events.push(Event::PointerMoved(at));
                input.events.push(Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() });
                input.events.push(Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() });
            }
            if step == 2 { input.events.push(Event::Text("2".into())); }
            if step == 3 { input.events.push(Event::Text("/".into())); }
            if step == 4 { input.events.push(Event::Text("1".into())); }
            if step == 5 {
                input.events.push(Event::Key { key: egui::Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() });
                input.events.push(Event::Key { key: egui::Key::Enter, physical_key: None, pressed: false, repeat: false, modifiers: Default::default() });
            }
            let mut output = ctx.run_ui(input, |ui| {
                let response = ui.add(number_drag(ui, &mut value, NumberKind::Length, 0.1, 2, id, None));
                rect = response.rect;
                remember_rect(id, rect);
            });
            output.textures_delta.clear();
            if step == 6 {
                show_after_return = FIELDS.with(|fields| fields.borrow().get(&id).map(|state| state.show_number));
            }
        }
        assert_eq!(value, 2.0);
        assert_eq!(show_after_return, Some(true), "Return should show the evaluated number");
        let state = FIELDS.with(|fields| fields.borrow().get(&id).cloned()).expect("field state");
        assert_eq!(state.expression.as_deref(), Some("2/1"));
        assert!(!state.show_number, "clicking the field should show its expression again");
    }
    #[test]
    fn expression_is_only_shown_while_focused() {
        assert_eq!(display_text(Some("40/2"), 20.0, 2), "40/2");
        assert_eq!(display_text(None, 20.0, 2), "20");
        assert_eq!(display_text(None, 20.0, 2), "20");
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
                let response = ui.add(number_drag(ui, &mut value, NumberKind::Length, 0.1, 2, id, None));
                rect = response.rect;
            });
            output.textures_delta.clear();
        }
        assert_eq!(value, 20.0);
    }

}
