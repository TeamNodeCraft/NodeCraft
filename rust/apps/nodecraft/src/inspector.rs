//! Schema-driven inspector widgets for graph node values.
//!
//! Known node fields get deliberate widgets (text, switch, drag-value, slider,
//! choice). Raw/unknown JSON fields are not shown.

use egui::{Response, Sense, Ui, Vec2};
use crate::ui_shape;
use serde_json::Value;
use nodecraft_session::GraphNode;

#[derive(Clone)]
pub struct Change {
    pub field: String,
    pub value: Value,
}

#[derive(Default)]
pub struct Output {
    pub changes: Vec<Change>,
    /// Expression edits to persist under the node's reserved JSON map.
    pub expressions: Vec<(String, Option<String>)>,
    pub choose_file: bool,
    pub choose_directory: bool,
    pub toggle_suppressed: bool,
    pub delete: bool,
    pub section_order: Option<Vec<String>>,
    pub begin_selection: Option<String>,
    pub toggle_sketch_edit: bool,
    pub choose_linked_part: bool,
    pub open_linked_part: bool,
}

/// Paint the inspector reset control at an exact header rect. This is separate
/// from layout so the caller can centre it against the whole header rather than
/// only the title row.
pub fn reset_icon_button_at(ui: &mut Ui, rect: egui::Rect) -> Response {
    let theme = crate::theme::get();
    let response = ui.interact(
        rect,
        egui::Id::new("inspector-reset"),
        Sense::click(),
    );
    if response.hovered() || response.is_pointer_button_down_on() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        let alpha = if response.is_pointer_button_down_on() { 44 } else { 26 };
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(8),
            theme.with_alpha(theme.highlight, alpha),
        );
    }
    let colour = if response.hovered() {
        theme.color(theme.text)
    } else {
        theme.color(theme.icon)
    };
    let svg = std::str::from_utf8(include_bytes!(
        "../../../../vendor/iconify-design/restore.svg"
    ))
    .unwrap_or_default()
    .replace("currentColor", &crate::theme::Theme::hex(colour.to_array()));
    let image = egui::Image::from_bytes(
        "bytes://nodecraft-inspector-reset.svg",
        svg.into_bytes(),
    )
    .fit_to_exact_size(Vec2::splat(17.0));
    image.paint_at(ui, rect.shrink(4.5));
    response.on_hover_text("Reset this node's settings to defaults")
}

#[derive(Clone)]
enum Widget {
    Text,
    Switch,
    Number {
        min: Option<f64>,
        max: Option<f64>,
        step: f64,
        kind: nodecraft_node_contract::NumberKind,
    },
    Choice(&'static [(&'static str, &'static str)]),
}

#[derive(Clone)]
struct FieldSpec {
    key: String,
    label: String,
    default: Value,
    widget: Widget,
}

fn field(key: &'static str, label: &'static str, default: Value, widget: Widget) -> FieldSpec {
    FieldSpec { key: key.to_owned(), label: label.to_owned(), default, widget }
}

fn text(key: &'static str, label: &'static str, default: &str) -> FieldSpec {
    field(key, label, Value::String(default.to_owned()), Widget::Text)
}

fn switch(key: &'static str, label: &'static str, default: bool) -> FieldSpec {
    field(key, label, Value::Bool(default), Widget::Switch)
}





fn choice(
    key: &'static str,
    label: &'static str,
    default: &'static str,
    choices: &'static [(&'static str, &'static str)],
) -> FieldSpec {
    field(key, label, Value::String(default.to_owned()), Widget::Choice(choices))
}

fn primitive_specs(node: &GraphNode) -> Vec<FieldSpec> {
    let kind = node
        .values
        .get("primitiveType")
        .and_then(Value::as_str)
        .unwrap_or("box");
    let mut specs = vec![choice(
        "primitiveType",
        "Primitive",
        "box",
        nodecraft_node_contract::PRIMITIVE_TYPES,
    )];
    specs.extend(contract_field_specs(nodecraft_node_contract::primitive_fields(kind)));
    specs
}

fn surface_loft_sections(ui: &mut Ui, sections: &[String], output: &mut Output) {
    let theme = crate::theme::get();
    ui.label(
        egui::RichText::new("SECTIONS")
            .size(9.0)
            .color(theme.with_alpha(theme.muted, 190)),
    );
    if sections.is_empty() {
        ui.label(
            egui::RichText::new("Connect sketch or spline profiles to the Sections port, in order.")
                .size(10.0)
                .color(theme.with_alpha(theme.muted, 220)),
        );
        return;
    }
    let mut order = sections.to_vec();
    let mut changed = false;
    for index in 0..order.len() {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("{}. {}", index + 1, order[index]))
                    .size(11.0)
                    .color(theme.color(theme.text)),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("↓").clicked() && index + 1 < order.len() {
                    order.swap(index, index + 1);
                    changed = true;
                }
                if ui.small_button("↑").clicked() && index > 0 {
                    order.swap(index, index - 1);
                    changed = true;
                }
            });
        });
    }
    if changed {
        output.section_order = Some(order);
    }
}

fn topology_references(
    ui: &mut Ui,
    node: &GraphNode,
    output: &mut Output,
    field: &str,
    title: &str,
    empty: &str,
    remove_hint: &str,
    clear_label: &str,
    clear_hint: &str,
    active: bool,
) {
    let theme = crate::theme::get();
    egui::Frame::new()
        .fill(theme.with_alpha(theme.control, 90))
        .stroke(egui::Stroke::new(1.0, theme.with_alpha(theme.border, 170)))
        .corner_radius(6.0)
        .inner_margin(8.0)
        .show(ui, |ui| {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(title)
                .size(9.0)
                .color(theme.with_alpha(theme.muted, 190)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let svg = include_str!("../../../../vendor/fluent/edit.svg");
            let colour = if active {
                crate::theme::Theme::hex(theme.highlight)
            } else {
                "#ffffff".to_owned()
            };
            let svg = crate::app::tint_svg(svg.as_bytes(), &colour);
            let icon = egui::Image::from_bytes(
                if active { "bytes://inspector-edit-active.svg" } else { "bytes://inspector-edit-idle.svg" },
                svg.into_bytes(),
            )
            .fit_to_exact_size(egui::vec2(14.0, 14.0));
            if ui
                .add(egui::Button::image(icon).frame(false))
                .on_hover_text("Edit selection")
                .clicked()
            {
                output.begin_selection = Some(field.to_owned());
            }
        });
    });
    let mut references: Vec<String> = node
        .values
        .get(field)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("id").and_then(Value::as_str))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if references.is_empty() {
        ui.label(
            egui::RichText::new(empty)
                .size(10.0)
                .color(theme.with_alpha(theme.muted, 220)),
        );
    } else {
        let mut remove = None;
        for (index, reference) in references.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(reference)
                        .size(10.0)
                        .color(theme.color(theme.text)),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").on_hover_text(remove_hint).clicked() {
                        remove = Some(index);
                    }
                });
            });
        }
        let mut changed = remove.is_some();
        if let Some(index) = remove {
            references.remove(index);
        }
        if !references.is_empty()
            && ui
                .small_button(clear_label)
                .on_hover_text(clear_hint)
                .clicked()
        {
            references.clear();
            changed = true;
        }
        if changed {
            output.changes.push(Change {
                field: field.to_owned(),
                value: Value::Array(
                    references
                        .into_iter()
                        .map(|id| serde_json::json!({ "id": id }))
                        .collect(),
                ),
            });
        }
    }
    });
}

fn spline_points(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    let theme = crate::theme::get();
    let mut points: Vec<(f64, f64)> = node
        .values
        .get("points")
        .and_then(Value::as_array)
        .map(|points| {
            points
                .iter()
                .filter_map(|point| {
                    let x = point.get("x").and_then(Value::as_f64)?;
                    let y = point.get("y").and_then(Value::as_f64)?;
                    Some((x, y))
                })
                .collect()
        })
        .unwrap_or_default();
    let control = node.values.get("control").and_then(Value::as_bool).unwrap_or(false);
    ui.label(
        egui::RichText::new(if control { "CONTROL POINTS" } else { "FIT POINTS" })
            .size(9.0)
            .color(theme.with_alpha(theme.muted, 190)),
    );
    let mut changed = false;
    let mut remove = None;
    for (index, (x, y)) in points.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("{}.", index + 1))
                    .size(10.0)
                    .color(theme.with_alpha(theme.muted, 220)),
            );
            let x_id = egui::Id::new(("spline-x", &node.id, index));
            let x_response = ui.add(
                crate::expression_input::number_drag(
                    ui,
                    x,
                    nodecraft_node_contract::NumberKind::Length,
                    0.5,
                    2,
                    x_id,
                    node.expressions
                        .get(&format!("points.{index}.x"))
                        .map(String::as_str),
                )
                .range(-10000.0..=10000.0)
                .prefix("X "),
            );
            crate::expression_input::remember_rect(x_id, x_response.rect);
            if let Some(expression) = crate::expression_input::take_expression_change(x_id) {
                output.expressions.push((format!("points.{index}.x"), expression));
            }
            let y_id = egui::Id::new(("spline-y", &node.id, index));
            let y_response = ui.add(
                crate::expression_input::number_drag(
                    ui,
                    y,
                    nodecraft_node_contract::NumberKind::Length,
                    0.5,
                    2,
                    y_id,
                    node.expressions
                        .get(&format!("points.{index}.y"))
                        .map(String::as_str),
                )
                .range(-10000.0..=10000.0)
                .prefix("Y "),
            );
            crate::expression_input::remember_rect(y_id, y_response.rect);
            if let Some(expression) = crate::expression_input::take_expression_change(y_id) {
                output.expressions.push((format!("points.{index}.y"), expression));
            }
            if ui.small_button("×").on_hover_text("Remove point").clicked() {
                remove = Some(index);
            }
            changed |= x_response.changed() || y_response.changed();
        });
    }
    if let Some(index) = remove {
        if points.len() > 2 {
            points.remove(index);
            changed = true;
        }
    }
    if ui.small_button("+ Add point").clicked() && points.len() < 500 {
        let (x, y) = points.last().copied().unwrap_or((0.0, 0.0));
        points.push((x + 20.0, y));
        changed = true;
    }
    if changed {
        let value = Value::Array(
            points
                .into_iter()
                .map(|(x, y)| serde_json::json!({"x": x, "y": y}))
                .collect(),
        );
        output.changes.push(Change { field: "points".to_owned(), value });
    }
}

fn parse_look_color(text: &str) -> Option<egui::Color32> {
    let hex = text.trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }
    let channel = |range: std::ops::Range<usize>| u8::from_str_radix(&hex[range], 16).ok();
    Some(egui::Color32::from_rgb(
        channel(0..2)?,
        channel(2..4)?,
        channel(4..6)?,
    ))
}

fn look_color(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    let current = node
        .values
        .get("color")
        .and_then(Value::as_str)
        .and_then(parse_look_color)
        .unwrap_or_else(|| egui::Color32::from_rgb(127, 160, 207));
    let mut color = current;
    ui.horizontal(|ui| {
        ui.label("Colour");
        if ui.color_edit_button_srgba(&mut color).changed() {
            output.changes.push(Change {
                field: "color".to_owned(),
                value: Value::String(format!(
                    "#{:02x}{:02x}{:02x}",
                    color.r(),
                    color.g(),
                    color.b()
                )),
            });
        }
        ui.colored_label(color, format!("#{:02x}{:02x}{:02x}", color.r(), color.g(), color.b()));
    });
}

fn contract_field_specs(fields: &[nodecraft_node_contract::NodeFieldSpec]) -> Vec<FieldSpec> {
    fields
        .iter()
        .map(|spec| {
            let default = match spec.default {
                nodecraft_node_contract::FieldDefault::Text(value) => Value::String(value.to_owned()),
                nodecraft_node_contract::FieldDefault::Bool(value) => Value::Bool(value),
                nodecraft_node_contract::FieldDefault::Number(value) => Value::from(value),
                nodecraft_node_contract::FieldDefault::Choice(value) => Value::String(value.to_owned()),
            };
            let widget = match spec.kind {
                nodecraft_node_contract::FieldKind::Text => Widget::Text,
                nodecraft_node_contract::FieldKind::Switch => Widget::Switch,
                nodecraft_node_contract::FieldKind::Number { min, max, step, kind } => {
                    Widget::Number { min, max, step, kind }
                }
                nodecraft_node_contract::FieldKind::Choice(choices) => Widget::Choice(choices),
            };
            FieldSpec {
                key: spec.key.to_owned(),
                label: spec.label.to_owned(),
                default,
                widget,
            }
        })
        .collect()
}

/// Map the contract's neutral field metadata onto this egui inspector.
fn contract_specs(node_type: &str) -> Option<Vec<FieldSpec>> {
    let fields = nodecraft_node_contract::fields(node_type)?;
    Some(contract_field_specs(fields))
}

fn known_specs(node: &GraphNode) -> Vec<FieldSpec> {
    if let Some(specs) = contract_specs(&node.node_type) {
        return specs;
    }
    match node.node_type.as_str() {
        "solid" | "primitive" => primitive_specs(node),
        "stepimporter" => vec![
            text("path", "STEP path", ""),
            switch("heal", "Heal", true),
        ],
        "exporter" => vec![
            text("fileName", "File name", "model"),
            text("directory", "Directory", ""),
            switch("overwrite", "Overwrite", false),
        ],
        _ => Vec::new(),
    }
}

/// Values restored by the inspector reset action. This replaces settings for the
/// node type while preserving asset/geometry payloads (STEP paths, sketch and
/// spline point data) that are not settings.
pub fn reset_values(node: &GraphNode) -> Vec<Change> {
    let mut values = serde_json::Map::new();
    for spec in known_specs(node) {
        values.insert(spec.key, spec.default);
    }
    match node.node_type.as_str() {
        "look" => {
            values.insert("color".to_owned(), Value::String("#7fa0cf".to_owned()));
        }
        "shell" | "offsetfaces" => {
            values.insert("faces".to_owned(), Value::Array(Vec::new()));
        }
        "chamfer" | "fillet" => {
            values.insert("edges".to_owned(), Value::Array(Vec::new()));
        }
        _ => {}
    }
    for key in ["path", "sketch", "points"] {
        if let Some(value) = node.values.get(key) {
            values.insert(key.to_owned(), value.clone());
        }
    }
    let mut changes: Vec<Change> = values
        .into_iter()
        .map(|(field, value)| Change { field, value })
        .collect();
    changes
}

fn current_value(node: &GraphNode, spec: &FieldSpec) -> Value {
    node.values.get(spec.key.as_str()).cloned().unwrap_or_else(|| spec.default.clone())
}

fn draw_switch(ui: &mut Ui, value: &mut bool) -> Response {
    let theme = crate::theme::get();
    let desired = Vec2::new(34.0, 18.0);
    let (rect, mut response) = ui.allocate_exact_size(desired, Sense::click());
    response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
    if response.clicked() {
        *value = !*value;
        response.mark_changed();
    }
    let t = if *value { 1.0 } else { 0.0 };
    let track = theme.color(theme.control).lerp_to_gamma(theme.color(theme.accent), t);
    let knob = theme.color(theme.text).lerp_to_gamma(theme.color(theme.control), t * 0.25);
    ui.painter().rect_filled(rect, rect.height() * 0.5, track);
    let radius = rect.height() * 0.5 - 2.0;
    let center = egui::Pos2::new(
        rect.left() + radius + 2.0 + (rect.width() - 2.0 * radius - 4.0) * t,
        rect.center().y,
    );
    ui.painter().circle_filled(center, radius, knob);
    response
}

fn draw_choice(ui: &mut Ui, key: &str, value: &mut String, choices: &[(&str, &str)]) -> bool {
    let selected = choices
        .iter()
        .find(|(candidate, _)| *candidate == value.as_str())
        .map(|(_, label)| *label)
        .unwrap_or(value.as_str());
    let mut changed = false;
    egui::ComboBox::from_id_salt(format!("inspector-{key}"))
        .selected_text(selected)
        .show_ui(ui, |ui| {
            for (candidate, label) in choices {
                if ui.selectable_value(value, (*candidate).to_owned(), *label).changed() {
                    changed = true;
                }
            }
        });
    changed
}

fn clamp_number(value: &mut f64, min: Option<f64>, max: Option<f64>) -> bool {
    let mut changed = false;
    if let Some(min) = min {
        if *value < min {
            *value = min;
            changed = true;
        }
    }
    if let Some(max) = max {
        if *value > max {
            *value = max;
            changed = true;
        }
    }
    changed
}

fn draw_number(
    ui: &mut Ui,
    number: &mut f64,
    step: f64,
    min: Option<f64>,
    max: Option<f64>,
    kind: nodecraft_node_contract::NumberKind,
    id: egui::Id,
    stored_expression: Option<&str>,
) -> (bool, Option<Option<String>>) {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let button_size = Vec2::new(22.0, 22.0);
        let theme = crate::theme::get();
        let minus = ui_shape::button(
            ui,
            "-",
            button_size.x,
            button_size.y,
            theme.color(theme.control),
            egui::Stroke::new(1.0, theme.with_alpha(theme.border, 190)),
            theme.color(theme.text),
        )
        .on_hover_text("Decrease");
        if minus.clicked() {
            *number -= step;
            clamp_number(number, min, max);
            changed = true;
        }
        // Keep the number entry centered in the remaining width. The DragValue
        // is a text entry first: clicking it types an exact number.
        let entry_width = (ui.available_width() - button_size.x * 2.0 - 8.0).max(40.0);
        let decimals = if step >= 1.0 {
            0
        } else if step >= 0.1 {
            1
        } else {
            2
        };
        let entry = ui.add_sized(
            [entry_width, button_size.y],
            crate::expression_input::number_drag(
                ui,
                number,
                kind,
                step,
                decimals,
                id,
                stored_expression,
            ),
        );
        crate::expression_input::remember_rect(id, entry.rect);
        if entry.changed() {
            clamp_number(number, min, max);
            changed = true;
        }
        let plus = ui_shape::button(
            ui,
            "+",
            button_size.x,
            button_size.y,
            theme.color(theme.control),
            egui::Stroke::new(1.0, theme.with_alpha(theme.border, 190)),
            theme.color(theme.text),
        )
        .on_hover_text("Increase");
        if plus.clicked() {
            *number += step;
            clamp_number(number, min, max);
            changed = true;
        }
    });
    (changed, crate::expression_input::take_expression_change(id))
}

pub fn node_title(node: &GraphNode) -> String {
    match node.node_type.as_str() {
        "solid" | "primitive" => {
            let kind = node.values.get("primitiveType").and_then(Value::as_str).unwrap_or("box");
            match kind {
                "box" => "Box".to_owned(),
                "cylinder" => "Cylinder".to_owned(),
                "cone" => "Cone".to_owned(),
                "sphere" => "Sphere".to_owned(),
                "torus" => "Torus".to_owned(),
                "wedge" => "Wedge".to_owned(),
                "pyramid" => "Pyramid".to_owned(),
                "partialSphere" => "Partial sphere".to_owned(),
                "partialTorus" => "Partial torus".to_owned(),
                other => title_case(other),
            }
        }
        "stepimporter" => "Import STEP".to_owned(),
        "surfaceloft" => "Surface Loft".to_owned(),
        "exporter" => "Export STEP".to_owned(),
        "translate" => "Transform".to_owned(),
        "group" => "Group".to_owned(),
        "offsetfaces" => "Offset Faces".to_owned(),
        "groupinput" => "Group Input".to_owned(),
        "groupoutput" => "Group Output".to_owned(),
        "boolean" => "Boolean".to_owned(),
        "combine" => "Combine to Part".to_owned(),
        "fuse" => "Fuse".to_owned(),
        "chamfer" => "Chamfer".to_owned(),
        "fillet" => "Fillet".to_owned(),
        "look" => "Look".to_owned(),
        other => title_case(&other.replace(['_', '-'], " ")),
    }
}

fn title_case(text: &str) -> String {
    text.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn node_badge(node: &GraphNode) -> String {
    match node.node_type.as_str() {
        "stepimporter" => "Import",
        "exporter" | "linkedpart" => "Output",
        "group" => "Subpatch",
        "offsetfaces" => "Feature",
        "groupinput" | "groupoutput" => "Interface",
        "translate" => "Transform",
        "boolean" | "fuse" => "Boolean",
        "combine" => "Part",
        "solid" | "primitive" => "Solid",
        "sketch" => "Sketch",
        "spline" => "Curve",
        "extrude" => "Operation",
        "chamfer" | "fillet" | "shell" | "look" => "Feature",
        "scatter" => "Generator",
        _ => "Node",
    }
    .to_owned()
}

fn section_title(node: &GraphNode) -> &'static str {
    match node.node_type.as_str() {
        "stepimporter" => "STEP FILE",
        "exporter" => "OUTPUT FILE",
        "translate" => "TRANSFORM",
        "boolean" | "fuse" => "OPERATION",
        "combine" => "SHAPES",
        "solid" | "primitive" => "PARAMETERS",
        "sketch" => "SKETCH",
        "spline" => "SPLINE",
        "extrude" => "EXTRUSION",
        "chamfer" => "CHAMFER",
        "fillet" => "FILLET",
        "look" => "LOOK",
        "shell" => "SHELL",
        "mirror" => "MIRROR",
        "scatter" => "SCATTER",
        "group" => "SUBPATCH",
        "offsetfaces" => "OFFSET",
        "groupinput" | "groupoutput" => "GROUP INTERFACE",
        _ => "VALUES",
    }
}

fn node_description(node: &GraphNode) -> Option<String> {
    let text = match node.node_type.as_str() {
        "stepimporter" => "Imports STEP solids and surfaces from AP203, AP214 or AP242 Part 21 files. Requires a local .step/.stp file.".to_owned(),
        "exporter" => "Writes the connected shape to a STEP file on disk.".to_owned(),
        "translate" => "Moves and rotates the incoming shape. Select it in the graph or use the viewport gizmo.".to_owned(),
        "solid" | "primitive" => "Creates a parametric primitive solid. Dimensions are in model units.".to_owned(),
        "boolean" | "fuse" => {
            let mode = node.values.get("mode").and_then(Value::as_str).unwrap_or("union");
            match mode {
                "subtract" => "A minus B: removes the second shape from the first.".to_owned(),
                "subtract-reverse" => "B minus A: removes the first shape from the second.".to_owned(),
                "intersect" => "Shared volume of both input shapes.".to_owned(),
                _ => "Combines both input shapes into one result.".to_owned(),
            }
        }
        "combine" => "Collects every connected shape into a single Part.".to_owned(),
        "mirror" => "Reflects the incoming shape across a world plane.".to_owned(),
        "chamfer" => "Cuts selected edges of the incoming shape.".to_owned(),
        "fillet" => "Rounds selected edges of the incoming shape.".to_owned(),
        "look" => "Applies a colour to every body in the incoming Part. The geometry is unchanged.".to_owned(),
        "shell" => "Hollows the incoming shape by removing selected faces.".to_owned(),
        "offsetfaces" => "Moves the selected faces along their normals.".to_owned(),
        "extrude" => "Extrudes a sketch profile by the given amount.".to_owned(),
        "scatter" => "Repeats a shape across a seeded area.".to_owned(),
        "group" => {
            let count = node.values.get("nodes").and_then(Value::as_array).map_or(0, Vec::len);
            let inputs = node.values.get("inputs").and_then(Value::as_array).map_or(0, Vec::len);
            let outputs = node.values.get("outputs").and_then(Value::as_array).map_or(0, Vec::len);
            format!(
                "Subpatch with {count} child node(s), {inputs} input(s) and {outputs} output(s). \
                 It is flattened into the parent graph during evaluation."
            )
        }
        _ => return None,
    };
    Some(text)
}

fn section_label(ui: &mut Ui, text: &str) {
    let theme = crate::theme::get();
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(text)
            .size(10.0)
            .strong()
            .color(theme.with_alpha(theme.muted, 220)),
    );
}

fn description(ui: &mut Ui, text: &str) {
    ui_shape::callout(ui, text);
}

/// Show an error for the selected node even when it is outside the viewed
/// target closure. The canvas glow says where; this says why.
fn linked_part_fields(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    let mut path = node.values.get("path").and_then(Value::as_str).unwrap_or_default().to_owned();
    ui.horizontal(|ui| {
        let button_width = 76.0;
        let path_width = (ui.available_width() - button_width - 8.0).max(80.0);
        if ui
            .add_sized(
                [path_width, 26.0],
                egui::TextEdit::singleline(&mut path).hint_text("Path to .nodecraft"),
            )
            .changed()
        {
            output.changes.push(Change { field: "path".to_owned(), value: Value::String(path.clone()) });
        }
        if ui.button("Choose…").clicked() {
            output.choose_linked_part = true;
        }
    });
    ui.add_space(6.0);
    let enabled = !path.is_empty();
    if ui.add_enabled(enabled, egui::Button::new("Open in new tab")).clicked() {
        output.open_linked_part = true;
    }
    description(ui, "Opens this .nodecraft document as an editable tab. Linked parts may themselves link further documents.");
}

fn step_schema(ui: &Ui, path: &str) -> Option<String> {
    let id = egui::Id::new(("step-schema", path));
    if let Some(cached) = ui.ctx().data(|data| data.get_temp::<Option<String>>(id)) {
        return cached;
    }
    let schema = nodecraft_step::read_schema(path).ok().flatten();
    ui.ctx().data_mut(|data| data.insert_temp(id, schema.clone()));
    schema
}

fn step_import_fields(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    let theme = crate::theme::get();
    let mut path = node.values.get("path").and_then(Value::as_str).unwrap_or_default().to_owned();
    ui.horizontal(|ui| {
        let button_width = 76.0;
        let path_width = (ui.available_width() - button_width - 8.0).max(80.0);
        let changed = ui
            .add_sized(
                [path_width, 26.0],
                egui::TextEdit::singleline(&mut path).hint_text("Path to .step or .stp"),
            )
            .changed();
        if changed {
            output.changes.push(Change { field: "path".to_owned(), value: Value::String(path.clone()) });
        }
        let button_colour = theme.color(theme.control);
        if ui_shape::button(
            ui,
            "Choose…",
            button_width,
            26.0,
            button_colour,
            egui::Stroke::new(1.0, theme.with_alpha(theme.border, 190)),
            theme.color(theme.text),
        )
        .clicked()
        {
            output.choose_file = true;
        }
    });
    if !path.is_empty() {
        if let Some(schema) = step_schema(ui, &path) {
            let protocol = nodecraft_step::application_protocol(&schema).unwrap_or("Other");
            ui.label(
                egui::RichText::new(format!("{protocol} · {schema}"))
                    .size(11.0)
                    .color(theme.color(theme.muted)),
            )
            .on_hover_text(
                "Supported: AP203/AP214/AP242 Part 21 B-rep import. \n\
                 AP242 PMI/GD&T, tessellated geometry and STEP XML (Part 28) are not yet supported.",
            );
        }
    }
    let mut heal = node.values.get("heal").and_then(Value::as_bool).unwrap_or(true);
    if ui.checkbox(&mut heal, "Heal imported B-rep").changed() {
        output.changes.push(Change { field: "heal".to_owned(), value: Value::Bool(heal) });
    }
    let _ = theme;
}

fn export_fields(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    let theme = crate::theme::get();
    let mut name = node
        .values
        .get("fileName")
        .and_then(Value::as_str)
        .unwrap_or("model")
        .to_owned();
    let mut directory = node
        .values
        .get("directory")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let mut overwrite = node
        .values
        .get("overwrite")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    egui::Grid::new(("exporter-grid", &node.id))
        .num_columns(2)
        .spacing([10.0, 7.0])
        .show(ui, |ui| {
            ui.label("File name");
            if ui
                .add(egui::TextEdit::singleline(&mut name).hint_text("model"))
                .changed()
            {
                output.changes.push(Change {
                    field: "fileName".to_owned(),
                    value: Value::String(name.clone()),
                });
            }
            ui.end_row();

            ui.label("Directory");
            ui.horizontal(|ui| {
                let button_width = 76.0;
                let path_width = (ui.available_width() - button_width - 8.0).max(80.0);
                if ui
                    .add_sized(
                        [path_width, 26.0],
                        egui::TextEdit::singleline(&mut directory)
                            .hint_text("STEP output folder"),
                    )
                    .changed()
                {
                    output.changes.push(Change {
                        field: "directory".to_owned(),
                        value: Value::String(directory.clone()),
                    });
                }
                if ui_shape::button(
                    ui,
                    "Choose…",
                    button_width,
                    26.0,
                    theme.color(theme.control),
                    egui::Stroke::new(1.0, theme.with_alpha(theme.border, 190)),
                    theme.color(theme.text),
                )
                .clicked()
                {
                    output.choose_directory = true;
                }
            });
            ui.end_row();

            ui.label("Overwrite");
            if draw_switch(ui, &mut overwrite).changed() {
                output.changes.push(Change {
                    field: "overwrite".to_owned(),
                    value: Value::Bool(overwrite),
                });
            }
            ui.end_row();
        });
    let _ = theme;
}

fn action_buttons(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    let theme = crate::theme::get();
    ui.add_space(12.0);
    let suppress_label = if node.suppressed { "Unsuppress node" } else { "Suppress node" };
    let width = ui.available_width();
    if ui_shape::button(
        ui,
        suppress_label,
        width,
        30.0,
        theme.with_alpha(theme.control, 220),
        egui::Stroke::new(1.0, theme.with_alpha(theme.border, 210)),
        theme.color(theme.text),
    )
    .clicked()
    {
        output.toggle_suppressed = true;
    }
    ui.add_space(6.0);
    let danger = egui::Color32::from_rgb(245, 132, 122);
    if ui_shape::button(
        ui,
        "Delete node",
        width,
        30.0,
        egui::Color32::from_rgba_unmultiplied(80, 28, 26, 120),
        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(245, 132, 122, 170)),
        danger,
    )
    .clicked()
    {
        output.delete = true;
    }
}

/// The loaded STEP file's display name, without its extension.
fn step_file_name(node: &GraphNode) -> Option<String> {
    let path = node.values.get("path").and_then(Value::as_str)?;
    let name = std::path::Path::new(path).file_stem()?.to_str()?;
    (!name.is_empty()).then(|| name.to_owned())
}

/// Draw the generated inspector for one node and return field and action changes.
pub fn show(
    ui: &mut Ui,
    node: &GraphNode,
    sections: &[String],
    active_selection: Option<&str>,
    sketch_editing: bool,
) -> Output {
    let mut output = Output::default();
    if node.node_type == "stepimporter" {
        if let Some(name) = step_file_name(node) {
            let theme = crate::theme::get();
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!("File name: {name}"))
                    .size(12.0)
                    .strong()
                    .color(theme.color(theme.text)),
            );
        }
        section_label(ui, section_title(node));
        step_import_fields(ui, node, &mut output);
    } else if node.node_type == "linkedpart" {
        section_label(ui, section_title(node));
        linked_part_fields(ui, node, &mut output);
    } else if node.node_type == "exporter" {
        section_label(ui, section_title(node));
        export_fields(ui, node, &mut output);
    } else {
        let specs = known_specs(node);
        section_label(ui, section_title(node));
        if node.node_type == "sketch" {
            let theme = crate::theme::get();
            ui.add_space(6.0);
            let label = if sketch_editing { "Finish Sketch" } else { "Edit Sketch" };
            if ui_shape::button(
                ui,
                label,
                ui.available_width(),
                30.0,
                if sketch_editing {
                    theme.color(theme.accent)
                } else {
                    theme.with_alpha(theme.control, 220)
                },
                egui::Stroke::new(1.0, theme.with_alpha(theme.border, 210)),
                if sketch_editing {
                    theme.color(theme.control)
                } else {
                    theme.color(theme.text)
                },
            )
            .clicked()
            {
                output.toggle_sketch_edit = true;
            }
            ui.add_space(8.0);
        }
        if node.node_type == "spline" {
            spline_points(ui, node, &mut output);
            ui.add_space(6.0);
        }
        if node.node_type == "surfaceloft" {
            surface_loft_sections(ui, sections, &mut output);
            ui.add_space(6.0);
        }
        if node.node_type == "look" {
            look_color(ui, node, &mut output);
            ui.add_space(6.0);
        }
        if node.node_type == "offsetfaces" {
            topology_references(
                ui,
                node,
                &mut output,
                "faces",
                "OFFSET FACES",
                "No faces selected. Select the Face filter in the viewport, then click faces to offset.",
                "Remove face",
                "Clear faces",
                "Clear faces",
                active_selection == Some("faces"),
            );
            ui.add_space(6.0);
        }
        if node.node_type == "shell" {
            topology_references(
                ui,
                node,
                &mut output,
                "faces",
                "OPENING FACES",
                "No openings. Select the Face filter in the viewport, then click faces to open the shell.",
                "Remove opening",
                "Clear openings",
                "Close the shell",
                active_selection == Some("faces"),
            );
            ui.add_space(6.0);
        }
        if node.node_type == "chamfer" || node.node_type == "fillet" {
            topology_references(
                ui,
                node,
                &mut output,
                "edges",
                "EDGES",
                "No edges. Click the edit icon, then click edges in the viewport to apply the treatment.",
                "Remove edge",
                "Clear edges",
                "Remove all edge selections",
                active_selection == Some("edges"),
            );
            ui.add_space(6.0);
        }
        egui::Grid::new(("inspector-grid", &node.id))
            .num_columns(2)
            .spacing([10.0, 7.0])
            .striped(false)
            .show(ui, |ui| {
                for spec in &specs {
                    let mut value = current_value(node, spec);
                    let mut changed = false;
                    ui.label(&spec.label);
                    match &spec.widget {
                        Widget::Text => {
                            let mut text = value.as_str().unwrap_or_default().to_owned();
                            if ui.add(egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY)).changed() {
                                value = Value::String(text);
                                changed = true;
                            }
                        }
                        Widget::Switch => {
                            let mut enabled = value.as_bool().unwrap_or(false);
                            if draw_switch(ui, &mut enabled).changed() {
                                value = Value::Bool(enabled);
                                changed = true;
                            }
                        }
                        Widget::Number { min, max, step, kind } => {
                            let mut number = value.as_f64().unwrap_or(0.0);
                            let field_id = egui::Id::new(("inspector-number", &node.id, &spec.key));
                            let stored_expression =
                                node.expressions.get(&spec.key).map(String::as_str);
                            let (number_changed, expression_change) = draw_number(
                                ui,
                                &mut number,
                                *step,
                                *min,
                                *max,
                                *kind,
                                field_id,
                                stored_expression,
                            );
                            if number_changed {
                                value = Value::from(number);
                                changed = true;
                            }
                            if let Some(expression) = expression_change {
                                output.expressions.push((spec.key.clone(), expression));
                            }
                        }
                        Widget::Choice(choices) => {
                            let mut selected = value.as_str().unwrap_or_default().to_owned();
                            if draw_choice(ui, &spec.key, &mut selected, choices) {
                                value = Value::String(selected);
                                changed = true;
                            }
                        }
                    }
                    ui.end_row();
                    if changed {
                        output.changes.push(Change { field: spec.key.clone(), value });
                    }
                }
            });
    }
    output
}

pub fn note(ui: &mut Ui, node: &GraphNode) {
    if let Some(text) = node_description(node) {
        ui.add_space(8.0);
        description(ui, &text);
    }
}

pub fn actions(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    action_buttons(ui, node, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(node_type: &str, values: Value) -> GraphNode {
        GraphNode { expressions: Default::default(),
            id: "n".into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        }
    }

    #[test]
    fn contract_fields_drive_static_inspector_specs() {
        let shell = known_specs(&node("shell", Value::Null));
        assert_eq!(shell.len(), 2);
        let thickness = shell.iter().find(|spec| spec.key == "thickness").expect("thickness");
        assert_eq!(thickness.default, serde_json::json!(2.0));
        assert!(matches!(thickness.widget, Widget::Number { min: Some(0.0), .. }));
        let direction = shell.iter().find(|spec| spec.key == "direction").expect("direction");
        assert_eq!(direction.default, serde_json::json!("inward"));
        assert!(matches!(direction.widget, Widget::Choice(_)));

        let mirror = known_specs(&node("mirror", Value::Null));
        let combine = mirror.iter().find(|spec| spec.key == "combine").expect("combine");
        assert_eq!(combine.default, serde_json::json!(true));
        assert!(matches!(combine.widget, Widget::Switch));

        let fillet = known_specs(&node("fillet", Value::Null));
        let continuity = fillet.iter().find(|spec| spec.key == "continuity").expect("continuity");
        assert_eq!(continuity.default, serde_json::json!(""));

        let box_specs = known_specs(&node("primitive", serde_json::json!({"primitiveType": "box"})));
        assert_eq!(box_specs.len(), 4);
        let length = box_specs.iter().find(|spec| spec.key == "length").expect("length");
        assert_eq!(length.default, serde_json::json!(40.0));
        assert!(matches!(length.widget, Widget::Number { min: Some(0.0), .. }));
        let cone = known_specs(&node("primitive", serde_json::json!({"primitiveType": "cone"})));
        assert!(cone.iter().any(|spec| spec.key == "bottomRadius"));
        assert!(cone.iter().any(|spec| spec.key == "topRadius"));
    }

    #[test]
    fn cad_dimensions_have_no_inspector_upper_bound() {
        let specs = known_specs(&node(
            "primitive",
            serde_json::json!({ "primitiveType": "box", "length": 5000.0 }),
        ));
        let length = specs.iter().find(|spec| spec.key == "length").expect("length");
        assert!(matches!(
            length.widget,
            Widget::Number { max: None, .. }
        ));
        let translate = known_specs(&node("translate", Value::Null));
        let x = translate.iter().find(|spec| spec.key == "x").expect("x");
        assert!(matches!(x.widget, Widget::Number { min: None, max: None, .. }));
    }

    #[test]
    fn step_file_name_drops_the_extension() {
        let step = node("stepimporter", serde_json::json!({ "path": "/tmp/models/MoonBaguette.stp" }));
        assert_eq!(step_file_name(&step).as_deref(), Some("MoonBaguette"));
        let no_path = node("stepimporter", serde_json::json!({}));
        assert_eq!(step_file_name(&no_path), None);
    }

    #[test]
    fn generate_translate_widgets() {
        let specs = known_specs(&node("translate", Value::Null));
        let keys: Vec<&str> = specs.iter().map(|spec| spec.key.as_str()).collect();
        assert_eq!(keys, ["space", "x", "y", "z", "rx", "ry", "rz"]);
        assert!(matches!(specs[1].widget, Widget::Number { .. }));
    }

    #[test]
    fn generate_primitive_widgets_from_type() {
        let cylinder = node("primitive", serde_json::json!({ "primitiveType": "cylinder" }));
        let specs = known_specs(&cylinder);
        let keys: Vec<&str> = specs.iter().map(|spec| spec.key.as_str()).collect();
        assert_eq!(keys, ["primitiveType", "radius", "height"]);
    }

    #[test]
    fn unknown_and_raw_fields_are_not_inspected() {
        let custom = node("custom", serde_json::json!({ "label": "x", "enabled": true, "size": 3 }));
        assert!(known_specs(&custom).is_empty());
        let extrude = node(
            "extrude",
            serde_json::json!({ "operation": "new", "faces": [{ "id": "face:1" }] }),
        );
        assert!(!known_specs(&extrude).iter().any(|spec| spec.key == "faces"));
    }

    #[test]
    fn boolean_and_fuse_generate_an_operation_inspector() {
        for node_type in ["boolean", "fuse"] {
            let specs = known_specs(&node(node_type, Value::Null));
            assert_eq!(specs.len(), 1);
            assert_eq!(specs[0].key, "mode");
            assert!(matches!(specs[0].widget, Widget::Choice(_)));
        }
    }

    #[test]
    fn reset_values_restores_settings_but_preserves_asset_data() {
        let look = GraphNode { expressions: Default::default(),
            id: "look".into(),
            node_type: "look".into(),
            values: serde_json::json!({"color": "#ff0000"}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let reset = reset_values(&look);
        assert!(reset.iter().any(|change| change.field == "color"
            && change.value == serde_json::json!("#7fa0cf")));

        let spline = GraphNode { expressions: Default::default(),
            id: "s".into(),
            node_type: "spline".into(),
            values: serde_json::json!({"degree": 5.0, "points": [[0.0, 0.0], [1.0, 1.0]]}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let reset = reset_values(&spline);
        assert!(reset.iter().any(|change| change.field == "degree" && change.value == serde_json::json!(3.0)));
        assert!(reset.iter().any(|change| change.field == "points"
            && change.value == serde_json::json!([[0.0, 0.0], [1.0, 1.0]])));

        let fillet = GraphNode { expressions: Default::default(),
            id: "f".into(),
            node_type: "fillet".into(),
            values: serde_json::json!({"length": 9.0, "edges": [{"id": "edge:1"}]}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let reset = reset_values(&fillet);
        assert!(reset.iter().any(|change| change.field == "edges"
            && change.value == serde_json::json!([])));
    }

}
