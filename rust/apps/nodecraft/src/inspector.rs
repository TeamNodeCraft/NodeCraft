//! Schema-driven inspector widgets for graph node values.
//!
//! Known node fields get deliberate widgets (text, switch, drag-value, slider,
//! choice). Raw/unknown JSON fields are not shown.

use crate::ui_shape;
use egui::{Response, Sense, Ui, Vec2};
use nodecraft_session::GraphNode;
use serde_json::Value;

#[derive(Clone, Debug)]
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
    /// Open a graph instance's definition.
    pub open_definition: bool,
    /// Rename the referenced local graph definition.
    pub graph_name: Option<String>,
    /// Resize the referenced local graph definition's interface.
    pub definition_ports: Option<(usize, usize)>,
    /// Resize the current document's root graph interface from a marker.
    pub root_interface_ports: Option<(Option<usize>, Option<usize>)>,
    /// Clone the referenced local definition for this instance.
    pub make_unique: bool,
    /// Import an external definition into the current document.
    pub make_local: bool,
    /// Resize the body input/output ports of an inline graph.
    pub graph_ports: Option<(usize, usize)>,
    /// Re-import the selected importer's changed source.
    pub reload_source: bool,
    /// Pick a replacement for a missing CAD original.
    pub locate_import: bool,
    /// Save the embedded copy beside the document and re-link the importer.
    pub save_import_copy: bool,
}

/// Paint the inspector reset control at an exact header rect. This is separate
/// from layout so the caller can centre it against the whole header rather than
/// only the title row.
pub fn reset_icon_button_at(ui: &mut Ui, rect: egui::Rect) -> Response {
    let theme = crate::theme::get();
    let response = ui.interact(rect, egui::Id::new("inspector-reset"), Sense::click());
    if response.hovered() || response.is_pointer_button_down_on() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        let alpha = if response.is_pointer_button_down_on() { 44 } else { 26 };
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(8),
            theme.with_alpha(theme.highlight, alpha),
        );
    }
    let colour = if response.hovered() { theme.color(theme.text) } else { theme.color(theme.icon) };
    let svg = std::str::from_utf8(include_bytes!("../../../../vendor/iconify-design/restore.svg"))
        .unwrap_or_default()
        .replace("currentColor", &crate::theme::Theme::hex(colour.to_array()));
    let image = egui::Image::from_bytes("bytes://nodecraft-inspector-reset.svg", svg.into_bytes())
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
    let kind = node.values.get("primitiveType").and_then(Value::as_str).unwrap_or("box");
    let mut specs =
        vec![choice("primitiveType", "Primitive", "box", nodecraft_node_contract::PRIMITIVE_TYPES)];
    specs.extend(contract_field_specs(nodecraft_node_contract::primitive_fields(kind)));
    specs
}

fn surface_loft_sections(ui: &mut Ui, sections: &[String], output: &mut Output) {
    let theme = crate::theme::get();
    ui.label(egui::RichText::new("SECTIONS").size(9.0).color(theme.with_alpha(theme.muted, 190)));
    if sections.is_empty() {
        ui.label(
            egui::RichText::new(
                "Connect sketch or spline profiles to the Sections port, in order.",
            )
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

/// The reference ids a topology field currently stores, in document order.
/// The inspector list and the viewport highlight read the same ids.
pub(crate) fn topology_reference_ids(node: &GraphNode, field: &str) -> Vec<String> {
    node.values
        .get(field)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("id").and_then(Value::as_str))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
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
    let mut references = topology_reference_ids(node, field);
    let title = if references.is_empty() {
        title.to_owned()
    } else {
        format!("{title}  {}", references.len())
    };
    egui::Frame::new()
        .fill(theme.with_alpha(theme.control, 90))
        .stroke(egui::Stroke::new(1.0, theme.with_alpha(theme.border, 170)))
        .corner_radius(6.0)
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(title).size(9.0).color(theme.with_alpha(theme.muted, 190)),
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
                        if active {
                            "bytes://inspector-edit-active.svg"
                        } else {
                            "bytes://inspector-edit-idle.svg"
                        },
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
            if references.is_empty() {
                ui.label(
                    egui::RichText::new(empty).size(10.0).color(theme.with_alpha(theme.muted, 220)),
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
                    && ui.small_button(clear_label).on_hover_text(clear_hint).clicked()
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

/// Whether a spline shows its control points. The contract owns the default;
/// this is the one read used by the point list label and any future reader.
fn spline_control_points(node: &GraphNode) -> bool {
    nodecraft_node_contract::resolve_bool("spline", &node.values, "control")
        .expect("spline.control is a contract switch")
}

fn spline_points(ui: &mut Ui, node: &GraphNode, output: &mut Output) -> Vec<IndexedVectorRow> {
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
    let control = spline_control_points(node);
    ui.label(
        egui::RichText::new(if control { "CONTROL POINTS" } else { "FIT POINTS" })
            .size(9.0)
            .color(theme.with_alpha(theme.muted, 190)),
    );
    let mut changed = false;
    let mut remove = None;
    let row_count = points.len();
    let mut laid_out = Vec::with_capacity(row_count);
    for (index, (x, y)) in points.iter_mut().enumerate() {
        let mut row_changed = false;
        let row =
            indexed_vector_row(ui, index, row_count, 2, "Remove point", |ui, field, width| {
                let (value, axis, prefix, id) = match field {
                    0 => (&mut *x, "x", "X ", egui::Id::new(("spline-x", &node.id, index))),
                    1 => (&mut *y, "y", "Y ", egui::Id::new(("spline-y", &node.id, index))),
                    _ => unreachable!("a spline point has exactly two values"),
                };
                let expression_key = format!("points.{index}.{axis}");
                let response = ui.add_sized(
                    [width, FIELD_ROW_HEIGHT],
                    crate::expression_input::number_drag(
                        ui,
                        value,
                        nodecraft_node_contract::NumberKind::Length,
                        0.5,
                        2,
                        id,
                        node.expressions.get(&expression_key).map(String::as_str),
                    )
                    .range(-10000.0..=10000.0)
                    .prefix(prefix),
                );
                crate::expression_input::remember_rect(id, response.rect);
                if let Some(expression) = crate::expression_input::take_expression_change(id) {
                    output.expressions.push((expression_key, expression));
                }
                row_changed |= response.changed();
            });
        changed |= row_changed;
        if row.remove.clicked() {
            remove = Some(index);
        }
        laid_out.push(row);
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
            points.into_iter().map(|(x, y)| serde_json::json!({"x": x, "y": y})).collect(),
        );
        output.changes.push(Change { field: "points".to_owned(), value });
    }
    laid_out
}

fn parse_look_color(text: &str) -> Option<egui::Color32> {
    let hex = text.trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }
    let channel = |range: std::ops::Range<usize>| u8::from_str_radix(&hex[range], 16).ok();
    Some(egui::Color32::from_rgb(channel(0..2)?, channel(2..4)?, channel(4..6)?))
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
                nodecraft_node_contract::FieldDefault::Text(value) => {
                    Value::String(value.to_owned())
                }
                nodecraft_node_contract::FieldDefault::Bool(value) => Value::Bool(value),
                nodecraft_node_contract::FieldDefault::Number(value) => Value::from(value),
                nodecraft_node_contract::FieldDefault::Choice(value) => {
                    Value::String(value.to_owned())
                }
            };
            let widget = match spec.kind {
                nodecraft_node_contract::FieldKind::Text => Widget::Text,
                nodecraft_node_contract::FieldKind::Switch => Widget::Switch,
                nodecraft_node_contract::FieldKind::Number { min, max, step, kind } => {
                    Widget::Number { min, max, step, kind }
                }
                nodecraft_node_contract::FieldKind::Choice(choices) => Widget::Choice(choices),
            };
            FieldSpec { key: spec.key.to_owned(), label: spec.label.to_owned(), default, widget }
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
        "stepimporter" => vec![text("path", "CAD path", ""), switch("heal", "Heal", true)],
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
    let changes: Vec<Change> =
        values.into_iter().map(|(field, value)| Change { field, value }).collect();
    changes
}

fn current_value(node: &GraphNode, spec: &FieldSpec) -> Value {
    node.values.get(spec.key.as_str()).cloned().unwrap_or_else(|| spec.default.clone())
}

/// One inspector control row. The label column and the control column are the
/// same width for every field, on every node: number boxes, dropdowns, text
/// entries and switches all line up instead of sizing themselves from their
/// longest label or current value.
const FIELD_ROW_HEIGHT: f32 = 24.0;
const FIELD_ROW_GAP: f32 = 7.0;
const FIELD_COLUMN_GAP: f32 = 10.0;
const LABEL_COLUMN_SHARE: f32 = 0.40;

fn field_columns_for(total: f32) -> (f32, f32) {
    let label = (total * LABEL_COLUMN_SHARE).clamp(72.0, 140.0);
    let control = (total - label - FIELD_COLUMN_GAP).max(64.0);
    (label, control)
}

fn field_columns(ui: &Ui) -> (f32, f32) {
    field_columns_for(ui.available_width())
}

fn field_row(ui: &mut Ui, label: &str, add_control: impl FnOnce(&mut Ui, f32)) {
    let (label_width, control_width) = field_columns(ui);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = FIELD_COLUMN_GAP;
        ui.set_height(FIELD_ROW_HEIGHT);
        ui.allocate_ui_with_layout(
            egui::vec2(label_width, FIELD_ROW_HEIGHT),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_width(label_width);
                ui.add(egui::Label::new(label).truncate()).on_hover_text(label);
            },
        );
        ui.allocate_ui_with_layout(
            egui::vec2(control_width, FIELD_ROW_HEIGHT),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_width(control_width);
                ui.spacing_mut().interact_size.y = FIELD_ROW_HEIGHT;
                add_control(ui, control_width);
            },
        );
    });
    ui.add_space(FIELD_ROW_GAP);
}

/// The laid-out columns of one indexed vector row. `fields` are the value cells
/// (X, Y, …) and `remove` is the remove button; both use the same padded content
/// width as `field_row`, so they line up with every other inspector control.
struct IndexedVectorRow {
    // Production only needs the remove response; the headless layout test reads
    // these rects to pin the columns. The test covers this exception.
    #[allow(dead_code)]
    index: egui::Rect,
    #[allow(dead_code)]
    fields: Vec<egui::Rect>,
    remove: Response,
}

/// Width of the right-aligned index column. The widest index is measured from
/// the same 10 pt face the label uses, with room for at least two digits so the
/// column does not resize as points are added or removed.
fn indexed_vector_index_width(ui: &Ui, rows: usize) -> f32 {
    let widest = format!("{}.", rows.max(99));
    ui.painter()
        .layout_no_wrap(widest, egui::FontId::proportional(10.0), ui.visuals().text_color())
        .size()
        .x
}

/// One inspector row for an indexed vector list: a fixed index column, `fields`
/// equal-width value cells, then the remove button on the padded content edge.
/// Callers add their own widgets through `add_field`; every value cell is handed
/// the same width so no field sizes itself from its text.
fn indexed_vector_row(
    ui: &mut Ui,
    index: usize,
    rows: usize,
    fields: usize,
    remove_hint: &str,
    mut add_field: impl FnMut(&mut Ui, usize, f32),
) -> IndexedVectorRow {
    let content_width = ui.available_width();
    let index_width = indexed_vector_index_width(ui, rows);
    // A remove button is square at the row height, leaving the rest of the
    // content width to the equal value cells.
    let remove_width = FIELD_ROW_HEIGHT;
    let field_width =
        ((content_width - index_width - remove_width - FIELD_COLUMN_GAP * (fields + 1) as f32)
            / fields.max(1) as f32)
            .max(1.0);
    let theme = crate::theme::get();
    let mut index_rect = egui::Rect::ZERO;
    let mut field_rects = Vec::with_capacity(fields);
    let mut remove = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = FIELD_COLUMN_GAP;
        ui.set_height(FIELD_ROW_HEIGHT);
        index_rect = ui
            .allocate_ui_with_layout(
                egui::vec2(index_width, FIELD_ROW_HEIGHT),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    ui.set_width(index_width);
                    ui.label(
                        egui::RichText::new(format!("{}.", index + 1))
                            .size(10.0)
                            .color(theme.with_alpha(theme.muted, 220)),
                    );
                },
            )
            .response
            .rect;
        for field in 0..fields {
            let response = ui.allocate_ui_with_layout(
                egui::vec2(field_width, FIELD_ROW_HEIGHT),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.set_width(field_width);
                    ui.spacing_mut().interact_size.y = FIELD_ROW_HEIGHT;
                    add_field(ui, field, field_width);
                },
            );
            field_rects.push(response.response.rect);
        }
        remove = Some(
            ui.allocate_ui_with_layout(
                egui::vec2(remove_width, FIELD_ROW_HEIGHT),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    ui.set_width(remove_width);
                    ui.small_button("×").on_hover_text(remove_hint)
                },
            )
            .inner,
        );
    });
    ui.add_space(FIELD_ROW_GAP);
    IndexedVectorRow {
        index: index_rect,
        fields: field_rects,
        remove: remove.expect("an indexed vector row always lays out its remove button"),
    }
}

/// Lay out one boolean row: the switch only needs its own width, so the label
/// gets the rest of the row minus the column gap. The switch is right-aligned
/// on the same edge as every other inspector control. Returns the label and
/// switch responses; `field_switch_row` exposes the switch response to callers.
/// `tooltip` replaces the label's default hover text when a row needs help text.
fn field_switch_row_layout(
    ui: &mut Ui,
    label: &str,
    value: &mut bool,
    tooltip: Option<&str>,
) -> (Response, Response) {
    let switch_width = crate::widgets::SWITCH_SIZE.x;
    let label_width = (ui.available_width() - switch_width - FIELD_COLUMN_GAP).max(0.0);
    let mut label_response = None;
    let mut switch_response = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = FIELD_COLUMN_GAP;
        ui.set_height(FIELD_ROW_HEIGHT);
        ui.allocate_ui_with_layout(
            egui::vec2(label_width, FIELD_ROW_HEIGHT),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_width(label_width);
                label_response = Some(
                    ui.add(egui::Label::new(label).truncate())
                        .on_hover_text(tooltip.unwrap_or(label)),
                );
            },
        );
        ui.allocate_ui_with_layout(
            egui::vec2(switch_width, FIELD_ROW_HEIGHT),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                ui.set_width(switch_width);
                ui.spacing_mut().interact_size.y = FIELD_ROW_HEIGHT;
                switch_response = Some(crate::widgets::switch(ui, value));
            },
        );
    });
    ui.add_space(FIELD_ROW_GAP);
    (
        label_response.expect("a switch row always lays out its label"),
        switch_response.expect("a switch row always lays out its switch"),
    )
}

/// One boolean inspector row. See `field_switch_row_layout` for the geometry.
fn field_switch_row(ui: &mut Ui, label: &str, value: &mut bool, tooltip: Option<&str>) -> Response {
    field_switch_row_layout(ui, label, value, tooltip).1
}

fn draw_choice(
    ui: &mut Ui,
    key: &str,
    value: &mut String,
    choices: &[(&str, &str)],
    width: f32,
) -> bool {
    let selected = choices
        .iter()
        .find(|(candidate, _)| *candidate == value.as_str())
        .map(|(_, label)| *label)
        .unwrap_or(value.as_str());
    let mut changed = false;
    egui::ComboBox::from_id_salt(format!("inspector-{key}"))
        .selected_text(selected)
        .width(width)
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
    focus: bool,
) -> (bool, Option<Option<String>>) {
    let mut changed = false;
    let control_width = ui.available_width();
    ui.horizontal(|ui| {
        // Use the field's full control width, not the nested horizontal's
        // available width: the nested layout subtracts its own indent.
        ui.set_width(control_width);
        ui.spacing_mut().item_spacing.x = 4.0;
        let button_size = Vec2::new(24.0, FIELD_ROW_HEIGHT);
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
        let entry_width = (control_width - button_size.x * 2.0 - 8.0).max(40.0);
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
        if focus {
            ui.memory_mut(|memory| memory.request_focus(entry.id));
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
    let metadata = nodecraft_node_contract::metadata(&node.node_type);
    if metadata
        .is_some_and(|metadata| metadata.icon_kind == nodecraft_node_contract::IconKind::Primitive)
    {
        let kind = node.values.get("primitiveType").and_then(Value::as_str).unwrap_or("box");
        return nodecraft_node_contract::primitive_label(kind)
            .map_or_else(|| title_case(kind), str::to_owned);
    }
    if let Some(metadata) = metadata {
        return metadata.label.to_owned();
    }
    title_case(&node.node_type.replace(['_', '-'], " "))
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
    nodecraft_node_contract::metadata(&node.node_type)
        .map_or("Node", |metadata| metadata.category)
        .to_owned()
}

fn section_title(node: &GraphNode) -> &'static str {
    nodecraft_node_contract::metadata(&node.node_type).map_or("VALUES", |metadata| metadata.section)
}

fn node_description(node: &GraphNode) -> Option<String> {
    nodecraft_node_contract::description(&node.node_type, &node.values)
}

fn graph_port_count(node: &GraphNode, field: &str) -> usize {
    node.values.get(field).and_then(|value| value.as_array()).map_or(0, Vec::len)
}

fn graph_instance_fields(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    let mut name = node.values.get("name").and_then(Value::as_str).unwrap_or("Graph").to_owned();
    let definition = match node.values.get("definition").and_then(Value::as_str) {
        Some(id) => format!("Local · {id}"),
        None => node
            .values
            .get("external")
            .and_then(|external| external.get("path"))
            .and_then(Value::as_str)
            .map(|path| format!("External · {path}"))
            .unwrap_or_else(|| "Unresolved".to_owned()),
    };
    let inputs = node.values.get("inputs").and_then(Value::as_array).map_or(0, Vec::len);
    let outputs = node.values.get("outputs").and_then(Value::as_array).map_or(0, Vec::len);
    let is_local = node.values.get("definition").and_then(Value::as_str).is_some();
    field_row(ui, "Name", |ui, control_width| {
        if ui
            .add_sized(
                [control_width, FIELD_ROW_HEIGHT],
                egui::TextEdit::singleline(&mut name).hint_text("Graph name"),
            )
            .changed()
        {
            output.graph_name = Some(name.clone());
        }
    });
    field_row(ui, "Definition", |ui, _| {
        ui.label(definition);
    });
    if is_local || node.values.get("external").is_some() {
        let mut input_count = inputs as f64;
        let mut output_count = outputs as f64;
        let mut changed = false;
        field_row(ui, "Num inputs", |ui, _| {
            let (edited, _) = draw_number(
                ui,
                &mut input_count,
                1.0,
                Some(0.0),
                Some(nodecraft_session::groups::MAX_GRAPH_PORTS as f64),
                nodecraft_node_contract::NumberKind::Unitless,
                egui::Id::new(("graph-definition-inputs", &node.id)),
                None,
                false,
            );
            changed |= edited;
        });
        field_row(ui, "Num outputs", |ui, _| {
            let (edited, _) = draw_number(
                ui,
                &mut output_count,
                1.0,
                Some(0.0),
                Some(nodecraft_session::groups::MAX_GRAPH_PORTS as f64),
                nodecraft_node_contract::NumberKind::Unitless,
                egui::Id::new(("graph-definition-outputs", &node.id)),
                None,
                false,
            );
            changed |= edited;
        });
        if changed {
            output.definition_ports =
                Some((input_count.round() as usize, output_count.round() as usize));
        }
    } else {
        field_row(ui, "Inputs", |ui, _| {
            ui.label(inputs.to_string());
        });
        field_row(ui, "Outputs", |ui, _| {
            ui.label(outputs.to_string());
        });
    }
    ui.add_space(6.0);
    if ui_shape::button(
        ui,
        "Open graph",
        ui.available_width(),
        30.0,
        crate::theme::get().with_alpha(crate::theme::get().control, 220),
        egui::Stroke::new(1.0, crate::theme::get().with_alpha(crate::theme::get().border, 210)),
        crate::theme::get().color(crate::theme::get().text),
    )
    .clicked()
    {
        output.open_definition = true;
    }
    if !is_local
        && ui_shape::button(
            ui,
            "Make local",
            ui.available_width(),
            30.0,
            crate::theme::get().with_alpha(crate::theme::get().control, 220),
            egui::Stroke::new(1.0, crate::theme::get().with_alpha(crate::theme::get().border, 210)),
            crate::theme::get().color(crate::theme::get().text),
        )
        .clicked()
    {
        output.make_local = true;
    }
    if is_local
        && ui_shape::button(
            ui,
            "Make unique",
            ui.available_width(),
            30.0,
            crate::theme::get().with_alpha(crate::theme::get().control, 220),
            egui::Stroke::new(1.0, crate::theme::get().with_alpha(crate::theme::get().border, 210)),
            crate::theme::get().color(crate::theme::get().text),
        )
        .clicked()
    {
        output.make_unique = true;
    }
}

fn graph_fields(ui: &mut Ui, node: &GraphNode, output: &mut Output, editable_ports: bool) {
    let mut name = node.values.get("name").and_then(Value::as_str).unwrap_or_default().to_owned();
    field_row(ui, "Name", |ui, control_width| {
        if ui
            .add_sized(
                [control_width, FIELD_ROW_HEIGHT],
                egui::TextEdit::singleline(&mut name).hint_text("Graph name"),
            )
            .changed()
        {
            output
                .changes
                .push(Change { field: "name".to_owned(), value: Value::String(name.clone()) });
        }
    });
    let inputs = graph_port_count(node, "inputs");
    let outputs = graph_port_count(node, "outputs");
    if !editable_ports {
        field_row(ui, "Num inputs", |ui, _| {
            ui.label(inputs.to_string());
        });
        field_row(ui, "Num outputs", |ui, _| {
            ui.label(outputs.to_string());
        });
        return;
    }
    let mut input_value = inputs as f64;
    let mut output_value = outputs as f64;
    let mut changed = false;
    field_row(ui, "Num inputs", |ui, _| {
        let (edited, _) = draw_number(
            ui,
            &mut input_value,
            1.0,
            Some(0.0),
            Some(nodecraft_session::groups::MAX_GRAPH_PORTS as f64),
            nodecraft_node_contract::NumberKind::Unitless,
            egui::Id::new(("graph-input-ports", &node.id)),
            None,
            false,
        );
        changed |= edited;
    });
    field_row(ui, "Num outputs", |ui, _| {
        let (edited, _) = draw_number(
            ui,
            &mut output_value,
            1.0,
            Some(0.0),
            Some(nodecraft_session::groups::MAX_GRAPH_PORTS as f64),
            nodecraft_node_contract::NumberKind::Unitless,
            egui::Id::new(("graph-output-ports", &node.id)),
            None,
            false,
        );
        changed |= edited;
    });
    if changed {
        output.graph_ports = Some((input_value.round() as usize, output_value.round() as usize));
    }
}

fn marker_ports(node: &GraphNode) -> Vec<String> {
    if let Some(ports) = node.values.get("ports").and_then(Value::as_array) {
        return ports.iter().filter_map(|port| port.as_str().map(str::to_owned)).collect();
    }
    node.values.get("port").and_then(Value::as_str).map(str::to_owned).into_iter().collect()
}

fn marker_interface_fields(ui: &mut Ui, node: &GraphNode, output: &mut Output, editable: bool) {
    let ports = marker_ports(node);
    if !editable {
        field_row(ui, "Ports", |ui, _| {
            let text = if ports.is_empty() { "None".to_owned() } else { ports.join(", ") };
            ui.label(text);
        });
        return;
    }
    let is_input = node.node_type == "groupinput";
    let label = if is_input { "Num inputs" } else { "Num outputs" };
    let mut count = ports.len() as f64;
    let mut changed = false;
    field_row(ui, label, |ui, _| {
        let (edited, _) = draw_number(
            ui,
            &mut count,
            1.0,
            Some(0.0),
            Some(nodecraft_session::groups::MAX_GRAPH_PORTS as f64),
            nodecraft_node_contract::NumberKind::Unitless,
            egui::Id::new(("root-interface-ports", &node.id)),
            None,
            false,
        );
        changed |= edited;
    });
    if changed {
        let count = Some(count.round() as usize);
        output.root_interface_ports = Some(if is_input { (count, None) } else { (None, count) });
    }
}

fn section_label(ui: &mut Ui, text: &str) {
    let theme = crate::theme::get();
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(text).size(10.0).strong().color(theme.with_alpha(theme.muted, 220)),
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
            output
                .changes
                .push(Change { field: "path".to_owned(), value: Value::String(path.clone()) });
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
    description(
        ui,
        "Opens this .nodecraft document as an editable tab. Linked parts may themselves link further documents.",
    );
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

fn step_import_field(key: &str) -> &'static nodecraft_node_contract::NodeFieldSpec {
    nodecraft_node_contract::field_spec("stepimporter", &Value::Null, key)
        .unwrap_or_else(|| panic!("stepimporter.{key} is a contract field"))
}

/// The one rule for the path-dependent controls: an importer with no path has
/// nothing to point at, embed or watch, so those controls are disabled together.
fn import_controls_enabled(path: &str) -> bool {
    !path.is_empty()
}

const MONTHS: [&str; 12] =
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{} KB", (bytes + KB / 2) / KB)
    } else {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    }
}

/// The short date shown for an embedded copy, e.g. `16 Nov`. The stored value
/// is UTC ISO-8601; only its date part matters here. `None` keeps callers from
/// inventing a date for older records.
fn short_embedded_date(embedded_at: &str) -> Option<String> {
    let date = embedded_at.split('T').next()?;
    let mut parts = date.split('-');
    let _year = parts.next()?;
    let month: usize = parts.next()?.parse().ok()?;
    let day: u32 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&month) || day == 0 {
        return None;
    }
    Some(format!("{day} {}", MONTHS[month - 1]))
}

fn embedded_copy_status_text(copy: &nodecraft_engine::import_io::EmbeddedCopyStatus) -> String {
    let mut details = format!("{}, {}", copy.file_name, human_size(copy.byte_length));
    if let Some(date) = copy.embedded_at.as_deref().and_then(short_embedded_date) {
        details.push_str(&format!(", embedded {date}"));
    }
    format!("Original not found \u{2014} using embedded copy ({details})")
}

/// One inspector action button with the same geometry and colours as Choose….
fn import_action_button(ui: &mut Ui, label: &str, width: f32) -> Response {
    let theme = crate::theme::get();
    ui_shape::button(
        ui,
        label,
        width,
        26.0,
        theme.color(theme.control),
        egui::Stroke::new(1.0, theme.with_alpha(theme.border, 190)),
        theme.color(theme.text),
    )
}

/// Draw the engine-status line directly under the path field, plus the actions
/// that make sense for it. This never stats or hashes a file itself.
fn import_status_fields(
    ui: &mut Ui,
    status: &nodecraft_engine::import_io::ImportStatus,
    output: &mut Output,
) {
    use nodecraft_engine::import_io::ImportStatus;
    let theme = crate::theme::get();
    match status {
        ImportStatus::Available { kind, path } => {
            let modifier = if *kind == nodecraft_engine::import_io::ImportSourceKind::Alternate {
                " (alternate path)"
            } else {
                ""
            };
            ui.label(
                egui::RichText::new(format!("{}{modifier}", path.display()))
                    .size(10.5)
                    .color(theme.color(theme.muted)),
            );
        }
        ImportStatus::Ingested { copy, .. } => {
            ui.label(
                egui::RichText::new(embedded_copy_status_text(copy))
                    .size(10.5)
                    .color(theme.color(theme.node_warning)),
            );
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                if import_action_button(ui, nodecraft_node_contract::STEP_IMPORT_LOCATE_LABEL, 74.0)
                    .clicked()
                {
                    output.locate_import = true;
                }
                if import_action_button(
                    ui,
                    nodecraft_node_contract::STEP_IMPORT_SAVE_COPY_LABEL,
                    116.0,
                )
                .clicked()
                {
                    output.save_import_copy = true;
                }
            });
        }
        ImportStatus::Missing { path } => {
            ui.label(
                egui::RichText::new(format!("Original not found: {}", path.display()))
                    .size(10.5)
                    .color(theme.color(theme.node_error)),
            );
            if import_action_button(ui, nodecraft_node_contract::STEP_IMPORT_LOCATE_LABEL, 74.0)
                .clicked()
            {
                output.locate_import = true;
            }
        }
    }
}

fn step_import_fields(
    ui: &mut Ui,
    node: &GraphNode,
    output: &mut Output,
    document_directory: Option<&std::path::Path>,
    source_changed: bool,
    import_status: Option<&nodecraft_engine::import_io::ImportStatus>,
) {
    let theme = crate::theme::get();
    let mut path = node.values.get("path").and_then(Value::as_str).unwrap_or_default().to_owned();
    ui.horizontal(|ui| {
        let button_width = 76.0;
        let path_width = (ui.available_width() - button_width - 8.0).max(80.0);
        let changed = ui
            .add_sized(
                [path_width, 26.0],
                egui::TextEdit::singleline(&mut path)
                    .hint_text("Path to .step, .stp, .FCStd or .3dm"),
            )
            .changed();
        if changed {
            output
                .changes
                .push(Change { field: "path".to_owned(), value: Value::String(path.clone()) });
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
    // An unconfigured importer is neutral, not an error: point at the next
    // action in the same muted style as the other secondary source notes.
    if path.is_empty() {
        ui.label(
            egui::RichText::new("Choose a CAD file").size(10.5).color(theme.color(theme.muted)),
        );
    }
    // The path-dependent controls share one enabled rule: no path, no source to
    // point at, embed or watch. Heal stays available.
    let controls_enabled = import_controls_enabled(&path);
    if let Some(status) = import_status {
        import_status_fields(ui, status, output);
    }
    if !path.is_empty()
        && let Some(schema) = step_schema(ui, &path)
    {
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
    if source_changed {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Source changed")
                    .size(11.0)
                    .color(theme.color(theme.node_warning)),
            );
            if ui_shape::button(
                ui,
                "Reload?",
                66.0,
                26.0,
                theme.color(theme.control),
                egui::Stroke::new(1.0, theme.with_alpha(theme.border, 190)),
                theme.color(theme.text),
            )
            .clicked()
            {
                output.reload_source = true;
            }
        });
    }
    let path_mode = step_import_field("pathMode");
    let path_mode_choices = match path_mode.kind {
        nodecraft_node_contract::FieldKind::Choice(choices) => choices,
        _ => unreachable!("stepimporter.pathMode is a choice field"),
    };
    let mut mode = nodecraft_engine::import_io::path_mode(&node.values).to_owned();
    let mut selected_mode = None;
    ui.add_enabled_ui(controls_enabled, |ui| {
        field_row(ui, path_mode.label, |ui, control_width| {
            if draw_choice(ui, path_mode.key, &mut mode, path_mode_choices, control_width) {
                selected_mode = Some(mode.clone());
            }
        });
    });
    if let Some(mode) = selected_mode
        && let Some((path, relative, warning)) =
            nodecraft_engine::import_io::switch_path_mode(&node.values, &mode, document_directory)
    {
        output
            .changes
            .push(Change { field: "pathMode".to_owned(), value: Value::String(mode.clone()) });
        output.changes.push(Change { field: "path".to_owned(), value: Value::String(path) });
        output.changes.push(Change {
            field: "relativePath".to_owned(),
            value: Value::String(relative.unwrap_or_default()),
        });
        output.changes.push(Change {
            field: "pathWarning".to_owned(),
            value: Value::String(warning.unwrap_or_default()),
        });
    }
    let mut ingest = nodecraft_node_contract::resolve_bool("stepimporter", &node.values, "ingest")
        .unwrap_or(false);
    let ingest_changed = ui
        .add_enabled_ui(controls_enabled, |ui| {
            field_switch_row(
                ui,
                step_import_field("ingest").label,
                &mut ingest,
                Some("Save a copy of the CAD file inside this document, used if the original goes missing."),
            )
        })
        .inner
        .changed();
    if ingest_changed {
        output.changes.push(Change { field: "ingest".to_owned(), value: Value::Bool(ingest) });
    }
    let mut watch = nodecraft_node_contract::resolve_bool("stepimporter", &node.values, "watch")
        .unwrap_or(false);
    let watch_tooltip = if ingest {
        "Re-import automatically when the CAD file changes on disk, and update the embedded copy."
    } else {
        "Re-import automatically when the CAD file changes on disk."
    };
    let watch_changed = ui
        .add_enabled_ui(controls_enabled, |ui| {
            field_switch_row(ui, step_import_field("watch").label, &mut watch, Some(watch_tooltip))
        })
        .inner
        .changed();
    if watch_changed {
        output.changes.push(Change { field: "watch".to_owned(), value: Value::Bool(watch) });
    }
    let mut heal = node.values.get("heal").and_then(Value::as_bool).unwrap_or(true);
    if field_switch_row(ui, "Heal imported B-rep", &mut heal, None).changed() {
        output.changes.push(Change { field: "heal".to_owned(), value: Value::Bool(heal) });
    }
}

fn export_fields(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    let theme = crate::theme::get();
    let mut name =
        node.values.get("fileName").and_then(Value::as_str).unwrap_or("model").to_owned();
    let mut directory =
        node.values.get("directory").and_then(Value::as_str).unwrap_or_default().to_owned();
    let mut overwrite = node.values.get("overwrite").and_then(Value::as_bool).unwrap_or(false);
    field_row(ui, "File name", |ui, control_width| {
        if ui
            .add_sized(
                [control_width, FIELD_ROW_HEIGHT],
                egui::TextEdit::singleline(&mut name).hint_text("model"),
            )
            .changed()
        {
            output
                .changes
                .push(Change { field: "fileName".to_owned(), value: Value::String(name.clone()) });
        }
    });

    field_row(ui, "Directory", |ui, control_width| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let button_width = 76.0;
            let path_width = (control_width - button_width - 6.0).max(60.0);
            if ui
                .add_sized(
                    [path_width, FIELD_ROW_HEIGHT],
                    egui::TextEdit::singleline(&mut directory).hint_text("STEP output folder"),
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
                FIELD_ROW_HEIGHT,
                theme.color(theme.control),
                egui::Stroke::new(1.0, theme.with_alpha(theme.border, 190)),
                theme.color(theme.text),
            )
            .clicked()
            {
                output.choose_directory = true;
            }
        });
    });

    if field_switch_row(ui, "Overwrite", &mut overwrite, None).changed() {
        output
            .changes
            .push(Change { field: "overwrite".to_owned(), value: Value::Bool(overwrite) });
    }
    let _ = theme;
}

fn action_buttons(ui: &mut Ui, node: &GraphNode, output: &mut Output) {
    let theme = crate::theme::get();
    if nodecraft_session::groups::is_graph_interface_node(node) {
        ui.add_space(10.0);
        ui.label(
            egui::RichText::new("Graph interface nodes are structural and cannot be deleted.")
                .size(11.0)
                .color(theme.with_alpha(theme.muted, 220)),
        );
        return;
    }
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
    editable_ports: bool,
    editable_root_ports: bool,
    document_directory: Option<&std::path::Path>,
    source_changed: bool,
    import_status: Option<&nodecraft_engine::import_io::ImportStatus>,
    focus_field: Option<&str>,
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
        step_import_fields(
            ui,
            node,
            &mut output,
            document_directory,
            source_changed,
            import_status,
        );
    } else if node.node_type == "linkedpart" {
        section_label(ui, section_title(node));
        linked_part_fields(ui, node, &mut output);
    } else if node.node_type == "exporter" {
        section_label(ui, section_title(node));
        export_fields(ui, node, &mut output);
    } else if node.node_type == "group" {
        section_label(ui, section_title(node));
        graph_fields(ui, node, &mut output, editable_ports);
    } else if node.node_type == "graph" {
        section_label(ui, section_title(node));
        graph_instance_fields(ui, node, &mut output);
    } else if matches!(node.node_type.as_str(), "groupinput" | "groupoutput") {
        section_label(ui, section_title(node));
        marker_interface_fields(ui, node, &mut output, editable_root_ports);
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
                if sketch_editing { theme.color(theme.control) } else { theme.color(theme.text) },
            )
            .clicked()
            {
                output.toggle_sketch_edit = true;
            }
            ui.add_space(8.0);
        }
        if node.node_type == "spline" {
            let _ = spline_points(ui, node, &mut output);
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
        for spec in &specs {
            let mut value = current_value(node, spec);
            let mut changed = false;
            if matches!(&spec.widget, Widget::Switch) {
                let mut enabled = value.as_bool().unwrap_or(false);
                if field_switch_row(ui, &spec.label, &mut enabled, None).changed() {
                    value = Value::Bool(enabled);
                    changed = true;
                }
            } else {
                field_row(ui, &spec.label, |ui, control_width| match &spec.widget {
                    Widget::Text => {
                        let mut text = value.as_str().unwrap_or_default().to_owned();
                        if ui
                            .add_sized(
                                [control_width, FIELD_ROW_HEIGHT],
                                egui::TextEdit::singleline(&mut text),
                            )
                            .changed()
                        {
                            value = Value::String(text);
                            changed = true;
                        }
                    }
                    Widget::Switch => unreachable!("switch rows use field_switch_row"),
                    Widget::Number { min, max, step, kind } => {
                        let mut number = value.as_f64().unwrap_or(0.0);
                        let field_id = egui::Id::new(("inspector-number", &node.id, &spec.key));
                        let stored_expression = node.expressions.get(&spec.key).map(String::as_str);
                        let (number_changed, expression_change) = draw_number(
                            ui,
                            &mut number,
                            *step,
                            *min,
                            *max,
                            *kind,
                            field_id,
                            stored_expression,
                            focus_field == Some(spec.key.as_str()),
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
                        if draw_choice(ui, &spec.key, &mut selected, choices, control_width) {
                            value = Value::String(selected);
                            changed = true;
                        }
                    }
                });
            }
            if changed {
                output.changes.push(Change { field: spec.key.clone(), value });
            }
        }
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
        GraphNode {
            expressions: Default::default(),
            id: "n".into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        }
    }

    #[test]
    fn switch_rows_leave_the_label_the_rest_of_the_row() {
        // The default inspector panel is 280 px wide with 16 px insets on each
        // side, so a field row gets 248 px.
        const CONTENT_WIDTH: f32 = 248.0;
        let ctx = egui::Context::default();
        let mut value = false;
        let mut labels = Vec::new();
        let mut switch_right = None;
        let mut control_right = None;
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(CONTENT_WIDTH, 200.0),
                )),
                ..Default::default()
            },
            |ui| {
                ui.set_width(CONTENT_WIDTH);
                for label in ["Auto-reload", "Embed file"] {
                    let (label_response, switch_response) =
                        field_switch_row_layout(ui, label, &mut value, None);
                    let galley = ui.painter().layout_no_wrap(
                        label.to_owned(),
                        egui::TextStyle::Body.resolve(ui.style()),
                        ui.visuals().text_color(),
                    );
                    labels.push((label, label_response.rect.width(), galley.size().x));
                    switch_right = Some(switch_response.rect.right());
                }
                field_row(ui, "Path", |ui, _| {
                    control_right = Some(ui.max_rect().right());
                });
            },
        );
        output.textures_delta.clear();
        let switch_right = switch_right.expect("switch laid out");
        let control_right = control_right.expect("control laid out");
        assert_eq!(
            switch_right, control_right,
            "a switch shares the right edge of the popup/control column"
        );
        for (label, available, needed) in labels {
            assert!(
                available >= needed,
                "{label:?} label has {available} px but needs {needed} px"
            );
        }
    }

    #[test]
    fn spline_point_rows_fill_and_share_the_inspector_columns() {
        // The default inspector panel is 280 px wide with 16 px insets on each
        // side, so the padded content width is 248 px.
        const CONTENT_WIDTH: f32 = 248.0;
        const ROWS: usize = 9;
        let ctx = egui::Context::default();
        let points: Vec<Value> = (0..ROWS)
            .map(|index| {
                serde_json::json!({
                    "x": -205.33 + index as f64,
                    "y": 50.4 - index as f64,
                })
            })
            .collect();
        let node = node("spline", serde_json::json!({ "points": points }));
        let mut output = Output::default();
        let mut rows = Vec::new();
        let mut popup_right = None;
        let mut frame = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(CONTENT_WIDTH, 1200.0),
                )),
                ..Default::default()
            },
            |ui| {
                ui.set_width(CONTENT_WIDTH);
                rows = spline_points(ui, &node, &mut output);
                let mut value = "a".to_owned();
                field_row(ui, "Popup", |ui, width| {
                    draw_choice(ui, "test-popup", &mut value, &[("a", "A")], width);
                    popup_right = Some(ui.max_rect().right());
                });
            },
        );
        frame.textures_delta.clear();

        assert_eq!(rows.len(), ROWS, "every point should lay out one row");
        let first = &rows[0];
        assert_eq!(first.fields.len(), 2, "a point row has X and Y fields");
        let popup_right = popup_right.expect("the popup row laid out");
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row.index.width(), first.index.width(), "row {index} index width");
            assert_eq!(row.index.right(), first.index.right(), "row {index} index right");
            assert_eq!(row.fields[0].left(), first.fields[0].left(), "row {index} X left");
            assert_eq!(row.fields[0].right(), first.fields[0].right(), "row {index} X right");
            assert_eq!(row.fields[1].left(), first.fields[1].left(), "row {index} Y left");
            assert_eq!(row.fields[1].right(), first.fields[1].right(), "row {index} Y right");
            assert_eq!(row.fields[0].width(), row.fields[1].width(), "row {index} X/Y widths");
            assert_eq!(
                row.remove.rect.right(),
                popup_right,
                "row {index} remove button shares the padded content edge"
            );
        }
    }

    #[test]
    fn import_cad_rows_read_their_labels_from_the_contract() {
        assert_eq!(step_import_field("pathMode").label, "Path");
        assert_eq!(step_import_field("ingest").label, "Embed file");
        assert_eq!(step_import_field("watch").label, "Auto-reload");
        assert_eq!(
            step_import_field("pathMode").kind,
            nodecraft_node_contract::FieldKind::Choice(&[
                ("relative", "Relative"),
                ("absolute", "Full path"),
            ])
        );
    }

    fn import_frame_texts(frame: &egui::FullOutput) -> Vec<String> {
        fn collect(shape: &egui::Shape, texts: &mut Vec<String>) {
            match shape {
                egui::Shape::Text(shape) => texts.push(shape.galley.text().to_owned()),
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, texts);
                    }
                }
                _ => {}
            }
        }
        let mut texts = Vec::new();
        for clipped in &frame.shapes {
            collect(&clipped.shape, &mut texts);
        }
        texts
    }

    /// Click inside the source controls at a known right-column x. The inspector
    /// lays switches out on the same right edge as the popup, so x = 225 lands on
    /// both without depending on their individual widths.
    fn click_import_control(ctx: &egui::Context, node: &GraphNode, y: f32) -> Vec<Change> {
        let at = egui::pos2(225.0, y);
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(248.0, 600.0),
            )),
            ..Default::default()
        };
        input.events.push(egui::Event::PointerMoved(at));
        input.events.push(egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        });
        input.events.push(egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        });
        let mut output = Output::default();
        let mut frame = ctx.run_ui(input, |ui| {
            ui.set_width(248.0);
            step_import_fields(ui, node, &mut output, None, false, None);
        });
        frame.textures_delta.clear();
        output.changes
    }

    #[test]
    fn empty_path_disables_the_source_controls_and_clicks_do_not_change_them() {
        assert!(!import_controls_enabled(""));
        assert!(import_controls_enabled("part.step"));
        let ctx = egui::Context::default();
        let node = node("stepimporter", serde_json::json!({ "path": "" }));
        let mut changes = Vec::new();
        // Cover the whole inspector: Path popup, Embed file, Auto-reload and Heal
        // all live in the right column. Only the three source controls are
        // asserted; Heal intentionally stays enabled.
        for step in 0..150 {
            changes.extend(click_import_control(&ctx, &node, step as f32 * 4.0));
        }
        assert!(
            !changes
                .iter()
                .any(|change| { matches!(change.field.as_str(), "pathMode" | "ingest" | "watch") }),
            "disabled source controls must not emit changes: {changes:?}"
        );
    }

    #[test]
    fn empty_path_shows_the_choose_a_cad_file_hint() {
        let ctx = egui::Context::default();
        let empty = node("stepimporter", serde_json::json!({ "path": "" }));
        let mut output = Output::default();
        let mut frame = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(248.0, 600.0),
                )),
                ..Default::default()
            },
            |ui| {
                ui.set_width(248.0);
                step_import_fields(ui, &empty, &mut output, None, false, None);
            },
        );
        let texts = import_frame_texts(&frame);
        frame.textures_delta.clear();
        assert!(
            texts.iter().any(|text| text == "Choose a CAD file"),
            "empty path hint missing, got {texts:?}"
        );

        let configured = node("stepimporter", serde_json::json!({ "path": "part.step" }));
        let mut output = Output::default();
        let mut frame = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(248.0, 600.0),
                )),
                ..Default::default()
            },
            |ui| {
                ui.set_width(248.0);
                step_import_fields(ui, &configured, &mut output, None, false, None);
            },
        );
        let texts = import_frame_texts(&frame);
        frame.textures_delta.clear();
        assert!(
            !texts.iter().any(|text| text == "Choose a CAD file"),
            "the hint is only for an unconfigured importer, got {texts:?}"
        );
    }

    #[test]
    fn a_path_enables_the_source_controls() {
        let ctx = egui::Context::default();
        let node = node(
            "stepimporter",
            serde_json::json!({ "path": "part.step", "pathMode": "relative" }),
        );
        let mut changes = Vec::new();
        // The switch rows sit below the path field and the Path popup row.
        for step in 0..35 {
            changes.extend(click_import_control(&ctx, &node, 55.0 + step as f32 * 2.0));
        }
        assert!(
            changes.iter().any(|change| matches!(change.field.as_str(), "ingest" | "watch")),
            "enabled switches should emit their change: {changes:?}"
        );
    }

    #[test]
    fn ingested_copy_status_names_the_copy_and_offers_both_actions() {
        use nodecraft_engine::import_io::{EmbeddedCopyStatus, ImportStatus};
        let copy = EmbeddedCopyStatus {
            file_name: "bracket.step".into(),
            byte_length: 18 * 1024,
            embedded_at: Some("2026-11-16T09:30:00Z".into()),
        };
        let status = ImportStatus::Ingested {
            missing: std::path::PathBuf::from("/models/bracket.step"),
            copy,
        };
        let node = node(
            "stepimporter",
            serde_json::json!({
                "path": "/models/bracket.step",
                "pathMode": "absolute",
                "ingest": true,
                "ingestHash": "abc",
            }),
        );
        let ctx = egui::Context::default();
        let mut output = Output::default();
        let mut frame = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(248.0, 600.0),
                )),
                ..Default::default()
            },
            |ui| {
                ui.set_width(248.0);
                step_import_fields(ui, &node, &mut output, None, false, Some(&status));
            },
        );
        let texts = import_frame_texts(&frame);
        frame.textures_delta.clear();
        assert!(
            texts.iter().any(|text| text.contains(
                "Original not found \u{2014} using embedded copy (bracket.step, 18 KB, embedded 16 Nov)"
            )),
            "warning status line missing, got {texts:?}"
        );
        assert!(
            texts.iter().any(|text| text == nodecraft_node_contract::STEP_IMPORT_LOCATE_LABEL),
            "Locate… button missing, got {texts:?}"
        );
        assert!(
            texts.iter().any(|text| text == nodecraft_node_contract::STEP_IMPORT_SAVE_COPY_LABEL),
            "Save copy as… button missing, got {texts:?}"
        );
        assert_eq!(
            embedded_copy_status_text(&EmbeddedCopyStatus {
                file_name: "bracket.step".into(),
                byte_length: 18 * 1024,
                embedded_at: None,
            }),
            "Original not found \u{2014} using embedded copy (bracket.step, 18 KB)",
            "a record without embeddedAt omits the date"
        );
    }

    #[test]
    fn missing_copy_status_uses_the_error_line_and_only_offers_locate() {
        use nodecraft_engine::import_io::ImportStatus;
        let status =
            ImportStatus::Missing { path: std::path::PathBuf::from("/models/bracket.step") };
        let node = node(
            "stepimporter",
            serde_json::json!({ "path": "/models/bracket.step", "pathMode": "absolute" }),
        );
        let ctx = egui::Context::default();
        let mut output = Output::default();
        let mut frame = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(248.0, 600.0),
                )),
                ..Default::default()
            },
            |ui| {
                ui.set_width(248.0);
                step_import_fields(ui, &node, &mut output, None, false, Some(&status));
            },
        );
        let texts = import_frame_texts(&frame);
        frame.textures_delta.clear();
        assert!(
            texts.iter().any(|text| text.contains("Original not found: /models/bracket.step")),
            "error status line missing, got {texts:?}"
        );
        assert!(
            texts.iter().any(|text| text == nodecraft_node_contract::STEP_IMPORT_LOCATE_LABEL),
            "Locate… button missing, got {texts:?}"
        );
        assert!(
            !texts.iter().any(|text| text == nodecraft_node_contract::STEP_IMPORT_SAVE_COPY_LABEL),
            "Save copy as… needs an embedded copy: {texts:?}"
        );
    }

    #[test]
    fn old_document_with_an_absolute_path_shows_full_path_without_a_change() {
        let ctx = egui::Context::default();
        let node = node("stepimporter", serde_json::json!({ "path": "/models/old/part.step" }));
        let mut inspector = Output::default();
        let mut frame = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(248.0, 600.0),
                )),
                ..Default::default()
            },
            |ui| {
                ui.set_width(248.0);
                step_import_fields(ui, &node, &mut inspector, None, false, None);
            },
        );
        assert!(inspector.changes.is_empty(), "showing a derived path mode must not emit a change");

        fn collect(shape: &egui::Shape, texts: &mut Vec<String>) {
            match shape {
                egui::Shape::Text(shape) => texts.push(shape.galley.text().to_owned()),
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, texts);
                    }
                }
                _ => {}
            }
        }
        let mut texts = Vec::new();
        for clipped in &frame.shapes {
            collect(&clipped.shape, &mut texts);
        }
        frame.textures_delta.clear();
        assert!(
            texts.iter().any(|text| text == "Full path"),
            "the path popup should show Full path, got {texts:?}"
        );
    }

    #[test]
    fn field_columns_are_shared_and_stable() {
        for width in [220.0, 280.0, 360.0, 480.0] {
            let (label, control) = field_columns_for(width);
            assert!(label >= 72.0 && label <= 140.0, "{width}: label {label}");
            assert!(control >= 64.0, "{width}: control {control}");
            assert!(
                label + control + FIELD_COLUMN_GAP <= width + f32::EPSILON,
                "{width}: columns overflow"
            );
            // The same width always produces the same columns. Labels and
            // current widget values are not inputs to the layout.
            assert_eq!((label, control), field_columns_for(width));
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

        let box_specs =
            known_specs(&node("primitive", serde_json::json!({"primitiveType": "box"})));
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
        assert!(matches!(length.widget, Widget::Number { max: None, .. }));
        let translate = known_specs(&node("translate", Value::Null));
        let x = translate.iter().find(|spec| spec.key == "x").expect("x");
        assert!(matches!(x.widget, Widget::Number { min: None, max: None, .. }));
    }

    #[test]
    fn step_file_name_drops_the_extension() {
        let step =
            node("stepimporter", serde_json::json!({ "path": "/tmp/models/MoonBaguette.stp" }));
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
        let custom =
            node("custom", serde_json::json!({ "label": "x", "enabled": true, "size": 3 }));
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
    fn spline_control_points_use_the_contract_default() {
        // The contract default for spline.control is false. This pins the
        // resolver path; the promoted contract-defaults-only rule flags a
        // private values.get(...).as_bool().unwrap_or(false) instead.
        let missing = node("spline", serde_json::json!({}));
        assert!(!spline_control_points(&missing));
        assert_eq!(
            spline_control_points(&missing),
            nodecraft_node_contract::resolve_bool("spline", &missing.values, "control")
                .expect("spline.control is a contract switch")
        );
        assert!(spline_control_points(&node("spline", serde_json::json!({"control": true}))));
        assert!(!spline_control_points(&node("spline", serde_json::json!({"control": false}))));
    }

    #[test]
    fn boolean_description_reads_mode_through_the_contract() {
        assert_eq!(
            node_description(&node("boolean", serde_json::json!({}))).as_deref(),
            Some("Combines both input shapes into one result.")
        );
        // The contract accepts the legacy `operation`/`op` aliases. A private
        // values.get("mode") read would fall back to union here.
        assert_eq!(
            node_description(&node("boolean", serde_json::json!({"operation": "subtract"})))
                .as_deref(),
            Some("A minus B: removes the second shape from the first.")
        );
        assert_eq!(
            node_description(&node("fuse", serde_json::json!({"op": "intersect"}))).as_deref(),
            Some("Shared volume of both input shapes.")
        );
    }

    #[test]
    fn reset_values_restores_settings_but_preserves_asset_data() {
        let look = GraphNode {
            expressions: Default::default(),
            id: "look".into(),
            node_type: "look".into(),
            values: serde_json::json!({"color": "#ff0000"}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let reset = reset_values(&look);
        assert!(
            reset
                .iter()
                .any(|change| change.field == "color"
                    && change.value == serde_json::json!("#7fa0cf"))
        );

        let spline = GraphNode {
            expressions: Default::default(),
            id: "s".into(),
            node_type: "spline".into(),
            values: serde_json::json!({"degree": 5.0, "points": [[0.0, 0.0], [1.0, 1.0]]}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let reset = reset_values(&spline);
        assert!(
            reset
                .iter()
                .any(|change| change.field == "degree" && change.value == serde_json::json!(3.0))
        );
        assert!(reset.iter().any(|change| change.field == "points"
            && change.value == serde_json::json!([[0.0, 0.0], [1.0, 1.0]])));

        let fillet = GraphNode {
            expressions: Default::default(),
            id: "f".into(),
            node_type: "fillet".into(),
            values: serde_json::json!({"length": 9.0, "edges": [{"id": "edge:1"}]}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let reset = reset_values(&fillet);
        assert!(
            reset
                .iter()
                .any(|change| change.field == "edges" && change.value == serde_json::json!([]))
        );
    }

    /// Pin the inspector metadata map before it moves into the contract table.
    /// The expected values are the ones this file produced before the move.
    #[test]
    fn inspector_metadata_map_is_pinned() {
        let cases: &[(&str, Value, &str, &str, &str, Option<&str>)] = &[
            (
                "primitive",
                serde_json::json!({"primitiveType": "box"}),
                "Box",
                "Solid",
                "PARAMETERS",
                Some("Creates a parametric primitive solid. Dimensions are in model units."),
            ),
            (
                "solid",
                serde_json::json!({"primitiveType": "cylinder"}),
                "Cylinder",
                "Solid",
                "PARAMETERS",
                Some("Creates a parametric primitive solid. Dimensions are in model units."),
            ),
            (
                "primitive",
                serde_json::json!({"primitiveType": "partialSphere"}),
                "Partial sphere",
                "Solid",
                "PARAMETERS",
                Some("Creates a parametric primitive solid. Dimensions are in model units."),
            ),
            (
                "stepimporter",
                serde_json::json!({}),
                "Import CAD",
                "Import",
                "CAD FILE",
                Some(
                    "Imports CAD solids with cadmpeg: STEP (AP203/AP214/AP242), FreeCAD .FCStd and Rhino .3dm are detected from the file contents.",
                ),
            ),
            (
                "exporter",
                serde_json::json!({}),
                "Export STEP",
                "Output",
                "OUTPUT FILE",
                Some("Writes the connected shape to a STEP file on disk."),
            ),
            (
                "translate",
                serde_json::json!({}),
                "Transform",
                "Transform",
                "TRANSFORM",
                Some(
                    "Moves and rotates the incoming shape. Select it in the graph or use the viewport gizmo.",
                ),
            ),
            ("surfaceloft", serde_json::json!({}), "Surface Loft", "Node", "VALUES", None),
            (
                "circularpattern",
                serde_json::json!({}),
                "Circular Pattern",
                "Pattern",
                "CIRCULAR PATTERN",
                Some("Repeats a shape around an axis."),
            ),
            (
                "group",
                serde_json::json!({
                    "nodes": [{"type": "box"}, {"type": "groupinput"}, {"type": "groupoutput"}],
                    "inputs": [{"name": "in"}],
                    "outputs": []
                }),
                "Graph",
                "Graph",
                "GRAPH",
                Some(
                    "Graph with 1 child node(s), 1 input port(s) and 0 output port(s). It is flattened into the parent graph during evaluation.",
                ),
            ),
            ("graph", serde_json::json!({}), "Graph", "Graph", "GRAPH", None),
            (
                "offsetfaces",
                serde_json::json!({}),
                "Offset Faces",
                "Feature",
                "OFFSET",
                Some("Moves the selected faces along their normals."),
            ),
            (
                "groupinput",
                serde_json::json!({}),
                "Graph Input",
                "Interface",
                "GRAPH INTERFACE",
                None,
            ),
            (
                "groupoutput",
                serde_json::json!({}),
                "Graph Output",
                "Interface",
                "GRAPH INTERFACE",
                None,
            ),
            (
                "boolean",
                serde_json::json!({}),
                "Boolean",
                "Boolean",
                "OPERATION",
                Some("Combines both input shapes into one result."),
            ),
            (
                "boolean",
                serde_json::json!({"mode": "subtract"}),
                "Boolean",
                "Boolean",
                "OPERATION",
                Some("A minus B: removes the second shape from the first."),
            ),
            (
                "boolean",
                serde_json::json!({"mode": "subtract-reverse"}),
                "Boolean",
                "Boolean",
                "OPERATION",
                Some("B minus A: removes the first shape from the second."),
            ),
            (
                "boolean",
                serde_json::json!({"mode": "intersect"}),
                "Boolean",
                "Boolean",
                "OPERATION",
                Some("Shared volume of both input shapes."),
            ),
            (
                "fuse",
                serde_json::json!({"op": "intersect"}),
                "Fuse",
                "Boolean",
                "OPERATION",
                Some("Shared volume of both input shapes."),
            ),
            (
                "combine",
                serde_json::json!({}),
                "Combine to Part",
                "Part",
                "SHAPES",
                Some("Collects every connected shape into a single Part."),
            ),
            (
                "chamfer",
                serde_json::json!({}),
                "Chamfer",
                "Feature",
                "CHAMFER",
                Some("Cuts selected edges of the incoming shape."),
            ),
            (
                "fillet",
                serde_json::json!({}),
                "Fillet",
                "Feature",
                "FILLET",
                Some("Rounds selected edges of the incoming shape."),
            ),
            (
                "look",
                serde_json::json!({}),
                "Look",
                "Feature",
                "LOOK",
                Some(
                    "Applies a colour to every body in the incoming Part. The geometry is unchanged.",
                ),
            ),
            (
                "shell",
                serde_json::json!({}),
                "Shell",
                "Feature",
                "SHELL",
                Some("Hollows the incoming shape by removing selected faces."),
            ),
            (
                "mirror",
                serde_json::json!({}),
                "Mirror",
                "Node",
                "MIRROR",
                Some("Reflects the incoming shape across a world plane."),
            ),
            (
                "scatter",
                serde_json::json!({}),
                "Scatter",
                "Generator",
                "SCATTER",
                Some("Repeats a shape across a seeded area."),
            ),
            ("sketch", serde_json::json!({}), "Sketch", "Sketch", "SKETCH", None),
            ("spline", serde_json::json!({}), "Spline", "Curve", "SPLINE", None),
            (
                "extrude",
                serde_json::json!({}),
                "Extrude",
                "Operation",
                "EXTRUSION",
                Some("Extrudes a sketch profile by the given amount."),
            ),
            ("loft", serde_json::json!({}), "Loft", "Node", "VALUES", None),
            ("linkedpart", serde_json::json!({}), "Linkedpart", "Output", "VALUES", None),
            ("time", serde_json::json!({}), "Time", "Node", "VALUES", None),
            ("sceneTransform", serde_json::json!({}), "SceneTransform", "Node", "VALUES", None),
            ("not-a-node", serde_json::json!({}), "Not A Node", "Node", "VALUES", None),
        ];
        for (node_type, values, title, badge, section, description) in cases {
            let node = node(node_type, values.clone());
            assert_eq!(node_title(&node), *title, "{node_type} title");
            assert_eq!(node_badge(&node), *badge, "{node_type} badge");
            assert_eq!(section_title(&node), *section, "{node_type} section");
            assert_eq!(node_description(&node).as_deref(), *description, "{node_type} description",);
        }
    }

    /// The transform gizmo's click asks the inspector to focus one field; the
    /// number entry takes the focus so the clicked axis or angle is the field
    /// the user is looking at.
    #[test]
    fn a_focus_request_focuses_the_number_entry() {
        let ctx = egui::Context::default();
        let mut value = 0.0;
        let mut focused = None;
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(248.0, 120.0),
                )),
                ..Default::default()
            },
            |ui| {
                ui.set_width(248.0);
                draw_number(
                    ui,
                    &mut value,
                    1.0,
                    None,
                    None,
                    nodecraft_node_contract::NumberKind::Unitless,
                    egui::Id::new(("inspector-number", "translate", "x")),
                    None,
                    true,
                );
                focused = ui.memory(|memory| memory.focused());
            },
        );
        output.textures_delta.clear();
        assert!(focused.is_some(), "a focus request focuses the number entry");
    }

    #[test]
    fn topology_list_reads_the_nodes_committed_face_references() {
        let node = node(
            "shell",
            serde_json::json!({"faces": [{"id": "face:2"}, {"id": "body:1:face:5"}]}),
        );
        assert_eq!(
            topology_reference_ids(&node, "faces"),
            vec!["face:2".to_owned(), "body:1:face:5".to_owned()],
        );
        assert!(topology_reference_ids(&node, "edges").is_empty());
    }
}
