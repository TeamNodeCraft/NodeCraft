//! Node-descriptor contract.
//!
//! One static description per node type: its ports, value kinds, output and
//! evaluation capabilities. Evaluation can migrate to descriptor-driven
//! dispatch without changing the graph or serialized documents.

use nodecraft_protocol::{Connection, Node};
use std::borrow::Cow;
use std::fmt;

mod metadata;
pub use metadata::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Shape,
    Part,
    /// A solid/Part value, but not a sketch.
    Geometry,
    Sketch,
    /// Accepts any geometry kind; used by ports that support both sketch and
    /// solid inputs (Extrude is the current example).
    Any,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortSpec {
    pub name: &'static str,
    pub kind: ValueKind,
    /// Whether the evaluator expects this port to be connected for a meaningful
    /// result. Documents may still contain an unconnected port temporarily.
    pub required: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeDescriptor {
    pub node_type: &'static str,
    pub inputs: &'static [PortSpec],
    pub output: ValueKind,
    pub exact: bool,
    pub mesh: bool,
    /// Input port whose result is shown as a ghost under this node's result
    /// while the node is selected. This is a display capability, independent of
    /// topology selection.
    pub preview_input: Option<&'static str>,
    /// Value field that must be non-empty for the preview to apply. Fillet and
    /// Chamfer use `edges`: an empty edge selection has no feature to ghost.
    pub preview_requires_value: Option<&'static str>,
}

const GEOMETRY: &[PortSpec] = &[PortSpec { name: "shape", kind: ValueKind::Geometry, required: true }];
const NO_PORTS: &[PortSpec] = &[];

macro_rules! descriptor {
    ($name:literal, $inputs:expr, $output:expr, $exact:literal, $mesh:literal) => {
        NodeDescriptor {
            node_type: $name,
            inputs: $inputs,
            output: $output,
            exact: $exact,
            mesh: $mesh,
            preview_input: None,
            preview_requires_value: None,
        }
    };
}

macro_rules! descriptor_with_preview {
    (
        $name:literal,
        $inputs:expr,
        $output:expr,
        $exact:literal,
        $mesh:literal,
        $preview:literal,
        $requires:literal
    ) => {
        NodeDescriptor {
            node_type: $name,
            inputs: $inputs,
            output: $output,
            exact: $exact,
            mesh: $mesh,
            preview_input: Some($preview),
            preview_requires_value: Some($requires),
        }
    };
}

static DESCRIPTORS: &[NodeDescriptor] = &[
    descriptor!("primitive", NO_PORTS, ValueKind::Shape, true, true),
    descriptor!("solid", NO_PORTS, ValueKind::Shape, true, true),
    descriptor!("stepimporter", NO_PORTS, ValueKind::Shape, true, true),
    descriptor!("translate", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("mirror", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("combine", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!(
        "boolean",
        &[
            PortSpec { name: "a", kind: ValueKind::Geometry, required: true },
            PortSpec { name: "b", kind: ValueKind::Geometry, required: true },
        ],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor!(
        "fuse",
        &[
            PortSpec { name: "a", kind: ValueKind::Geometry, required: true },
            PortSpec { name: "b", kind: ValueKind::Geometry, required: true },
        ],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor_with_preview!("chamfer", GEOMETRY, ValueKind::Geometry, true, true, "shape", "edges"),
    descriptor_with_preview!("fillet", GEOMETRY, ValueKind::Geometry, true, true, "shape", "edges"),
    descriptor!("look", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("shell", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("offsetfaces", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!(
        "extrude",
        &[
            PortSpec { name: "profile", kind: ValueKind::Any, required: true },
            // Older documents connected Extrude through a generic shape port.
            PortSpec { name: "shape", kind: ValueKind::Any, required: false },
        ],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor!(
        "loft",
        &[
            PortSpec { name: "bottom", kind: ValueKind::Sketch, required: true },
            PortSpec { name: "top", kind: ValueKind::Sketch, required: false },
        ],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor!(
        "surfaceloft",
        &[PortSpec { name: "sections", kind: ValueKind::Sketch, required: true }],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor!("scatter", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("circularpattern", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("sketch", NO_PORTS, ValueKind::Sketch, true, true),
    descriptor!("spline", NO_PORTS, ValueKind::Sketch, true, true),
    descriptor!("exporter", GEOMETRY, ValueKind::Empty, true, false),
    descriptor!("linkedpart", NO_PORTS, ValueKind::Empty, false, false),
    descriptor!("graph", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("group", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("groupinput", NO_PORTS, ValueKind::Geometry, true, true),
    descriptor!("groupoutput", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("time", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("sceneTransform", GEOMETRY, ValueKind::Geometry, true, true),
];


// ---------------------------------------------------------------------------
// Neutral inspector field metadata
// ---------------------------------------------------------------------------

/// How a node value should be edited. UI crates map these to widgets; the
/// contract never depends on egui.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumberKind {
    Length,
    Angle,
    Unitless,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldKind {
    Text,
    Switch,
    Number { min: Option<f64>, max: Option<f64>, step: f64, kind: NumberKind },
    Choice(&'static [(&'static str, &'static str)]),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldDefault {
    Text(&'static str),
    Bool(bool),
    Number(f64),
    Choice(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeFieldSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub default: FieldDefault,
    pub kind: FieldKind,
}

const fn field_switch(key: &'static str, label: &'static str, default: bool) -> NodeFieldSpec {
    NodeFieldSpec { key, label, default: FieldDefault::Bool(default), kind: FieldKind::Switch }
}

const fn field_number(
    key: &'static str,
    label: &'static str,
    default: f64,
    min: Option<f64>,
    max: Option<f64>,
    step: f64,
) -> NodeFieldSpec {
    NodeFieldSpec {
        key,
        label,
        default: FieldDefault::Number(default),
        kind: FieldKind::Number { min, max, step, kind: NumberKind::Length },
    }
}

const fn field_angle(
    key: &'static str,
    label: &'static str,
    default: f64,
    min: Option<f64>,
    max: Option<f64>,
    step: f64,
) -> NodeFieldSpec {
    NodeFieldSpec {
        key,
        label,
        default: FieldDefault::Number(default),
        kind: FieldKind::Number { min, max, step, kind: NumberKind::Angle },
    }
}

const fn field_unitless(
    key: &'static str,
    label: &'static str,
    default: f64,
    min: Option<f64>,
    max: Option<f64>,
    step: f64,
) -> NodeFieldSpec {
    NodeFieldSpec {
        key,
        label,
        default: FieldDefault::Number(default),
        kind: FieldKind::Number { min, max, step, kind: NumberKind::Unitless },
    }
}

const fn field_choice(
    key: &'static str,
    label: &'static str,
    default: &'static str,
    choices: &'static [(&'static str, &'static str)],
) -> NodeFieldSpec {
    NodeFieldSpec { key, label, default: FieldDefault::Choice(default), kind: FieldKind::Choice(choices) }
}

const TRANSLATE_FIELDS: &[NodeFieldSpec] = &[
    field_choice("space", "Space", "world", &[("world", "World"), ("local", "Local")]),
    field_number("x", "Position X", 0.0, None, None, 0.01),
    field_number("y", "Position Y", 0.0, None, None, 0.01),
    field_number("z", "Position Z", 0.0, None, None, 0.01),
    field_angle("rx", "Rotation X°", 0.0, Some(-180.0), Some(180.0), 0.1),
    field_angle("ry", "Rotation Y°", 0.0, Some(-180.0), Some(180.0), 0.1),
    field_angle("rz", "Rotation Z°", 0.0, Some(-180.0), Some(180.0), 0.1),
];

const BOOLEAN_FIELDS: &[NodeFieldSpec] = &[field_choice(
    "mode",
    "Operation",
    "union",
    &[("union", "Union"), ("subtract", "Subtract A − B"), ("subtract-reverse", "Subtract B − A"), ("intersect", "Intersect")],
)];

const EXTRUDE_FIELDS: &[NodeFieldSpec] = &[
    field_number("amount", "Amount", 10.0, Some(0.0), None, 0.1),
    field_choice("operation", "Operation", "new", &[("new", "New"), ("join", "Join"), ("cut", "Cut"), ("intersect", "Intersect")]),
];

const LOFT_FIELDS: &[NodeFieldSpec] = &[field_unitless("ruled", "Ruled", 0.0, Some(0.0), Some(1.0), 1.0)];

const CHAMFER_FIELDS: &[NodeFieldSpec] = &[
    field_number("length", "Distance", 2.0, Some(0.0), None, 0.1),
    field_choice("chamferType", "Type", "equal", &[("equal", "Equal distance"), ("two-distance", "Two distances"), ("distance-angle", "Distance + angle")]),
    field_number("distance2", "Distance 2", 2.0, Some(0.0), None, 0.1),
    field_angle("angle", "Angle°", 45.0, Some(0.1), Some(179.9), 0.1),
    field_switch("tangentChain", "Tangent chain", true),
    field_choice("chamferCorner", "Corner", "mitre", &[("mitre", "Mitre"), ("facet", "Facet")]),
    field_switch("flip", "Flip distances", false),
];

const FILLET_FIELDS: &[NodeFieldSpec] = &[
    field_number("length", "Radius", 2.0, Some(0.0), None, 0.1),
    field_choice("continuity", "Continuity", "", &[("", "Circular (G1)"), ("G2", "Curvature (G2)"), ("G3", "Curvature (G3)")]),
    field_choice("cornerTurn", "Corner turn", "mitre", &[("mitre", "Mitre"), ("swept", "Swept")]),
    field_switch("tangentChain", "Tangent chain", true),
];

const OFFSET_FIELDS: &[NodeFieldSpec] = &[field_number("distance", "Distance", 1.0, Some(0.0), None, 0.1)];

const SHELL_FIELDS: &[NodeFieldSpec] = &[
    field_number("thickness", "Thickness", 2.0, Some(0.0), None, 0.1),
    field_choice("direction", "Direction", "inward", &[("inward", "Inward"), ("outward", "Outward")]),
];

const MIRROR_FIELDS: &[NodeFieldSpec] = &[
    field_choice("plane", "Plane", "xz", &[("xy", "XY"), ("xz", "XZ"), ("yz", "YZ")]),
    field_number("offset", "Offset", 0.0, None, None, 0.01),
    field_switch("combine", "Combine", true),
];

const SCATTER_FIELDS: &[NodeFieldSpec] = &[
    field_switch("combine", "Combine", true),
    field_unitless("count", "Count", 1000.0, None, None, 0.1),
    field_number("spanX", "Span X", 240.0, None, None, 0.1),
    field_number("spanY", "Span Y", 160.0, None, None, 0.1),
    field_unitless("seed", "Seed", 1.0, None, None, 0.1),
    field_choice("distribution", "Distribution", "random", &[("random", "Random"), ("stratified", "Stratified"), ("grid", "Grid")]),
    field_unitless("minScale", "Min scale", 0.75, None, None, 0.1),
    field_unitless("maxScale", "Max scale", 1.25, None, None, 0.1),
];

const CIRCULAR_PATTERN_FIELDS: &[NodeFieldSpec] = &[
    field_unitless("count", "Count", 6.0, None, None, 1.0),
    field_angle("angle", "Angle°", 360.0, None, None, 1.0),
    field_choice("axis", "Axis", "z", &[("x", "X"), ("y", "Y"), ("z", "Z")]),
    field_number("centerX", "Center X", 0.0, None, None, 0.01),
    field_number("centerY", "Center Y", 0.0, None, None, 0.01),
    field_number("centerZ", "Center Z", 0.0, None, None, 0.01),
];

const SPLINE_FIELDS: &[NodeFieldSpec] = &[
    field_unitless("degree", "Degree", 3.0, Some(1.0), Some(9.0), 1.0),
    field_switch("control", "Control points", false),
    field_switch("periodic", "Closed", false),
];

/// CAD import source policy. `path` itself is a selector key, not a field: it
/// is an asset reference, not a setting with a default.
const STEP_IMPORTER_FIELDS: &[NodeFieldSpec] = &[
    field_choice(
        "pathMode",
        "Path",
        "relative",
        &[("relative", "Relative"), ("absolute", "Full path")],
    ),
    field_switch("ingest", "Embed file", false),
    field_switch("watch", "Auto-reload", false),
];

/// Inspector action labels for a CAD source that needs re-linking. Kept beside
/// the field labels so the import UI has one copy of its user-facing strings.
pub const STEP_IMPORT_LOCATE_LABEL: &str = "Locate…";
pub const STEP_IMPORT_SAVE_COPY_LABEL: &str = "Save copy as…";


pub const PRIMITIVE_TYPES: &[(&str, &str)] = &[
    ("box", "Box"),
    ("cylinder", "Cylinder"),
    ("cone", "Cone"),
    ("sphere", "Sphere"),
    ("torus", "Torus"),
    ("wedge", "Wedge"),
    ("pyramid", "Pyramid"),
    ("partialSphere", "Partial sphere"),
    ("partialTorus", "Partial torus"),
];

const BOX_FIELDS: &[NodeFieldSpec] = &[
    field_number("length", "Length", 40.0, Some(0.0), None, 0.1),
    field_number("width", "Width", 40.0, Some(0.0), None, 0.1),
    field_number("height", "Height", 40.0, Some(0.0), None, 0.1),
];

const CYLINDER_FIELDS: &[NodeFieldSpec] = &[
    field_number("radius", "Radius", 20.0, Some(0.0), None, 0.1),
    field_number("height", "Height", 40.0, Some(0.0), None, 0.1),
];

const CONE_FIELDS: &[NodeFieldSpec] = &[
    field_number("bottomRadius", "Bottom radius", 20.0, Some(0.0), None, 0.1),
    field_number("topRadius", "Top radius", 0.0, Some(0.0), None, 0.1),
    field_number("height", "Height", 40.0, Some(0.0), None, 0.1),
];

const SPHERE_FIELDS: &[NodeFieldSpec] = &[field_number("radius", "Radius", 20.0, Some(0.0), None, 0.1)];

const TORUS_FIELDS: &[NodeFieldSpec] = &[
    field_number("majorRadius", "Major radius", 15.0, Some(0.0), None, 0.1),
    field_number("minorRadius", "Minor radius", 5.0, Some(0.0), None, 0.1),
];

const WEDGE_FIELDS: &[NodeFieldSpec] = &[
    field_number("length", "Length", 40.0, Some(0.0), None, 0.1),
    field_number("width", "Width", 30.0, Some(0.0), None, 0.1),
    field_number("height", "Height", 25.0, Some(0.0), None, 0.1),
    field_number("topLength", "Top length", 0.0, Some(0.0), None, 0.1),
];

const PYRAMID_FIELDS: &[NodeFieldSpec] = &[
    field_number("length", "Base length", 40.0, Some(0.0), None, 0.1),
    field_number("height", "Height", 40.0, Some(0.0), None, 0.1),
];

const PARTIAL_SPHERE_FIELDS: &[NodeFieldSpec] = &[
    field_number("radius", "Radius", 20.0, Some(0.0), None, 0.1),
    field_angle("startAngle", "Start angle°", 0.0, Some(-180.0), Some(180.0), 1.0),
    field_angle("endAngle", "End angle°", 90.0, Some(-180.0), Some(180.0), 1.0),
];

const PARTIAL_TORUS_FIELDS: &[NodeFieldSpec] = &[
    field_number("majorRadius", "Major radius", 15.0, Some(0.0), None, 0.1),
    field_number("minorRadius", "Minor radius", 5.0, Some(0.0), None, 0.1),
    field_angle("startAngle", "Start angle°", 0.0, Some(-180.0), Some(180.0), 1.0),
    field_angle("endAngle", "End angle°", 90.0, Some(-180.0), Some(180.0), 1.0),
];

/// Fields for a primitive kind. Unknown kinds fall back to the box so editing a
/// partially written document never loses its numeric controls.
#[must_use]
pub fn primitive_fields(kind: &str) -> &'static [NodeFieldSpec] {
    match kind {
        "cylinder" => CYLINDER_FIELDS,
        "cone" => CONE_FIELDS,
        "sphere" => SPHERE_FIELDS,
        "torus" => TORUS_FIELDS,
        "wedge" => WEDGE_FIELDS,
        "pyramid" => PYRAMID_FIELDS,
        "partialSphere" => PARTIAL_SPHERE_FIELDS,
        "partialTorus" => PARTIAL_TORUS_FIELDS,
        _ => BOX_FIELDS,
    }
}

/// Node types with a static contract field table. The table is the one list
/// [`fields`], the inspector, and the contract-defaults lint read; adding a
/// field table here makes it visible to all three.
const STATIC_FIELD_TABLES: &[(&str, &[NodeFieldSpec])] = &[
    ("translate", TRANSLATE_FIELDS),
    ("boolean", BOOLEAN_FIELDS),
    ("fuse", BOOLEAN_FIELDS),
    ("extrude", EXTRUDE_FIELDS),
    ("loft", LOFT_FIELDS),
    ("surfaceloft", LOFT_FIELDS),
    ("chamfer", CHAMFER_FIELDS),
    ("fillet", FILLET_FIELDS),
    ("offsetfaces", OFFSET_FIELDS),
    ("shell", SHELL_FIELDS),
    ("mirror", MIRROR_FIELDS),
    ("scatter", SCATTER_FIELDS),
    ("circularpattern", CIRCULAR_PATTERN_FIELDS),
    ("spline", SPLINE_FIELDS),
    ("stepimporter", STEP_IMPORTER_FIELDS),
];

/// Neutral field metadata for node types with static inspector fields.
/// Dynamic field sets (primitive, stepimporter, exporter) stay with their
/// dedicated UI until a conditional-field model is needed.
#[must_use]
pub fn fields(node_type: &str) -> Option<&'static [NodeFieldSpec]> {
    STATIC_FIELD_TABLES
        .iter()
        .find(|(name, _)| *name == node_type)
        .map(|(_, fields)| *fields)
}

/// Every field spec the contract defines, both static tables and each primitive
/// kind's dynamic table. This is the authoritative field vocabulary for lint
/// rules and docs; it deliberately excludes selector keys that are not fields
/// themselves (for example `primitiveType`).
pub fn field_specs() -> impl Iterator<Item = &'static NodeFieldSpec> {
    STATIC_FIELD_TABLES
        .iter()
        .flat_map(|(_, fields)| fields.iter())
        .chain(
            PRIMITIVE_TYPES
                .iter()
                .flat_map(|(kind, _)| primitive_fields(kind).iter()),
        )
}

/// Every distinct field key the contract defines, sorted. The
/// `contract-defaults-only` ast-grep rule's `FIELD` regex is generated from
/// this list and checked by a test in this crate.
#[must_use]
pub fn field_keys() -> Vec<&'static str> {
    let mut keys: Vec<&'static str> = field_specs().map(|spec| spec.key).collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// The effective value of a contract field: the stored value when the reader
/// can interpret it, otherwise the contract default.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldValue<'a> {
    Bool(bool),
    Number(f64),
    Text(Cow<'a, str>),
}

impl NodeFieldSpec {
    /// The contract default as a typed value.
    #[must_use]
    pub fn default_value(self) -> FieldValue<'static> {
        match self.default {
            FieldDefault::Text(value) | FieldDefault::Choice(value) => {
                FieldValue::Text(Cow::Borrowed(value))
            }
            FieldDefault::Bool(value) => FieldValue::Bool(value),
            FieldDefault::Number(value) => FieldValue::Number(value),
        }
    }

    /// The contract default as JSON, for filling a node's values in tests or
    /// for an explicit reset action.
    #[must_use]
    pub fn default_json(self) -> serde_json::Value {
        match self.default {
            FieldDefault::Text(value) | FieldDefault::Choice(value) => {
                serde_json::Value::String(value.to_owned())
            }
            FieldDefault::Bool(value) => serde_json::Value::Bool(value),
            FieldDefault::Number(value) => serde_json::Value::from(value),
        }
    }
}

/// Field metadata for a node. Primitive (and legacy `solid`) nodes have a
/// dynamic field set selected by `primitiveType`; every other node type uses
/// the static table in [`fields`].
#[must_use]
pub fn fields_for(node_type: &str, values: &serde_json::Value) -> &'static [NodeFieldSpec] {
    match node_type {
        "primitive" | "solid" => {
            let kind = values
                .get("primitiveType")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            primitive_fields(kind)
        }
        _ => fields(node_type).unwrap_or(&[]),
    }
}

/// The contract metadata for one field, including dynamic primitive fields.
#[must_use]
pub fn field_spec(
    node_type: &str,
    values: &serde_json::Value,
    key: &str,
) -> Option<&'static NodeFieldSpec> {
    fields_for(node_type, values)
        .iter()
        .find(|spec| spec.key == key)
}

/// Legacy keys whose values older documents stored in place of the contract
/// key. The contract owns this migration knowledge so every reader shares it.
fn legacy_field_keys(node_type: &str, key: &str) -> &'static [&'static str] {
    match (node_type, key) {
        ("boolean" | "fuse", "mode") => &["operation", "op"],
        _ => &[],
    }
}

fn parse_field_value<'a>(
    spec: &NodeFieldSpec,
    raw: &'a serde_json::Value,
) -> Option<FieldValue<'a>> {
    match spec.default {
        FieldDefault::Bool(_) => match raw {
            serde_json::Value::Bool(value) => Some(FieldValue::Bool(*value)),
            // Older documents encoded the mirror/scatter Combine switch as a
            // string; `join` is the historical name for "combine".
            serde_json::Value::String(value) if spec.key == "combine" => {
                Some(FieldValue::Bool(value == "true" || value == "join"))
            }
            _ => None,
        },
        FieldDefault::Number(_) => raw.as_f64().map(FieldValue::Number),
        FieldDefault::Text(_) | FieldDefault::Choice(_) => {
            raw.as_str().map(|value| FieldValue::Text(Cow::Borrowed(value)))
        }
    }
}

/// Resolve one contract field from a node's values.
///
/// This is the single read-time default path for node fields. A missing key
/// (or a value in an encoding this contract cannot interpret) resolves to the
/// contract default; a stored value always wins over it. Legacy boolean `mode:
/// "join"` and the Boolean `operation`/`op` aliases are accepted here too.
///
/// Defaults are resolved here and are not injected into a node's stored
/// `values`. That keeps old documents (and their content hashes) unchanged,
/// while evaluating exactly what the inspector displays. A saved mirror without
/// `combine` therefore now combines, matching the inspector's Combine switch.
#[must_use]
pub fn resolve_field<'a>(
    node_type: &str,
    values: &'a serde_json::Value,
    key: &str,
) -> Option<FieldValue<'a>> {
    let spec = field_spec(node_type, values, key)?;
    let stored = values
        .get(key)
        .or_else(|| {
            legacy_field_keys(node_type, key)
                .iter()
                .find_map(|alias| values.get(*alias))
        });
    if let Some(raw) = stored
        && let Some(value) = parse_field_value(spec, raw)
    {
        return Some(value);
    }
    // A missing Combine field on an old document may still say "join" through
    // the former shared mode key.
    if matches!(spec.default, FieldDefault::Bool(_))
        && key == "combine"
        && values.get("mode").and_then(serde_json::Value::as_str) == Some("join")
    {
        return Some(FieldValue::Bool(true));
    }
    Some(spec.default_value())
}

/// Resolve a contract switch.
#[must_use]
pub fn resolve_bool(
    node_type: &str,
    values: &serde_json::Value,
    key: &str,
) -> Option<bool> {
    match resolve_field(node_type, values, key)? {
        FieldValue::Bool(value) => Some(value),
        _ => None,
    }
}

/// Resolve a contract number.
#[must_use]
pub fn resolve_number(
    node_type: &str,
    values: &serde_json::Value,
    key: &str,
) -> Option<f64> {
    match resolve_field(node_type, values, key)? {
        FieldValue::Number(value) => Some(value),
        _ => None,
    }
}

/// Resolve a contract text or choice field.
#[must_use]
pub fn resolve_text<'a>(
    node_type: &str,
    values: &'a serde_json::Value,
    key: &str,
) -> Option<Cow<'a, str>> {
    match resolve_field(node_type, values, key)? {
        FieldValue::Text(value) => Some(value),
        _ => None,
    }
}

/// Resolve a contract choice. Choices are strings; this names the intent at
/// call sites that select an enum-like value.
#[must_use]
pub fn resolve_choice<'a>(
    node_type: &str,
    values: &'a serde_json::Value,
    key: &str,
) -> Option<Cow<'a, str>> {
    resolve_text(node_type, values, key)
}

/// Every contract default for a node as JSON, keyed by field. Used by reset
/// actions and by tests that compare a missing value with its explicit default.
#[must_use]
pub fn default_values(
    node_type: &str,
    values: &serde_json::Value,
) -> serde_json::Map<String, serde_json::Value> {
    fields_for(node_type, values)
        .iter()
        .map(|spec| (spec.key.to_owned(), spec.default_json()))
        .collect()
}

#[must_use]
pub fn descriptor(node_type: &str) -> Option<&'static NodeDescriptor> {
    DESCRIPTORS.iter().find(|descriptor| descriptor.node_type == node_type)
}

/// The input port whose mesh is ghosted under a selected node's result.
#[must_use]
pub fn preview_input(node_type: &str) -> Option<&'static str> {
    descriptor(node_type)?.preview_input
}

/// The preview input port, if the node currently has the values that make its
/// preview meaningful. An empty edge selection means there is no feature to
/// ghost, regardless of the descriptor capability.
#[must_use]
pub fn preview_input_if_active(node_type: &str, values: &serde_json::Value) -> Option<&'static str> {
    let descriptor = descriptor(node_type)?;
    let input = descriptor.preview_input?;
    if let Some(field) = descriptor.preview_requires_value {
        let active = values
            .get(field)
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| !items.is_empty());
        if !active {
            return None;
        }
    }
    Some(input)
}


/// Node types whose exact result is a Part and whose selected result can be
/// displayed by the Part display path. This is capability metadata, independent
/// of the geometry implementation.
const PART_DISPLAY_TYPES: &[&str] = &[
    "solid", "primitive", "translate", "mirror", "combine", "boolean", "fuse",
    "stepimporter", "extrude", "sketch", "spline", "loft", "surfaceloft",
    "shell", "offsetfaces", "scatter", "circularpattern", "chamfer", "fillet", "look",
];

/// Whether this node type owns a CAD source path that can be ingested,
/// watched or rewritten relative to the document folder. A contract capability
/// so session document policy does not add its own node-type branch.
#[must_use]
pub fn import_source_node(node_type: &str) -> bool {
    matches!(node_type, "stepimporter")
}

/// Whether this node type has a Part result that the viewport can display.
#[must_use]
pub fn part_display_supported(node_type: &str) -> bool {
    PART_DISPLAY_TYPES.contains(&node_type)
}

/// Node types that are only drawn by a 2D overlay and must not produce a 3D
/// display body even though they have a Part display capability.
#[must_use]
pub fn overlay_only(node_type: &str) -> bool {
    node_type == "sketch"
}


/// Topology field a node collects from the viewport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionKind {
    Edge,
    Face,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionSpec {
    pub node_type: &'static str,
    pub field: &'static str,
    pub kind: SelectionKind,
}

const SELECTION_SPECS: &[SelectionSpec] = &[
    SelectionSpec { node_type: "chamfer", field: "edges", kind: SelectionKind::Edge },
    SelectionSpec { node_type: "fillet", field: "edges", kind: SelectionKind::Edge },
    SelectionSpec { node_type: "shell", field: "faces", kind: SelectionKind::Face },
    SelectionSpec { node_type: "extrude", field: "faces", kind: SelectionKind::Face },
];

/// The topology selection a node opts into, if any.
#[must_use]
pub fn selection_spec(node_type: &str) -> Option<SelectionSpec> {
    SELECTION_SPECS.iter().copied().find(|spec| spec.node_type == node_type)
}

pub fn descriptors() -> impl Iterator<Item = &'static NodeDescriptor> {
    DESCRIPTORS.iter()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractError {
    UnknownNodeType { node: String, node_type: String },
    UnknownInput { node: String, node_type: String, input: String },
    MissingSource { node: String, source: String },
    MissingOutput { node: String, source: String },
    WrongKind {
        node: String,
        input: String,
        expected: ValueKind,
        actual: ValueKind,
    },
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNodeType { node, node_type } => {
                write!(f, "node {node:?} has unknown type {node_type:?}")
            }
            Self::UnknownInput { node, node_type, input } => {
                write!(f, "node {node:?} ({node_type}) has no input port {input:?}")
            }
            Self::MissingSource { node, source } => {
                write!(f, "connection to {node:?} references missing source {source:?}")
            }
            Self::MissingOutput { node: _, source } => {
                write!(f, "connection source {source:?} has no geometry output")
            }
            Self::WrongKind { node, input, expected, actual } => {
                write!(f, "connection to {node:?}.{input} expects {expected:?}, source is {actual:?}")
            }
        }
    }
}

impl std::error::Error for ContractError {}

/// Validate port names and value kinds. Connectivity requirements are checked by
/// the evaluator, not here, because partial documents are valid while editing.
pub fn validate_connections(
    document_nodes: &[Node],
    document_connections: &[Connection],
) -> Result<(), ContractError> {
    let mut nodes = std::collections::BTreeMap::new();
    for node in document_nodes {
        if descriptor(&node.node_type).is_none() {
            return Err(ContractError::UnknownNodeType {
                node: node.id.clone(),
                node_type: node.node_type.clone(),
            });
        }
        nodes.insert(node.id.as_str(), node);
    }
    for connection in document_connections {
        validate_connection(connection, &nodes)?;
    }
    Ok(())
}

fn accepts(port: ValueKind, output: ValueKind) -> bool {
    match port {
        ValueKind::Any => output != ValueKind::Empty,
        ValueKind::Geometry => matches!(output, ValueKind::Shape | ValueKind::Part | ValueKind::Geometry),
        _ => port == output,
    }
}

type NodeMap<'a> = std::collections::BTreeMap<&'a str, &'a Node>;

fn validate_connection(
    connection: &Connection,
    nodes: &NodeMap<'_>,
) -> Result<(), ContractError> {
    let Some(target) = nodes.get(connection.to.as_str()) else {
        return Ok(());
    };
    let target_descriptor = descriptor(&target.node_type).expect("target validated");
    let Some(port) = target_descriptor
        .inputs
        .iter()
        .find(|port| port.name == connection.input)
        .copied()
    else {
        return Err(ContractError::UnknownInput {
            node: target.id.clone(),
            node_type: target.node_type.clone(),
            input: connection.input.clone(),
        });
    };
    let Some(source) = nodes.get(connection.from.as_str()) else {
        return Err(ContractError::MissingSource {
            node: target.id.clone(),
            source: connection.from.clone(),
        });
    };
    let source_descriptor = descriptor(&source.node_type).expect("source validated");
    if source_descriptor.output == ValueKind::Empty {
        return Err(ContractError::MissingOutput {
            node: target.id.clone(),
            source: source.id.clone(),
        });
    }
    if !accepts(port.kind, source_descriptor.output) {
        return Err(ContractError::WrongKind {
            node: target.id.clone(),
            input: connection.input.clone(),
            expected: port.kind,
            actual: source_descriptor.output,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_table_is_unique_and_covers_supported_nodes() {
        let mut seen = std::collections::BTreeSet::new();
        for descriptor in descriptors() {
            assert!(seen.insert(descriptor.node_type), "duplicate {}", descriptor.node_type);
            assert!(!descriptor.inputs.iter().any(|port| port.name.is_empty()));
        }
        for node_type in [
            "primitive", "solid", "stepimporter", "translate", "mirror", "combine", "boolean",
            "fuse", "chamfer", "fillet", "look", "shell", "extrude", "loft", "surfaceloft",
            "scatter", "circularpattern", "sketch", "spline", "exporter", "linkedpart", "time",
            "sceneTransform",
        ] {
            assert!(descriptor(node_type).is_some(), "missing descriptor for {node_type}");
        }
    }

    fn node(id: &str, node_type: &str) -> Node {
        Node {
            id: id.into(),
            node_type: node_type.into(),
            values: serde_json::json!({}),
            suppressed: false,
            cache: None,
        }
    }

    fn link(from: &str, to: &str, input: &str) -> Connection {
        Connection { from: from.into(), to: to.into(), input: input.into(), out: "shape".into() }
    }

    #[test]
    fn preview_inputs_are_declared_by_feature_descriptors() {
        assert_eq!(preview_input("fillet"), Some("shape"));
        assert_eq!(preview_input("chamfer"), Some("shape"));
        assert_eq!(preview_input("translate"), None);
        assert_eq!(
            preview_input_if_active("fillet", &serde_json::json!({"edges": []})),
            None
        );
        assert_eq!(
            preview_input_if_active("fillet", &serde_json::json!({"edges": [{"id": "edge:0"}]})),
            Some("shape")
        );
        // Every declared preview port must exist on the same descriptor.
        for descriptor in descriptors() {
            let Some(port) = descriptor.preview_input else { continue };
            assert!(
                descriptor.inputs.iter().any(|input| input.name == port),
                "{} previews missing input {port}",
                descriptor.node_type
            );
        }
    }

    #[test]
    fn connection_kinds_and_ports_validate() {
        let nodes = vec![
            node("sketch", "sketch"),
            node("extrude", "extrude"),
            node("box", "primitive"),
            node("boolean", "boolean"),
        ];
        let connections = vec![
            link("sketch", "extrude", "profile"),
            link("box", "boolean", "a"),
            link("extrude", "boolean", "b"),
        ];
        assert_eq!(validate_connections(&nodes, &connections), Ok(()));
    }

    #[test]
    fn geometry_ports_accept_shape_and_part_but_not_sketch() {
        assert!(accepts(ValueKind::Geometry, ValueKind::Shape));
        assert!(accepts(ValueKind::Geometry, ValueKind::Part));
        assert!(!accepts(ValueKind::Geometry, ValueKind::Sketch));
        assert!(accepts(ValueKind::Any, ValueKind::Sketch));
        assert!(!accepts(ValueKind::Any, ValueKind::Empty));
    }

    #[test]
    fn sketch_cannot_connect_to_shape_port() {
        let nodes = vec![node("s", "sketch"), node("t", "translate")];
        let connections = vec![link("s", "t", "shape")];
        assert!(matches!(
            validate_connections(&nodes, &connections),
            Err(ContractError::WrongKind { .. })
        ));
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;

    #[test]
    fn part_display_capabilities_are_known_node_types() {
        for node_type in PART_DISPLAY_TYPES {
            assert!(descriptor(node_type).is_some(), "part display type {node_type:?} has no descriptor");
        }
        assert!(overlay_only("sketch"));
        assert!(!overlay_only("spline"));
        assert_eq!(selection_spec("fillet").map(|spec| spec.field), Some("edges"));
        assert_eq!(selection_spec("shell").map(|spec| spec.field), Some("faces"));
        assert!(selection_spec("primitive").is_none());
        for spec in SELECTION_SPECS {
            assert!(descriptor(spec.node_type).is_some(), "selection type has no descriptor");
        }
        for node_type in ["translate", "boolean", "extrude", "loft", "chamfer", "fillet", "shell", "mirror", "scatter", "circularpattern", "spline", "stepimporter"] {
            let fields = fields(node_type).unwrap_or_else(|| panic!("missing fields for {node_type}"));
            let mut keys = std::collections::BTreeSet::new();
            for field in fields {
                assert!(keys.insert(field.key), "duplicate field {} on {node_type}", field.key);
            }
        }
        assert!(fields("primitive").is_none());
        let mut primitive_keys = std::collections::BTreeSet::new();
        for (kind, _) in PRIMITIVE_TYPES {
            let fields = primitive_fields(kind);
            assert!(!fields.is_empty(), "primitive {kind} has no fields");
            for field in fields {
                assert!(primitive_keys.insert((*kind, field.key)), "duplicate primitive field {} on {kind}", field.key);
            }
        }
    }

    /// Walk every contract field and check the one resolver returns the
    /// contract default both when the key is missing and when it is explicitly
    /// written as the default. A reader that keeps its own fallback now has a
    /// single expectation to meet: an empty `values` object and one filled with
    /// these defaults must resolve identically.
    #[test]
    fn every_contract_field_resolves_to_its_default() {
        fn check(node_type: &str, base: &serde_json::Value, fields: &[NodeFieldSpec]) {
            let mut defaults = base.as_object().cloned().unwrap_or_default();
            defaults.extend(default_values(node_type, base));
            let defaults = serde_json::Value::Object(defaults);
            for spec in fields {
                let missing = resolve_field(node_type, base, spec.key)
                    .unwrap_or_else(|| panic!("{node_type}.{} did not resolve", spec.key));
                let explicit = resolve_field(node_type, &defaults, spec.key)
                    .unwrap_or_else(|| panic!("{node_type}.{} default did not resolve", spec.key));
                assert_eq!(missing, spec.default_value(), "{node_type}.{} missing value", spec.key);
                assert_eq!(explicit, spec.default_value(), "{node_type}.{} explicit default", spec.key);
                match spec.default {
                    FieldDefault::Bool(value) => {
                        assert_eq!(resolve_bool(node_type, base, spec.key), Some(value));
                    }
                    FieldDefault::Number(value) => {
                        assert_eq!(resolve_number(node_type, base, spec.key), Some(value));
                    }
                    FieldDefault::Text(value) | FieldDefault::Choice(value) => {
                        assert_eq!(
                            resolve_text(node_type, base, spec.key).as_deref(),
                            Some(value)
                        );
                        assert_eq!(
                            resolve_choice(node_type, base, spec.key).as_deref(),
                            Some(value)
                        );
                    }
                }
            }
        }

        for node_type in [
            "translate", "boolean", "fuse", "extrude", "loft", "surfaceloft",
            "chamfer", "fillet", "offsetfaces", "shell", "mirror", "scatter", "circularpattern", "spline",
            "stepimporter",
        ] {
            let base = serde_json::json!({});
            check(node_type, &base, fields(node_type).expect("static fields"));
        }
        for (kind, _) in PRIMITIVE_TYPES {
            let base = serde_json::json!({"primitiveType": kind});
            let fields = fields_for("primitive", &base);
            assert!(!fields.is_empty(), "primitive {kind} has no fields");
            check("primitive", &base, fields);
        }
    }

    /// The `contract-defaults-only` ast-grep rule must name exactly the fields
    /// the contract defines. This test generates that field alternation from
    /// [`field_keys`] and compares it with the rule's `FIELD` constraint. On a
    /// drift it prints the regex to paste, so the rule never grows a private
    /// second list.
    #[test]
    fn contract_defaults_rule_tracks_every_contract_field() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let rule = root.join("ast-grep/rules/contract-defaults-only.yml");
        if !rule.is_file() {
            eprintln!(
                "skipping contract-defaults rule sync: private rule is absent from {}",
                root.display()
            );
            return;
        }
        let text = std::fs::read_to_string(&rule)
            .unwrap_or_else(|error| panic!("read {}: {error}", rule.display()));
        let expected = format!("^\"({})\"$", field_keys().join("|"));
        let field_constraint = text
            .split_once("  FIELD:")
            .map(|(_, rest)| rest)
            .unwrap_or_else(|| panic!("{} has no FIELD constraint", rule.display()));
        let actual = field_constraint
            .lines()
            .find_map(|line| line.trim().strip_prefix("regex: "))
            .map(|value| value.trim().trim_matches('\''))
            .unwrap_or_else(|| panic!("{} FIELD constraint has no regex", rule.display()));
        assert_eq!(
            actual,
            expected,
            "ast-grep FIELD regex drifted from nodecraft-node-contract::field_keys(); \
             replace the contract-defaults-only FIELD constraint with:\n  FIELD:\n    regex: '{expected}'"
        );
    }

    /// A Circular Pattern's fields resolve to the inspector defaults even when
    /// an older document stored none of them.
    #[test]
    fn circular_pattern_defaults_match_the_inspector() {
        let values = serde_json::json!({});
        assert_eq!(resolve_number("circularpattern", &values, "count"), Some(6.0));
        assert_eq!(resolve_number("circularpattern", &values, "angle"), Some(360.0));
        assert_eq!(resolve_choice("circularpattern", &values, "axis").as_deref(), Some("z"));
        for key in ["centerX", "centerY", "centerZ"] {
            assert_eq!(
                resolve_number("circularpattern", &values, key),
                Some(0.0),
                "circularpattern.{key}",
            );
        }
        let explicit = serde_json::json!({
            "count": 6.0, "angle": 360.0, "axis": "z",
            "centerX": 0.0, "centerY": 0.0, "centerZ": 0.0,
        });
        for key in ["count", "angle", "centerX", "centerY", "centerZ"] {
            assert_eq!(
                resolve_number("circularpattern", &explicit, key),
                resolve_number("circularpattern", &values, key),
                "circularpattern.{key} explicit default",
            );
        }
        assert_eq!(resolve_choice("circularpattern", &explicit, "axis").as_deref(), Some("z"));
    }

    #[test]
    fn resolver_accepts_the_legacy_encodings_readers_used() {
        let mirror = serde_json::json!({});
        assert_eq!(resolve_bool("mirror", &mirror, "combine"), Some(true));
        assert_eq!(resolve_choice("mirror", &mirror, "plane").as_deref(), Some("xz"));
        assert_eq!(
            resolve_bool("mirror", &serde_json::json!({"combine": "join"}), "combine"),
            Some(true)
        );
        assert_eq!(
            resolve_bool("mirror", &serde_json::json!({"combine": "true"}), "combine"),
            Some(true)
        );
        assert_eq!(
            resolve_bool("mirror", &serde_json::json!({"combine": "false"}), "combine"),
            Some(false)
        );
        assert_eq!(
            resolve_bool("mirror", &serde_json::json!({"mode": "join"}), "combine"),
            Some(true)
        );
        assert_eq!(
            resolve_choice(
                "boolean",
                &serde_json::json!({"operation": "subtract"}),
                "mode"
            )
            .as_deref(),
            Some("subtract")
        );
        assert_eq!(
            resolve_choice("fuse", &serde_json::json!({"op": "intersect"}), "mode").as_deref(),
            Some("intersect")
        );
    }
}
