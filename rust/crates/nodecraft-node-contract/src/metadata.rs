//! Presentation metadata for node types, as data.
//!
//! The engine and the UI both need to answer "what is this node type called,
//! what does it look like, where does it belong?". This module is the one table
//! for those answers. It contains no egui code and no icon bytes: `icon` is a
//! key the UI maps to its own asset.
//!
//! Every contract node type has one row. The unknown fallback lives in the
//! readers, which keep their previous defaults ("Node", "VALUES", a box icon).

use serde_json::Value;

use super::{PRIMITIVE_TYPES, resolve_choice};

/// The tool-dock category a node type is listed under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockSection {
    Create,
    Modify,
    Boolean,
    Pattern,
    Organise,
    Animate,
    Io,
}

/// Where a node type's icon key comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IconKind {
    /// `NodeMetadata::icon` is the key.
    Static,
    /// The key follows the primitive's `primitiveType`.
    Primitive,
    /// The key follows the Boolean mode (`union`/`subtract`/...).
    BooleanMode,
}

/// How a node type's description is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptionKind {
    /// `NodeMetadata::description` is the text (empty means no description).
    Static,
    /// The text follows the Boolean mode.
    BooleanMode,
    /// `NodeMetadata::description` is a template with `{count}`, `{inputs}` and
    /// `{outputs}` placeholders.
    GroupSummary,
}

/// One tool-dock entry, in the table so the dock order is not a UI arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DockPlacement {
    pub section: DockSection,
    /// Group heading inside the section. Empty means "no heading".
    pub group: &'static str,
    /// Order inside the group. Lower values come first.
    pub order: u16,
    /// The entry owns a split submenu (the default Solid item).
    pub split: bool,
    /// Primitive selector for a primitive-family entry.
    pub primitive: Option<&'static str>,
}

/// One node type's display metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeMetadata {
    pub node_type: &'static str,
    /// Icon key, resolved by the UI to its own asset.
    pub icon: &'static str,
    pub icon_kind: IconKind,
    /// Inspector badge, for example "Feature" or "Output".
    pub category: &'static str,
    /// Inspector section heading, for example "FILLET".
    pub section: &'static str,
    /// Inspector/dock display label.
    pub label: &'static str,
    /// Node-canvas card title. Some card titles are shorter than the inspector
    /// label (the Boolean/Fuse pair is the clearest example).
    pub card_label: &'static str,
    /// Tool-dock label. It normally matches the inspector label; Linked Part is
    /// the one historical exception.
    pub dock_label: &'static str,
    /// Static description, or a template for [`DescriptionKind::GroupSummary`].
    pub description: &'static str,
    pub description_kind: DescriptionKind,
    pub dock: Option<DockPlacement>,
}

/// A primitive kind's icon key and tool-dock order. The label stays in
/// [`PRIMITIVE_TYPES`], the contract's one primitive label table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimitiveMetadata {
    pub kind: &'static str,
    pub icon: &'static str,
    pub dock_order: u16,
    /// Tool-dock label; the Partial kinds capitalize the second word here.
    pub dock_label: &'static str,
}

const fn plain(
    node_type: &'static str,
    icon: &'static str,
    category: &'static str,
    section: &'static str,
    label: &'static str,
    card_label: &'static str,
    description: &'static str,
) -> NodeMetadata {
    NodeMetadata {
        node_type,
        icon,
        icon_kind: IconKind::Static,
        category,
        section,
        label,
        card_label,
        dock_label: label,
        description,
        description_kind: DescriptionKind::Static,
        dock: None,
    }
}

const fn with_dock_label(mut metadata: NodeMetadata, dock_label: &'static str) -> NodeMetadata {
    metadata.dock_label = dock_label;
    metadata
}

const fn with_icon(mut metadata: NodeMetadata, icon_kind: IconKind) -> NodeMetadata {
    metadata.icon_kind = icon_kind;
    metadata
}

const fn with_description(
    mut metadata: NodeMetadata,
    description_kind: DescriptionKind,
) -> NodeMetadata {
    metadata.description_kind = description_kind;
    metadata
}

const fn docked(
    mut metadata: NodeMetadata,
    section: DockSection,
    group: &'static str,
    order: u16,
    split: bool,
    primitive: Option<&'static str>,
) -> NodeMetadata {
    metadata.dock = Some(DockPlacement { section, group, order, split, primitive });
    metadata
}

/// The one node-type metadata table.
pub const NODE_METADATA: &[NodeMetadata] = &[
    docked(
        with_icon(
            plain(
                "solid",
                "box",
                "Solid",
                "PARAMETERS",
                "Solid",
                "Solid",
                "Creates a parametric primitive solid. Dimensions are in model units.",
            ),
            IconKind::Primitive,
        ),
        DockSection::Create,
        "",
        10,
        true,
        Some("box"),
    ),
    with_icon(
        plain(
            "primitive",
            "box",
            "Solid",
            "PARAMETERS",
            "Solid",
            "Solid",
            "Creates a parametric primitive solid. Dimensions are in model units.",
        ),
        IconKind::Primitive,
    ),
    docked(
        plain(
            "stepimporter",
            "import",
            "Import",
            "CAD FILE",
            "Import CAD",
            "Import STEP",
            "Imports CAD solids with cadmpeg: STEP (AP203/AP214/AP242), FreeCAD .FCStd and Rhino .3dm are detected from the file contents.",
        ),
        DockSection::Io,
        "",
        0,
        false,
        None,
    ),
    docked(
        plain(
            "translate",
            "translate",
            "Transform",
            "TRANSFORM",
            "Transform",
            "Transform",
            "Moves and rotates the incoming shape. Select it in the graph or use the viewport gizmo.",
        ),
        DockSection::Modify,
        "",
        0,
        false,
        None,
    ),
    docked(
        plain(
            "mirror",
            "mirror",
            "Node",
            "MIRROR",
            "Mirror",
            "Mirror",
            "Reflects the incoming shape across a world plane.",
        ),
        DockSection::Pattern,
        "",
        0,
        false,
        None,
    ),
    docked(
        plain(
            "combine",
            "combine",
            "Part",
            "SHAPES",
            "Combine to Part",
            "Combine to Part",
            "Collects every connected shape into a single Part.",
        ),
        DockSection::Organise,
        "",
        0,
        false,
        None,
    ),
    docked(
        with_description(
            with_icon(
                plain(
                    "boolean",
                    "union",
                    "Boolean",
                    "OPERATION",
                    "Boolean",
                    "Boolean",
                    "Combines both input shapes into one result.",
                ),
                IconKind::BooleanMode,
            ),
            DescriptionKind::BooleanMode,
        ),
        DockSection::Boolean,
        "",
        0,
        false,
        None,
    ),
    with_description(
        with_icon(
            plain(
                "fuse",
                "union",
                "Boolean",
                "OPERATION",
                "Fuse",
                "Boolean",
                "Combines both input shapes into one result.",
            ),
            IconKind::BooleanMode,
        ),
        DescriptionKind::BooleanMode,
    ),
    docked(
        plain(
            "chamfer",
            "chamfer",
            "Feature",
            "CHAMFER",
            "Chamfer",
            "Chamfer",
            "Cuts selected edges of the incoming shape.",
        ),
        DockSection::Modify,
        "",
        1,
        false,
        None,
    ),
    docked(
        plain(
            "fillet",
            "fillet",
            "Feature",
            "FILLET",
            "Fillet",
            "Fillet",
            "Rounds selected edges of the incoming shape.",
        ),
        DockSection::Modify,
        "",
        2,
        false,
        None,
    ),
    docked(
        plain(
            "shell",
            "shell",
            "Feature",
            "SHELL",
            "Shell",
            "Shell",
            "Hollows the incoming shape by removing selected faces.",
        ),
        DockSection::Modify,
        "",
        3,
        false,
        None,
    ),
    docked(
        plain(
            "offsetfaces",
            "offsetfaces",
            "Feature",
            "OFFSET",
            "Offset Faces",
            "Node",
            "Moves the selected faces along their normals.",
        ),
        DockSection::Modify,
        "",
        4,
        false,
        None,
    ),
    docked(
        plain(
            "extrude",
            "extrude",
            "Operation",
            "EXTRUSION",
            "Extrude",
            "Extrude",
            "Extrudes a sketch profile by the given amount.",
        ),
        DockSection::Create,
        "Features",
        20,
        false,
        None,
    ),
    docked(
        plain("loft", "loft", "Node", "VALUES", "Loft", "Loft", ""),
        DockSection::Create,
        "Features",
        21,
        false,
        None,
    ),
    docked(
        plain("surfaceloft", "surfaceloft", "Node", "VALUES", "Surface Loft", "Surface Loft", ""),
        DockSection::Create,
        "Features",
        22,
        false,
        None,
    ),
    docked(
        plain(
            "scatter",
            "scatter",
            "Generator",
            "SCATTER",
            "Scatter",
            "Scatter",
            "Repeats a shape across a seeded area.",
        ),
        DockSection::Pattern,
        "",
        2,
        false,
        None,
    ),
    docked(
        plain(
            "circularpattern",
            "circular-pattern",
            "Pattern",
            "CIRCULAR PATTERN",
            "Circular Pattern",
            "Circular Pattern",
            "Repeats a shape around an axis.",
        ),
        DockSection::Pattern,
        "",
        1,
        false,
        None,
    ),
    docked(
        plain("sketch", "sketch", "Sketch", "SKETCH", "Sketch", "Sketch", ""),
        DockSection::Create,
        "Sketch",
        0,
        false,
        None,
    ),
    docked(
        plain("spline", "spline", "Curve", "SPLINE", "Spline", "Spline", ""),
        DockSection::Create,
        "Sketch",
        1,
        false,
        None,
    ),
    docked(
        plain(
            "exporter",
            "export",
            "Output",
            "OUTPUT FILE",
            "Export STEP",
            "Export STEP",
            "Writes the connected shape to a STEP file on disk.",
        ),
        DockSection::Io,
        "",
        2,
        false,
        None,
    ),
    docked(
        with_dock_label(
            plain("linkedpart", "linked", "Output", "VALUES", "Linkedpart", "Linked Part", ""),
            "Linked Part",
        ),
        DockSection::Io,
        "",
        1,
        false,
        None,
    ),
    docked(
        plain("graph", "graph", "Graph", "GRAPH", "Graph", "Graph", ""),
        DockSection::Organise,
        "",
        1,
        false,
        None,
    ),
    with_description(
        plain(
            "group",
            "group",
            "Graph",
            "GRAPH",
            "Graph",
            "Graph",
            "Graph with {count} child node(s), {inputs} input port(s) and {outputs} output port(s). It is flattened into the parent graph during evaluation.",
        ),
        DescriptionKind::GroupSummary,
    ),
    plain("groupinput", "box", "Interface", "GRAPH INTERFACE", "Graph Input", "Graph Input", ""),
    plain("groupoutput", "box", "Interface", "GRAPH INTERFACE", "Graph Output", "Graph Output", ""),
    plain("time", "box", "Node", "VALUES", "Time", "Node", ""),
    plain("sceneTransform", "box", "Node", "VALUES", "SceneTransform", "Node", ""),
    docked(
        plain(
            "look",
            "look",
            "Feature",
            "LOOK",
            "Look",
            "Look",
            "Applies a colour to every body in the incoming Part. The geometry is unchanged.",
        ),
        DockSection::Organise,
        "",
        2,
        false,
        None,
    ),
    // Legacy document node types: the canvas and inspector still resolve them so
    // an older file keeps its icon and title.
    plain("box", "box", "Node", "VALUES", "Box", "Box", ""),
    plain("cylinder", "cylinder", "Node", "VALUES", "Cylinder", "Cylinder", ""),
    plain("cone", "cone", "Node", "VALUES", "Cone", "Cone", ""),
    plain("sphere", "sphere", "Node", "VALUES", "Sphere", "Sphere", ""),
    plain("torus", "torus", "Node", "VALUES", "Torus", "Torus", ""),
    plain("wedge", "wedge", "Node", "VALUES", "Wedge", "Wedge", ""),
];

/// Primitive kind icon keys and dock order. Labels remain in [`PRIMITIVE_TYPES`].
pub const PRIMITIVE_METADATA: &[PrimitiveMetadata] = &[
    PrimitiveMetadata { kind: "box", icon: "box", dock_order: 0, dock_label: "Box" },
    PrimitiveMetadata { kind: "sphere", icon: "sphere", dock_order: 1, dock_label: "Sphere" },
    PrimitiveMetadata { kind: "cylinder", icon: "cylinder", dock_order: 2, dock_label: "Cylinder" },
    PrimitiveMetadata { kind: "cone", icon: "cone", dock_order: 3, dock_label: "Cone" },
    PrimitiveMetadata { kind: "torus", icon: "torus", dock_order: 4, dock_label: "Torus" },
    PrimitiveMetadata { kind: "pyramid", icon: "pyramid", dock_order: 5, dock_label: "Pyramid" },
    PrimitiveMetadata { kind: "wedge", icon: "wedge", dock_order: 6, dock_label: "Wedge" },
    PrimitiveMetadata {
        kind: "partialSphere",
        icon: "sphere",
        dock_order: 7,
        dock_label: "Partial Sphere",
    },
    PrimitiveMetadata {
        kind: "partialTorus",
        icon: "torus",
        dock_order: 8,
        dock_label: "Partial Torus",
    },
];

const BOOLEAN_ICONS: &[(&str, &str)] = &[
    ("intersect", "intersection"),
    ("intersection", "intersection"),
    ("subtract-reverse", "subtract-reverse"),
    ("subtract-b-a", "subtract-reverse"),
    ("subtract", "subtract"),
    ("difference", "subtract"),
    ("subtract-a-b", "subtract"),
];

const BOOLEAN_DESCRIPTIONS: &[(&str, &str)] = &[
    ("subtract", "A minus B: removes the second shape from the first."),
    ("subtract-reverse", "B minus A: removes the first shape from the second."),
    ("intersect", "Shared volume of both input shapes."),
];

/// The metadata row for a node type.
#[must_use]
pub fn metadata(node_type: &str) -> Option<&'static NodeMetadata> {
    NODE_METADATA.iter().find(|metadata| metadata.node_type == node_type)
}

/// Every metadata row.
pub fn metadata_rows() -> impl Iterator<Item = &'static NodeMetadata> {
    NODE_METADATA.iter()
}

/// The primitive-kind metadata for a `primitiveType` selector.
#[must_use]
pub fn primitive_metadata(kind: &str) -> Option<&'static PrimitiveMetadata> {
    PRIMITIVE_METADATA.iter().find(|metadata| metadata.kind == kind)
}

/// The label for a primitive kind, from the contract's label table.
#[must_use]
pub fn primitive_label(kind: &str) -> Option<&'static str> {
    PRIMITIVE_TYPES.iter().find(|(name, _)| *name == kind).map(|(_, label)| *label)
}

/// The icon key for a Boolean mode. Unknown modes use the Union icon, matching
/// the canvas's previous fallback.
#[must_use]
pub fn boolean_icon(mode: &str) -> &'static str {
    BOOLEAN_ICONS.iter().find(|(name, _)| *name == mode).map_or("union", |(_, icon)| *icon)
}

/// The description text for a Boolean mode. Unknown modes use Union's text.
#[must_use]
pub fn boolean_description(mode: &str) -> &'static str {
    BOOLEAN_DESCRIPTIONS
        .iter()
        .find(|(name, _)| *name == mode)
        .map_or("Combines both input shapes into one result.", |(_, text)| *text)
}

/// The description shown for a node, or `None` when the type has none.
///
/// Boolean and Group descriptions depend on the node's values; both are
/// resolved here from data so every UI reader shows the same text.
#[must_use]
pub fn description(node_type: &str, values: &Value) -> Option<String> {
    let metadata = metadata(node_type)?;
    match metadata.description_kind {
        DescriptionKind::Static => {
            (!metadata.description.is_empty()).then(|| metadata.description.to_owned())
        }
        DescriptionKind::BooleanMode => {
            let mode = resolve_choice(node_type, values, "mode")
                .unwrap_or_else(|| std::borrow::Cow::Borrowed("union"));
            Some(boolean_description(mode.as_ref()).to_owned())
        }
        DescriptionKind::GroupSummary => {
            let count = values.get("nodes").and_then(Value::as_array).map_or(0, |nodes| {
                nodes
                    .iter()
                    .filter(|node| {
                        !matches!(
                            node.get("type").and_then(Value::as_str),
                            Some("groupinput" | "groupoutput")
                        )
                    })
                    .count()
            });
            let inputs = values.get("inputs").and_then(Value::as_array).map_or(0, Vec::len);
            let outputs = values.get("outputs").and_then(Value::as_array).map_or(0, Vec::len);
            Some(
                metadata
                    .description
                    .replace("{count}", &count.to_string())
                    .replace("{inputs}", &inputs.to_string())
                    .replace("{outputs}", &outputs.to_string()),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every contract node type has one metadata row. The engine has a
    /// companion test for its registry ids.
    #[test]
    fn every_contract_node_type_has_metadata() {
        for descriptor in crate::descriptors() {
            assert!(
                metadata(descriptor.node_type).is_some(),
                "contract node type {:?} has no metadata row",
                descriptor.node_type
            );
        }
    }

    #[test]
    fn metadata_rows_are_unique_and_complete() {
        let mut seen = std::collections::BTreeSet::new();
        for row in metadata_rows() {
            assert!(seen.insert(row.node_type), "duplicate metadata row {}", row.node_type);
            assert!(!row.icon.is_empty(), "{} has no icon key", row.node_type);
            assert!(!row.category.is_empty(), "{} has no category", row.node_type);
            assert!(!row.section.is_empty(), "{} has no section", row.node_type);
            assert!(!row.label.is_empty(), "{} has no label", row.node_type);
            assert!(!row.card_label.is_empty(), "{} has no card label", row.node_type);
            assert!(!row.dock_label.is_empty(), "{} has no dock label", row.node_type);
            if let Some(dock) = row.dock {
                assert!(
                    !dock.split || dock.primitive.is_some(),
                    "{} split dock entry has no primitive",
                    row.node_type
                );
            }
        }
    }

    #[test]
    fn every_primitive_kind_has_metadata_and_a_label() {
        for (kind, label) in PRIMITIVE_TYPES {
            let metadata = primitive_metadata(kind)
                .unwrap_or_else(|| panic!("primitive {kind} has no metadata"));
            assert!(!metadata.icon.is_empty());
            assert!(!metadata.dock_label.is_empty());
            assert_eq!(primitive_label(kind), Some(*label));
        }
        let mut orders: Vec<u16> = PRIMITIVE_METADATA.iter().map(|row| row.dock_order).collect();
        orders.sort_unstable();
        orders.dedup();
        assert_eq!(orders.len(), PRIMITIVE_METADATA.len(), "duplicate primitive dock order");
    }

    #[test]
    fn descriptions_keep_the_value_selected_text() {
        let union = serde_json::json!({});
        assert_eq!(
            description("boolean", &union).as_deref(),
            Some("Combines both input shapes into one result."),
        );
        assert_eq!(
            description("boolean", &serde_json::json!({"mode": "subtract"})).as_deref(),
            Some("A minus B: removes the second shape from the first."),
        );
        assert_eq!(
            description("boolean", &serde_json::json!({"mode": "subtract-reverse"})).as_deref(),
            Some("B minus A: removes the first shape from the second."),
        );
        assert_eq!(
            description("fuse", &serde_json::json!({"op": "intersect"})).as_deref(),
            Some("Shared volume of both input shapes."),
        );
        let group = serde_json::json!({
            "nodes": [{"type": "box"}, {"type": "groupinput"}],
            "inputs": [{"name": "in"}],
            "outputs": []
        });
        assert_eq!(
            description("group", &group).as_deref(),
            Some(
                "Graph with 1 child node(s), 1 input port(s) and 0 output port(s). It is flattened into the parent graph during evaluation."
            ),
        );
        assert_eq!(description("graph", &serde_json::json!({})), None);
    }
}
