//! Assembly product-structure parsing: transforms, occurrences, nesting and real files.

use std::collections::BTreeSet;

use step_assemblies::{
    Assembly, AssemblyError, Node, Transform, parse_assembly, parse_assembly_file,
};

// ---------------------------------------------------------------------------
// Minimal STEP writer for fixtures
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Step {
    lines: Vec<String>,
}

impl Step {
    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, body: impl AsRef<str>) -> u32 {
        self.lines.push(body.as_ref().to_string());
        self.lines.len() as u32
    }

    fn text(&self) -> String {
        let mut out = String::from(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
             FILE_NAME('','',(''),(''),'step-assemblies-test','','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n",
        );
        for (index, line) in self.lines.iter().enumerate() {
            out.push_str(&format!("#{}={};\n", index + 1, line));
        }
        out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        out
    }

    fn parse(&self) -> Result<Assembly, AssemblyError> {
        parse_assembly(&self.text())
    }
}

/// `AXIS2_PLACEMENT_3D` with the given origin and local Z / X directions.
fn frame(step: &mut Step, origin: [f64; 3], z: [f64; 3], x: [f64; 3]) -> u32 {
    let point = step.add(format!(
        "CARTESIAN_POINT('',({},{},{}))",
        number(origin[0]),
        number(origin[1]),
        number(origin[2])
    ));
    let axis = step.add(format!(
        "DIRECTION('',({},{},{}))",
        number(z[0]),
        number(z[1]),
        number(z[2])
    ));
    let reference = step.add(format!(
        "DIRECTION('',({},{},{}))",
        number(x[0]),
        number(x[1]),
        number(x[2])
    ));
    step.add(format!("AXIS2_PLACEMENT_3D('',#{point},#{axis},#{reference})"))
}

fn number(value: f64) -> String {
    if value == 0.0 {
        "0.".into()
    } else if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        format!("{value:?}")
    }
}

/// A `PRODUCT` -> formation -> `PRODUCT_DEFINITION` -> `PRODUCT_DEFINITION_SHAPE` ->
/// `SHAPE_REPRESENTATION` chain. Returns `(definition, representation)`.
fn product(step: &mut Step, product_id: &str, name: &str, representation_items: &[u32]) -> (u32, u32) {
    let product = step.add(format!("PRODUCT('{product_id}','{name}','',())"));
    let formation = step.add(format!("PRODUCT_DEFINITION_FORMATION('','',#{product})"));
    let definition = step.add(format!("PRODUCT_DEFINITION('design','',#{formation},$)"));
    let shape = step.add(format!("PRODUCT_DEFINITION_SHAPE('','',#{definition})"));
    let items: Vec<String> = representation_items.iter().map(|item| format!("#{item}")).collect();
    let representation = step.add(format!(
        "SHAPE_REPRESENTATION('{name}',({}),$)",
        items.join(",")
    ));
    step.add(format!(
        "SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})"
    ));
    (definition, representation)
}

/// A CAx-IF representation relationship: `rep_1` = child, `rep_2` = parent, with
/// `ITEM_DEFINED_TRANSFORMATION(child_item, parent_item)`.
fn relationship(
    step: &mut Step,
    child_representation: u32,
    parent_representation: u32,
    child_item: u32,
    parent_item: u32,
) -> u32 {
    let transformation = step.add(format!(
        "ITEM_DEFINED_TRANSFORMATION('','',#{child_item},#{parent_item})"
    ));
    step.add(format!(
        "(REPRESENTATION_RELATIONSHIP('','',#{child_representation},#{parent_representation})\n\
         REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{transformation})\n\
         SHAPE_REPRESENTATION_RELATIONSHIP())"
    ))
}

/// A non-CAx-IF relationship with the parent representation listed first (the layout
/// `as1_pe.stp` uses).
fn relationship_parent_first(
    step: &mut Step,
    parent_representation: u32,
    child_representation: u32,
    parent_item: u32,
    child_item: u32,
) -> u32 {
    let transformation = step.add(format!(
        "ITEM_DEFINED_TRANSFORMATION('','',#{parent_item},#{child_item})"
    ));
    step.add(format!(
        "(REPRESENTATION_RELATIONSHIP('','',#{parent_representation},#{child_representation})\n\
         REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{transformation})\n\
         SHAPE_REPRESENTATION_RELATIONSHIP())"
    ))
}

/// A `NEXT_ASSEMBLY_USAGE_OCCURRENCE` + occurrence shape + context-dependent shape
/// representation. Returns the NAUO entity id.
fn occurrence(
    step: &mut Step,
    id: &str,
    name: &str,
    placement_name: &str,
    parent_definition: u32,
    child_definition: u32,
    relationship: u32,
) -> u32 {
    let nauo = step.add(format!(
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE('{id}','{name}','',#{parent_definition},#{child_definition},$)"
    ));
    let shape = step.add(format!("PRODUCT_DEFINITION_SHAPE('{placement_name}','',#{nauo})"));
    step.add(format!(
        "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{relationship},#{shape})"
    ));
    nauo
}

fn assert_world_consistent(assembly: &Assembly) {
    fn check(node: &Node, parent_world: Transform) {
        let expected = parent_world.multiply(&node.local_transform);
        assert!(
            node.world_transform.approx_eq(&expected, 1e-12),
            "node {}: world {:?} != parent*local {:?}",
            node.name,
            node.world_transform.rows(),
            expected.rows()
        );
        for child in &node.children {
            check(child, node.world_transform);
        }
    }
    for root in &assembly.roots {
        check(root, Transform::IDENTITY);
    }
}

// ---------------------------------------------------------------------------
// Fixtures / tests
// ---------------------------------------------------------------------------

#[test]
fn parses_parent_child_transform() {
    let mut step = Step::new();
    let parent_placement = frame(&mut step, [10.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let child_placement = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (child_definition, child_representation) =
        product(&mut step, "child", "PART_A", &[child_placement]);
    let (root_definition, root_representation) =
        product(&mut step, "root", "ROOT", &[parent_placement]);
    let relation = relationship(
        &mut step,
        child_representation,
        root_representation,
        child_placement,
        parent_placement,
    );
    occurrence(
        &mut step,
        "occ-1",
        "Child occurrence",
        "Placement A",
        root_definition,
        child_definition,
        relation,
    );

    let assembly = step.parse().expect("parse assembly");
    assert_eq!(assembly.roots.len(), 1);
    let root = &assembly.roots[0];
    assert_eq!(root.name, "ROOT");
    assert_eq!(root.product_name, "ROOT");
    assert!(root.local_transform.is_identity(1e-12));
    assert!(root.world_transform.is_identity(1e-12));
    assert_eq!(root.children.len(), 1);
    let child = &root.children[0];
    assert_eq!(child.name, "Placement A", "occurrence placement name wins");
    assert_eq!(child.product_name, "PART_A");
    assert_eq!(child.occurrence_id.as_deref(), Some("occ-1"));
    assert_eq!(child.occurrence_name.as_deref(), Some("Child occurrence"));
    assert_eq!(child.placement_name.as_deref(), Some("Placement A"));
    assert_eq!(child.definition_id, Some(child_definition));
    assert_eq!(child.representation_id, Some(child_representation));
    assert_eq!(child.local_transform.translation(), [10.0, 0.0, 0.0]);
    assert_eq!(child.world_transform.translation(), [10.0, 0.0, 0.0]);
    assert_eq!(
        child.world_transform.transform_point([1.0, 2.0, 3.0]),
        [11.0, 2.0, 3.0]
    );
    assert_world_consistent(&assembly);
    assert_eq!(assembly.node_count(), 2);
    assert!(assembly.warnings.is_empty(), "{:?}", assembly.warnings);
}

#[test]
fn applies_rotation_and_translation_from_the_placements() {
    let mut step = Step::new();
    // Parent frame: origin (5,5,0), local X = +Y (rotation 90 degrees about Z).
    let parent_placement = frame(&mut step, [5.0, 5.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
    let child_placement = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (child_definition, child_representation) =
        product(&mut step, "child", "PART_A", &[child_placement]);
    let (root_definition, root_representation) =
        product(&mut step, "root", "ROOT", &[parent_placement]);
    let relation = relationship(
        &mut step,
        child_representation,
        root_representation,
        child_placement,
        parent_placement,
    );
    occurrence(
        &mut step,
        "occ-1",
        "Child",
        "Placement",
        root_definition,
        child_definition,
        relation,
    );

    let assembly = step.parse().expect("parse assembly");
    let child = &assembly.roots[0].children[0];
    // Local +X maps to parent +Y, then the frame origin (5,5,0) is added.
    let mapped = child.world_transform.transform_point([1.0, 0.0, 0.0]);
    assert!((mapped[0] - 5.0).abs() < 1e-12, "{mapped:?}");
    assert!((mapped[1] - 6.0).abs() < 1e-12, "{mapped:?}");
    assert!((mapped[2] - 0.0).abs() < 1e-12, "{mapped:?}");
    assert_eq!(child.world_transform.translation(), [5.0, 5.0, 0.0]);
}

#[test]
fn supports_multiple_occurrences_of_the_same_part() {
    let mut step = Step::new();
    let child_placement = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let parent_placement_a = frame(&mut step, [10.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let parent_placement_b = frame(&mut step, [20.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (child_definition, child_representation) =
        product(&mut step, "child", "PART_A", &[child_placement]);
    let (root_definition, root_representation) = product(
        &mut step,
        "root",
        "ROOT",
        &[parent_placement_a, parent_placement_b],
    );
    let relation_a = relationship(
        &mut step,
        child_representation,
        root_representation,
        child_placement,
        parent_placement_a,
    );
    let relation_b = relationship(
        &mut step,
        child_representation,
        root_representation,
        child_placement,
        parent_placement_b,
    );
    occurrence(
        &mut step,
        "occ-1",
        "First",
        "Placement A",
        root_definition,
        child_definition,
        relation_a,
    );
    occurrence(
        &mut step,
        "occ-2",
        "Second",
        "Placement B",
        root_definition,
        child_definition,
        relation_b,
    );

    let assembly = step.parse().expect("parse assembly");
    let root = &assembly.roots[0];
    assert_eq!(root.children.len(), 2, "same part placed twice");
    assert_eq!(root.children[0].definition_id, root.children[1].definition_id);
    assert_eq!(
        root.children[0].representation_id,
        root.children[1].representation_id
    );
    assert_eq!(root.children[0].name, "Placement A");
    assert_eq!(root.children[1].name, "Placement B");
    assert_eq!(root.children[0].world_transform.translation(), [10.0, 0.0, 0.0]);
    assert_eq!(root.children[1].world_transform.translation(), [20.0, 0.0, 0.0]);
    assert_eq!(assembly.node_count(), 3);
    assert_world_consistent(&assembly);
}

#[test]
fn accumulates_world_transforms_through_nested_assemblies() {
    let mut step = Step::new();
    let c_frame = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let b_child_frame = c_frame;
    let b_parent_frame = frame(&mut step, [5.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let root_parent_frame = frame(&mut step, [10.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (c_definition, c_representation) = product(&mut step, "c", "PART_C", &[c_frame]);
    let (b_definition, b_representation) =
        product(&mut step, "b", "SUB_B", &[b_child_frame, b_parent_frame]);
    let (root_definition, root_representation) =
        product(&mut step, "root", "ROOT", &[root_parent_frame]);
    // root -> B: child item is an identity placement in B's rep, parent item is in root.
    let root_to_b = relationship(
        &mut step,
        b_representation,
        root_representation,
        b_child_frame,
        root_parent_frame,
    );
    occurrence(
        &mut step,
        "occ-b",
        "B occurrence",
        "B placement",
        root_definition,
        b_definition,
        root_to_b,
    );
    // B -> C.
    let b_to_c = relationship(
        &mut step,
        c_representation,
        b_representation,
        c_frame,
        b_parent_frame,
    );
    occurrence(
        &mut step,
        "occ-c",
        "C occurrence",
        "C placement",
        b_definition,
        c_definition,
        b_to_c,
    );

    let assembly = step.parse().expect("parse assembly");
    let root = &assembly.roots[0];
    assert_eq!(root.children.len(), 1);
    let sub = &root.children[0];
    assert_eq!(sub.product_name, "SUB_B");
    assert_eq!(sub.world_transform.translation(), [10.0, 0.0, 0.0]);
    assert_eq!(sub.children.len(), 1);
    let part = &sub.children[0];
    assert_eq!(part.product_name, "PART_C");
    assert_eq!(part.local_transform.translation(), [5.0, 0.0, 0.0]);
    assert_eq!(part.world_transform.translation(), [15.0, 0.0, 0.0]);
    assert_eq!(part.world_transform.transform_point([1.0, 0.0, 0.0]), [16.0, 0.0, 0.0]);
    assert_eq!(assembly.node_count(), 3);
    assert_world_consistent(&assembly);
}

#[test]
fn handles_relationships_with_the_parent_representation_listed_first() {
    let mut step = Step::new();
    let child_frame = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let parent_frame = frame(&mut step, [7.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (child_definition, child_representation) =
        product(&mut step, "child", "PART_A", &[child_frame]);
    let (root_definition, root_representation) =
        product(&mut step, "root", "ROOT", &[parent_frame]);
    let relation = relationship_parent_first(
        &mut step,
        root_representation,
        child_representation,
        parent_frame,
        child_frame,
    );
    occurrence(
        &mut step,
        "occ-1",
        "Child",
        "Placement",
        root_definition,
        child_definition,
        relation,
    );
    let assembly = step.parse().expect("parse assembly");
    let child = &assembly.roots[0].children[0];
    assert_eq!(child.world_transform.translation(), [7.0, 0.0, 0.0]);
}

#[test]
fn missing_placement_relationship_falls_back_to_identity_with_a_warning() {
    let mut step = Step::new();
    let child_frame = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let mut root_frame = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (child_definition, _child_representation) =
        product(&mut step, "child", "PART_A", &[child_frame]);
    let (root_definition, _root_representation) =
        product(&mut step, "root", "ROOT", &[root_frame]);
    // Occurs in the file order, but without CDSR/relationship.
    let _ = step.add(format!(
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE('occ-1','Child','',#{root_definition},#{child_definition},$)"
    ));
    let assembly = step.parse().expect("parse assembly");
    let child = &assembly.roots[0].children[0];
    assert!(child.local_transform.is_identity(1e-12));
    assert!(assembly.warnings.iter().any(|warning| warning.contains("no placement relationship")));
    let _ = &mut root_frame;
}

#[test]
fn ignores_plain_supplemental_representation_relationships() {
    let mut step = Step::new();
    let frame = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (definition, representation) = product(&mut step, "part", "PART_A", &[frame]);
    let brep = step.add(format!("ADVANCED_BREP_SHAPE_REPRESENTATION('PART_A',(#{frame}),$)"));
    step.add(format!(
        "SHAPE_REPRESENTATION_RELATIONSHIP('','',#{representation},#{brep})"
    ));
    let assembly = step.parse().expect("parse assembly");
    assert_eq!(assembly.roots.len(), 1);
    assert_eq!(assembly.roots[0].definition_id, Some(definition));
    assert_eq!(assembly.roots[0].children.len(), 0);
    assert_eq!(assembly.representations.len(), 2);
}

#[test]
fn reports_cyclic_assembly_structure() {
    let mut step = Step::new();
    let frame = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (a_definition, a_representation) = product(&mut step, "a", "A", &[frame]);
    let (b_definition, b_representation) = product(&mut step, "b", "B", &[frame]);
    let a_to_b = relationship(&mut step, b_representation, a_representation, frame, frame);
    occurrence(&mut step, "ab", "B in A", "", a_definition, b_definition, a_to_b);
    let b_to_a = relationship(&mut step, a_representation, b_representation, frame, frame);
    occurrence(&mut step, "ba", "A in B", "", b_definition, a_definition, b_to_a);
    let error = step.parse().expect_err("cycle must fail");
    assert!(matches!(error, AssemblyError::Malformed(_)), "{error}");
    assert!(error.to_string().contains("cyclic"), "{error}");
}

#[test]
fn reports_structural_errors() {
    assert!(matches!(
        parse_assembly("not a step file"),
        Err(AssemblyError::Step(_))
    ));
    let no_products = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;\nEND-ISO-10303-21;\n";
    assert!(matches!(parse_assembly(no_products), Err(AssemblyError::Malformed(_))));

    let mut step = Step::new();
    let frame = frame(&mut step, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (definition, _) = product(&mut step, "part", "PART_A", &[frame]);
    let relation = relationship(&mut step, 99, 99, frame, frame);
    occurrence(&mut step, "bad", "Bad", "", definition, 12345, relation);
    let error = step.parse().expect_err("missing child definition");
    assert!(matches!(error, AssemblyError::Malformed(_)), "{error}");
}

#[test]
fn leaf_only_file_becomes_a_single_root() {
    let mut step = Step::new();
    let frame = frame(&mut step, [1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let (definition, representation) = product(&mut step, "part", "PART_A", &[frame]);
    let assembly = step.parse().expect("parse assembly");
    assert_eq!(assembly.roots.len(), 1);
    assert_eq!(assembly.roots[0].name, "PART_A");
    assert_eq!(assembly.roots[0].definition_id, Some(definition));
    assert_eq!(assembly.roots[0].representation_id, Some(representation));
    assert!(assembly.roots[0].world_transform.is_identity(1e-12));
}

// ---------------------------------------------------------------------------
// Real STEP fixtures from the repository (skipped when absent)
// ---------------------------------------------------------------------------

fn repository_file(name: &str) -> Option<std::path::PathBuf> {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for candidate in [
        manifest.join("../../../tests/Sample AP203 Data Files").join(name),
        manifest.join("../../../tests/Sample AP203 Data Files").join(name),
    ] {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn node_names(assembly: &Assembly) -> BTreeSet<String> {
    assembly.walk().iter().map(|node| node.name.clone()).collect()
}

/// Leaf nodes (placed parts) with their world translations.
fn leaf_translations(assembly: &Assembly) -> Vec<(String, [f64; 3])> {
    fn collect(node: &Node, out: &mut Vec<(String, [f64; 3])>) {
        if node.children.is_empty() {
            out.push((node.product_name.clone(), node.world_transform.translation()));
            return;
        }
        for child in &node.children {
            collect(child, out);
        }
    }
    let mut out = Vec::new();
    for root in &assembly.roots {
        collect(root, &mut out);
    }
    out
}

fn has_translation(leaves: &[(String, [f64; 3])], product: &str, expected: [f64; 3]) -> bool {
    leaves.iter().any(|(name, translation)| {
        name == product
            && translation
                .iter()
                .zip(expected.iter())
                .all(|(actual, expected)| (actual - expected).abs() < 1e-6)
    })
}

#[test]
fn parses_the_as1_pe_assembly_tree() {
    let Some(path) = repository_file("as1_pe.stp") else {
        eprintln!("skipping as1_pe.stp: fixture not present");
        return;
    };
    let assembly = parse_assembly_file(&path).expect("parse as1_pe.stp");
    assert_eq!(assembly.roots.len(), 1);
    let root = &assembly.roots[0];
    assert_eq!(root.name, "AS1_ASM");
    assert_eq!(root.product_name, "AS1_ASM");
    assert_eq!(root.children.len(), 4, "top-level instances");
    // Two occurrences of the L-BRACKET_ASM sub-assembly, each expanding its five parts.
    let sub_assemblies: Vec<&Node> = root
        .children
        .iter()
        .filter(|child| child.product_name == "L-BRACKET_ASM")
        .collect();
    assert_eq!(sub_assemblies.len(), 2);
    for sub in &sub_assemblies {
        assert_eq!(sub.children.len(), 5);
    }
    assert_eq!(assembly.node_count(), 1 + 4 + 2 * 5);
    assert_world_consistent(&assembly);
    for node in assembly.walk() {
        for row in node.world_transform.rows() {
            assert!(row.iter().all(|value| value.is_finite()));
        }
    }
    assert!(
        node_names(&assembly).iter().all(|name| !name.trim().is_empty()),
        "every node needs a name: {:?}",
        node_names(&assembly)
    );
    // Ground truth: smlibpy's `step_reader.interpret` places the same twelve bodies at
    // these file-unit translations (the file declares inches; smlibpy reports mm, i.e.
    // exactly 25.4x these numbers).
    let leaves = leaf_translations(&assembly);
    assert_eq!(leaves.len(), 12, "placed part bodies");
    for (product, translation) in [
        ("PLATE", [0.0, 0.0, 0.0]),
        ("L-BRACKET", [35.0, 0.0, 0.0]),
        ("L-BRACKET", [-135.0, 0.0, 0.0]),
        ("BOLT", [15.0, 10.0, 0.0]),
        ("BOLT", [-7.5, 10.0, -12.990_381]),
        ("BOLT", [-7.5, 10.0, 12.990_381]),
        ("BOLT", [-115.0, 10.0, 0.0]),
        ("NUT", [35.0, 40.0, 0.0]),
        ("NUT", [-135.0, 40.0, 0.0]),
        ("ROD", [-152.0, 40.0, 0.0]),
    ] {
        assert!(
            has_translation(&leaves, product, translation),
            "missing {product} at {translation:?} in {leaves:?}"
        );
    }
    assert!(assembly.warnings.is_empty(), "{:?}", assembly.warnings);
}

#[test]
fn parses_other_repository_assemblies_without_panic() {
    // (file, products, occurrences/nodes, root product name)
    let cases = [
        ("moon_buggy_asm.stp", 20usize, 30usize, "Moon Buggy"),
        ("weldment_asm_solid.stp", 34, 109, "WELDMENT_ASM"),
        ("rear.stp", 49, 50, "REAR_LEFT_QUARTER_ASM"),
        ("cubcylso.stp", 2, 2, "*MASTER"),
    ];
    for (name, products, nodes, root_name) in cases {
        let Some(path) = repository_file(name) else { continue };
        let assembly = parse_assembly_file(&path)
            .unwrap_or_else(|error| panic!("{name}: parse failed: {error}"));
        assert_eq!(assembly.products.len(), products, "{name}: products");
        assert_eq!(assembly.node_count(), nodes, "{name}: nodes");
        assert_eq!(assembly.roots.len(), 1, "{name}: roots");
        assert_eq!(assembly.roots[0].product_name, root_name, "{name}: root name");
        assert_world_consistent(&assembly);
        for node in assembly.walk() {
            assert!(!node.name.trim().is_empty(), "{name}: unnamed node");
        }
    }
}
