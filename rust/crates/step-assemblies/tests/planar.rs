//! Placed planar solids from STEP assemblies: synthetic assemblies with independent
//! transforms, unit scaling, repeated instances, and explicit unsupported errors for
//! curved real-world fixtures.
//!
//! Synthetic parts are produced by writing a box with `nodecraft_step::write_box`, then
//! copying its geometry/topology entities into a hand-built assembly product graph. That
//! keeps the B-rep exact and the fixture small while still exercising the public
//! `nodecraft_step::read_planar_solids` extraction end to end.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use nodecraft_step::{StepFile as ParsedStepFile, Value, write_box};
use step_assemblies::planar::{PlanarInstance, UNSUPPORTED_ERROR_PREFIX, planar_instances};

// ---------------------------------------------------------------------------
// Minimal STEP assembly writer
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
             FILE_NAME('','',(''),(''),'step-assemblies-planar-test','','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n",
        );
        for (index, line) in self.lines.iter().enumerate() {
            out.push_str(&format!("#{}={};\n", index + 1, line));
        }
        out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        out
    }

    fn write(&self, name: &str) -> PathBuf {
        let path = temp_path(name);
        std::fs::write(&path, self.text()).unwrap();
        path
    }
}

/// Unique temp path: tests run in parallel threads of one process, so the pid alone is
/// not unique enough for the shared box-source files.
fn temp_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "step-planar-{}-{serial}-{tag}.step",
        std::process::id()
    ))
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

fn default_frame(step: &mut Step, origin: [f64; 3]) -> u32 {
    frame(step, origin, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
}

/// `PRODUCT` -> formation -> `PRODUCT_DEFINITION` -> `PRODUCT_DEFINITION_SHAPE` ->
/// `SHAPE_REPRESENTATION` chain. Returns `(definition, representation)`.
fn product(
    step: &mut Step,
    product_id: &str,
    name: &str,
    representation_items: &[u32],
) -> (u32, u32) {
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

/// CAx-IF representation relationship with `ITEM_DEFINED_TRANSFORMATION(child, parent)`.
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

/// `NEXT_ASSEMBLY_USAGE_OCCURRENCE` + occurrence shape + context-dependent shape.
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

fn length_unit(step: &mut Step, unit: &str) {
    match unit {
        "MILLI" => {
            step.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))");
        }
        "INCH" => {
            step.add("(NAMED_UNIT(*) SI_UNIT($,.INCH.))");
        }
        other => panic!("unsupported test unit {other}"),
    }
}

// ---------------------------------------------------------------------------
// Real planar box geometry, copied into the synthetic assembly
// ---------------------------------------------------------------------------

const COPIED_ENTITIES: &[&str] = &[
    "CARTESIAN_POINT",
    "DIRECTION",
    "VECTOR",
    "LINE",
    "CIRCLE",
    "AXIS2_PLACEMENT_3D",
    "PLANE",
    "VERTEX_POINT",
    "EDGE_CURVE",
    "ORIENTED_EDGE",
    "EDGE_LOOP",
    "FACE_OUTER_BOUND",
    "FACE_BOUND",
    "ADVANCED_FACE",
    "CLOSED_SHELL",
    "MANIFOLD_SOLID_BREP",
];

struct CopiedBox {
    solid: u32,
    shell: u32,
}

/// Write a box with the crate's own planar writer, then copy its geometry entities into
/// `step`, renumbering entity ids. The box is centred on the origin.
fn copy_box(step: &mut Step, size: [f64; 3], label: &str) -> CopiedBox {
    let source_path = temp_path(&format!("source-{label}"));
    write_box(source_path.to_str().unwrap(), size[0], size[1], size[2]).unwrap();
    let text = std::fs::read_to_string(&source_path).unwrap();
    let _ = std::fs::remove_file(&source_path);
    let file = ParsedStepFile::parse(&text).unwrap();

    let ordered: Vec<(u32, &nodecraft_step::Entity)> = file
        .entities
        .iter()
        .filter(|(_, entity)| COPIED_ENTITIES.contains(&entity.name.as_str()))
        .map(|(id, entity)| (*id, entity))
        .collect();
    let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
    for (index, (old_id, _)) in ordered.iter().enumerate() {
        remap.insert(*old_id, step.lines.len() as u32 + 1 + index as u32);
    }

    let mut solid = None;
    let mut shell = None;
    for (old_id, entity) in &ordered {
        let args: Vec<String> = entity
            .args
            .iter()
            .map(|value| value_text(value, &remap))
            .collect();
        let new_id = step.add(format!("{}({})", entity.name, args.join(",")));
        assert_eq!(new_id, remap[old_id], "entity id prediction drifted");
        match entity.name.as_str() {
            "MANIFOLD_SOLID_BREP" => solid = Some(new_id),
            "CLOSED_SHELL" => shell = Some(new_id),
            _ => {}
        }
    }
    CopiedBox {
        solid: solid.expect("box writer emits MANIFOLD_SOLID_BREP"),
        shell: shell.expect("box writer emits CLOSED_SHELL"),
    }
}

fn value_text(value: &Value, remap: &BTreeMap<u32, u32>) -> String {
    match value {
        Value::Ref(id) => match remap.get(id) {
            Some(new_id) => format!("#{new_id}"),
            None => panic!("copied geometry references entity #{id} outside the copied set"),
        },
        Value::Str(text) => format!("'{}'", text.replace('\'', "''")),
        Value::Number(number) => real(*number),
        Value::Enum(name) => format!(".{name}."),
        Value::List(values) => format!(
            "({})",
            values
                .iter()
                .map(|value| value_text(value, remap))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Typed(name, values) => {
            if values.is_empty() {
                name.clone()
            } else {
                format!(
                    "{name}({})",
                    values
                        .iter()
                        .map(|value| value_text(value, remap))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        }
        Value::Null => "$".into(),
    }
}

fn real(value: f64) -> String {
    if value == 0.0 {
        return "0.".into();
    }
    let text = format!("{value:?}");
    match text.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => {
            let mantissa = if mantissa.contains('.') {
                mantissa.to_string()
            } else {
                format!("{mantissa}.")
            };
            format!("{mantissa}E{exponent}")
        }
        None => {
            if text.contains('.') {
                text
            } else {
                format!("{text}.")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Assembly fixtures
// ---------------------------------------------------------------------------

/// Two boxes placed by independent occurrences: `PART_A` translated by `(10, 0, 0)` and
/// `PART_B` rotated 90 degrees about +Z and translated by `(0, 20, 0)`.
fn two_part_assembly(unit: &str) -> Step {
    let mut step = Step::new();
    length_unit(&mut step, unit);

    let part_a = copy_box(&mut step, [10.0, 4.0, 20.0], "a");
    let part_b = copy_box(&mut step, [6.0, 6.0, 6.0], "b");
    let child_a = default_frame(&mut step, [0.0, 0.0, 0.0]);
    let child_b = default_frame(&mut step, [0.0, 0.0, 0.0]);
    let (definition_a, representation_a) =
        product(&mut step, "part-a", "PART_A", &[child_a, part_a.solid]);
    let (definition_b, representation_b) =
        product(&mut step, "part-b", "PART_B", &[child_b, part_b.solid]);

    let parent_a = default_frame(&mut step, [10.0, 0.0, 0.0]);
    let parent_b = frame(
        &mut step,
        [0.0, 20.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    );
    let (root_definition, root_representation) =
        product(&mut step, "root", "ROOT", &[parent_a, parent_b]);

    let relation_a = relationship(
        &mut step,
        representation_a,
        root_representation,
        child_a,
        parent_a,
    );
    let relation_b = relationship(
        &mut step,
        representation_b,
        root_representation,
        child_b,
        parent_b,
    );
    occurrence(
        &mut step,
        "occ-a",
        "PART_A occurrence",
        "Placement A",
        root_definition,
        definition_a,
        relation_a,
    );
    occurrence(
        &mut step,
        "occ-b",
        "PART_B occurrence",
        "Placement B",
        root_definition,
        definition_b,
        relation_b,
    );
    step
}

fn repository_file(name: &str) -> Option<PathBuf> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for directory in ["Sample AP203 Data Files"] {
        let candidate = manifest.join("../../../tests").join(directory).join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn parse_instances(step: &Step, name: &str) -> Result<Vec<PlanarInstance>, String> {
    let path = step.write(name);
    let result = planar_instances(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    result
}

fn by_product<'a>(instances: &'a [PlanarInstance], product: &str) -> &'a PlanarInstance {
    instances
        .iter()
        .find(|instance| instance.product_name == product)
        .unwrap_or_else(|| panic!("no instance for {product}: {instances:?}"))
}

fn assert_translation(instance: &PlanarInstance, expected: [f64; 3]) {
    let actual = instance.world_transform.translation();
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() < 1e-6,
            "{}: translation[{axis}] {} != {}",
            instance.name,
            actual[axis],
            expected[axis]
        );
    }
}

fn assert_bounds(instance: &PlanarInstance, min: [f64; 3], max: [f64; 3]) {
    let (actual_min, actual_max) = instance.body.bounds().expect("body has points");
    for axis in 0..3 {
        assert!(
            (actual_min[axis] - min[axis]).abs() < 1e-6,
            "{}: min[{axis}] {} != {}",
            instance.name,
            actual_min[axis],
            min[axis]
        );
        assert!(
            (actual_max[axis] - max[axis]).abs() < 1e-6,
            "{}: max[{axis}] {} != {}",
            instance.name,
            actual_max[axis],
            max[axis]
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn real_planar_part_becomes_one_identity_instance() {
    let Some(path) = repository_file("block.stp") else {
        eprintln!("skipping block.stp: fixture not present");
        return;
    };
    let instances = planar_instances(path.to_str().unwrap()).expect("block.stp is a planar part");
    assert_eq!(instances.len(), 1);
    let instance = &instances[0];
    assert_eq!(instance.body.faces.len(), 6);
    assert_eq!(instance.body.points.len(), 8);
    assert_eq!(instance.product_name, "Product");
    assert!(!instance.name.trim().is_empty());
    assert!(instance.world_transform.is_identity(1e-9));
    // The identity instance must carry exactly the body the plain planar reader returns.
    let bodies = nodecraft_step::read_planar_solids(path.to_str().unwrap()).unwrap();
    assert_eq!(bodies.len(), 1);
    assert_eq!(instance.body.bounds(), bodies[0].bounds());
}

#[test]
fn synthetic_assembly_places_two_planar_parts() {
    let step = two_part_assembly("MILLI");
    let instances = parse_instances(&step, "two-parts").expect("planar assembly");
    assert_eq!(instances.len(), 2, "{instances:#?}");

    let part_a = by_product(&instances, "PART_A");
    assert_eq!(part_a.body.faces.len(), 6);
    assert_eq!(part_a.name, "Placement A");
    assert_eq!(part_a.world_transform.translation(), [10.0, 0.0, 0.0]);
    // Box 10 x 4 x 20 centred on the origin, then translated by +10 in x.
    assert_bounds(part_a, [5.0, -2.0, -10.0], [15.0, 2.0, 10.0]);

    let part_b = by_product(&instances, "PART_B");
    assert_eq!(part_b.body.faces.len(), 6);
    assert_eq!(part_b.name, "Placement B");
    assert_eq!(part_b.world_transform.translation(), [0.0, 20.0, 0.0]);
    // 6-cube rotated 90 degrees about +Z (local X -> world Y), then translated +20 in y.
    assert_bounds(part_b, [-3.0, 17.0, -3.0], [3.0, 23.0, 3.0]);
}

#[test]
fn world_transform_field_matches_transformed_body() {
    let step = two_part_assembly("MILLI");
    let instances = parse_instances(&step, "transform-field").unwrap();
    for instance in &instances {
        // Both boxes are centred on their local origin, so the transformed body centre is
        // the world translation.
        let (min, max) = instance.body.bounds().unwrap();
        for axis in 0..3 {
            let centre = (min[axis] + max[axis]) * 0.5;
            assert!(
                (centre - instance.world_transform.translation()[axis]).abs() < 1e-6,
                "{}: centre {centre} != translation {:?}",
                instance.name,
                instance.world_transform.translation()
            );
        }
    }
}

#[test]
fn inch_units_scale_both_bodies_and_transforms() {
    let step = two_part_assembly("INCH");
    let instances = parse_instances(&step, "inch").unwrap();
    let part_a = by_product(&instances, "PART_A");
    assert_translation(part_a, [254.0, 0.0, 0.0]);
    assert_bounds(
        part_a,
        [5.0 * 25.4, -2.0 * 25.4, -10.0 * 25.4],
        [15.0 * 25.4, 2.0 * 25.4, 10.0 * 25.4],
    );

    let part_b = by_product(&instances, "PART_B");
    assert_translation(part_b, [0.0, 20.0 * 25.4, 0.0]);
    assert_bounds(
        part_b,
        [-3.0 * 25.4, 17.0 * 25.4, -3.0 * 25.4],
        [3.0 * 25.4, 23.0 * 25.4, 3.0 * 25.4],
    );
}

#[test]
fn repeated_parts_get_independent_transformed_bodies() {
    let mut step = Step::new();
    length_unit(&mut step, "MILLI");
    let part = copy_box(&mut step, [10.0, 4.0, 20.0], "repeat");
    let child = default_frame(&mut step, [0.0, 0.0, 0.0]);
    let (part_definition, part_representation) =
        product(&mut step, "part", "PART", &[child, part.solid]);

    let first_parent = default_frame(&mut step, [10.0, 0.0, 0.0]);
    let second_parent = default_frame(&mut step, [20.0, 0.0, 0.0]);
    let (root_definition, root_representation) =
        product(&mut step, "root", "ROOT", &[first_parent, second_parent]);
    let first_relation = relationship(
        &mut step,
        part_representation,
        root_representation,
        child,
        first_parent,
    );
    let second_relation = relationship(
        &mut step,
        part_representation,
        root_representation,
        child,
        second_parent,
    );
    occurrence(
        &mut step,
        "occ-1",
        "first",
        "First placement",
        root_definition,
        part_definition,
        first_relation,
    );
    occurrence(
        &mut step,
        "occ-2",
        "second",
        "Second placement",
        root_definition,
        part_definition,
        second_relation,
    );

    let instances = parse_instances(&step, "repeat").unwrap();
    assert_eq!(instances.len(), 2, "{instances:#?}");
    assert!(instances.iter().all(|instance| instance.solid_id == instances[0].solid_id));
    let first = instances
        .iter()
        .find(|instance| instance.name == "First placement")
        .expect("first instance");
    let second = instances
        .iter()
        .find(|instance| instance.name == "Second placement")
        .expect("second instance");
    assert_bounds(first, [5.0, -2.0, -10.0], [15.0, 2.0, 10.0]);
    assert_bounds(second, [15.0, -2.0, -10.0], [25.0, 2.0, 10.0]);
    assert_ne!(first.body.points, second.body.points);
}

#[test]
fn multiple_solids_in_one_representation_become_separate_instances() {
    let mut step = Step::new();
    length_unit(&mut step, "MILLI");
    let first = copy_box(&mut step, [2.0, 2.0, 2.0], "solid-a");
    let second = copy_box(&mut step, [4.0, 4.0, 4.0], "solid-b");
    // One product whose shape representation holds two solids. With no assembly
    // occurrences it becomes a single root leaf, so both solids share its transform.
    let (_definition, _representation) =
        product(&mut step, "multi", "MULTI", &[first.solid, second.solid]);

    let instances = parse_instances(&step, "multi-solid").unwrap();
    assert_eq!(instances.len(), 2, "{instances:#?}");
    let mut solid_ids: Vec<u32> = instances.iter().map(|instance| instance.solid_id).collect();
    solid_ids.sort_unstable();
    assert_eq!(solid_ids, vec![first.solid, second.solid]);
    // Both bodies share the leaf transform (identity).
    for instance in &instances {
        assert!(instance.world_transform.is_identity(1e-9));
        assert_eq!(instance.body.faces.len(), 6);
    }
    let small = instances.iter().find(|instance| instance.solid_id == first.solid).unwrap();
    let large = instances.iter().find(|instance| instance.solid_id == second.solid).unwrap();
    assert_bounds(small, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    assert_bounds(large, [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
}

#[test]
fn leaf_without_solid_geometry_is_an_explicit_unsupported_error() {
    let mut step = Step::new();
    length_unit(&mut step, "MILLI");
    let _orphan = copy_box(&mut step, [2.0, 2.0, 2.0], "orphan");
    // A product whose representation has no items: the file still contains a solid, but
    // no leaf instance selects it.
    let _ = product(&mut step, "empty", "EMPTY", &[]);
    let error = parse_instances(&step, "no-geometry").expect_err("must not silently drop");
    assert!(error.starts_with(UNSUPPORTED_ERROR_PREFIX), "{error}");
    assert!(error.contains("no leaf instance"), "{error}");
}

#[test]
fn mapped_item_leaf_geometry_is_an_explicit_unsupported_error() {
    let mut step = Step::new();
    length_unit(&mut step, "MILLI");
    let solid = copy_box(&mut step, [2.0, 2.0, 2.0], "mapped-peer");
    let _peer = product(&mut step, "peer", "PEER", &[solid.solid]);

    let mapped = step.add("MAPPED_ITEM('',$,$)");
    let _mapped_part = product(&mut step, "mapped", "MAPPED_PART", &[mapped]);

    let error = parse_instances(&step, "mapped").expect_err("mapped geometry is unsupported");
    assert!(error.starts_with(UNSUPPORTED_ERROR_PREFIX), "{error}");
    assert!(error.contains("MAPPED_ITEM"), "{error}");
}

#[test]
fn void_shell_is_an_explicit_unsupported_error() {
    let mut step = Step::new();
    length_unit(&mut step, "MILLI");
    let part = copy_box(&mut step, [2.0, 2.0, 2.0], "void");
    let void_shell = step.add("OPEN_SHELL('',())");
    let solid = step.add(format!(
        "BREP_WITH_VOIDS('',#{},(#{void_shell}))",
        part.shell
    ));
    let _ = product(&mut step, "voided", "VOIDED", &[solid]);

    let error = parse_instances(&step, "void").expect_err("void shells are unsupported");
    assert!(error.starts_with(UNSUPPORTED_ERROR_PREFIX), "{error}");
    assert!(error.contains("void"), "{error}");
}

#[test]
fn curved_real_assemblies_report_explicit_unsupported_errors() {
    let mut checked = 0;
    for name in [
        "as1_pe.stp",
        "moon_buggy_asm.stp",
        "weldment_asm_solid.stp",
        "vaccase_asm_solid.stp",
    ] {
        let Some(path) = repository_file(name) else {
            continue;
        };
        checked += 1;
        match planar_instances(path.to_str().unwrap()) {
            Ok(instances) => panic!(
                "{name}: expected unsupported geometry, got {} planar instances",
                instances.len()
            ),
            Err(error) => assert!(
                error.starts_with(UNSUPPORTED_ERROR_PREFIX),
                "{name}: {error}"
            ),
        }
    }
    if checked == 0 {
        eprintln!("skipping curved assembly fixtures: none present");
    }
}

#[test]
fn syntax_errors_keep_their_own_message() {
    let path = temp_path("syntax");
    std::fs::write(&path, "ISO-10303-21;HEADER;ENDSEC;DATA;#1=CARTESIAN_POINT('',(0.,0.,0.)")
        .unwrap();
    let error = planar_instances(path.to_str().unwrap()).expect_err("truncated file");
    let _ = std::fs::remove_file(&path);
    assert!(error.contains("STEP parse error"), "{error}");
    assert!(!error.starts_with(UNSUPPORTED_ERROR_PREFIX), "{error}");
}

#[test]
fn missing_file_is_an_io_error() {
    let path = std::env::temp_dir().join("step-planar-does-not-exist.step");
    let error = planar_instances(path.to_str().unwrap()).expect_err("missing file");
    assert!(!error.starts_with(UNSUPPORTED_ERROR_PREFIX), "{error}");
    assert!(error.contains("No such file") || error.contains("os error"), "{error}");
}
