//! Derived/product/context entity output tests for assemblies.

use std::fs;

#[test]
fn assembly_output_has_nauo_and_transform_entities() {
    let dir = std::env::temp_dir().join(format!("sn-derived-asm-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let body = nodecraft_step::PlanarBody {
        points: vec![[0.,0.,0.],[2.,0.,0.],[2.,2.,0.],[0.,2.,0.],[0.,0.,1.],[2.,0.,1.],[2.,2.,1.],[0.,2.,1.]],
        faces: vec![vec![0,3,2,1],vec![4,5,6,7],vec![0,1,5,4],vec![1,2,6,5],vec![2,3,7,6],vec![3,0,4,7]],
        holes: vec![vec![];6],
    };
    let part = step_assemblies::PlanarPart::placed("part", body, step_assemblies::Transform::IDENTITY);
    let path = dir.join("assembly.step");
    step_assemblies::write_planar_assembly(path.to_str().unwrap(), &[part]).unwrap();
    let text = fs::read_to_string(&path).unwrap();
    for required in ["NEXT_ASSEMBLY_USAGE_OCCURRENCE", "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION", "ITEM_DEFINED_TRANSFORMATION", "PRODUCT_DEFINITION_SHAPE"] {
        assert!(text.contains(required), "missing {required}");
    }
}
