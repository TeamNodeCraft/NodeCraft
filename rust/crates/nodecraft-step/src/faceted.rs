//! Minimal `FACETED_BREP` reader for tessellated STEP bodies.

use crate::{PlanarBody, StepFile, Value};

fn refs(entity: &crate::Entity, index: usize) -> Vec<u32> {
    match entity.args.get(index) {
        Some(Value::List(values)) => values.iter().filter_map(|value| match value { Value::Ref(id) => Some(*id), _ => None }).collect(),
        Some(Value::Ref(id)) => vec![*id],
        _ => Vec::new(),
    }
}

fn point(file: &StepFile, id: u32) -> Option<[f64; 3]> {
    let entity = file.entities.get(&id)?;
    let Value::List(values) = entity.args.get(1)? else { return None };
    let mut out = [0.0; 3];
    for (index, value) in values.iter().take(3).enumerate() {
        let Value::Number(number) = value else { return None };
        out[index] = *number;
    }
    Some(out)
}

fn face_points(file: &StepFile, face_id: u32) -> Vec<[f64; 3]> {
    let Some(face) = file.entities.get(&face_id) else { return Vec::new() };
    for bound_id in refs(face, 1) {
        let Some(bound) = file.entities.get(&bound_id) else { continue };
        let loops = refs(bound, 1);
        let Some(loop_id) = loops.first().copied() else { continue };
        let Some(loop_) = file.entities.get(&loop_id) else { continue };
        let points: Vec<[f64; 3]> = refs(loop_, 1).into_iter().filter_map(|id| point(file, id)).collect();
        if points.len() >= 3 { return points; }
    }
    Vec::new()
}

/// Reads each `FACETED_BREP` as a planar body.
pub fn read_faceted_breps(path: &str) -> Result<Vec<PlanarBody>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text)?;
    let mut bodies = Vec::new();
    for brep in file.entities.values().filter(|entity| entity.name == "FACETED_BREP") {
        let Some(shell_id) = refs(brep, 1).first().copied() else { continue };
        let Some(shell) = file.entities.get(&shell_id) else { continue };
        let mut body = PlanarBody::default();
        let mut lookup: std::collections::BTreeMap<[i64; 3], usize> = std::collections::BTreeMap::new();
        for face_id in refs(shell, 1) {
            let points = face_points(&file, face_id);
            if points.len() < 3 { continue; }
            let mut indices = Vec::new();
            for point in points {
                let key = [(point[0] * 1e7).round() as i64, (point[1] * 1e7).round() as i64, (point[2] * 1e7).round() as i64];
                let index = *lookup.entry(key).or_insert_with(|| { body.points.push(point); body.points.len() - 1 });
                if indices.last().copied() != Some(index) { indices.push(index); }
            }
            if indices.first() == indices.last() { indices.pop(); }
            if indices.len() >= 3 { body.push_face(indices, Vec::new()); }
        }
        if !body.faces.is_empty() {
            if body.volume().is_finite() && body.volume() < 0.0 { body.repair_orientation()?; }
            bodies.push(body);
        }
    }
    Ok(bodies)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TETRA: &str = "DATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(1.,0.,0.));\n#3=CARTESIAN_POINT('',(0.,1.,0.));\n#4=CARTESIAN_POINT('',(0.,0.,1.));\n#10=POLY_LOOP('',(#1,#2,#3));\n#11=POLY_LOOP('',(#1,#3,#4));\n#12=POLY_LOOP('',(#1,#4,#2));\n#13=POLY_LOOP('',(#2,#4,#3));\n#20=FACE_OUTER_BOUND('',#10,.T.);\n#21=FACE_OUTER_BOUND('',#11,.T.);\n#22=FACE_OUTER_BOUND('',#12,.T.);\n#23=FACE_OUTER_BOUND('',#13,.T.);\n#30=FACE('',(#20));\n#31=FACE('',(#21));\n#32=FACE('',(#22));\n#33=FACE('',(#23));\n#40=CLOSED_SHELL('',(#30,#31,#32,#33));\n#50=FACETED_BREP('tetra',#40);\nENDSEC;";

    #[test]
    fn reads_tetrahedron_faceted_brep() {
        let path = std::env::temp_dir().join(format!("sn-faceted-{}.step", std::process::id()));
        std::fs::write(&path, TETRA).unwrap();
        let bodies = read_faceted_breps(path.to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0].faces.len(), 4);
        assert!((bodies[0].volume() - 1.0 / 6.0).abs() < 1e-9, "{}", bodies[0].volume());
    }
}
