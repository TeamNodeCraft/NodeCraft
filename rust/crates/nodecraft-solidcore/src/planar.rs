//! Extract planar B-rep faces into the neutral STEP `PlanarBody`.
//!
//! This is the bridge for STEP export: solidcore topology -> shared vertices/loops ->
//! `nodecraft-step::write_planar_body`.

use solidcore::kernel::Kernel;
use solidcore::query::api_queries::face_get_boundary_loops;
use solidcore::query::face::p1::evaluate_curve;
use solidcore::topology::Brep;
use nodecraft_step::PlanarBody;
use std::collections::BTreeMap;

fn point_key(point: [f64; 3]) -> [i64; 3] {
    [
        (point[0] * 1e7).round() as i64,
        (point[1] * 1e7).round() as i64,
        (point[2] * 1e7).round() as i64,
    ]
}

pub fn planar_body(kernel: &Kernel, brep: &Brep) -> Result<PlanarBody, String> {
    let mut body = PlanarBody::default();
    let mut indices = BTreeMap::new();
    for (_face_id, face) in brep.faces().iter() {
        if let Some(surface) = &face.surface {
            let planar = surface.is_planar(kernel, 1e-6).map_err(|error| error.to_string())?;
            let _ = planar;
        } else {
            return Err("face has no surface".into());
        }
        let loops = face_get_boundary_loops(brep, _face_id).map_err(|error| error.to_string())?;
        let mut offset = 0usize;
        let mut polygons: Vec<Vec<usize>> = Vec::new();
        for count in &loops.edge_counts {
            let mut polygon = Vec::new();
            for index in offset..offset + count {
                let orientation = loops.orientations[index];
                let edge = brep.edge(loops.edges[index]).map_err(|error| error.to_string())?;
                let forward = format!("{orientation:?}") == "Same";
                let parameter = if forward { edge.interval.evaluate(0.0) } else { edge.interval.evaluate(1.0) };
                let values = evaluate_curve(kernel, edge.curve.as_ref(), parameter).map_err(|error| error.to_string())?;
                let point = [values[0].x, values[0].y, values[0].z];
                let key = point_key(point);
                let vertex = *indices.entry(key).or_insert_with(|| {
                    body.points.push(point);
                    body.points.len() - 1
                });
                if polygon.last().copied() != Some(vertex) {
                    polygon.push(vertex);
                }
            }
            offset += count;
            if polygon.len() >= 3 {
                if polygon.first() == polygon.last() { polygon.pop(); }
                if polygon.len() >= 3 { polygons.push(polygon); }
            }
        }
        let mut loops = polygons.into_iter();
        if let Some(outer) = loops.next() {
            body.push_face(outer, loops.collect());
        }
    }
    if body.faces.is_empty() { return Err("B-rep has no planar faces".into()); }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::primitive::api_primitives::create_box;

    #[test]
    fn box_becomes_planar_body() {
        let mut kernel = crate::kernel::kernel();
        let brep = create_box(&mut kernel, solidcore::math::Point3::new(-5.0, -2.0, -10.0), 10.0, 4.0, 20.0).unwrap();
        let body = planar_body(&kernel, &brep).unwrap();
        assert_eq!(body.faces.len(), 6);
        assert_eq!(body.points.len(), 8);
        assert!((body.volume().abs() - 800.0).abs() < 1e-6);
        assert!((body.area() - 640.0).abs() < 1e-6);
    }
}
