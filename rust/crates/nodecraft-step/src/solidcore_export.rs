//! Adapter from NodeCraft's planar body to the official SolidCore `step-export` crate.
//!
//! The official writer is intentionally narrower than the legacy writer: one
//! `MANIFOLD_SOLID_BREP`, one closed shell, planar faces, line edges. Callers opt in with
//! `NODECRAFT_STEP_WRITER=solidcore`; the legacy writer remains the default until the
//! upstream writer covers more of NodeCraft's export surface.

use step_import::model::{
    Body, BodyKind, Bound, Curve, Frame, Loop, Model, OrientedEdge, Shell, Surface, Vertex,
};

use crate::{planar_face_normals, PlanarBody};

fn unit(value: [f64; 3]) -> Result<[f64; 3], String> {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if !(length.is_finite() && length > 0.0) {
        return Err("degenerate planar direction".into());
    }
    Ok([value[0] / length, value[1] / length, value[2] / length])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn frame(origin: [f64; 3], normal: [f64; 3]) -> Result<Frame, String> {
    let z = unit(normal)?;
    let candidate = if z[0].abs() > 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let x = unit([
        candidate[0] - z[0] * dot(candidate, z),
        candidate[1] - z[1] * dot(candidate, z),
        candidate[2] - z[2] * dot(candidate, z),
    ])?;
    let y = cross(z, x);
    Ok(Frame { origin, x, y, z })
}

fn model(body: &PlanarBody) -> Result<Model, String> {
    if body.points.is_empty() || body.faces.is_empty() {
        return Err("planar body is empty".into());
    }
    let normals = planar_face_normals(body)?;
    let vertices = body
        .points
        .iter()
        .enumerate()
        .map(|(index, point)| Vertex {
            step_id: index as u64 + 1,
            point: *point,
        })
        .collect::<Vec<_>>();
    let mut edges = Vec::new();
    let mut edge_by_pair = std::collections::BTreeMap::new();
    for face_index in 0..body.faces.len() {
        for points in body.loops(face_index) {
            for index in 0..points.len() {
                let a = points[index];
                let b = points[(index + 1) % points.len()];
                if a == b {
                    return Err("degenerate planar edge".into());
                }
                let key = if a < b { (a, b) } else { (b, a) };
                if let std::collections::btree_map::Entry::Vacant(entry) = edge_by_pair.entry(key) {
                    let start = *body.points.get(key.0).ok_or("missing planar point")?;
                    let end = *body.points.get(key.1).ok_or("missing planar point")?;
                    let delta = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                    let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                    if !(length.is_finite() && length > 0.0) {
                        return Err("degenerate planar edge".into());
                    }
                    let dir = [delta[0] / length, delta[1] / length, delta[2] / length];
                    edges.push(step_import::model::Edge {
                        step_id: edges.len() as u64 + 1,
                        start: key.0,
                        end: key.1,
                        curve: Curve::Line {
                            origin: start,
                            dir,
                            magnitude: length,
                        },
                        same_sense: true,
                        t_start: 0.0,
                        t_end: length,
                    });
                    entry.insert(edges.len() - 1);
                }
            }
        }
    }
    let mut faces = Vec::with_capacity(body.faces.len());
    for face_index in 0..body.faces.len() {
        let mut loops = Vec::new();
        let mut outer = true;
        for points in body.loops(face_index) {
            let mut oriented = Vec::with_capacity(points.len());
            for index in 0..points.len() {
                let a = points[index];
                let b = points[(index + 1) % points.len()];
                let key = if a < b { (a, b) } else { (b, a) };
                let edge = *edge_by_pair.get(&key).ok_or("missing planar edge")?;
                oriented.push(OrientedEdge {
                    edge,
                    forward: a < b,
                });
            }
            loops.push(Loop {
                outer,
                orientation: true,
                bound: Bound::Edges(oriented),
            });
            outer = false;
        }
        let first = body.faces[face_index].first().copied().ok_or("empty planar face")?;
        let origin = *body.points.get(first).ok_or("missing planar point")?;
        faces.push(step_import::model::Face {
            step_id: face_index as u64 + 1,
            surface: Surface::Plane {
                frame: frame(origin, normals[face_index])?,
            },
            same_sense: true,
            loops,
        });
    }
    Ok(Model {
        pcurves: std::collections::BTreeMap::new(),
        file_pcurves: step_import::model::FilePcurves::default(),
        header: step_import::part21::Header::default(),
        diagnostics: Vec::new(),
        schemas: vec!["CONFIG_CONTROL_DESIGN".into()],
        bodies: vec![Body {
            step_id: 1,
            name: None,
            kind: BodyKind::ManifoldSolid,
            unit_mm: 1.0,
            uncertainty_mm: None,
            shells: vec![Shell {
                step_id: 1,
                closed: true,
                orientation: true,
                faces: (0..faces.len()).collect(),
            }],
            faces,
            edges,
            vertices,
        }],
        skipped: Vec::new(),
        assembly_instances: 0,
    })
}

/// Writes a planar closed body as a welded `FACETED_BREP` through `step-export`.
pub fn write_faceted_body(path: &str, body: &PlanarBody) -> Result<(), String> {
    if body.points.is_empty() || body.faces.is_empty() { return Err("planar body is empty".into()); }
    let text = step_export::write_faceted_display(&model(body)?).map_err(|error| error.to_string())?;
    std::fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}

/// Writes a planar closed body through the official SolidCore `step-export` crate.
pub fn write_planar_body(path: &str, body: &PlanarBody) -> Result<(), String> {
    let repaired;
    let body = if body.volume().is_finite() && body.volume() < 0.0 {
        repaired = {
            let mut copy = body.clone();
            copy.repair_orientation()?;
            copy
        };
        &repaired
    } else {
        body
    };
    let text = step_export::write(&model(body)?).map_err(|error| error.to_string())?;
    std::fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exports_a_box_through_the_official_writer() {
        let body = PlanarBody {
            points: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [2.0, 2.0, 2.0],
                [0.0, 2.0, 2.0],
            ],
            faces: vec![
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![1, 2, 6, 5],
                vec![2, 3, 7, 6],
                vec![3, 0, 4, 7],
            ],
            holes: vec![vec![]; 6],
        };
        let path = std::env::temp_dir().join(format!("sn-solidcore-export-{}.step", std::process::id()));
        write_planar_body(path.to_str().unwrap(), &body).unwrap();
        let imported = crate::read_planar_solids(path.to_str().unwrap()).unwrap();
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].faces.len(), 6);
        assert_eq!(imported[0].points.len(), 8);
    }
}


/// Write an exact SolidCore B-rep through the neutral model and the analytic/NURBS
/// `step-export` writer.
pub fn write_brep_exact(path: &str, brep: &solidcore::topology::Brep) -> Result<(), String> {
    let model = step_brep::brep_to_model(brep).map_err(|error| error.to_string())?;
    let text = step_export::write(&model).map_err(|error| error.to_string())?;
    std::fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}
