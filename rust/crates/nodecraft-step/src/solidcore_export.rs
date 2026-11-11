//! NodeCraft's STEP writers: the upstream neutral-model writer for planar/faceted
//! bodies, and the exact record writer for evaluated B-reps.
//!
//! `step-export` writes the neutral planar/faceted model: one `MANIFOLD_SOLID_BREP`
//! with planar faces and line edges (or a welded `FACETED_BREP` for the display
//! writer). [`crate::write_planar_body`] and [`crate::write_faceted_brep`] route
//! through it via [`write_planar_body`] / [`write_faceted_body`].
//!
//! Exact B-rep export ([`write_brep_exact`] / [`write_breps_exact`]) uses
//! `solidcore-step`'s `write::topology` instead, which writes the schema-4 record
//! through `solidcore-convert` without going through this neutral planar model.

use solidcore_record::brep::Brep as RecordBrep;
use solidcore_record::step::model::{
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
                    edges.push(solidcore_record::step::model::Edge {
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
        faces.push(solidcore_record::step::model::Face {
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
        file_pcurves: solidcore_record::step::model::FilePcurves::default(),
        header: solidcore_record::step::p21::Header::default(),
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


/// Write an exact SolidCore B-rep as one STEP part.
///
/// The body is converted to its schema-4 record with `solidcore-convert` and written
/// by `solidcore-step`'s `write::topology`, which states the exact analytic/NURBS
/// geometry and topology of every face.
pub fn write_brep_exact(path: &str, brep: &solidcore::topology::Brep) -> Result<(), String> {
    write_breps_exact(path, std::slice::from_ref(brep))
}

/// Write several exact SolidCore solids as one STEP Part.
///
/// Each body is converted to its record form independently and written as its own
/// named product; the exchange file shares the header, unit context and geometry
/// writer between them, exactly as upstream's exporter does.
pub fn write_breps_exact(path: &str, breps: &[solidcore::topology::Brep]) -> Result<(), String> {
    if breps.is_empty() {
        return Err("Part has no solids".into());
    }
    let stem = file_stem(path);
    let mut kernel = solidcore::kernel::Kernel::default();
    let mut records = Vec::with_capacity(breps.len());
    for (index, brep) in breps.iter().enumerate() {
        let mut copy = brep
            .clone_standing_for_self()
            .map_err(|error| format!("STEP export copy: {error}"))?;
        let (record, _order) = solidcore_convert::brep::to_record(&mut kernel, &mut copy)
            .map_err(|error| format!("STEP export record: {error}"))?;
        if record.regions.len() <= 1 && record.shells.is_empty() {
            return Err(format!("body {} has no STEP geometry", index + 1));
        }
        let name = if breps.len() == 1 { stem.clone() } else { format!("{stem}_{}", index + 1) };
        records.push((name, record));
    }
    let text = write_records(&records, &stem)?;
    std::fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}

/// Writes records as one AP242 exchange file: header, unit/uncertainty context and
/// one product per record with its exact solids/surface model.
fn write_records(records: &[(String, RecordBrep)], file_name: &str) -> Result<String, String> {
    use std::fmt::Write as _;

    use solidcore_step::p21::Param;
    use solidcore_step::write::{GeomWriter, Writer, string, topology};

    fn s(text: &str) -> Param {
        Param::Str(text.to_string())
    }
    fn r(id: u64) -> Param {
        Param::Ref(id)
    }
    fn refs(ids: &[u64]) -> Param {
        Param::List(ids.iter().map(|id| r(*id)).collect())
    }

    /// The least distance uncertainty written (mm).
    const MIN_UNCERTAINTY: f64 = 1e-6;

    /// The distance uncertainty a record states: its own tolerance and those of every
    /// edge and vertex it holds (STEP keeps no tolerance per edge, so the context's
    /// value is the only place an edge's deviation can go).
    fn record_uncertainty(record: &RecordBrep) -> f64 {
        std::iter::once(record.tolerance)
            .chain(record.edges.iter().map(|edge| edge.tolerance))
            .chain(record.vertices.iter().map(|vertex| vertex.tolerance))
            .filter(|tolerance| tolerance.is_finite())
            .fold(MIN_UNCERTAINTY, f64::max)
    }

    let mut w = Writer::new();
    let app = w.add_distinct("APPLICATION_CONTEXT", &[s("managed model based 3d engineering")]);
    w.add_distinct(
        "APPLICATION_PROTOCOL_DEFINITION",
        &[s("international standard"), s("ap242_managed_model_based_3d_engineering"), Param::Int(2014), r(app)],
    );
    let pctx = w.add_distinct("PRODUCT_CONTEXT", &[s(""), r(app), s("mechanical")]);
    let pdctx = w.add_distinct("PRODUCT_DEFINITION_CONTEXT", &[s("part definition"), r(app), s("design")]);
    let mm = w.add_complex(&[
        ("LENGTH_UNIT", vec![]),
        ("NAMED_UNIT", vec![Param::Derived]),
        ("SI_UNIT", vec![Param::Enum("MILLI".into()), Param::Enum("METRE".into())]),
    ]);
    let rad = w.add_complex(&[
        ("NAMED_UNIT", vec![Param::Derived]),
        ("PLANE_ANGLE_UNIT", vec![]),
        ("SI_UNIT", vec![Param::Null, Param::Enum("RADIAN".into())]),
    ]);
    let sr = w.add_complex(&[
        ("NAMED_UNIT", vec![Param::Derived]),
        ("SI_UNIT", vec![Param::Null, Param::Enum("STERADIAN".into())]),
        ("SOLID_ANGLE_UNIT", vec![]),
    ]);
    let uncertainty = records.iter().map(|(_, record)| record_uncertainty(record)).fold(MIN_UNCERTAINTY, f64::max);
    let unc = w.add_distinct(
        "UNCERTAINTY_MEASURE_WITH_UNIT",
        &[Param::Typed("LENGTH_MEASURE".into(), vec![Param::Real(uncertainty)]), r(mm), s("distance_accuracy_value"), s("confusion accuracy")],
    );
    let ctx = w.add_complex(&[
        ("GEOMETRIC_REPRESENTATION_CONTEXT", vec![Param::Int(3)]),
        ("GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT", vec![refs(&[unc])]),
        ("GLOBAL_UNIT_ASSIGNED_CONTEXT", vec![refs(&[mm, rad, sr])]),
        ("REPRESENTATION_CONTEXT", vec![s("Context #1"), s("3D Context with UNIT and UNCERTAINTY")]),
    ]);
    let mut product_ids = Vec::with_capacity(records.len());
    let mut wrote_geometry = false;
    for (name, record) in records {
        let written = topology::brep(&mut GeomWriter::new(&mut w), record, name)
            .map_err(|error| format!("STEP export of '{name}': {error}"))?;
        let (solids, sheets) = (written.solids, written.sheets);
        if !solids.is_empty() || sheets.is_some() {
            wrote_geometry = true;
        }
        let prod = w.add_distinct("PRODUCT", &[s(name), s(name), s(""), refs(&[pctx])]);
        product_ids.push(prod);
        let pdf = w.add_distinct("PRODUCT_DEFINITION_FORMATION", &[s(""), s(""), r(prod)]);
        let pd = w.add_distinct("PRODUCT_DEFINITION", &[s("design"), s(""), r(pdf), r(pdctx)]);
        let pds = w.add_distinct("PRODUCT_DEFINITION_SHAPE", &[s(""), s(""), r(pd)]);
        // The product's own shape representation carries the solid items directly, as
        // the previous writer did: NodeCraft's assembly reader and other consumers walk
        // SHAPE_DEFINITION_REPRESENTATION to find a product's geometry, and this file has
        // no assembly occurrences to place through representation relationships.
        if !solids.is_empty() {
            let rep = w.add_distinct("ADVANCED_BREP_SHAPE_REPRESENTATION", &[s(name), refs(&solids), r(ctx)]);
            w.add_distinct("SHAPE_DEFINITION_REPRESENTATION", &[r(pds), r(rep)]);
        }
        if let Some(sheets) = sheets {
            let rep = w.add_distinct("MANIFOLD_SURFACE_SHAPE_REPRESENTATION", &[s(name), refs(&[sheets]), r(ctx)]);
            w.add_distinct("SHAPE_DEFINITION_REPRESENTATION", &[r(pds), r(rep)]);
        }
    }
    if !wrote_geometry {
        return Err("STEP export produced no bodies".into());
    }
    if !product_ids.is_empty() {
        w.add_distinct("PRODUCT_RELATED_PRODUCT_CATEGORY", &[s("part"), s(""), refs(&product_ids)]);
    }

    let mut text = String::new();
    text.push_str("ISO-10303-21;\nHEADER;\n");
    text.push_str("FILE_DESCRIPTION((''),'2;1');\n");
    let _ = writeln!(
        text,
        "FILE_NAME({},'',(''),(''),'NodeCraft','nodecraft-step','');",
        string(file_name)
    );
    text.push_str(
        "FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }'));\nENDSEC;\nDATA;\n",
    );
    text.push_str(&w.data());
    text.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
    Ok(text)
}

/// The file name (without extension) used for the written product and header.
fn file_stem(path: &str) -> String {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let stem = base
        .strip_suffix(".step")
        .or_else(|| base.strip_suffix(".stp"))
        .or_else(|| base.strip_suffix(".STEP"))
        .or_else(|| base.strip_suffix(".STP"))
        .unwrap_or(base);
    if stem.trim().is_empty() { "part".to_string() } else { stem.trim().to_string() }
}

#[cfg(test)]
mod part_export_tests {
    use super::*;

    #[test]
    fn writes_multiple_solids_as_one_part() {
        use solidcore::kernel::Kernel;
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives as api;

        let mut kernel = Kernel::default();
        let a = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
        let b = api::create_box(&mut kernel, Point3::new(20.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
        let path = std::env::temp_dir().join(format!(
            "nodecraft-part-{}-{}.step",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        write_breps_exact(path.to_str().unwrap(), &[a, b]).expect("write part");
        let text = std::fs::read_to_string(&path).expect("read part");
        assert_eq!(text.matches("MANIFOLD_SOLID_BREP").count(), 2);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn exact_export_round_trips_through_the_step_reader() {
        use solidcore::kernel::Kernel;
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives as api;

        let mut kernel = Kernel::default();
        let box_ = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
        let cylinder = api::create_cylinder(&mut kernel, Point3::new(0.0, 0.0, 0.0), 3.0, 8.0).unwrap();
        let path = std::env::temp_dir().join(format!(
            "nodecraft-exact-roundtrip-{}-{}.step",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        write_breps_exact(path.to_str().unwrap(), &[box_, cylinder]).expect("write part");
        let text = std::fs::read_to_string(&path).expect("read part");
        let exchange = solidcore_step::p21::parse(&text).expect("parse written STEP");
        let solids = solidcore_step::topology::read_solids(&exchange);
        assert!(solids.failures.is_empty(), "{:?}", solids.failures);
        assert_eq!(solids.solids.len(), 2);
        // The box keeps its exact corners and the cylinder its three faces: the writer
        // writes the record's topology, not a re-tessellation.
        let mut box_bounds = None;
        let mut cylinder_faces = None;
        for solid in &solids.solids {
            let mut min = [f64::INFINITY; 3];
            let mut max = [f64::NEG_INFINITY; 3];
            for vertex in &solid.brep.vertices {
                for axis in 0..3 {
                    min[axis] = min[axis].min(vertex.point[axis]);
                    max[axis] = max[axis].max(vertex.point[axis]);
                }
            }
            if solid.brep.faces.len() == 6 {
                box_bounds = Some((min, max));
            } else {
                cylinder_faces = Some(solid.brep.faces.len());
            }
        }
        assert_eq!(box_bounds, Some(([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])));
        assert_eq!(cylinder_faces, Some(3));
        let _ = std::fs::remove_file(path);
    }
}
