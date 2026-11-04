//! Builds a faceted solidcore `Brep` from an indexed triangle mesh.
//!
//! This is the first STEP-IO adapter stage: it turns imported planar STEP geometry
//! (already triangulated) into a real solidcore B-rep so volume/area queries and
//! future booleans can consume it. Curved STEP import should later replace the
//! faceted faces with analytic/NURBS faces.

use std::collections::BTreeMap;

use solidcore::geom::attribute::{AttributeStore, ObjectKey};
use solidcore::geom::extent1d::Extent1d;
use solidcore::geom::extent2d::Extent2d;
use solidcore::error::EntityRef;
use solidcore::geom::object::ObjectRef;
use solidcore::kernel::Kernel;
use solidcore::math::{Point3, Vec3};
use solidcore::nurbs::b_spline_curve::BSplineCurve;
use solidcore::nurbs::b_spline_surface::BSplineSurface;
use solidcore::surface::surface::Surface;
use solidcore::topology::brep::p10::{make_topology_from_data, MakeTopologyOptions, TopologyEnv, HEALER_VERSION};
use solidcore::topology::brep_data::{
    BrepData, EdgeData, EuData, FaceData, FaceuseData, Keyed, LoopData, RegionData, ShellData, VertexData,
};
use solidcore::topology::Brep;

fn key_of(entity: EntityRef) -> ObjectRef {
    let (type_code, index) = match entity {
        EntityRef::Face(id) => (16002, u64::from(id.index())),
        EntityRef::Edge(id) => (16016, u64::from(id.index())),
        EntityRef::Edgeuse(id) => (16005, u64::from(id.index())),
        EntityRef::Vertex(id) => (16021, u64::from(id.index())),
        EntityRef::Vertexuse(id) => (16006, u64::from(id.index())),
        EntityRef::Loop(id) => (16003, u64::from(id.index())),
        EntityRef::Loopuse(id) => (16004, u64::from(id.index())),
        EntityRef::Faceuse(id) => (16022, u64::from(id.index())),
        EntityRef::Shell(id) => (16013, u64::from(id.index())),
        EntityRef::Region(id) => (16012, u64::from(id.index())),
        EntityRef::Brep(_) => (16011, 0),
    };
    ObjectRef { type_code, key: ObjectKey(1_000_000 + type_code as u64 * 100_000 + index) }
}

fn line(a: Point3, b: Point3) -> Result<solidcore::curve::curve::Curve, String> {
    Ok(solidcore::curve::curve::Curve::from(BSplineCurve::create_line_segment(3, a, b).map_err(|e| e.to_string())?))
}

fn triangle_plane(a: Point3, b: Point3, c: Point3) -> Result<Surface, String> {
    let p11 = b + (c - a);
    Ok(Surface::from(BSplineSurface::create_bilinear_surface(a, b, c, p11).map_err(|e| e.to_string())?))
}

/// Builds a closed faceted solid from shared vertices and triangles.
pub fn brep_from_triangles(points: &[[f64; 3]], triangles: &[[u32; 3]]) -> Result<Brep, String> {
    if points.is_empty() || triangles.is_empty() { return Err("mesh is empty".into()); }
    let points: Vec<Point3> = points.iter().map(|p| Point3::new(p[0], p[1], p[2])).collect();
    let mut data = BrepData::new(1, solidcore::topology::brep_data::DataSourceType::Unknown);
    data.healer_version = HEALER_VERSION;
    data.regions.push(RegionData { start_shell: 0, num_shells: 1, is_void: 0, ..RegionData::zeroed() });
    data.shells.push(ShellData { shell_type: 0, faceuse_start: 0, num_faceuses: triangles.len() as u64, ..ShellData::zeroed() });
    for point in &points { data.vertices.push(VertexData { point: *point, zone_tol_3d: 1e-7, ..VertexData::zeroed() }); }
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for (face_index, triangle) in triangles.iter().enumerate() {
        data.faceuses.push(FaceuseData { face: face_index as u64, orientation: 1, ..FaceuseData::zeroed() });
        data.surfaces.push(Keyed { key: ObjectKey(500 + face_index as u64), object: triangle_plane(points[triangle[0] as usize], points[triangle[1] as usize], points[triangle[2] as usize])? });
        data.faces.push(FaceData { surface: face_index as u64, start_loop: face_index as u64, num_loops: 1, uv_domain: Extent2d::from_bounds(0.0, 0.0, 1.0, 1.0), rectangular_trim: 0, zone_tol_3d: 1e-4, ..FaceData::zeroed() });
        data.loops.push(LoopData { loop_type: 0, start_eu: data.edgeuses.len() as u64, num_eu: 3, ..LoopData::zeroed() });
        for side in 0..3 {
            let a = triangle[side]; let b = triangle[(side + 1) % 3];
            let key = (a.min(b), a.max(b));
            let edge = edges.iter().position(|candidate| *candidate == key).unwrap_or_else(|| {
                edges.push(key);
                data.curves_3d.push(Keyed { key: ObjectKey(900 + edges.len() as u64), object: line(points[key.0 as usize], points[key.1 as usize]).unwrap() });
                data.edges.push(EdgeData { curve: edges.len() as u64 - 1, start_vertex: key.0 as u64, end_vertex: key.1 as u64, prim_eu: data.edgeuses.len() as i64, interval: Extent1d::from_raw(0.0, 1.0), zone_tol_3d: 1e-6, ..EdgeData::zeroed() });
                edges.len() - 1
            });
            data.edgeuses.push(EuData { eu_type: 0, edge: edge as u64, loop_: face_index as u64, orientation: i32::from(a < b), uv_curve: 999_999, ..EuData::zeroed() });
        }
    }
    let mut kernel = crate::kernel::kernel();
    let mut brep = Brep::new_with_infinite_region().map_err(|e| e.to_string())?;
    let mut store = AttributeStore::new();
    let mut records = solidcore::topology::topology::NotifyRecords::new();
    let mut env = TopologyEnv { all_attributes: &[], attributes: &mut store, key_of: &key_of, records: &mut records, next_key: 1 };
    make_topology_from_data(&mut kernel, &mut brep, &mut data, &mut env, MakeTopologyOptions::default()).map_err(|e| e.to_string())?;
    Ok(brep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::mass::integrator::properties::{brep_area, brep_volume, PropertyAttributes};
    use solidcore::topology::Brep;

    fn box_mesh() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let points = vec![
            [-1.,-1.,-1.],[1.,-1.,-1.],[1.,1.,-1.],[-1.,1.,-1.],
            [-1.,-1.,1.],[1.,-1.,1.],[1.,1.,1.],[-1.,1.,1.],
        ];
        let triangles = vec![
            [0,2,1],[0,3,2], [4,5,6],[4,6,7], [0,1,5],[0,5,4],
            [1,2,6],[1,6,5], [2,3,7],[2,7,6], [3,0,4],[3,4,7],
        ];
        (points, triangles)
    }

    #[test]
    fn faceted_box_has_volume_and_area() {
        let (points, triangles) = box_mesh();
        let mut brep = brep_from_triangles(&points, &triangles).unwrap();
        let mut kernel = crate::kernel::kernel();
        let store = AttributeStore::new();
        let attrs = PropertyAttributes { store: &store, key_of: &|entity| key_of(entity).key };
        let volume = brep_volume(&kernel, &mut brep, 1e-6, &attrs).unwrap();
        let area = brep_area(&kernel, &mut brep, 1e-6).unwrap();
        assert!((volume.abs() - 8.0).abs() < 1e-4, "volume {volume}");
        assert!((area - 24.0).abs() < 1e-4, "area {area}");
    }
}

/// Converts a STEP planar body (outer loops + holes) into a solidcore `Brep`.
///
/// The B-rep is faceted (triangulated planar faces) but geometrically exact; it is
/// suitable for volume/area queries and as a staging representation until exact
/// STEP face topology is preserved.
pub fn brep_from_planar_body(body: &nodecraft_step::PlanarBody) -> Result<Brep, String> {
    let triangles = body.triangulate()?;
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut remap = BTreeMap::new();
    for (index, point) in body.points.iter().enumerate() {
        remap.insert(index, points.len() as u32);
        points.push(*point);
    }
    let triangles: Vec<[u32; 3]> = triangles.iter().map(|triangle| [remap[&triangle[0]], remap[&triangle[1]], remap[&triangle[2]]]).collect();
    brep_from_triangles(&points, &triangles)
}

#[cfg(test)]
#[cfg(test)]
mod planar_tests {
    use super::*;
    use solidcore::mass::integrator::properties::{brep_area, brep_volume, PropertyAttributes};
    use solidcore::topology::Brep;

    #[test]
    fn block_step_becomes_solidcore_brep() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/Sample AP203 Data Files/block.stp");
        if !std::path::Path::new(path).exists() { return; }
        let bodies = nodecraft_step::read_planar_solids(path).unwrap();
        let mut brep = brep_from_planar_body(&bodies[0]).unwrap();
        let mut kernel = crate::kernel::kernel();
        let store = AttributeStore::new();
        let attrs = PropertyAttributes { store: &store, key_of: &|entity| key_of(entity).key };
        let volume = brep_volume(&kernel, &mut brep, 1e-6, &attrs).unwrap();
        let area = brep_area(&kernel, &mut brep, 1e-6).unwrap();
        assert!((volume.abs() - 60.0).abs() < 1e-4, "volume {volume}");
        assert!((area - 94.0).abs() < 1e-4, "area {area}");
        let mesh = crate::curved::brep_display_mesh(&kernel, &brep, &crate::curved::CurvedTessOptions::default()).unwrap();
        assert!(!mesh.faces.is_empty());
        assert!(!mesh.vertices.is_empty());
    }
}

/// Builds a solidcore Brep from a STEP planar body and tessellates it for display.
pub fn planar_body_display_mesh(body: &nodecraft_step::PlanarBody) -> Result<nodecraft_protocol::Mesh, String> {
    let brep = brep_from_planar_body_exact(body).or_else(|_| brep_from_planar_body(body))?;
    let kernel = crate::kernel::kernel();
    crate::curved::brep_display_mesh(&kernel, &brep, &crate::curved::CurvedTessOptions::default())
}


fn loop_normal(points: &[[f64; 3]], loop_points: &[usize]) -> Vec3 {
    let origin = Point3::new(points[loop_points[0]][0], points[loop_points[0]][1], points[loop_points[0]][2]);
    let mut normal = Vec3::new(0.0, 0.0, 0.0);
    for index in 1..loop_points.len() - 1 {
        let a = Point3::new(points[loop_points[index]][0], points[loop_points[index]][1], points[loop_points[index]][2]) - origin;
        let b = Point3::new(points[loop_points[index + 1]][0], points[loop_points[index + 1]][1], points[loop_points[index + 1]][2]) - origin;
        normal = normal + a.cross(b);
    }
    normal
}

struct PlaneBasis { origin: Point3, u: Vec3, v: Vec3, lu: f64, lv: f64, surface: Surface }

fn plane_basis(points: &[[f64; 3]], outer: &[usize]) -> Result<PlaneBasis, String> {
    let p = |index: usize| Point3::new(points[index][0], points[index][1], points[index][2]);
    let origin = p(outer[0]);
    let mut e1 = None;
    let mut e2 = None;
    for index in 1..outer.len() {
        let candidate = p(outer[index]) - origin;
        if e1.is_none() { e1 = Some(candidate); continue; }
        if e1.unwrap().cross(candidate).length() > 1e-12 { e2 = Some(candidate); break; }
    }
    let e1 = e1.ok_or("planar face has no second point")?;
    let e2 = e2.ok_or("planar face has no non-collinear third point")?;
    let lu = e1.length();
    let u = e1 * (1.0 / lu);
    let v = e2 - u * e2.dot(u); let v = v * (1.0 / v.length());
    let lv = e2.dot(v);
    let surface = Surface::from(solidcore::surface::plane::Plane::new(
        origin,
        u,
        v,
        solidcore::math::Vec2::new(lu, lv),
        &Extent2d::from_bounds(0.0, 0.0, 1.0, 1.0),
    ));
    Ok(PlaneBasis { origin, u, v, lu, lv, surface })
}

impl PlaneBasis {
    fn uv(&self, point: Point3) -> [f64; 2] {
        let d = point - self.origin;
        [d.dot(self.u) / self.lu, d.dot(self.v) / self.lv]
    }
}

fn uv_line(a: [f64; 2], b: [f64; 2]) -> Result<solidcore::curve::curve::Curve, String> {
    Ok(solidcore::curve::curve::Curve::from(BSplineCurve::create_line_segment(2, Point3::new(a[0], a[1], 0.0), Point3::new(b[0], b[1], 0.0)).map_err(|e| e.to_string())?))
}

/// Exact planar STEP faces with per-edge UV trim curves.
pub fn brep_from_planar_body_exact(body: &nodecraft_step::PlanarBody) -> Result<Brep, String> {
    if body.points.is_empty() || body.faces.is_empty() { return Err("planar body is empty".into()); }
    let mut data = BrepData::new(1, solidcore::topology::brep_data::DataSourceType::Unknown);
    data.healer_version = HEALER_VERSION;
    data.regions.push(RegionData { start_shell: 0, num_shells: 1, is_void: 0, ..RegionData::zeroed() });
    data.shells.push(ShellData { shell_type: 0, faceuse_start: 0, num_faceuses: body.faces.len() as u64, ..ShellData::zeroed() });
    for point in &body.points { data.vertices.push(VertexData { point: Point3::new(point[0], point[1], point[2]), zone_tol_3d: 1e-7, ..VertexData::zeroed() }); }
    let mut edges: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (face_index, outer) in body.faces.iter().enumerate() {
        let basis = plane_basis(&body.points, outer)?;
        let loop_n = {
            let mut normal = Vec3::new(0.0, 0.0, 0.0);
            let origin = basis.origin;
            for index in 1..outer.len() - 1 {
                let a = Point3::new(body.points[outer[index]][0], body.points[outer[index]][1], body.points[outer[index]][2]) - origin;
                let b = Point3::new(body.points[outer[index + 1]][0], body.points[outer[index + 1]][1], body.points[outer[index + 1]][2]) - origin;
                normal = normal + a.cross(b);
            }
            normal
        };
        let orientation = i32::from(loop_n.dot(basis.u.cross(basis.v)) >= 0.0);
        data.faceuses.push(FaceuseData { face: face_index as u64, orientation, ..FaceuseData::zeroed() });
        data.surfaces.push(Keyed { key: ObjectKey(700 + face_index as u64), object: basis.surface.clone() });
        data.faces.push(FaceData { surface: face_index as u64, start_loop: data.loops.len() as u64, num_loops: 1 + body.face_holes(face_index).len() as u64, uv_domain: Extent2d::from_bounds(0.0, 0.0, 1.0, 1.0), rectangular_trim: 0, zone_tol_3d: 1e-6, ..FaceData::zeroed() });
        for (hole_index, loop_points) in std::iter::once(outer).chain(body.face_holes(face_index).iter()).enumerate() {
            let loop_points: Vec<usize> = if hole_index > 0 && loop_normal(&body.points, loop_points).dot(basis.u.cross(basis.v)) >= 0.0 {
                loop_points.iter().rev().copied().collect()
            } else { loop_points.clone() };
            let loop_index = data.loops.len() as u64;
            data.loops.push(LoopData { loop_type: 0, start_eu: data.edgeuses.len() as u64, num_eu: loop_points.len() as u64, ..LoopData::zeroed() });
            for index in 0..loop_points.len() {
                let a = loop_points[index]; let b = loop_points[(index + 1) % loop_points.len()];
                let key = if a < b { (a, b) } else { (b, a) };
                let edge = if let Some(existing) = edges.get(&key) { *existing } else {
                    let e = edges.len();
                    data.curves_3d.push(Keyed { key: ObjectKey(900 + e as u64), object: line(data.vertices[key.0].point, data.vertices[key.1].point).unwrap() });
                    data.edges.push(EdgeData { curve: e as u64, start_vertex: key.0 as u64, end_vertex: key.1 as u64, prim_eu: data.edgeuses.len() as i64, interval: Extent1d::from_raw(0.0, 1.0), zone_tol_3d: 1e-6, ..EdgeData::zeroed() });
                    edges.insert(key, e);
                    e
                };
                let uv_a = basis.uv(data.vertices[key.0].point);
                let uv_b = basis.uv(data.vertices[key.1].point);
                let uv_index = data.uv_curves.len() as u64;
                data.uv_curves.push(Keyed { key: ObjectKey(1200 + uv_index), object: uv_line(uv_a, uv_b)? });
                data.edgeuses.push(EuData { eu_type: 0, edge: edge as u64, loop_: loop_index, orientation: i32::from(a < b), uv_curve: uv_index, ..EuData::zeroed() });
            }
        }
    }
    let mut kernel = crate::kernel::kernel();
    let mut brep = Brep::new_with_infinite_region().map_err(|e| e.to_string())?;
    let mut store = AttributeStore::new();
    let mut records = solidcore::topology::topology::NotifyRecords::new();
    let mut env = TopologyEnv { all_attributes: &[], attributes: &mut store, key_of: &key_of, records: &mut records, next_key: 1 };
    make_topology_from_data(&mut kernel, &mut brep, &mut data, &mut env, MakeTopologyOptions::default()).map_err(|e| e.to_string())?;
    Ok(brep)
}

#[cfg(test)]
mod exact_planar_tests {
    use super::*;
    use solidcore::mass::integrator::properties::{brep_area, brep_volume, PropertyAttributes};

    #[test]
    fn exact_planar_face_with_hole_has_trimmed_area() {
        let body = nodecraft_step::PlanarBody {
            points: vec![[0.,0.,0.],[4.,0.,0.],[4.,4.,0.],[0.,4.,0.],[1.,1.,0.],[3.,1.,0.],[3.,3.,0.],[1.,3.,0.]],
            faces: vec![vec![0,1,2,3]],
            holes: vec![vec![vec![4,5,6,7]]],
        };
        let mut brep = brep_from_planar_body_exact(&body).unwrap();
        let mut kernel = crate::kernel::kernel();
        let area = brep_area(&kernel, &mut brep, 1e-6).unwrap();
        assert!((area - 12.0).abs() < 1e-4, "area {area}");
    }

    #[test]
    fn exact_block_has_six_faces_and_exact_metrics() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/Sample AP203 Data Files/block.stp");
        if !std::path::Path::new(path).exists() { return; }
        let bodies = nodecraft_step::read_planar_solids(path).unwrap();
        let mut brep = brep_from_planar_body_exact(&bodies[0]).unwrap();
        let kernel = crate::kernel::kernel();
        let store = AttributeStore::new();
        let attrs = PropertyAttributes { store: &store, key_of: &|entity| key_of(entity).key };
        let volume = brep_volume(&kernel, &mut brep, 1e-6, &attrs).unwrap().abs();
        let area = brep_area(&kernel, &mut brep, 1e-6).unwrap();
        assert_eq!(brep.faces().len(), 6);
        assert!((volume - 60.0).abs() < 1e-4, "volume {volume}");
        assert!((area - 94.0).abs() < 1e-4, "area {area}");
    }
}
