//! STEP swept and offset surface parsing.

use nodecraft_step::{StepFile, Value};

fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    (len > 1e-12).then(|| [v[0]/len, v[1]/len, v[2]/len])
}
fn point(file: &StepFile, id: u32) -> Option<[f64; 3]> {
    let entity = file.entities.get(&id)?;
    let Value::List(values) = entity.args.get(1)? else { return None };
    let mut out = [0.0; 3];
    for (slot, value) in out.iter_mut().zip(values) { let Value::Number(number) = value else { return None }; *slot = *number; }
    Some(out)
}
fn placement_frame(file: &StepFile, id: u32) -> Result<([f64; 3], [f64; 3], [f64; 3]), String> {
    let entity = file.entities.get(&id).ok_or_else(|| format!("#{id}: missing placement"))?;
    let location = match entity.args.get(1) { Some(Value::Ref(location)) => point(file, *location).ok_or("bad placement location")?, _ => return Err("placement missing location".into()) };
    let z = match entity.args.get(2) {
        Some(Value::Ref(axis)) => { let entity = file.entities.get(axis).ok_or("bad placement axis")?; let Value::List(values) = entity.args.get(1).ok_or("bad axis direction")? else { return Err("bad axis direction".into()) }; let mut d=[0.0;3]; for (slot,value) in d.iter_mut().zip(values){let Value::Number(n)=value else{return Err("bad axis".into())};*slot=*n;} unit(d).ok_or("zero axis")? }
        _ => [0.0,0.0,1.0],
    };
    let x = match entity.args.get(3) {
        Some(Value::Ref(reference)) => { let entity = file.entities.get(reference).ok_or("bad reference")?; let Value::List(values) = entity.args.get(1).ok_or("bad reference direction")? else { return Err("bad reference direction".into()) }; let mut d=[0.0;3]; for (slot,value) in d.iter_mut().zip(values){let Value::Number(n)=value else{return Err("bad reference".into())};*slot=*n;} unit(d).ok_or("zero reference")? }
        _ => [1.0,0.0,0.0],
    };
    let dot = x[0]*z[0]+x[1]*z[1]+x[2]*z[2];
    let x = unit([x[0]-z[0]*dot, x[1]-z[1]*dot, x[2]-z[2]*dot]).ok_or("degenerate reference")?;
    Ok((location, z, x))
}


#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceOfRevolution { pub id: u32, pub swept_curve: u32, pub axis_position: u32 }

impl SurfaceOfRevolution {
    /// Evaluates the surface at curve parameter `u` and rotation angle `v`.
    /// Samples a `(u_segments+1) × (v_segments+1)` grid.
    pub fn sample(&self, file: &StepFile, u_segments: usize, v_segments: usize, v_start: f64, v_end: f64) -> Result<Vec<Vec<[f64; 3]>>, String> {
        if u_segments == 0 || v_segments == 0 { return Err("sample counts must be positive".into()); }
        let curve = crate::parse_curve(file, self.swept_curve)?;
        let (u0, u1) = curve.domain();
        let mut rows = Vec::with_capacity(u_segments + 1);
        for i in 0..=u_segments {
            let u = u0 + (u1-u0)*i as f64/u_segments as f64;
            let mut row = Vec::with_capacity(v_segments + 1);
            for j in 0..=v_segments { row.push(self.point_at(file, u, v_start + (v_end-v_start)*j as f64/v_segments as f64)?); }
            rows.push(row);
        }
        Ok(rows)
    }

    pub fn point_at(&self, file: &StepFile, u: f64, v: f64) -> Result<[f64; 3], String> {
        let curve = crate::parse_curve(file, self.swept_curve)?;
        let p = curve.point_at(u)?;
        let (origin, axis, _) = placement_frame(file, self.axis_position)?;
        let local = [p[0]-origin[0], p[1]-origin[1], p[2]-origin[2]];
        let cos = v.cos(); let sin = v.sin();
        let dot = local[0]*axis[0]+local[1]*axis[1]+local[2]*axis[2];
        let cross = [axis[1]*local[2]-axis[2]*local[1], axis[2]*local[0]-axis[0]*local[2], axis[0]*local[1]-axis[1]*local[0]];
        Ok([
            origin[0] + local[0]*cos + cross[0]*sin + axis[0]*dot*(1.0-cos),
            origin[1] + local[1]*cos + cross[1]*sin + axis[1]*dot*(1.0-cos),
            origin[2] + local[2]*cos + cross[2]*sin + axis[2]*dot*(1.0-cos),
        ])
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceOfLinearExtrusion { pub id: u32, pub swept_curve: u32, pub extrusion_axis: u32 }

impl SurfaceOfLinearExtrusion {
    /// Evaluates the surface at curve parameter `u` and extrusion distance `v`.
    /// Samples a `(u_segments+1) × (v_segments+1)` grid.
    pub fn sample(&self, file: &StepFile, u_segments: usize, v_segments: usize, v_start: f64, v_end: f64) -> Result<Vec<Vec<[f64; 3]>>, String> {
        if u_segments == 0 || v_segments == 0 { return Err("sample counts must be positive".into()); }
        let curve = crate::parse_curve(file, self.swept_curve)?;
        let (u0, u1) = curve.domain();
        let mut rows = Vec::with_capacity(u_segments + 1);
        for i in 0..=u_segments {
            let u = u0 + (u1-u0)*i as f64/u_segments as f64;
            let mut row = Vec::with_capacity(v_segments + 1);
            for j in 0..=v_segments { row.push(self.point_at(file, u, v_start + (v_end-v_start)*j as f64/v_segments as f64)?); }
            rows.push(row);
        }
        Ok(rows)
    }

    pub fn point_at(&self, file: &StepFile, u: f64, v: f64) -> Result<[f64; 3], String> {
        let curve = crate::parse_curve(file, self.swept_curve)?;
        let p = curve.point_at(u)?;
        let axis = file.entities.get(&self.extrusion_axis).ok_or("missing extrusion axis")?;
        let Value::List(values) = axis.args.get(1).ok_or("bad extrusion axis")? else { return Err("bad extrusion axis".into()) };
        let mut direction = [0.0; 3];
        for (slot, value) in direction.iter_mut().zip(values) { let Value::Number(number) = value else { return Err("bad extrusion direction".into()) }; *slot = *number; }
        let direction = unit(direction).ok_or("zero extrusion direction")?;
        Ok([p[0]+direction[0]*v, p[1]+direction[1]*v, p[2]+direction[2]*v])
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct OffsetSurface { pub id: u32, pub basis_surface: u32, pub distance: f64, pub self_intersect: bool }

impl OffsetSurface {
    /// Evaluates the offset at `(u, v)` using a finite-difference surface normal.
    /// Samples a `(u_segments+1) × (v_segments+1)` grid over the basis domain.
    pub fn sample(&self, file: &StepFile, u_segments: usize, v_segments: usize) -> Result<Vec<Vec<[f64; 3]>>, String> {
        if u_segments == 0 || v_segments == 0 { return Err("sample counts must be positive".into()); }
        let basis = crate::parse_surface(file, self.basis_surface)?;
        let (u0,u1) = basis.u_domain(); let (v0,v1) = basis.v_domain();
        let mut rows = Vec::with_capacity(u_segments+1);
        for i in 0..=u_segments {
            let u = u0 + (u1-u0)*i as f64/u_segments as f64;
            let mut row = Vec::with_capacity(v_segments+1);
            for j in 0..=v_segments { row.push(self.point_at(file, u, v0 + (v1-v0)*j as f64/v_segments as f64)?); }
            rows.push(row);
        }
        Ok(rows)
    }

    pub fn point_at(&self, file: &StepFile, u: f64, v: f64) -> Result<[f64; 3], String> {
        let surface = crate::parse_surface(file, self.basis_surface)?;
        let point = surface.point_at(u, v)?;
        let epsilon = 1e-5;
        let du = surface.point_at(u + epsilon, v)?;
        let dv = surface.point_at(u, v + epsilon)?;
        let tangent_u = [du[0]-point[0], du[1]-point[1], du[2]-point[2]];
        let tangent_v = [dv[0]-point[0], dv[1]-point[1], dv[2]-point[2]];
        let normal = [tangent_u[1]*tangent_v[2]-tangent_u[2]*tangent_v[1], tangent_u[2]*tangent_v[0]-tangent_u[0]*tangent_v[2], tangent_u[0]*tangent_v[1]-tangent_u[1]*tangent_v[0]];
        let length = (normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]).sqrt();
        if length <= 1e-12 { return Err("offset surface has a degenerate normal".into()); }
        Ok([point[0] + normal[0]/length*self.distance, point[1] + normal[1]/length*self.distance, point[2] + normal[2]/length*self.distance])
    }
}

fn entity<'a>(file: &'a StepFile, id: u32, kind: &str) -> Result<&'a nodecraft_step::Entity, String> {
    let entity = file.entities.get(&id).ok_or_else(|| format!("#{id}: no such entity"))?;
    if entity.name != kind { return Err(format!("#{id}: expected {kind}, got {}", entity.name)); }
    Ok(entity)
}
fn reference(entity: &nodecraft_step::Entity, index: usize, what: &str) -> Result<u32, String> {
    match entity.args.get(index) { Some(Value::Ref(id)) => Ok(*id), _ => Err(format!("{what}: missing reference")) }
}

pub fn parse_surface_of_revolution(file: &StepFile, id: u32) -> Result<SurfaceOfRevolution, String> {
    let entity = entity(file, id, "SURFACE_OF_REVOLUTION")?;
    Ok(SurfaceOfRevolution { id, swept_curve: reference(entity, 1, "swept curve")?, axis_position: reference(entity, 2, "axis position")? })
}
pub fn parse_surface_of_linear_extrusion(file: &StepFile, id: u32) -> Result<SurfaceOfLinearExtrusion, String> {
    let entity = entity(file, id, "SURFACE_OF_LINEAR_EXTRUSION")?;
    Ok(SurfaceOfLinearExtrusion { id, swept_curve: reference(entity, 1, "swept curve")?, extrusion_axis: reference(entity, 2, "extrusion axis")? })
}
pub fn parse_offset_surface(file: &StepFile, id: u32) -> Result<OffsetSurface, String> {
    let entity = entity(file, id, "OFFSET_SURFACE")?;
    let basis_surface = reference(entity, 1, "basis surface")?;
    let distance = match entity.args.get(2) { Some(Value::Number(value)) => *value, _ => return Err("offset distance must be a number".into()) };
    let self_intersect = matches!(entity.args.get(3), Some(Value::Enum(flag)) if flag == "T");
    Ok(OffsetSurface { id, basis_surface, distance, self_intersect })
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEXT: &str = "DATA;\n#1=LINE('',#2,#3);\n#2=CARTESIAN_POINT('',(0.,0.,0.));\n#3=VECTOR('',#4,1.);\n#4=DIRECTION('',(1.,0.,0.));\n#5=AXIS2_PLACEMENT_3D('',#2,#4,#4);\n#6=SURFACE_OF_REVOLUTION('',#1,#5);\n#7=SURFACE_OF_LINEAR_EXTRUSION('',#1,#4);\n#8=OFFSET_SURFACE('',#6,2.5,.F.);\nENDSEC;";
    #[test]
    fn parses_swept_surfaces() {
        let file = StepFile::parse(TEXT).unwrap();
        assert_eq!(parse_surface_of_revolution(&file, 6).unwrap().swept_curve, 1);
        assert_eq!(parse_surface_of_linear_extrusion(&file, 7).unwrap().extrusion_axis, 4);
    }
    #[test]
    fn parses_offset_surface() {
        let file = StepFile::parse(TEXT).unwrap();
        let surface = parse_offset_surface(&file, 8).unwrap();
        assert_eq!(surface.basis_surface, 6);
        assert_eq!(surface.distance, 2.5);
        assert!(!surface.self_intersect);
    }
    #[test]
    fn evaluates_offset_surface() {
        let file = StepFile::parse("DATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(0.,1.,0.));\n#3=CARTESIAN_POINT('',(1.,0.,0.));\n#4=CARTESIAN_POINT('',(1.,1.,0.));\n#10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,\n(2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);\n#11=OFFSET_SURFACE('',#10,2.0,.F.);\nENDSEC;").unwrap();
        let surface = parse_offset_surface(&file, 11).unwrap();
        let point = surface.point_at(&file, 0.5, 0.5).unwrap();
        assert!((point[2]-2.0).abs() < 1e-3, "{point:?}");
    }

    #[test]
    fn evaluates_linear_extrusion() {
        let file = StepFile::parse("DATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(2.,0.,0.));\n#3=DIRECTION('',(0.,0.,1.));\n#10=B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#2),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);\n#11=SURFACE_OF_LINEAR_EXTRUSION('',#10,#3);\nENDSEC;").unwrap();
        let surface = parse_surface_of_linear_extrusion(&file, 11).unwrap();
        let point = surface.point_at(&file, 0.5, 3.0).unwrap();
        assert!((point[0]-1.0).abs() < 1e-9 && (point[2]-3.0).abs() < 1e-9, "{point:?}");
    }

    #[test]
    fn evaluates_surface_of_revolution() {
        let file = StepFile::parse("DATA;\n#0=CARTESIAN_POINT('',(0.,0.,0.));\n#1=CARTESIAN_POINT('',(1.,0.,0.));\n#2=CARTESIAN_POINT('',(2.,0.,0.));\n#3=DIRECTION('',(0.,0.,1.));\n#4=DIRECTION('',(1.,0.,0.));\n#5=AXIS2_PLACEMENT_3D('',#0,#3,#4);\n#10=B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#2),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);\n#11=SURFACE_OF_REVOLUTION('',#10,#5);\nENDSEC;").unwrap();
        let surface = parse_surface_of_revolution(&file, 11).unwrap();
        let point = surface.point_at(&file, 0.0, std::f64::consts::FRAC_PI_2).unwrap();
        assert!((point[0]).abs() < 1e-9 && (point[1]-1.0).abs() < 1e-9, "{point:?}");
    }

    #[test]
    fn rejects_wrong_kind() {
        let file = StepFile::parse(TEXT).unwrap();
        assert!(parse_offset_surface(&file, 6).is_err());
    }
}
