//! Exact before/after feature comparison.
//!
//! This is intentionally independent of any node type: callers supply the
//! evaluated before and after Parts, and receive the two material-difference
//! meshes. Feature nodes opt in by supplying those Parts; the comparison layer
//! never inspects node values.

use nodecraft_protocol::Mesh;

use crate::Part;

/// Material removed (before minus after) and material added (after minus before).
pub struct DifferenceMeshes {
    pub removed: Mesh,
    pub added: Mesh,
}

/// Compute exact difference meshes for one feature.
///
/// Parts are flattened to their realized solids. A multi-solid Part is unioned
/// into one operand first, so the result is independent of input body count.
pub fn compare_parts(before: &Part, after: &Part) -> Result<DifferenceMeshes, String> {
    let before = union_solids(before.realized_solids()?)?;
    let after = union_solids(after.realized_solids()?)?;
    let kernel = nodecraft_solidcore::solidcore::kernel::Kernel::default();
    let options = nodecraft_solidcore::curved::CurvedTessOptions::default();
    // Identical (or rigid-equivalent) bodies have no material difference. Boolean
    // difference of coincident solids is both pointless and numerically hostile.
    let before_mesh = nodecraft_solidcore::curved::brep_display_mesh(&kernel, &before, &options)
        .map_err(|error| error.to_string())?;
    let after_mesh = nodecraft_solidcore::curved::brep_display_mesh(&kernel, &after, &options)
        .map_err(|error| error.to_string())?;
    let before_volume = mesh_volume(&before_mesh);
    let after_volume = mesh_volume(&after_mesh);
    if (before_volume - after_volume).abs() <= before_volume.abs().max(1.0) * 1.0e-6 {
        return Ok(DifferenceMeshes { removed: Mesh::default(), added: Mesh::default() });
    }
    let removed = nodecraft_bool::boolean_breps(nodecraft_bool::BooleanOp::Difference, before.clone(), after.clone())?;
    let added = nodecraft_bool::boolean_breps(nodecraft_bool::BooleanOp::Difference, after, before)?;
    Ok(DifferenceMeshes {
        // One side of a pure cut/extrude difference is legitimately empty. An
        // empty BRep is a valid zero-triangle result, not a comparison failure.
        removed: nodecraft_solidcore::curved::brep_display_mesh(&kernel, &removed, &options)
            .unwrap_or_default(),
        added: nodecraft_solidcore::curved::brep_display_mesh(&kernel, &added, &options)
            .unwrap_or_default(),
    })
}

pub(crate) fn mesh_volume(mesh: &Mesh) -> f64 {
    mesh.faces.iter().map(|triangle| {
        let a = mesh.vertices[triangle[0] as usize];
        let b = mesh.vertices[triangle[1] as usize];
        let c = mesh.vertices[triangle[2] as usize];
        let cross = [
            f64::from(b[1]) * f64::from(c[2]) - f64::from(b[2]) * f64::from(c[1]),
            f64::from(b[2]) * f64::from(c[0]) - f64::from(b[0]) * f64::from(c[2]),
            f64::from(b[0]) * f64::from(c[1]) - f64::from(b[1]) * f64::from(c[0]),
        ];
        (f64::from(a[0]) * cross[0] + f64::from(a[1]) * cross[1] + f64::from(a[2]) * cross[2]) / 6.0
    }).sum()
}

fn union_solids(mut solids: Vec<nodecraft_solidcore::solidcore::topology::Brep>) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let mut result = solids.drain(..).next().ok_or("comparison Part has no solids")?;
    for solid in solids {
        result = nodecraft_bool::boolean_breps(nodecraft_bool::BooleanOp::Union, result, solid)?;
    }
    Ok(result)
}
