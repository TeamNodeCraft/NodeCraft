//! The solidcore calls a real (B-rep) edge treatment needs, and the port units that
//! own them.
//!
//! These names are taken from `solidcore`'s `plan/closure.md` entry-point list and
//! `plan/units/blend.fillet_executive_p1.toml`, and from the signatures recorded in
//! `harness/callgraph/units.json` (upstream commit `c9c5979`, solidcore `main`). They
//! are reported verbatim in [`crate::BlendError::Unsupported`] so the error names the
//! work that is actually missing.

/// SM_API entry points for circular fillets (constant radius, per-edge radii).
pub const FILLET_API: &[&str] = &[
    "SmApiFilletEdges(SmBrep*, SmTArray<SmEdge*>*, double radius, long, long, double)",
    "SmApiFilletEdgesPerEdge(SmBrep*, SmTArray<SmEdge*>*, SmTArray<double>*, long, double)",
    "SmApiCircularFillet",
];

/// SM_API entry points for chamfers (equal, two-distance, distance-angle, bevels).
pub const CHAMFER_API: &[&str] = &[
    "SmApiChamferFillet(SmBrep*, SmTArray<SmEdge*>*, double, long, long, double)",
    "SmApiFilletEdges(SmBrep*, SmTArray<SmEdge*>*, double, long, long, double)",
    "SmApiSetBevelCorners(SmBrep*, SmTArray<SmVertex*>*, long)",
];

/// Executive pieces behind every fillet/chamfer call.
pub const EXECUTIVE: &[&str] = &[
    "SmFilletExecutive::SetFilletParameters",
    "SmFilletExecutive::CreateFilletCorners",
    "SmFilletExecutive::DoFilleting",
    "SmFilletExecutive::DoPiecewiseMerge",
    "BlendExecutiveAccess adapter (SmBrep <-> FilletExecutive)",
];

/// Variable-radius / G2 / G3 fillet machinery.
pub const LAW_CONTINUITY: &[&str] = &[
    "SmBSplineFilletLaw constructor",
    "SmFilletLaw::Evaluate on the executive's law path",
    "blend.fillet_standard_solver_p2 (G2/G3 law setup)",
];

/// The solidcore port units that own the symbols above and are still partial or
/// unported at the locked revision (`Cargo.lock` `6ecdaaa`).
pub const PORT_UNITS: &[&str] = &[
    "blend.fillet_executive_p1",
    "blend.fillet_executive_p2",
    "blend.fillet_corner_p1",
    "blend.fillet_corner_p2",
    "blend.fillet_standard_solver_p2",
    "blend.fillet_geom_p3",
    "blend.fillet_geom_p5",
    "blend.fillet_geom_p6",
];
