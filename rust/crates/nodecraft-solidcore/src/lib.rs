//! SolidCore STEP readers, planar/analytic B-rep adapters and the kernel profile.

pub mod mesh_brep;
pub mod step_analytic;
pub mod step_nurbs;
pub mod step_mixed;
pub mod step_nurbs_faces;
pub mod step_curved;
pub mod step_import;
pub use step_curved::analytic_surface_from_step;
pub use step_mixed::{brep_from_mixed_step, brep_from_mixed_step_file};
pub use step_import::{StepBrepImport, StepBrepKind, brep_from_step};
pub use step_analytic::brep_from_analytic_step;
pub use step_nurbs::{bspline_curve_to_solidcore, bspline_surface_to_solidcore};
pub use step_nurbs_faces::{
    UvRectangle, brep_from_bspline_sheet, brep_from_bspline_sheet_file,
    brep_from_bspline_uv_rectangle,
};
pub use mesh_brep::{brep_from_planar_body, brep_from_planar_body_exact, brep_from_triangles};
pub mod error;
pub mod kernel;
pub mod planar;
pub use solidcore;

use solidcore::kernel::Kernel;
