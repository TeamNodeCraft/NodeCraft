//! NodeCraft's STEP/CAD adapter.
//!
//! There is one route in and one route out:
//!
//! * [`cadir::read_cad`] decodes a CAD file with cadmpeg into CADIR and reads every body
//!   region through `solidcore-cadir` into the schema-4 record the display mesher uses.
//! * [`cadir_export`] maps schema-4 records (or a display mesh for the faceted format) into
//!   CADIR and lets cadmpeg's STEP encoder write the file.
//!
//! No module in this crate writes or parses STEP Part 21 text, entity ids, headers, units
//! or styles by hand. Names, colours, layers, occurrences and placements are CADIR data.

pub mod cadir;
pub mod cadir_export;

pub use cadir::{
    CAD_EXTS, CadBody, CadFailure, CadImport, CadSolidTypeReport, CadTypeReport, MAX_INPUT_BYTES,
    application_protocol, is_cad_path, read_cad, read_schema,
};
pub use cadir_export::{
    ExportBody, StepExportOptions, bodies_from_breps, record_bounds, record_of_brep,
    record_volume, write_brep_exact, write_breps_exact, write_breps_exact_with_options, write_step,
    write_step_faceted, write_step_with_options,
};
