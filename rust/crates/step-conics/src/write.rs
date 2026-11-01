//! Writing conic STEP entities: per-entity strings and a minimal ISO-10303-21 writer.

use crate::{Axis2Placement3d, Conic, ConicError, Trim};

/// A minimal ISO-10303-21 writer for conic entities.
///
/// Entities are numbered from `#1` upwards. Every emitted line round-trips through
/// [`crate::ConicScene::parse`]: placements keep `$` for omitted axis/reference
/// directions, and the writer never emits an exponent-free value it cannot represent
/// (non-finite values are rejected before a line is added).
#[derive(Debug, Clone, Default)]
pub struct ConicWriter {
    next: u32,
    entities: Vec<String>,
}

impl ConicWriter {
    /// An empty writer whose first entity will be `#1`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The id the next pushed entity will get.
    #[must_use]
    pub const fn next_id(&self) -> u32 {
        self.next + 1
    }

    /// The emitted entity lines, without the file header/trailer.
    #[must_use]
    pub fn entities(&self) -> &[String] {
        &self.entities
    }

    /// Appends a raw entity body as `#id=body;` and returns its id.
    ///
    /// The caller is responsible for the body being a single STEP entity; this is the
    /// hook for embedding conic entities in a larger file.
    pub fn push_entity(&mut self, body: impl AsRef<str>) -> u32 {
        self.next += 1;
        let id = self.next;
        self.entities.push(format!("#{id}={};", body.as_ref()));
        id
    }

    /// Writes a `CARTESIAN_POINT`, the optional `DIRECTION`s and the
    /// `AXIS2_PLACEMENT_3D`; returns the placement id.
    ///
    /// # Errors
    ///
    /// [`ConicError::InvalidGeometry`] when the placement does not resolve (non-finite
    /// components, zero axis/reference direction, reference direction parallel to the
    /// axis).
    pub fn push_placement(
        &mut self,
        placement: &Axis2Placement3d,
    ) -> Result<u32, ConicError> {
        placement
            .resolve()
            .map_err(|error| ConicError::InvalidGeometry {
                entity: "AXIS2_PLACEMENT_3D",
                why: error.to_string(),
            })?;
        let [x, y, z] = placement.location;
        let point = self.push_entity(format!(
            "CARTESIAN_POINT('',({},{},{}))",
            real(x),
            real(y),
            real(z)
        ));
        let axis = placement.axis.map(|axis| {
            let [x, y, z] = axis;
            self.push_entity(format!("DIRECTION('',({},{},{}))", real(x), real(y), real(z)))
        });
        let reference = placement.ref_direction.map(|reference| {
            let [x, y, z] = reference;
            self.push_entity(format!("DIRECTION('',({},{},{}))", real(x), real(y), real(z)))
        });
        let axis = reference_arg(axis);
        let reference = reference_arg(reference);
        Ok(self.push_entity(format!(
            "AXIS2_PLACEMENT_3D('',#{point},{axis},{reference})"
        )))
    }

    /// Writes a placement and then the conic; returns the conic entity id.
    ///
    /// # Errors
    ///
    /// Those of [`Self::push_placement`] and [`Conic::step_entity`].
    pub fn push_conic(&mut self, conic: &Conic) -> Result<u32, ConicError> {
        // Validate before mutating the writer, so a failed push adds nothing.
        conic.step_entity(0)?;
        let placement = self.push_placement(conic.placement())?;
        let body = conic.step_entity(placement)?;
        Ok(self.push_entity(body))
    }

    /// Writes a placement, the basis conic and a parameter `TRIMMED_CURVE`; returns the
    /// `TRIMMED_CURVE` id.
    ///
    /// # Errors
    ///
    /// Those of [`Self::push_conic`] and [`Trim::step_trimmed_curve`].
    pub fn push_trimmed_conic(
        &mut self,
        conic: &Conic,
        trim: &Trim,
    ) -> Result<u32, ConicError> {
        // Validate before mutating the writer, so a failed push adds nothing.
        trim.step_trimmed_curve(0)?;
        let basis = self.push_conic(conic)?;
        let body = trim.step_trimmed_curve(basis)?;
        Ok(self.push_entity(body))
    }

    /// The complete ISO-10303-21 text: header, the emitted entity lines, trailer.
    #[must_use]
    pub fn finish(&self) -> String {
        let mut out = String::from(
            "ISO-10303-21;\n\
             HEADER;\n\
             FILE_DESCRIPTION((''),'2;1');\n\
             FILE_NAME('','',(''),(''),'step-conics','step-conics','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n\
             ENDSEC;\n\
             DATA;\n",
        );
        for line in &self.entities {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        out
    }
}

/// Writes `conics` as a complete ISO-10303-21 file whose conic entities round-trip
/// through [`crate::ConicScene::parse`].
///
/// # Errors
///
/// [`ConicError::InvalidGeometry`] for a conic that cannot be represented as STEP (a
/// non-positive or non-finite axis/parameter value).
pub fn write_conics(conics: &[Conic]) -> Result<String, ConicError> {
    let mut writer = ConicWriter::new();
    for conic in conics {
        writer.push_conic(conic)?;
    }
    Ok(writer.finish())
}

fn reference_arg(id: Option<u32>) -> String {
    id.map_or_else(|| "$".to_owned(), |id| format!("#{id}"))
}

/// A STEP real: `10.` for whole numbers, shortest round-tripping decimal otherwise,
/// with `E` exponents (never `e`) as ISO-10303-21 permits.
pub(crate) fn real(value: f64) -> String {
    if value == 0.0 {
        return "0.".to_owned();
    }
    let text = format!("{value:?}");
    match text.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => {
            let mantissa = if mantissa.contains('.') {
                mantissa.to_owned()
            } else {
                format!("{mantissa}.")
            };
            format!("{mantissa}E{exponent}")
        }
        None => {
            if text.contains('.') {
                text
            } else {
                format!("{text}.")
            }
        }
    }
}
