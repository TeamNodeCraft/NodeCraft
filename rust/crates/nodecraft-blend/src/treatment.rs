//! The NodeCraft Edge-node parameter model (`chamfer` node in the graph).
//!
//! The app's Edge node stores these values (see `TYPES.chamfer` and the
//! inspector additions):
//!
//! | JSON value | Meaning | Default |
//! |---|---|---|
//! | `mode` | `"chamfer"` or `"fillet"` | `"chamfer"` |
//! | `chamferType` | `"equal"`, `"two-distance"`, `"distance-angle"` (chamfer only) | `"equal"` |
//! | `continuity` | `""` (circular/G1), `"G2"`, `"G3"` (fillet only) | `""` |
//! | `length` | chamfer distance 1, or fillet radius | `2` |
//! | `distance2` | second chamfer distance (two-distance) | `length` |
//! | `angle` | chamfer angle in degrees (distance-angle) | `45` |
//! | `cornerTurn` | `"mitre"` or `"swept"` for the corners between circular fillets | `"mitre"` |
//!
//! [`EdgeTreatment::from_node_values`] parses that object strictly (unknown enum
//! strings and non-numeric numbers are [`BlendError::InvalidInput`]) and
//! [`EdgeTreatment::validate`] rejects the combinations solidcore cannot do today with
//! [`BlendError::Unsupported`] naming the missing calls.

use serde_json::Value;

use crate::error::BlendError;
use crate::missing::{CHAMFER_API, LAW_CONTINUITY};

/// Edge node operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMode {
    /// Break the edge with a plane (chamfer).
    Chamfer,
    /// Round the edge with a constant-radius arc (circular fillet, G1).
    Fillet,
}

impl EdgeMode {
    /// Parse the node value (`"chamfer"` / `"fillet"`).
    pub fn parse(value: &str) -> Result<Self, BlendError> {
        match value {
            "chamfer" => Ok(Self::Chamfer),
            "fillet" => Ok(Self::Fillet),
            other => Err(BlendError::InvalidInput(format!(
                "unknown edge mode {other:?}; expected \"chamfer\" or \"fillet\""
            ))),
        }
    }

    /// The node value of this mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chamfer => "chamfer",
            Self::Fillet => "fillet",
        }
    }
}

/// Chamfer width specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChamferType {
    /// Equal distance on both adjacent faces.
    Equal,
    /// Independent distances on the two faces.
    TwoDistance,
    /// Distance on the first face plus the angle of the chamfer plane (degrees).
    DistanceAngle,
}

impl ChamferType {
    /// Parse the node value (`"equal"`, `"two-distance"`, `"distance-angle"`).
    pub fn parse(value: &str) -> Result<Self, BlendError> {
        match value {
            "equal" => Ok(Self::Equal),
            "two-distance" => Ok(Self::TwoDistance),
            "distance-angle" => Ok(Self::DistanceAngle),
            other => Err(BlendError::InvalidInput(format!(
                "unknown chamfer type {other:?}; expected \"equal\", \"two-distance\" or \
                 \"distance-angle\""
            ))),
        }
    }

    /// The node value of this chamfer type.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::TwoDistance => "two-distance",
            Self::DistanceAngle => "distance-angle",
        }
    }
}

/// Fillet continuity of the edge node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Continuity {
    /// Circular (G1, tangent-continuous) fillet.
    Circular,
    /// G2 curvature-continuous fillet, using solidcore's blend-curve cross-section.
    G2,
    /// G3 curvature-continuous fillet, using solidcore's blend-curve cross-section.
    G3,
}

impl Continuity {
    /// Parse the node value (`""`, `"G2"`, `"G3"`).
    pub fn parse(value: &str) -> Result<Self, BlendError> {
        match value {
            "" | "G1" | "circular" | "Circular" => Ok(Self::Circular),
            "G2" => Ok(Self::G2),
            "G3" => Ok(Self::G3),
            other => Err(BlendError::InvalidInput(format!(
                "unknown fillet continuity {other:?}; expected \"\", \"G2\" or \"G3\""
            ))),
        }
    }

    /// The node value of this continuity.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Circular => "",
            Self::G2 => "G2",
            Self::G3 => "G3",
        }
    }
}

/// How two circular fillets turn where they meet at a sharp side edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CornerTurn {
    /// The rounds cross and meet in a mitre crease.
    Mitre,
    /// Each round ends on its section and the corner is the section revolved about
    /// the side edge.
    Swept,
}

impl CornerTurn {
    /// Parse the node value (`"mitre"` or `"swept"`).
    pub fn parse(value: &str) -> Result<Self, BlendError> {
        match value {
            "mitre" => Ok(Self::Mitre),
            "swept" => Ok(Self::Swept),
            other => Err(BlendError::InvalidInput(format!(
                "unknown fillet corner turn {other:?}; expected \"mitre\" or \"swept\""
            ))),
        }
    }

    /// The node value of this corner turn.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mitre => "mitre",
            Self::Swept => "swept",
        }
    }
}

/// One Edge-node treatment: the parsed, validated parameters of a chamfer or fillet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeTreatment {
    /// Chamfer or fillet.
    pub mode: EdgeMode,
    /// Chamfer width type (ignored for fillets).
    pub chamfer_type: ChamferType,
    /// Fillet continuity (ignored for chamfers, except that G2/G3 are rejected there).
    pub continuity: Continuity,
    /// Chamfer distance 1 or fillet radius (node `length`), in model units.
    pub length: f64,
    /// Second chamfer distance (node `distance2`), defaulting to [`Self::length`].
    pub distance2: Option<f64>,
    /// Chamfer angle in degrees (node `angle`), defaulting to `45`.
    pub angle: Option<f64>,
    /// How circular fillets meet at a sharp side edge (node `cornerTurn`).
    pub corner_turn: CornerTurn,
}

impl Default for EdgeTreatment {
    fn default() -> Self {
        Self {
            mode: EdgeMode::Chamfer,
            chamfer_type: ChamferType::Equal,
            continuity: Continuity::Circular,
            length: 2.0,
            distance2: None,
            angle: None,
            corner_turn: CornerTurn::Mitre,
        }
    }
}

impl EdgeTreatment {
    /// Parse a NodeCraft Edge-node `values` object.
    ///
    /// Missing fields use the app defaults; malformed fields are errors (never a
    /// silent fallback), because a wrong unit would place a wrong cut.
    pub fn from_node_values(values: &Value) -> Result<Self, BlendError> {
        if !values.is_object() {
            return Err(BlendError::InvalidInput(
                "edge node values must be a JSON object".into(),
            ));
        }
        let mode = match string_field(values, "mode")? {
            Some(text) => EdgeMode::parse(&text)?,
            None => EdgeMode::Chamfer,
        };
        let chamfer_type = match string_field(values, "chamferType")? {
            Some(text) => ChamferType::parse(&text)?,
            None => ChamferType::Equal,
        };
        let continuity = match string_field(values, "continuity")? {
            Some(text) => Continuity::parse(&text)?,
            None => Continuity::Circular,
        };
        let corner_turn = match string_field(values, "cornerTurn")? {
            Some(text) => CornerTurn::parse(&text)?,
            None => CornerTurn::Mitre,
        };
        let length = number_field(values, "length")?.unwrap_or(2.0);
        let distance2 = number_field(values, "distance2")?;
        let angle = number_field(values, "angle")?;
        Ok(Self {
            mode,
            chamfer_type,
            continuity,
            length,
            distance2,
            angle,
            corner_turn,
        })
    }

    /// Reject parameter combinations this crate cannot produce.
    ///
    /// The returned [`BlendError::Unsupported`] names the solidcore API calls / port
    /// units that are missing, so the node error is actionable.
    pub fn validate(&self) -> Result<(), BlendError> {
        if !self.length.is_finite() || self.length <= 0.0 {
            return Err(BlendError::InvalidInput(format!(
                "edge treatment length/radius must be a positive finite number, got {}",
                self.length
            )));
        }
        match self.mode {
            EdgeMode::Chamfer => {
                if self.continuity != Continuity::Circular {
                    return Err(BlendError::missing_api(
                        "chamfer continuity",
                        "a chamfer has no G2/G3 continuity; set continuity to \"\" or switch \
                         mode to fillet",
                        LAW_CONTINUITY.to_vec(),
                    ));
                }
                match self.chamfer_type {
                    ChamferType::Equal => Ok(()),
                    ChamferType::TwoDistance => {
                        let distance2 = self.distance2.unwrap_or(self.length);
                        if !distance2.is_finite() || distance2 <= 0.0 {
                            return Err(BlendError::InvalidInput(format!(
                                "second chamfer distance must be a positive finite number, \
                                 got {distance2}"
                            )));
                        }
                        Ok(())
                    }
                    ChamferType::DistanceAngle => {
                        let angle = self.angle.unwrap_or(45.0);
                        if !angle.is_finite() || angle <= 0.0 || angle >= 90.0 {
                            return Err(BlendError::missing_api(
                                "chainfer distance-angle",
                                format!(
                                    "convex chamfer distance-angle needs 0 < angle < 90 degrees, \
                                     got {angle}"
                                ),
                                CHAMFER_API.to_vec(),
                            ));
                        }
                        Ok(())
                    }
                }
            }
            EdgeMode::Fillet => {
                if self.corner_turn == CornerTurn::Swept && self.continuity != Continuity::Circular
                {
                    return Err(BlendError::missing_api(
                        "swept fillet corner",
                        "a swept fillet corner is only available for circular (G1) fillets",
                        LAW_CONTINUITY.to_vec(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// The two chamfer distances `(on face A, on face B)` after validation.
    ///
    /// The distance-angle construction is the standard convex one: with perpendicular
    /// adjacent faces the second distance is `length * tan(angle)`.
    pub fn chamfer_distances(&self) -> Result<(f64, f64), BlendError> {
        self.validate()?;
        if self.mode != EdgeMode::Chamfer {
            return Err(BlendError::InvalidInput(
                "chamfer distances requested for a fillet treatment".into(),
            ));
        }
        match self.chamfer_type {
            ChamferType::Equal => Ok((self.length, self.length)),
            ChamferType::TwoDistance => Ok((self.length, self.distance2.unwrap_or(self.length))),
            ChamferType::DistanceAngle => {
                let angle = self.angle.unwrap_or(45.0).to_radians();
                Ok((self.length, self.length * angle.tan()))
            }
        }
    }
}

/// An optional string field; a present non-string is an error.
fn string_field(values: &Value, key: &str) -> Result<Option<String>, BlendError> {
    match values.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(other) => Err(BlendError::InvalidInput(format!(
            "edge node field {key:?} must be a string, got {other}"
        ))),
    }
}

/// An optional numeric field; a present non-number is an error.
fn number_field(values: &Value, key: &str) -> Result<Option<f64>, BlendError> {
    match values.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number.as_f64().map(Some).ok_or_else(|| {
            BlendError::InvalidInput(format!(
                "edge node field {key:?} is not representable: {number}"
            ))
        }),
        Some(other) => Err(BlendError::InvalidInput(format!(
            "edge node field {key:?} must be a number, got {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swept_corners_parse_and_are_limited_to_circular_fillets() {
        let fillet = EdgeTreatment::from_node_values(&serde_json::json!({
            "mode": "fillet",
            "continuity": "",
            "cornerTurn": "swept",
            "length": 2.0
        }))
        .expect("valid fillet node");
        assert_eq!(fillet.corner_turn, CornerTurn::Swept);
        assert!(fillet.validate().is_ok());

        let g2 = EdgeTreatment::from_node_values(&serde_json::json!({
            "mode": "fillet",
            "continuity": "G2",
            "cornerTurn": "swept",
            "length": 2.0
        }))
        .expect("parse G2");
        assert!(g2.validate().is_err(), "swept is only for circular fillets");

        let unknown = EdgeTreatment::from_node_values(&serde_json::json!({
            "mode": "fillet",
            "cornerTurn": "rounded"
        }));
        assert!(unknown.is_err(), "unknown turn is refused");
    }
}
