//! Every `solidcore_ops::Error` as a sentence a user can act on.
//!
//! Ops hands its failures over as data: a variant, with the numbers and entities the kernel
//! found. This module is the one place that turns each into the sentence a node shows, with the
//! numbers (millimetres, radians and degrees), the entity the kernel names where it names one,
//! and a way out. Nothing here formats a value with `Debug`: an `Option` is worded, a point is
//! written out, and the inner enums that have `Display` give their own words.
//!
//! The match has no wildcard. `solidcore_ops::Error` is not non-exhaustive, so a variant the
//! kernel adds fails to compile until it is worded here; the variant-level `#[non_exhaustive]`
//! ones (`NotSolidBecause`, `Invariant`) are matched with `..`.

use solidcore::prism::profile::ProfileRefusal;
use solidcore_ops::sweep::{CornerRefusal, PathCheck};
use solidcore_ops::{Error as OpsError, Unblendable};

use crate::SessionError;

/// The operation that met an error, so the message can name the feature the user ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpKind {
    /// A conversion, a proof or a measurement, not a user feature operation.
    Body,
    /// A fillet, a chamfer or a blend.
    Blend,
    Boolean,
    Check,
    Extrude,
    Offset,
    Shell,
}

/// The message a node shows for every way an ops call can fail.
pub(crate) fn map_error(op: OpKind, error: OpsError) -> SessionError {
    if op == OpKind::Shell {
        if let Some(refusal) = shell_refusal(&error) {
            return refusal;
        }
    }
    match error {
        // A cancellation is not a failure: it is the caller's own token, and the session
        // rebuilds the worker rather than showing anything.
        OpsError::Cancelled => SessionError::Cancelled,

        // The kernel's own status. Input it refused is the user's to fix; any other status is
        // one ops does not translate, so it is a kernel failure to report.
        OpsError::Status(code) if code == solidcore_ops::error::INVALID_INPUT => {
            SessionError::Invalid(format!(
                "SolidCore refused the input as invalid (kernel status {code}). Check the sizes and the selected faces and edges."
            ))
        }
        OpsError::Status(code) => SessionError::Kernel(format!(
            "SolidCore failed with kernel status {code}. If it repeats on the same input, please report it."
        )),

        OpsError::Refused { operation, reason } => SessionError::Invalid(format!(
            "SolidCore refused the input in {operation}: {reason}. Check the geometry and the sizes this node uses."
        )),

        OpsError::Argument => SessionError::Invalid(
            "An argument to this operation is not finite or is out of range. Check the node's sizes and selections."
                .to_owned(),
        ),

        OpsError::Empty => SessionError::Invalid(
            "The result is empty: the operation left no faces. Check the bodies and sizes this node uses, or use a smaller change."
                .to_owned(),
        ),

        OpsError::NotSolid => SessionError::Invalid(
            "SolidCore built a result that is not a closed solid. This is usually a kernel limitation on this geometry; try a slightly different size."
                .to_owned(),
        ),

        // A `#[non_exhaustive]` tuple variant cannot be destructured from outside ops, so the
        // whole error's `Display` gives the kernel's test clause, e.g. "it is inside out".
        error @ OpsError::NotSolidBecause { .. } => SessionError::Invalid(format!(
            "SolidCore built a result that is not a closed solid ({error}). This is usually a kernel limitation on this geometry; try a slightly different size."
        )),

        OpsError::NoRoom => SessionError::Invalid(
            "No room for this blend: another face is in the way of the size asked. Use a smaller size, or blend fewer edges at once."
                .to_owned(),
        ),

        OpsError::WallsTouch => SessionError::Invalid(
            "The offset's walls meet instead of crossing: the body is too thin for this shell or offset. Use a thinner wall, or take fewer faces."
                .to_owned(),
        ),

        OpsError::TooLarge { size, kind, at, face, why } => {
            let mut message = format!(
                "The blend is too large for this geometry: the largest {} that fits is {}{}.",
                kind.word(),
                length(size),
                near(at)
            );
            if let Some(face) = face {
                message.push_str(&format!(
                    " The face that limits it is at {}.",
                    format_point(face)
                ));
            }
            if let Some(why) = why {
                message.push_str(&format!(" It fails because {}.", why.words()));
            }
            message.push_str(" Use a smaller size, or blend fewer edges at once.");
            SessionError::Invalid(message)
        }

        OpsError::WallFolds { largest: Some(x) } if x.is_finite() && x >= 0.005 => SessionError::Invalid(format!(
            "Shell too thick for this body: the largest that fits is {x:.2} mm. Curved faces such as a G2 fillet's limit the thickness; use a thinner shell, or a larger or G1 fillet."
        )),
        // A bound of zero (or no bound at all) says no wall fits, so it is written without a
        // number: suggesting `0.00 mm` would read as a thickness the user could choose.
        OpsError::WallFolds { .. } => SessionError::Invalid(
            "Shell too thick for this body's curved faces; try a thinner shell.".to_owned(),
        ),

        OpsError::Unsupported(what) => SessionError::Invalid(format!(
            "This operation is not supported yet: {what}. Try a different edge, face or size."
        )),

        OpsError::CornerUnsupported { why, at } => {
            let mut message = format!(
                "Blend corner not supported yet for edges that meet this way{}.",
                at_point(at)
            );
            if let Some(why) = why {
                message.push_str(&format!(" The kernel says: {why}."));
            }
            message.push_str(" Blend fewer edges at once, or use a different corner treatment.");
            SessionError::Invalid(message)
        }

        OpsError::CornerRefused { corner } => SessionError::Invalid(format!(
            "The kernel does not set back corner {corner}: it sets a corner back only where every edge meeting it is blended, all convex or all concave. Blend every edge meeting the corner, or blend fewer edges at once."
        )),

        OpsError::TurnRefused { room, why, at } => {
            let kind = if room {
                "no radius builds the turn between these rounds"
            } else {
                "this kind of turn between rounds is not built yet"
            };
            SessionError::Invalid(format!(
                "The blend's rounds cannot turn where they meet{}: {kind}. {why}. Blend fewer edges at once, or use a different corner treatment.",
                near(at)
            ))
        }

        OpsError::RoundsCross { at, fit } => {
            let mut message = match at.split_first() {
                Some((first, rest)) if rest.is_empty() => format!(
                    "The rounds leave a face whose boundary crosses itself at one point, near {}.",
                    format_point(*first)
                ),
                Some((first, rest)) => format!(
                    "The rounds leave a face whose boundary crosses itself at {} points, the first near {}.",
                    rest.len() + 1,
                    format_point(*first)
                ),
                None => "The rounds leave a face whose boundary crosses itself.".to_owned(),
            };
            if let Some(fit) = fit {
                message.push_str(&format!(
                    " The largest size that leaves no crossing is {}.",
                    length(fit)
                ));
            }
            message.push_str(" Use a smaller size.");
            SessionError::Invalid(message)
        }

        OpsError::TaperEndUnsupported { vertex, lean } => {
            let side = match lean {
                Some(lean) => format!("the side there leans {lean:.3} mm past the plane square to the edge"),
                None => "the side there is not straight".to_owned(),
            };
            SessionError::Invalid(format!(
                "The band's room is not known at vertex {vertex}: {side}. Use a different edge, or a constant taper."
            ))
        }

        OpsError::FacesCloseAtBothEnds => SessionError::Invalid(
            "The faces close to a point at both ends of the edge, so no band fits. Pick another edge."
                .to_owned(),
        ),

        OpsError::RoomRunsOut { at, largest } => SessionError::Invalid(format!(
            "The room runs out {at:.2} of the way along the edge: the largest radius that fits there is {}. Use a smaller radius, or shorten the variable radius.",
            length(largest)
        )),

        OpsError::NoEffect => SessionError::Invalid(
            "The operation did nothing: the size is too small to change the part. Use a larger size."
                .to_owned(),
        ),

        OpsError::CurveDoesNotCut => SessionError::Invalid(
            "The curves do not divide the face into regions: a curve that reaches neither the face's boundary nor another curve leaves a slit, and a curve off the face cuts nothing. Check the curves."
                .to_owned(),
        ),

        OpsError::NotExact => SessionError::Invalid(
            "The result is not exact in the input's geometry. Try a slightly different size, or simplify the geometry near the operation."
                .to_owned(),
        ),

        OpsError::Profile(refusal) => SessionError::Invalid(format!(
            "The planar profile is refused: {}. Close its loops, and check that every segment is finite and of some length.",
            profile_refusal(refusal)
        )),

        OpsError::Degenerate => SessionError::Invalid(
            "The geometry has no value where one is needed (a normal at a pole or an apex, a zero-length vector, or a collapsed edge). Adjust the model near that point and try again."
                .to_owned(),
        ),

        OpsError::UnsureRegion { region } => SessionError::Invalid(format!(
            "SolidCore cannot prove whether region {region} is material, and the answer depends on it. Loosen the tolerance, or change the geometry near that region."
        )),

        OpsError::DraftCornerRefused { why, at: point } => SessionError::Invalid(format!(
            "The draft's corner{} cannot be closed: {}. Use a shorter draft, or a different corner treatment.",
            at_point(point),
            corner_refusal(why)
        )),

        OpsError::PathUndecided { piece, check } => SessionError::Invalid(format!(
            "The sweep path is undecided at piece {piece}: the kernel cannot tell {}. Adjust the path, or loosen its tolerance.",
            path_check(check)
        )),

        OpsError::PathGap { join } => SessionError::Invalid(format!(
            "The sweep path's pieces {join} and {} do not meet. Move them together, or rebuild the path as one chain.",
            join.saturating_add(1)
        )),

        OpsError::BoundaryGap { join, gap } => SessionError::Invalid(format!(
            "The patch's boundary curves {join} and {} do not meet: they are {} apart. Move them together, or check the surface's tolerance.",
            join.saturating_add(1),
            length(gap)
        )),

        OpsError::NoCrossing { u, v, count } => SessionError::Invalid(format!(
            "The curve network's u-curve {u} and v-curve {v} cross {count} times, not once. Move the curves so they cross once, or rebuild the network."
        )),

        OpsError::NetworkGap { u, v, gap } => SessionError::Invalid(format!(
            "The curve network's u-curve {u} and v-curve {v} are up to {} apart where they should cross. Move them together, or check the model's tolerance.",
            length(gap)
        )),

        OpsError::PastTolerance { bound } => SessionError::Invalid(format!(
            "The change would move the curve by up to {}. Loosen the tolerance, or make a smaller change.",
            length(bound)
        )),

        OpsError::OffsetPastTolerance { tolerance, worst } => SessionError::Invalid(format!(
            "The offset strays up to {} from the distance asked, past the tolerance {}. Use a smaller offset distance, or loosen the tolerance.",
            length(worst),
            length(tolerance)
        )),

        OpsError::TooFar { largest: Some(largest) } => SessionError::Invalid(format!(
            "The extension cannot run as far as asked: the longest that works is {}. Use a shorter extension.",
            length(largest)
        )),
        OpsError::TooFar { largest: None } => SessionError::Invalid(
            "The extension cannot run as far as asked: it reaches a pole, an apex or a full turn. Use a shorter extension."
                .to_owned(),
        ),

        OpsError::RailPastTolerance { asked, reached, wires } => SessionError::Invalid(format!(
            "The loft's guided edges were asked within {} of their rails, but at {wires} wires the bound the kernel proved is only {}. Loosen the tolerance, or use fewer wires.",
            length(asked),
            length(reached)
        )),

        // The kernel's own inconsistency: a defect, not the input's.
        OpsError::Exception => SessionError::Kernel(
            "The kernel raised an internal exception during this operation. This is a kernel defect, not a problem with your input; please report it."
                .to_owned(),
        ),

        // The kernel's class and fields are its own: a defect is the kernel's fault; anything
        // else is a fact of the input the kernel's check met.
        OpsError::Invariant { defect, operation, reason, .. } => {
            if defect {
                SessionError::Kernel(format!(
                    "The kernel broke one of its own checks in {operation}: {reason}. This is a kernel defect, not a problem with your input; please report it."
                ))
            } else {
                SessionError::Invalid(format!(
                    "The kernel's check did not hold in {operation}: {reason}. Check this geometry, and report it if it repeats."
                ))
            }
        }

        OpsError::ContactUndecided { other, lower, upper } => SessionError::Invalid(format!(
            "Contact with body {other} is undecided: the distance between the bodies lies between {} and {}. Move the bodies apart, or loosen the tolerance.",
            length(lower),
            length(upper)
        )),

        OpsError::DefinitionNotCertified { face, eps } => SessionError::Invalid(format!(
            "The blend face {face} cannot be carried into the kernel: its definition certifies its net only within {}. Rebuild the blend, and report this geometry if it repeats.",
            length(eps)
        )),

        OpsError::Reentered => SessionError::Kernel(
            "The operation was called from inside another kernel call, or from the failure callback. This is a session defect, not a problem with your input; please report it."
                .to_owned(),
        ),

        OpsError::Record(reason) => SessionError::Kernel(format!(
            "The body could not be carried into the kernel as a record: {reason}. This is an internal conversion failure; please report it."
        )),

        OpsError::Internal(reason) => SessionError::Kernel(if reason.is_empty() {
            "The kernel panicked inside this operation without a message. This is a kernel defect, not a problem with your input; please report it."
                .to_owned()
        } else {
            format!(
                "The kernel panicked inside this operation: {reason}. This is a kernel defect, not a problem with your input; please report it."
            )
        }),

        OpsError::Unported(what) => SessionError::Invalid(format!(
            "This operation needs a kernel step that is not ported yet: {what}. Try a different edge, face or size."
        )),

        OpsError::EdgeNotBlendable { edge, reason } => SessionError::Invalid(format!(
            "Edge {edge} has no blend: {}. Pick another edge.",
            unblendable(reason)
        )),

        OpsError::ContinuesInto { edge, continues_into } => SessionError::Invalid(format!(
            "The blend of edge {edge} cannot end where its edge runs on into edge {continues_into}. Blend edge {continues_into} as well, or end the blend before it."
        )),

        OpsError::RoundTooSmall { edge, turn, tolerance, smallest } => SessionError::Invalid(format!(
            "The round of edge {edge} is too small to change the part: it lies within the edge's tolerance ({}). The faces' normals turn {} across the edge, and the smallest radius that rounds it is {}. Use at least that radius.",
            length(tolerance),
            angle(turn),
            length(smallest)
        )),

        OpsError::SetbackTooShort { vertex, least, at: point, builds } => {
            let mut message = format!(
                "The setback at {} is too short{}: it stops short of where the rounds beside the edge cross it. Use at least {}.",
                vertex_or_corner(vertex),
                at_point(point),
                length(least)
            );
            if builds {
                message.push_str(" That size builds the corner.");
            } else {
                message.push_str(" If the corner still fails, blend fewer edges at once.");
            }
            SessionError::Invalid(message)
        }

        OpsError::SetbackTooLong { vertex, largest } => SessionError::Invalid(format!(
            "The setback at vertex {vertex} does not fit on its edges or rounds. A setback below {} fits. Use a shorter setback, or blend fewer edges at the corner.",
            length(largest)
        )),

        OpsError::ChamferAngleTooLarge { edge, angle: asked, largest } => SessionError::Invalid(format!(
            "The chamfer angle on edge {edge} is too large: {} runs past the other face instead of meeting it. An angle below {} fits.",
            angle(asked),
            angle(largest)
        )),

        OpsError::OffsetRefused { face, why } => SessionError::Invalid(match face {
            Some(face) => format!(
                "The offset is refused at face {face}: {why}. Try a different distance, or offset fewer faces."
            ),
            None => format!(
                "The offset is refused: {why}. Try a different distance, or offset fewer faces."
            ),
        }),

        // The kernel's own class of the failure decides whether it is a defect to report or a
        // refusal to work around; its phrase is the kernel's statement of it, printed as given.
        OpsError::BlendFailed { class, status, phrase } => {
            let why = if phrase.is_empty() { "the kernel gave no reason" } else { phrase.as_str() };
            if class.is_defect() {
                SessionError::Kernel(format!(
                    "The blend failed in a kernel defect (status {status}): {why}. Please report it."
                ))
            } else if class.sized() {
                SessionError::Invalid(format!(
                    "The blend failed (status {status}): {why}. Use a smaller size, or a different corner treatment."
                ))
            } else {
                SessionError::Invalid(format!(
                    "The blend failed (status {status}): {why}. Try a different edge, size or corner treatment."
                ))
            }
        }

        // The certificate ran out of the work it is allowed, not out of room: no size fits
        // better, so the way out is a smaller or simpler blend, not a different fit.
        OpsError::BlendNotCertified { halvings, stations } => SessionError::Invalid(format!(
            "SolidCore could not certify the blend's surface within its tolerance in the work allowed ({halvings} spans halved, {stations} stations). This is a kernel work-budget limit on this geometry, not a size that does not fit; try a slightly smaller blend, or fewer edges at once."
        )),

        // A control-point edit meets the seam: the two faces' boundary rows along the edge are
        // not one curve, so moving a point on one face would open it. The kernel names the edge
        // (its record index) and why the rows differ.
        OpsError::SeamMismatch { edge, why } => SessionError::Invalid(format!(
            "A control-point edit would move edge {edge}, whose two faces' rows along it are not one curve ({why}), so the seam would open. Edit the points away from that edge."
        )),

        // A control-point edit moves a vertex ending an edge that cannot follow it: the edge is
        // not a line between planar faces the edit keeps, so its end would leave a gap.
        OpsError::EdgeLeftBehind { edge } => SessionError::Invalid(format!(
            "A control-point edit would move edge {edge}, which cannot follow its end (it is not a line between planar faces the edit keeps) and would leave a gap at the vertex. Edit the points away from that edge."
        )),
    }
}

/// A shell's own wording for the refusals it can meet: the message names the shell's wall, never
/// the blend's "no room" or the offset's "refused", and never prints a raw kernel status.
/// `None` leaves the variant to the op-neutral wording above.
fn shell_refusal(error: &OpsError) -> Option<SessionError> {
    let message = match error {
        OpsError::NoRoom => {
            "The wall is thicker than the part allows here. Use a thinner shell, or open fewer faces."
                .to_owned()
        }
        OpsError::WallsTouch => {
            "The shell's walls meet instead of crossing: the wall is thicker than the part allows here. Use a thinner shell, or open fewer faces."
                .to_owned()
        }
        // A bound that would round to `0.00 mm` is no useful written thickness: it says no wall
        // fits, and the message says so without a number.
        OpsError::WallFolds { largest: Some(x) } if x.is_finite() && *x >= 0.005 => format!(
            "The wall is thicker than the part allows here (max about {x:.2} mm). Use a thinner shell, or open fewer faces."
        ),
        // A zero (or non-finite) bound says no wall fits: never print `0.00 mm` as if it were
        // a thickness the user could ask for.
        OpsError::WallFolds { .. } => {
            "The wall is thicker than the part allows here: no wall fits this body. Use a thinner shell, or open fewer faces."
                .to_owned()
        }
        OpsError::Status(code) if *code == 1001 => {
            "The shell can't open this face: it is on a curved or filleted part of the body where the kernel can't make the opening. Open a planar face, or shell with no faces open."
                .to_owned()
        }
        OpsError::Status(_) => {
            "The shell could not be built on this body. Try a thinner wall, or open fewer faces."
                .to_owned()
        }
        OpsError::OffsetRefused { face, why } => match face {
            Some(face) => format!(
                "The shell's wall is refused at face {face}: {why}. Try a thinner shell, or open fewer faces."
            ),
            None => format!(
                "The shell's wall is refused: {why}. Try a thinner shell, or open fewer faces."
            ),
        },
        OpsError::NotSolid => {
            "The shell's wall would not be a closed solid on this body. Try a thinner shell, or open fewer faces."
                .to_owned()
        }
        OpsError::NotSolidBecause { .. } => format!(
            "The shell's wall would not be a closed solid on this body ({error}). Try a thinner shell, or open fewer faces."
        ),
        OpsError::Argument => {
            "The shell's wall or openings are not valid for this body. Check the thickness and reselect the faces to remove."
                .to_owned()
        }
        OpsError::Empty => {
            "The shell left no wall on this body. Open fewer faces, or use a thinner shell."
                .to_owned()
        }
        OpsError::Unsupported(what) => format!(
            "The shell can't measure this body's wall: {what}. Try a thinner shell."
        ),
        _ => return None,
    };
    Some(SessionError::Invalid(message))
}

/// A point in millimetres, as a message writes it out.
fn format_point(point: [f64; 3]) -> String {
    format!("({:.2}, {:.2}, {:.2}) mm", point[0], point[1], point[2])
}

/// A length in millimetres: part sizes to a hundredth, tolerances in scientific notation, so
/// that both a 2 mm blend and a 1e-5 mm tolerance read at their own scale.
fn length(value: f64) -> String {
    if value != 0.0 && value.abs() < 0.01 {
        format!("{value:.1e} mm")
    } else {
        format!("{value:.2} mm")
    }
}

/// An angle in radians, with its degrees for a user who thinks in them.
fn angle(radians: f64) -> String {
    format!("{radians:.3} rad ({:.1}°)", radians.to_degrees())
}

/// ` at (…) mm`, or nothing when the kernel names no point.
fn at_point(point: Option<[f64; 3]>) -> String {
    match point {
        Some(point) => format!(" at {}", format_point(point)),
        None => String::new(),
    }
}

/// `, near (…) mm`, or nothing when the kernel names no point.
fn near(point: Option<[f64; 3]>) -> String {
    match point {
        Some(point) => format!(", near {}", format_point(point)),
        None => String::new(),
    }
}

/// `vertex 3`, or `a corner` when the kernel no longer holds the vertex.
fn vertex_or_corner(vertex: Option<usize>) -> String {
    match vertex {
        Some(vertex) => format!("vertex {vertex}"),
        None => "a corner".to_owned(),
    }
}

/// Why an edge has no blend, as a clause: `Unblendable` has no `Display`.
fn unblendable(reason: Unblendable) -> &'static str {
    match reason {
        Unblendable::Smooth => "its faces meet smoothly along it, so there is nothing to round",
        Unblendable::NotTwoFaces => "it is not between two faces",
        Unblendable::Tangent => {
            "its faces are tangent, or folded back on each other, where the blend is set up"
        }
        Unblendable::Ambiguous => "which side to blend cannot be told",
    }
}

/// Which comparison of a path is undecided, as a clause: `PathCheck` has no `Display`.
fn path_check(check: PathCheck) -> &'static str {
    match check {
        PathCheck::Gap => "whether a piece starts where the one before ends",
        PathCheck::Length => "whether a piece is long enough to count",
        PathCheck::Forward => "whether a line goes forward along the path's chord",
        PathCheck::OffLine => "whether a point lies on the path's chord",
        PathCheck::Circle => "whether an arc lies on the first arc's circle",
        PathCheck::Reached => "whether an arc ends where the run of arcs has turned to",
        PathCheck::Turn => "whether a segment leaves in the direction the one before arrives in",
        PathCheck::Open => "whether the path's ends are apart",
    }
}

/// Why a draft corner could not be closed, as a clause: `CornerRefusal` has no `Display`.
fn corner_refusal(reason: CornerRefusal) -> &'static str {
    match reason {
        CornerRefusal::Apart => "the continued curves do not meet",
        CornerRefusal::NotExact => {
            "a curve there is neither a line nor an arc, so its wall has no exact continuation"
        }
        CornerRefusal::WallCloses => "a wall closes before the far end",
    }
}

/// Why a planar profile is refused, as a clause: `ProfileRefusal` has no `Display`.
fn profile_refusal(refusal: ProfileRefusal) -> &'static str {
    match refusal {
        ProfileRefusal::Budget => {
            "a loop is not closed, is empty, or has more segments than the budget allows"
        }
        ProfileRefusal::LineOrArc => "a line or circular arc is not finite, or has no length",
        ProfileRefusal::FreeForm => {
            "a free-form segment is not finite or has no length, or a conic weight is not positive"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OpKind, map_error};
    use crate::SessionError;
    use solidcore::error::ErrorKind;
    use solidcore::error::offset::LoopCrossing;
    use solidcore::math::Point3;
    use solidcore::primitive::api_primitives::create_plane;
    use solidcore::prism::profile::ProfileRefusal;
    use solidcore_ops::sweep::{CornerRefusal, PathCheck};
    use solidcore_ops::{
        Body, Error as OpsError, FitKind, OffsetRefusal, RoomReason, Unblendable, Unsoundness,
    };

    /// No message shows a value the way `Debug` would: none has `Some(`, a bare `None`, or the
    /// opening of a struct dump (`{ `). A user reads sentences, so each ends like one.
    fn assert_readable(op: OpKind, error: OpsError) {
        let message = map_error(op, error).to_string();
        assert!(!message.is_empty(), "an empty message");
        for forbidden in ["Some(", "None", "{ "] {
            assert!(!message.contains(forbidden), "the message shows {forbidden}: {message}");
        }
        if message != "cancelled" {
            assert!(message.ends_with('.'), "not a full sentence: {message}");
        }
    }

    /// Every `solidcore_ops::Error` this crate can build, mapped: readable text, no `Debug`.
    ///
    /// `Invariant` is `#[non_exhaustive]` and holds the kernel's own state, so it cannot be
    /// built outside solidcore-ops; its two arms (a defect to report, and a check that did not
    /// hold) are covered by review, not here. `NotSolidBecause` is also `#[non_exhaustive]`,
    /// so it is not built by hand either: it comes from a real call below.
    #[test]
    fn every_constructible_ops_error_maps_to_a_readable_sentence() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let sheet =
            create_plane(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0).expect("a sheet");
        let sheet_body = Body::from_brep(&mut kernel, sheet).expect("a sheet body");
        let (checked, _) =
            solidcore_ops::session::with_kernel(&mut kernel, || sheet_body.check_solid());
        let not_solid = checked.expect_err("a sheet is not a closed solid");
        assert!(matches!(not_solid, OpsError::NotSolidBecause { .. }));

        let errors = [
            OpsError::Status(1007),
            OpsError::Status(1001),
            OpsError::Refused { operation: "measure", reason: "the accuracy is not finite" },
            OpsError::Argument,
            OpsError::Empty,
            OpsError::NotSolid,
            OpsError::NoRoom,
            OpsError::WallsTouch,
            OpsError::TooLarge {
                size: 6.828,
                kind: FitKind::Radius,
                at: Some([1.0, 2.0, 3.0]),
                face: Some([4.0, 5.0, 6.0]),
                why: Some(RoomReason::BallDoesNotFit),
            },
            OpsError::TooLarge {
                size: 2.0,
                kind: FitKind::Distance,
                at: None,
                face: None,
                why: None,
            },
            OpsError::TooLarge {
                size: 0.5,
                kind: FitKind::Scale,
                at: Some([0.0, 0.0, 0.0]),
                face: None,
                why: Some(RoomReason::BandsOverlap),
            },
            OpsError::TooLarge {
                size: 3.0,
                kind: FitKind::Chord,
                at: None,
                face: None,
                why: Some(RoomReason::CutsThePart),
            },
            OpsError::WallFolds { largest: Some(1.25) },
            OpsError::WallFolds { largest: None },
            OpsError::Unsupported("a variable radius that is zero inside its edge"),
            OpsError::CornerUnsupported {
                why: Some("the corner is a saddle".to_owned()),
                at: Some([1.0, 2.0, 3.0]),
            },
            OpsError::CornerUnsupported { why: None, at: None },
            OpsError::CornerRefused { corner: 3 },
            OpsError::TurnRefused {
                room: true,
                why: "the wall turns away".to_owned(),
                at: Some([1.0, 2.0, 3.0]),
            },
            OpsError::TurnRefused {
                room: false,
                why: "the corner is a saddle".to_owned(),
                at: None,
            },
            OpsError::RoundsCross { at: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], fit: Some(2.5) },
            OpsError::RoundsCross { at: vec![[1.0, 2.0, 3.0]], fit: None },
            OpsError::RoundsCross { at: Vec::new(), fit: None },
            OpsError::TaperEndUnsupported { vertex: 4, lean: Some(0.25) },
            OpsError::TaperEndUnsupported { vertex: 5, lean: None },
            OpsError::FacesCloseAtBothEnds,
            OpsError::RoomRunsOut { at: 0.75, largest: 1.5 },
            OpsError::NoEffect,
            OpsError::CurveDoesNotCut,
            OpsError::NotExact,
            OpsError::Profile(ProfileRefusal::Budget),
            OpsError::Profile(ProfileRefusal::LineOrArc),
            OpsError::Profile(ProfileRefusal::FreeForm),
            OpsError::Degenerate,
            OpsError::UnsureRegion { region: 7 },
            OpsError::DraftCornerRefused { why: CornerRefusal::Apart, at: Some([0.0, 0.0, 0.0]) },
            OpsError::DraftCornerRefused { why: CornerRefusal::NotExact, at: None },
            OpsError::DraftCornerRefused {
                why: CornerRefusal::WallCloses,
                at: Some([2.0, 1.0, 0.0]),
            },
            OpsError::PathUndecided { piece: 2, check: PathCheck::Gap },
            OpsError::PathUndecided { piece: 3, check: PathCheck::Length },
            OpsError::PathUndecided { piece: 4, check: PathCheck::Forward },
            OpsError::PathUndecided { piece: 5, check: PathCheck::OffLine },
            OpsError::PathUndecided { piece: 6, check: PathCheck::Circle },
            OpsError::PathUndecided { piece: 7, check: PathCheck::Reached },
            OpsError::PathUndecided { piece: 8, check: PathCheck::Turn },
            OpsError::PathUndecided { piece: 9, check: PathCheck::Open },
            OpsError::PathGap { join: 3 },
            OpsError::BoundaryGap { join: 4, gap: 1.0e-4 },
            OpsError::NoCrossing { u: 1, v: 2, count: 0 },
            OpsError::NetworkGap { u: 1, v: 2, gap: 5.0e-5 },
            OpsError::PastTolerance { bound: 0.002 },
            OpsError::OffsetPastTolerance { tolerance: 1.0e-5, worst: 3.2e-4 },
            OpsError::TooFar { largest: Some(12.5) },
            OpsError::TooFar { largest: None },
            OpsError::RailPastTolerance { asked: 0.001, reached: 0.05, wires: 16 },
            OpsError::Exception,
            OpsError::ContactUndecided { other: 1, lower: 0.001, upper: 0.002 },
            OpsError::Cancelled,
            OpsError::DefinitionNotCertified { face: 2, eps: 1.0e-6 },
            OpsError::Reentered,
            OpsError::Record("the face's loop has no record id".to_owned()),
            OpsError::Internal("a trim curve has no surface".to_owned()),
            OpsError::Internal(String::new()),
            OpsError::Unported("the screw sweep"),
            OpsError::EdgeNotBlendable { edge: 0, reason: Unblendable::Smooth },
            OpsError::EdgeNotBlendable { edge: 1, reason: Unblendable::NotTwoFaces },
            OpsError::EdgeNotBlendable { edge: 2, reason: Unblendable::Tangent },
            OpsError::EdgeNotBlendable { edge: 3, reason: Unblendable::Ambiguous },
            OpsError::ContinuesInto { edge: 2, continues_into: 5 },
            OpsError::RoundTooSmall {
                edge: 1,
                turn: std::f64::consts::FRAC_PI_2,
                tolerance: 1.0e-5,
                smallest: 0.2,
            },
            OpsError::SetbackTooShort {
                vertex: Some(3),
                least: 2.5,
                at: Some([1.0, 0.0, 0.0]),
                builds: true,
            },
            OpsError::SetbackTooShort { vertex: None, least: 1.0, at: None, builds: false },
            OpsError::SetbackTooLong { vertex: 3, largest: 4.0 },
            OpsError::ChamferAngleTooLarge { edge: 1, angle: 1.2, largest: 0.9 },
            OpsError::OffsetRefused {
                face: Some(2),
                why: OffsetRefusal::NotCertified { distance: 1.5 },
            },
            OpsError::OffsetRefused {
                face: None,
                why: OffsetRefusal::Collapses { distance: 2.0, radius: 1.0 },
            },
            OpsError::OffsetRefused {
                face: Some(0),
                why: OffsetRefusal::LoopsCross { crossing: LoopCrossing::AtLoopVertex },
            },
            OpsError::OffsetRefused {
                face: None,
                why: OffsetRefusal::DistanceUnresolved { distance: 1.0e-9, resolution: 1.0e-5 },
            },
            OpsError::OffsetRefused {
                face: Some(4),
                why: OffsetRefusal::ToleranceUnresolved { tolerance: 1.0e-9 },
            },
            OpsError::OffsetRefused { face: None, why: OffsetRefusal::SeamOpened },
            OpsError::OffsetRefused { face: Some(1), why: OffsetRefusal::ExactCurveUncut },
            OpsError::OffsetRefused { face: None, why: OffsetRefusal::FacesApart },
            OpsError::OffsetRefused { face: Some(5), why: OffsetRefusal::ReachUncertified },
            OpsError::OffsetRefused { face: None, why: OffsetRefusal::LoopGapUncertified },
            OpsError::OffsetRefused {
                face: Some(0),
                why: OffsetRefusal::Unsound { what: Unsoundness::GeometryUnchecked },
            },
            OpsError::OffsetRefused {
                face: None,
                why: OffsetRefusal::Phrase {
                    class: ErrorKind::Failed,
                    defect: false,
                    phrase: "the offset routine failed".to_owned(),
                },
            },
            OpsError::BlendFailed {
                class: ErrorKind::Failed,
                status: 1001,
                phrase: "the blend's section meets no face".to_owned(),
            },
            OpsError::BlendFailed {
                class: ErrorKind::StageFailed,
                status: 1001,
                phrase: String::new(),
            },
            OpsError::BlendNotCertified { halvings: 3, stations: 9 },
            OpsError::SeamMismatch {
                edge: 2,
                why: solidcore_ops::surface_cvs::SeamMismatch::Degree { a: 2, b: 3 },
            },
            OpsError::EdgeLeftBehind { edge: 4 },
        ];
        for error in errors {
            assert_readable(OpKind::Body, error);
        }

        // The kernel now names the face that limits a too-large blend, and the vertex and
        // reason of a corner it does not build: both are worded, not dropped.
        let limited = map_error(
            OpKind::Body,
            OpsError::TooLarge {
                size: 6.828,
                kind: FitKind::Radius,
                at: Some([1.0, 2.0, 3.0]),
                face: Some([4.0, 5.0, 6.0]),
                why: Some(RoomReason::BallDoesNotFit),
            },
        )
        .to_string();
        assert!(
            limited.contains("The face that limits it is at (4.00, 5.00, 6.00) mm."),
            "the limiting face is not written out: {limited}"
        );
        let corner = map_error(
            OpKind::Body,
            OpsError::CornerUnsupported {
                why: Some("the corner is a saddle".to_owned()),
                at: Some([1.0, 2.0, 3.0]),
            },
        )
        .to_string();
        assert!(
            corner.contains("The kernel says: the corner is a saddle."),
            "the corner's reason is not written out: {corner}"
        );

        assert_eq!(map_error(OpKind::Body, OpsError::Cancelled), SessionError::Cancelled);
        assert_readable(OpKind::Body, not_solid);
    }

    /// A shell's refusal reads as the shell's: it names the wall, never the blend's "no room",
    /// the offset's "refused", or a raw kernel status, and it always ends like a sentence.
    #[test]
    fn shell_refusals_use_shell_wording_not_blend_wording() {
        let errors = [
            OpsError::NoRoom,
            OpsError::WallsTouch,
            OpsError::WallFolds { largest: Some(0.0) },
            OpsError::WallFolds { largest: Some(0.001) },
            OpsError::WallFolds { largest: Some(f64::NAN) },
            OpsError::WallFolds { largest: Some(1.25) },
            OpsError::WallFolds { largest: None },
            OpsError::Status(1001),
            OpsError::Status(1007),
            OpsError::OffsetRefused {
                face: Some(23),
                why: OffsetRefusal::NotCertified { distance: 1.5 },
            },
            OpsError::OffsetRefused { face: None, why: OffsetRefusal::SeamOpened },
            OpsError::NotSolid,
            OpsError::Argument,
            OpsError::Empty,
            OpsError::Unsupported("a wall measured on a body with no thickness"),
        ];
        for error in errors {
            let message = map_error(OpKind::Shell, error).to_string();
            assert!(
                message.contains("shell") || message.contains("wall"),
                "the shell message names neither the shell nor its wall: {message}"
            );
            for forbidden in ["blend", "edges", "0.00", "kernel status", "offset is refused"] {
                assert!(
                    !message.contains(forbidden),
                    "the shell message shows {forbidden}: {message}"
                );
            }
            assert!(message.ends_with('.'), "not a full sentence: {message}");
        }

        let measured =
            map_error(OpKind::Shell, OpsError::WallFolds { largest: Some(1.25) }).to_string();
        assert!(measured.contains("1.25 mm"), "the wall bound is not written out: {measured}");
    }
}
