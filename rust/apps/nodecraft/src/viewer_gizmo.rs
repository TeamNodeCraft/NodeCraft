//! The transform triad: X/Y/Z arrows, XY/YZ/XZ plane squares, X/Y/Z
//! rotation rings and a centre ball. Adapted from SolidCraft's
//! `crates/ui-egui/src/gizmo.rs`, synced with SolidCraft c2ffd48e
//! (MIT OR Apache-2.0).

use egui::{Color32, Pos2, Rect, Shape, Stroke, vec2};
use solidcraft_geom::Ring;
use solidcore_record::Vec3;
use solidcraft_geom::prelude::*;
use solidcraft_viewport::Camera;

/// World ↔ screen mapping for the current viewport frame.
#[derive(Clone, Copy)]
pub(crate) struct Proj {
    camera: Camera,
    rect: Rect,
    radius: f64,
    shift: [f32; 2],
}

impl Proj {
    fn to_screen(&self, point: Vec3) -> Option<Pos2> {
        let (x, y, _) = self.camera.to_screen(
            point,
            f64::from(self.rect.width()),
            f64::from(self.rect.height()),
            self.radius,
        )?;
        let shifted = vec2(
            x as f32 + self.shift[0] * self.rect.width() * 0.5,
            y as f32 - self.shift[1] * self.rect.height() * 0.5,
        );
        Some(self.rect.min + shifted)
    }

    fn ray(&self, point: Pos2) -> (Vec3, Vec3) {
        self.camera.ray(
            f64::from(point.x - self.rect.min.x)
                - f64::from(self.shift[0]) * f64::from(self.rect.width()) * 0.5,
            f64::from(point.y - self.rect.min.y)
                + f64::from(self.shift[1]) * f64::from(self.rect.height()) * 0.5,
            f64::from(self.rect.width()),
            f64::from(self.rect.height()),
        )
    }
}

fn projected_polygon(points: &[Pos2], fill: Color32, stroke: Stroke) -> Vec<Shape> {
    let mut mesh = egui::Mesh::default();
    for (index, point) in (0u32..).zip(points) {
        mesh.colored_vertex(*point, fill);
        if index >= 2 {
            mesh.add_triangle(0, index - 1, index);
        }
    }
    let mut shapes = vec![Shape::mesh(mesh)];
    shapes.extend(
        points
            .iter()
            .zip(points.iter().skip(1).chain(points.first()))
            .map(|(a, b)| Shape::line_segment([*a, *b], stroke)),
    );
    shapes
}

/// The node-side gizmo values.
#[derive(Clone)]
pub struct Gizmo {
    pub node_id: String,
    pub pivot: [f64; 3],
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
    /// Linear inverse of the downstream chain, mapping world deltas back to
    /// this node's parent frame.
    pub world_to_local: [[f64; 3]; 3],
    /// World directions of the node's local X/Y/Z axes.
    pub axes: [[f64; 3]; 3],
    /// True when `axes` include the node's own turn (`space: local`); a ring
    /// then turns about its intrinsic axis, so the drag composes on the right.
    pub local_axes: bool,
}

/// Persistent drag snapshot for the wrapper.
#[derive(Default)]
pub struct GizmoState {
    node_id: Option<String>,
    start_values: Option<[f64; 3]>,
    start_rotation: Option<[f64; 3]>,
    dragging: bool,
}

impl GizmoState {
    #[cfg(test)]
    pub fn is_dragging(&self) -> bool {
        self.dragging
    }

    pub fn reset(&mut self) {
        self.node_id = None;
        self.start_values = None;
        self.start_rotation = None;
        self.dragging = false;
    }
}

fn triad_for(gizmo: &Gizmo) -> Triad {
    Triad {
        center: Vec3::new(gizmo.pivot[0], gizmo.pivot[1], gizmo.pivot[2]),
        translate: true,
        rotate: true,
        axes: gizmo
            .axes
            .iter()
            .map(|axis| Vec3::new(axis[0], axis[1], axis[2]))
            .collect::<Vec<_>>()
            .try_into()
            .expect("three axes"),
        local_axes: gizmo.local_axes,
    }
}

/// Whether `point` is over one of the triad's handles. The viewport reads this
/// before picking so a click-to-focus does not also pick the model behind it.
pub(crate) fn contains_pointer(
    gizmo: &Gizmo,
    camera: &Camera,
    rect: Rect,
    shift: [f32; 2],
    radius: f64,
    point: Pos2,
) -> bool {
    let proj = Proj { camera: *camera, rect, radius, shift };
    let triad = triad_for(gizmo);
    mm_per_px(&proj, triad.center)
        .and_then(|scale| shapes(&proj, &triad, scale * LEN_PX))
        .is_some_and(|shapes| hit(&shapes, point).is_some())
}

/// What one frame of the transform triad produced.
#[derive(Clone, Debug, Default)]
pub struct GizmoUpdate {
    /// Absolute field values to write onto the Translate node.
    pub changes: Vec<(String, f64)>,
    /// The drag started this frame; the values still hold their pre-drag state.
    pub started: bool,
    /// A drag is in progress after this frame.
    pub dragging: bool,
    /// The drag was released this frame.
    pub ended: bool,
    /// A click without a drag landed on an arrow or ring: the inspector field
    /// it drives (`x`/`y`/`z` or `rx`/`ry`/`rz`), to focus without moving it.
    /// A plane square or the ball drives several fields and leaves it unset.
    pub focus: Option<String>,
}

/// Draw the SolidCraft triad and map its drag changes onto the Translate node.
pub fn show(
    ui: &mut egui::Ui,
    rect: Rect,
    camera: &Camera,
    shift: [f32; 2],
    radius: f64,
    gizmo: &mut Gizmo,
    state: &mut GizmoState,
) -> GizmoUpdate {
    let id = egui::Id::new(("solidcraft-transform-gizmo", gizmo.node_id.as_str()));
    if state.node_id.as_deref() != Some(gizmo.node_id.as_str()) {
        // Drop the old node's pointer state too, or selecting it again later
        // would resume a drag that ended when the selection moved.
        if let Some(previous) = state.node_id.take() {
            let previous = egui::Id::new(("solidcraft-transform-gizmo", previous.as_str()));
            ui.data_mut(|data| data.remove::<State>(previous));
        }
        // The wrapper may have been reset while this node was hidden, so this
        // node's own state can also be stale.
        ui.data_mut(|data| data.remove::<State>(id));
        state.node_id = Some(gizmo.node_id.clone());
        state.start_values = None;
        state.start_rotation = None;
        state.dragging = false;
    }
    let was_dragging = state.dragging;
    let proj = Proj { camera: *camera, rect, radius, shift };
    let triad = triad_for(gizmo);
    let painter = ui.painter().with_clip_rect(rect);
    let frame = solid_show(ui, &painter, &proj, id, &triad, TRANSLATE_SNAP_STEP);
    let focus = frame.clicked.and_then(focus_field);
    let Some(drag) = frame.drag else {
        state.dragging = false;
        // The pointer can vanish between frames (release outside the window);
        // report that as the end of the drag so previews can be retired.
        return GizmoUpdate { ended: was_dragging, focus, ..GizmoUpdate::default() };
    };
    if drag.started || state.start_values.is_none() {
        state.start_values = Some([gizmo.x, gizmo.y, gizmo.z]);
        state.start_rotation = Some([gizmo.rx, gizmo.ry, gizmo.rz]);
    }
    // A press alone is not a drag yet: `started` is reported only once the
    // pointer passes egui's click threshold, so a click opens no preview.
    state.dragging = drag.started || was_dragging;
    let mut changes = Vec::new();
    let values = state.start_values.unwrap_or([gizmo.x, gizmo.y, gizmo.z]);
    let rotations = state.start_rotation.unwrap_or([gizmo.rx, gizmo.ry, gizmo.rz]);
    // A click selects the field its handle drives but moves nothing.
    if !drag.started && was_dragging && frame.clicked.is_none() {
        match drag.change {
            Change::Translate(delta) => {
                let world = [delta.x, delta.y, delta.z];
                let local = gizmo
                    .world_to_local
                    .iter()
                    .map(|row| row[0] * world[0] + row[1] * world[1] + row[2] * world[2])
                    .collect::<Vec<_>>();
                for (index, field) in ["x", "y", "z"].iter().enumerate() {
                    let value = values[index] + local[index];
                    match index {
                        0 => gizmo.x = value,
                        1 => gizmo.y = value,
                        _ => gizmo.z = value,
                    }
                    changes.push(((*field).to_owned(), value));
                }
            }
            Change::Rotate { axis, angle } => {
                // A ring turns about the axis it shows: put the drag's turn on
                // top of the angles the drag started with, then write the XYZ
                // angles our Translate node evaluates. The dragged angle stays
                // within one turn (360 reads 0).
                let (next, changed) = turned_angles(rotations, axis, angle, triad.local_axes);
                for (index, field) in ["rx", "ry", "rz"].iter().enumerate() {
                    if !changed[index] {
                        continue;
                    }
                    let value = next[index];
                    match index {
                        0 => gizmo.rx = value,
                        1 => gizmo.ry = value,
                        _ => gizmo.rz = value,
                    }
                    changes.push(((*field).to_owned(), value));
                }
            }
        }
    }
    if drag.done {
        state.start_values = None;
        state.start_rotation = None;
        state.dragging = false;
    }
    GizmoUpdate {
        changes,
        started: drag.started,
        dragging: state.dragging,
        ended: drag.done && was_dragging,
        focus,
    }
}

/// A part of the triad.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Handle {
    /// Arrow along X, Y or Z (0, 1, 2).
    Axis(usize),
    /// Square in the plane across that axis (Plane(2) is XY).
    Plane(usize),
    /// Ring turning about that axis.
    Ring(usize),
    /// Centre ball: free drag in the view plane.
    Ball,
}

/// The inspector field a click on `handle` selects: an arrow drives one
/// position, a ring one rotation. The plane squares and ball drive several.
fn focus_field(handle: Handle) -> Option<String> {
    match handle {
        Handle::Axis(i) => Some(["x", "y", "z"][i.min(2)].to_owned()),
        Handle::Ring(i) => Some(["rx", "ry", "rz"][i.min(2)].to_owned()),
        Handle::Plane(_) | Handle::Ball => None,
    }
}

/// What a drag has done so far.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Change {
    Translate(Vec3),
    /// About X, Y or Z through the centre, in radians.
    Rotate {
        axis: usize,
        angle: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    pub handle: Handle,
    pub change: Change,
    /// The first frame of the drag (take a snapshot of the values now).
    pub started: bool,
    /// The drag ended this frame.
    pub done: bool,
}

/// One frame of pointer work on the triad: a drag, a click without one, or both.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Frame {
    drag: Option<Drag>,
    clicked: Option<Handle>,
}

/// Where the triad is and which handles it shows.
#[derive(Clone, Copy, Debug)]
pub struct Triad {
    pub center: Vec3,
    pub translate: bool,
    pub rotate: bool,
    pub axes: [Vec3; 3],
    /// See [`Gizmo::local_axes`].
    pub local_axes: bool,
}

#[derive(Clone, Copy, Debug)]
struct State {
    handle: Handle,
    center: Vec3,
    axes: [Vec3; 3],
    /// Grab point on the plane (planes, ball) or parameter along the axis.
    grab: Vec3,
    t0: f64,
    /// Ring: last angle seen and the angle turned so far.
    last: f64,
    turned: f64,
    /// The drag has passed egui's click threshold and was reported as started.
    started: bool,
}

/// The local axes a Translate node's angles turn about.
const DIRS: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// The order the Translate node applies its angles, left to right in the
/// product (see [`turn_of`]): about X first, then Y, then Z. This is also
/// `nodecraft_exact::Affine3::from_rotation`'s mesh/B-rep order.
const ANGLE_ORDER: [usize; 3] = [2, 1, 0];

/// Arrow length on screen (px).
pub const LEN_PX: f64 = 110.0;

/// How near a handle a press still takes it (px): the widest, the centre ball.
pub const HIT_TOLERANCE: f32 = 9.0;

/// Snap steps used only while the snap modifier is held. Keep them here so a
/// future snapping preference can replace the modifier policy without hunting
/// through each drag branch.
const TRANSLATE_SNAP_STEP: f64 = 1.0;
const ROTATE_SNAP_STEP: f64 = std::f64::consts::PI / 36.0; // 5 degrees

/// Ring radius as a share of the arrow length.
const RING: f64 = 0.62;

/// In-plane directions of the ring about an axis (right-handed: e1 × e2 = axis).
fn ring_basis(axes: &[Vec3; 3], i: usize) -> (Vec3, Vec3) {
    (axes[(i + 1) % 3], axes[(i + 2) % 3])
}

/// Where round a ring (degrees) a press takes that ring, clear of the handles
/// over it. Tests use it to grab a ring the same way the pointer does.
#[cfg(test)]
fn ring_grip(proj: &Proj, t: &Triad, ring: usize) -> Option<f64> {
    let l = mm_per_px(proj, t.center)? * LEN_PX;
    let sh = shapes(proj, t, l)?;
    let (e1, e2) = ring_basis(&t.axes, ring.min(2));
    (0..72).map(|k| f64::from(k) * 5.0).find(|a| {
        let th = a.to_radians();
        proj.to_screen(t.center + (e1 * th.cos() + e2 * th.sin()) * (RING * l)).is_some_and(|p| hit(&sh, p) == Some(Handle::Ring(ring.min(2))))
    })
}

/// Distance from `p` to the screen segment a–b.
fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    (a + ab * t).distance(p)
}

fn in_quad(p: Pos2, q: &Ring<Pos2>) -> bool {
    if q.len() < 3 {
        return false;
    }
    let mut sign = 0.0f32;
    for (&a, &b) in q.edges() {
        let c = (b - a).x * (p - a).y - (b - a).y * (p - a).x;
        if c.abs() < 1e-6 {
            continue;
        }
        if sign == 0.0 {
            sign = c.signum();
        } else if c.signum() != sign {
            return false;
        }
    }
    true
}

/// Where the view ray through `pos` meets the plane through `o` with normal `n`.
pub(crate) fn on_plane(proj: &Proj, pos: Pos2, o: Vec3, n: Vec3) -> Option<Vec3> {
    let (ro, rd) = proj.ray(pos);
    let den = rd.dot(n);
    if den.abs() < 1e-9 {
        return None;
    }
    Some(ro + rd * ((o - ro).dot(n) / den))
}

/// Parameter along the line `o + d t` closest to the view ray through `pos`.
fn along(proj: &Proj, pos: Pos2, o: Vec3, d: Vec3) -> Option<f64> {
    let (ro, r) = proj.ray(pos);
    let w0 = o - ro;
    let (b, dd, e) = (d.dot(r), d.dot(w0), r.dot(w0));
    let den = 1.0 - b * b;
    if den.abs() < 1e-6 {
        return None;
    }
    Some((b * e - dd) / den)
}

/// The triad's shapes: (handle, screen polygon or polyline, is it a filled area).
struct Shapes {
    arrows: Vec<(usize, Pos2, Pos2)>,
    planes: Vec<(usize, Ring<Pos2>)>,
    rings: Vec<(usize, Vec<Pos2>)>,
    ball: Pos2,
}

fn shapes(proj: &Proj, t: &Triad, l: f64) -> Option<Shapes> {
    let c = t.center;
    let ball = proj.to_screen(c)?;
    let mut s = Shapes { arrows: Vec::new(), planes: Vec::new(), rings: Vec::new(), ball };
    if t.translate {
        for (i, d) in t.axes.iter().enumerate() {
            if let (Some(a), Some(b)) = (proj.to_screen(c + *d * (0.18 * l)), proj.to_screen(c + *d * l)) {
                s.arrows.push((i, a, b));
            }
        }
        for i in 0..3 {
            let (e1, e2) = ring_basis(&t.axes, i);
            let q: Ring<Pos2> = [(0.25, 0.25), (0.45, 0.25), (0.45, 0.45), (0.25, 0.45)]
                .iter()
                .filter_map(|(a, b)| proj.to_screen(c + e1 * (a * l) + e2 * (b * l)))
                .collect();
            if q.len() == 4 {
                s.planes.push((i, q));
            }
        }
    }
    if t.rotate {
        for i in 0..3 {
            let (e1, e2) = ring_basis(&t.axes, i);
            let pts: Vec<Pos2> = (0..=64)
                .map(|k| k as f64 / 64.0 * std::f64::consts::TAU)
                .filter_map(|th| proj.to_screen(c + (e1 * th.cos() + e2 * th.sin()) * (RING * l)))
                .collect();
            if pts.len() > 2 {
                s.rings.push((i, pts));
            }
        }
    }
    Some(s)
}

fn hit(s: &Shapes, p: Pos2) -> Option<Handle> {
    if s.ball.distance(p) < HIT_TOLERANCE {
        return Some(Handle::Ball);
    }
    if let Some((i, ..)) = s.arrows.iter().find(|(_, a, b)| seg_dist(p, *a, *b) < 6.0) {
        return Some(Handle::Axis(*i));
    }
    if let Some((i, _)) = s.planes.iter().find(|(_, q)| in_quad(p, q)) {
        return Some(Handle::Plane(*i));
    }
    // From some views two rings lie almost on top of each other: the nearest
    // takes the press.
    let near = |pts: &Vec<Pos2>| pts.windows(2).map(|w| seg_dist(p, w[0], w[1])).fold(f32::MAX, f32::min);
    s.rings
        .iter()
        .map(|(i, pts)| (*i, near(pts)))
        .filter(|r| r.1 < 5.0)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| Handle::Ring(i))
}

/// Millimetres per screen pixel at a point.
fn mm_per_px(proj: &Proj, c: Vec3) -> Option<f64> {
    let (_, view) = proj.ray(proj.to_screen(c)?);
    let u = view.any_perp().normalized()?;
    let (a, b) = (proj.to_screen(c)?, proj.to_screen(c + u)?);
    let px = f64::from(a.distance(b));
    (px > 1e-9).then(|| 1.0 / px)
}

/// Whether the current modifiers request snapping. Continuous dragging is the
/// default; snapping is opt-in.
///
/// This is deliberately the one policy point. A future "snap while dragging"
/// preference should change this helper (and the step constants), not add
/// modifier checks to individual axis/plane/ring branches.
fn snap_requested(modifiers: egui::Modifiers) -> bool {
    modifiers.shift
}

fn snap(v: f64, step: f64, free: bool) -> f64 {
    if free || step <= 0.0 { v } else { (v / step).round() * step }
}

/// A 3×3 rotation (rows) about a unit axis.
fn rotation(axis: Vec3, angle: f64) -> [[f64; 3]; 3] {
    let (c, s) = (angle.cos(), angle.sin());
    let t = 1.0 - c;
    let (x, y, z) = (axis.x, axis.y, axis.z);
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

/// The turn three angles about X, Y and Z make, as the product in `order`: the
/// rightmost goes first. Read the other way round, each turns about the axes
/// the ones left of it leave.
fn turn_of(a: [f64; 3], order: [usize; 3]) -> [[f64; 3]; 3] {
    let none = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    order.iter().fold(none, |m, i| mul(&m, &rotation(DIRS[(*i).min(2)], a[(*i).min(2)])))
}

/// An angle within one turn of zero, keeping its sign; a full turn is 0.
fn within_turn(a: f64) -> f64 {
    let r = a % std::f64::consts::TAU;
    if std::f64::consts::TAU - r.abs() < 1e-6 { 0.0 } else { r }
}

/// The order that lets a drag on `ring` add to its own angle, so every angle
/// stays what was dragged. The rings hold still on the axes the triad shows,
/// so a ring's turn goes on top of the turns made: its angle has to sit at the
/// left end of the product. A ring whose angle is still zero can go there; `None`
/// when it has an angle and a later turn lies on top of it, and no sum of
/// angles makes the turn: the turns so far have to be kept and its angle start
/// again.
fn ring_order(angles: [f64; 3], order: [usize; 3], ring: usize) -> Option<[usize; 3]> {
    let i = ring.min(2);
    let at = order.iter().position(|o| *o == i)?;
    if order[..at].iter().all(|o| angles[(*o).min(2)].abs() < 1e-12) {
        return Some(order);
    }
    if angles[i].abs() >= 1e-12 {
        return None;
    }
    let mut rest = order.into_iter().filter(|o| *o != i);
    let [a, b] = [rest.next()?, rest.next()?];
    Some([i, a, b])
}

/// The XYZ angles (radians) that make the same turn as `m`, in the node's
/// fixed [`ANGLE_ORDER`]. Half a turn about X or Y has only `rx + rz` to show;
/// keep `rz` at zero there.
fn euler_xyz(m: &[[f64; 3]; 3]) -> [f64; 3] {
    let ry = (-m[2][0]).clamp(-1.0, 1.0).asin();
    if m[2][1].abs() + m[2][2].abs() < 1e-9 {
        [(-m[1][2]).atan2(m[1][1]), ry, 0.0]
    } else {
        [m[2][1].atan2(m[2][2]), ry, m[1][0].atan2(m[0][0])]
    }
}

/// The XYZ angles (degrees) after a turn of `angle` radians about `axis`, put
/// on top of the `start` angles the drag began with. `changed` says which
/// fields differ, so a ring that adds to its own angle still writes one field.
///
/// A world-space ring is left-multiplied: it turns about the parent axis the
/// triad shows. A local-space ring is right-multiplied: the triad's axes
/// already include the node's turn, so the ring shows where its own local axis
/// has ended up.
fn turned_angles(start: [f64; 3], axis: usize, angle: f64, local_axes: bool) -> ([f64; 3], [bool; 3]) {
    let i = axis.min(2);
    let start_rad = start.map(f64::to_radians);
    // Rx is the rightmost angle in the node's product, so a local X ring can
    // always add to its own field.
    let direct = if local_axes {
        i == 0
    } else {
        ring_order(start_rad, ANGLE_ORDER, i) == Some(ANGLE_ORDER)
    };
    if direct {
        // The ring can take its turn at the end the product turns on: add to
        // its own angle, within one turn.
        let mut next = start_rad;
        next[i] = within_turn(next[i] + angle);
        let mut changed = [false; 3];
        changed[i] = true;
        return (next.map(f64::to_degrees), changed);
    }
    // Keep the turns made and put this ring's turn on top, then read the
    // equivalent XYZ angles our node evaluates.
    let turn = rotation(DIRS[i], angle);
    let start_matrix = turn_of(start_rad, ANGLE_ORDER);
    let matrix = if local_axes {
        mul(&start_matrix, &turn)
    } else {
        mul(&turn, &start_matrix)
    };
    let next = euler_xyz(&matrix).map(|a| within_turn(a).to_degrees());
    let mut changed = [false; 3];
    for (index, changed) in changed.iter_mut().enumerate() {
        *changed = (next[index] - start_rad[index].to_degrees()).abs() > 1e-9;
    }
    (next, changed)
}

/// Draw the triad and run a drag on it. `step` is the distance snap step.
fn solid_show(ui: &mut egui::Ui, painter: &egui::Painter, proj: &Proj, id: egui::Id, t: &Triad, step: f64) -> Frame {
    let palette = crate::theme::get();
    let accent = palette.color(palette.accent);
    let axis_colours = [
        palette.color(palette.axis_x),
        palette.color(palette.axis_y),
        palette.color(palette.axis_z),
    ];
    let Some(l) = mm_per_px(proj, t.center).map(|scale| scale * LEN_PX) else {
        return Frame { drag: None, clicked: None };
    };
    let Some(sh) = shapes(proj, t, l) else {
        return Frame { drag: None, clicked: None };
    };
    let state: Option<State> = ui.data(|d| d.get_temp(id));
    let pointer = ui.input(|i| i.pointer.hover_pos());
    let primary_down = ui.input(|i| i.pointer.primary_down());
    let primary_pressed = ui.input(|i| i.pointer.primary_pressed());
    let hovered = if state.is_none() { pointer.filter(|_| ui.rect_contains_pointer(proj.rect)).and_then(|p| hit(&sh, p)) } else { None };
    let active = state.map(|s| s.handle).or(hovered);
    let hot = |h: Handle| active == Some(h);
    let col = |h: Handle, i: usize| if hot(h) { accent } else { axis_colours[i.min(2)] };
    for (i, pts) in &sh.rings {
        painter.add(Shape::line(pts.clone(), Stroke::new(if hot(Handle::Ring(*i)) { 3.0 } else { 1.8 }, col(Handle::Ring(*i), *i))));
    }
    for (i, q) in &sh.planes {
        let c = col(Handle::Plane(*i), *i);
        painter.extend(projected_polygon(q, c.gamma_multiply(if hot(Handle::Plane(*i)) { 0.7 } else { 0.35 }), Stroke::new(1.0, c)));
    }
    for (i, a, b) in &sh.arrows {
        arrow(painter, *a, *b, col(Handle::Axis(*i), *i), hot(Handle::Axis(*i)));
    }
    let bc = if hot(Handle::Ball) { accent } else { Color32::from_gray(235) };
    painter.circle(sh.ball, 6.0, bc, Stroke::new(1.5, Color32::from_gray(60)));
    // The handle under the press itself, owned as `hovered` is (this layer,
    // inside the view): a press and release in one frame can end with the
    // pointer somewhere else, even off the triad.
    let pressed = || {
        let at = ui.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::PointerButton { button: egui::PointerButton::Primary, pressed: true, pos, .. } => Some(*pos),
                _ => None,
            })
        })?;
        let owned = ui.clip_rect().intersect(proj.rect).contains(at) && ui.ctx().layer_id_at(at) == Some(ui.layer_id());
        owned.then(|| hit(&sh, at)).flatten()
    };
    let pressed_handle = pressed();
    let Some(p) = pointer else {
        // The pointer can vanish (release outside the window). Drop the drag
        // state as well, so its return cannot finish the old gesture from a
        // stale grab point.
        if state.is_some() {
            ui.data_mut(|data| data.remove::<State>(id));
        }
        return Frame { drag: None, clicked: if state.is_none() { pressed_handle } else { None } };
    };
    if active.is_none() {
        return Frame { drag: None, clicked: if state.is_none() { pressed_handle } else { None } };
    }
    // A gizmo is a direct pointer tool, not a general overlay. Use the pointer
    // state directly instead of creating a drag-sensing widget: a widget kept
    // alive on hover steals primary drags from the viewport, while creating it
    // only on the press frame loses egui's drag-start transition.
    if state.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }
    // Continuous by default so the gizmo follows the pointer smoothly. Shift
    // snaps using the step constants above; see `snap_requested`.
    let free = !ui.input(|i| snap_requested(i.modifiers));
    let view = proj.ray(sh.ball).1;
    let plane_normal = |axes: &[Vec3; 3], h: Handle| match h {
        Handle::Plane(i) | Handle::Ring(i) => axes[i],
        _ => view,
    };
    let mut started = false;
    let mut st = match state {
        Some(s) => s,
        None => {
            // A press and release that arrive in one frame never begins a
            // drag; it is a click on the handle under the press.
            if !primary_pressed || !primary_down {
                return Frame { drag: None, clicked: pressed_handle };
            }
            let Some(h) = hovered else {
                return Frame { drag: None, clicked: None };
            };
            let from = ui.input(|i| i.pointer.press_origin()).unwrap_or(p);
            let c = t.center;
            let mut s = State {
                handle: h,
                center: c,
                axes: t.axes,
                grab: c,
                t0: 0.0,
                last: 0.0,
                turned: 0.0,
                started: false,
            };
            match h {
                Handle::Axis(i) => {
                    let Some(t0) = along(proj, from, c, s.axes[i]) else {
                        return Frame { drag: None, clicked: None };
                    };
                    s.t0 = t0;
                }
                Handle::Ring(i) => {
                    let Some(q) = on_plane(proj, from, c, s.axes[i]).map(|q| q - c) else {
                        return Frame { drag: None, clicked: None };
                    };
                    let (e1, e2) = ring_basis(&s.axes, i);
                    s.last = q.dot(e2).atan2(q.dot(e1));
                }
                _ => {
                    let Some(grab) = on_plane(proj, from, c, plane_normal(&s.axes, h)) else {
                        return Frame { drag: None, clicked: None };
                    };
                    s.grab = grab;
                }
            }
            s
        }
    };
    let done = state.is_some() && !primary_down;
    // A press alone is still a click candidate. Report the drag only once the
    // pointer has moved far enough that egui no longer calls it a click, so a
    // click starts no preview and writes no value.
    if !st.started && !done && ui.input(|i| i.pointer.is_decidedly_dragging()) {
        st.started = true;
        started = true;
    }
    // The frame's events can hold the release and further movement — the hand
    // keeps moving as the button goes up, or input queued during a busy frame.
    // The release event carries where the button was let go (#8265): take the
    // final change from there, not from where the mouse has since got to.
    let p = if done {
        ui.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::PointerButton { button: egui::PointerButton::Primary, pressed: false, pos, .. } => Some(*pos),
                _ => None,
            })
        })
        .unwrap_or(p)
    } else {
        p
    };
    // A release without a drag is a click: select the field the handle drives,
    // move nothing. `primary_clicked` is egui's own click rule (time and
    // distance from the press), so it matches what the rest of the UI calls a
    // click.
    let clicked = done && ui.input(|i| i.pointer.primary_clicked());
    let change = if clicked {
        Change::Translate(Vec3::ZERO)
    } else {
        match st.handle {
            Handle::Axis(i) => {
                let v = along(proj, p, st.center, st.axes[i]).map(|v| v - st.t0).unwrap_or(0.0);
                Change::Translate(st.axes[i] * snap(v, step, free))
            }
            Handle::Ring(i) => {
                if let Some(q) = on_plane(proj, p, st.center, st.axes[i]).map(|q| q - st.center) {
                    let (e1, e2) = ring_basis(&st.axes, i);
                    let th = q.dot(e2).atan2(q.dot(e1));
                    let mut d = th - st.last;
                    while d > std::f64::consts::PI {
                        d -= std::f64::consts::TAU;
                    }
                    while d < -std::f64::consts::PI {
                        d += std::f64::consts::TAU;
                    }
                    st.turned += d;
                    st.last = th;
                }
                Change::Rotate { axis: i, angle: snap(st.turned, ROTATE_SNAP_STEP, free) }
            }
            h => {
                let d = on_plane(proj, p, st.center, plane_normal(&st.axes, h)).map(|q| q - st.grab).unwrap_or(Vec3::ZERO);
                Change::Translate(Vec3::new(snap(d.x, step, free), snap(d.y, step, free), snap(d.z, step, free)))
            }
        }
    };
    let clicked = clicked.then_some(st.handle);
    if done {
        ui.data_mut(|d| d.remove::<State>(id));
    } else {
        ui.data_mut(|d| d.insert_temp(id, st));
    }
    Frame { drag: Some(Drag { handle: st.handle, change, started, done }), clicked }
}

fn arrow(painter: &egui::Painter, a: Pos2, b: Pos2, col: Color32, hot: bool) {
    let v = b - a;
    let len = v.length();
    if len < 1.0 {
        return;
    }
    let u = v / len;
    let n = vec2(-u.y, u.x);
    painter.line_segment([a, b], Stroke::new(if hot { 3.0 } else { 2.0 }, col));
    let head = 12.0f32.min(len * 0.5);
    painter.add(Shape::convex_polygon(vec![b + u * 2.0, b - u * head + n * head * 0.4, b - u * head - n * head * 0.4], col, Stroke::NONE));
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapping_is_opt_in_through_one_modifier_policy() {
        assert!(!snap_requested(egui::Modifiers::default()));
        assert!(snap_requested(egui::Modifiers { shift: true, ..Default::default() }));
    }

    #[test]
    fn translation_snaps_to_whole_units_unless_free() {
        assert_eq!(snap(1.4, 1.0, false), 1.0);
        assert_eq!(snap(1.6, 1.0, false), 2.0);
        assert_eq!(snap(1.4, 1.0, true), 1.4);
        assert_eq!(snap(7.4, 5.0, false), 5.0);
    }

    #[test]
    fn gizmo_state_resets_its_drag_snapshot() {
        let mut state = GizmoState::default();
        state.node_id = Some("translate".into());
        state.start_values = Some([1.0, 2.0, 3.0]);
        state.start_rotation = Some([4.0, 5.0, 6.0]);
        state.dragging = true;
        state.reset();
        assert!(state.node_id.is_none());
        assert!(state.start_values.is_none());
        assert!(state.start_rotation.is_none());
        assert!(!state.is_dragging());
    }

    fn screen_rect() -> Rect {
        Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0))
    }

    fn test_proj() -> Proj {
        Proj { camera: Camera::default(), rect: screen_rect(), radius: 100.0, shift: [0.0, 0.0] }
    }

    fn test_triad() -> Triad {
        Triad {
            center: Vec3::ZERO,
            translate: true,
            rotate: true,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            local_axes: false,
        }
    }

    fn test_gizmo() -> Gizmo {
        Gizmo {
            node_id: "translate".into(),
            pivot: [0.0; 3],
            x: 0.0,
            y: 0.0,
            z: 0.0,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
            world_to_local: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            local_axes: false,
        }
    }

    fn arrow_length(proj: &Proj) -> f64 {
        mm_per_px(proj, Vec3::ZERO).expect("centre projects") * LEN_PX
    }

    fn pointer_at(p: Pos2) -> egui::Event {
        egui::Event::PointerMoved(p)
    }

    fn button_at(p: Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos: p,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        }
    }

    fn frame(
        ctx: &egui::Context,
        events: Vec<egui::Event>,
        gizmo: &mut Gizmo,
        state: &mut GizmoState,
    ) -> GizmoUpdate {
        let rect = screen_rect();
        let camera = Camera::default();
        let mut update = GizmoUpdate::default();
        let mut output = ctx.run_ui(
            egui::RawInput { screen_rect: Some(rect), events, ..Default::default() },
            |ui| {
                update = show(ui, rect, &camera, [0.0, 0.0], 100.0, gizmo, state);
            },
        );
        output.textures_delta.clear();
        update
    }

    fn value(update: &GizmoUpdate, field: &str) -> f64 {
        update
            .changes
            .iter()
            .find(|(name, _)| name == field)
            .map(|(_, value)| *value)
            .unwrap_or_else(|| panic!("no {field} change in {:?}", update.changes))
    }

    fn close(actual: f64, expected: f64) {
        // The gizmo projects through f32 screen positions, so a round-trip
        // world distance carries a few millionths of a millimetre.
        assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
    }

    /// A press and release on an arrow with no drag focuses that axis' field
    /// and writes nothing, whether the events share a frame or not.
    #[test]
    fn a_click_on_an_arrow_focuses_its_field_without_moving() {
        let proj = test_proj();
        let triad = test_triad();
        let l = arrow_length(&proj);
        for (axis, field) in [(0, "x"), (1, "y"), (2, "z")] {
            let at = proj
                .to_screen(Vec3::ZERO + DIRS[axis] * (0.7 * l))
                .expect("arrow point projects");

            // Press and release in one frame (input queued during a busy
            // frame arrives together).
            let mut ctx = egui::Context::default();
            let mut gizmo = test_gizmo();
            let mut state = GizmoState::default();
            let update = frame(
                &ctx,
                vec![pointer_at(at), button_at(at, true), button_at(at, false)],
                &mut gizmo,
                &mut state,
            );
            assert_eq!(update.focus.as_deref(), Some(field), "one-frame click on {field}");
            assert!(update.changes.is_empty(), "a click on {field} moves nothing");
            assert!(!update.started && !update.dragging && !update.ended);

            // Press and release in frames of their own.
            let mut ctx = egui::Context::default();
            let mut gizmo = test_gizmo();
            let mut state = GizmoState::default();
            let press = frame(&ctx, vec![pointer_at(at), button_at(at, true)], &mut gizmo, &mut state);
            assert!(!press.started && !press.dragging, "a press alone is not a drag on {field}");
            let release = frame(&ctx, vec![button_at(at, false)], &mut gizmo, &mut state);
            assert_eq!(release.focus.as_deref(), Some(field), "click on {field}");
            assert!(release.changes.is_empty(), "a click on {field} moves nothing");
            assert!(!release.ended, "a click never started a drag on {field}");

            // The pointer can end somewhere else before the release is read.
            let mut ctx = egui::Context::default();
            let mut gizmo = test_gizmo();
            let mut state = GizmoState::default();
            let away = proj
                .to_screen(Vec3::new(20.0, 20.0, 20.0))
                .expect("away point projects");
            let update = frame(
                &ctx,
                vec![pointer_at(at), button_at(at, true), button_at(at, false), pointer_at(away)],
                &mut gizmo,
                &mut state,
            );
            assert_eq!(update.focus.as_deref(), Some(field), "off-triad click on {field}");
            assert!(update.changes.is_empty());
        }
    }

    /// A click on a ring focuses its angle field; a plane square drives several
    /// fields, so it leaves the focus as it was.
    #[test]
    fn a_click_on_a_ring_focuses_its_angle() {
        let proj = test_proj();
        let triad = test_triad();
        let grip = ring_grip(&proj, &triad, 2).expect("ring z has a free grip");
        let l = arrow_length(&proj);
        let (e1, e2) = ring_basis(&triad.axes, 2);
        let th = grip.to_radians();
        let at = proj
            .to_screen(Vec3::ZERO + (e1 * th.cos() + e2 * th.sin()) * (RING * l))
            .expect("ring point projects");
        let mut ctx = egui::Context::default();
        let mut gizmo = test_gizmo();
        let mut state = GizmoState::default();
        let update = frame(
            &ctx,
            vec![pointer_at(at), button_at(at, true), button_at(at, false)],
            &mut gizmo,
            &mut state,
        );
        assert_eq!(update.focus.as_deref(), Some("rz"));
        assert!(update.changes.is_empty());
    }

    /// A plane square moves along two axes, so a click on it selects no single
    /// value.
    #[test]
    fn a_click_on_a_plane_leaves_the_focus_unset() {
        let proj = test_proj();
        let l = arrow_length(&proj);
        let at = proj
            .to_screen(Vec3::X * (0.35 * l) + Vec3::Y * (0.35 * l))
            .expect("plane point projects");
        let mut ctx = egui::Context::default();
        let mut gizmo = test_gizmo();
        let mut state = GizmoState::default();
        let update = frame(
            &ctx,
            vec![pointer_at(at), button_at(at, true), button_at(at, false)],
            &mut gizmo,
            &mut state,
        );
        assert_eq!(update.focus, None);
        assert!(update.changes.is_empty());
    }

    /// The viewport's pick blocker uses the same hit test as the triad.
    #[test]
    fn contains_pointer_matches_the_handles() {
        let proj = test_proj();
        let gizmo = test_gizmo();
        let l = arrow_length(&proj);
        let handle = proj
            .to_screen(Vec3::X * (0.7 * l))
            .expect("arrow point projects");
        assert!(contains_pointer(
            &gizmo,
            &proj.camera,
            proj.rect,
            proj.shift,
            proj.radius,
            handle
        ));
        let empty = proj
            .to_screen(Vec3::new(-0.4 * l, -0.4 * l, 0.0))
            .expect("empty point projects");
        assert!(!contains_pointer(
            &gizmo,
            &proj.camera,
            proj.rect,
            proj.shift,
            proj.radius,
            empty
        ));
    }

    /// Two rings can lie almost on top of each other on screen: the nearer
    /// takes the press, not whichever was pushed into `rings` first.
    #[test]
    fn the_nearest_ring_takes_the_press() {
        let p = Pos2::new(100.0, 100.0);
        let sh = Shapes {
            arrows: Vec::new(),
            planes: Vec::new(),
            rings: vec![
                (0, vec![Pos2::new(80.0, 103.0), Pos2::new(120.0, 103.0)]),
                (1, vec![Pos2::new(80.0, 101.0), Pos2::new(120.0, 101.0)]),
            ],
            ball: Pos2::new(500.0, 500.0),
        };
        assert_eq!(hit(&sh, p), Some(Handle::Ring(1)));
    }

    /// A ring's drag turns about the axis it shows: the new turn goes on top of
    /// the angles the drag started with, whichever fields that needs.
    #[test]
    fn a_drag_turns_about_the_ring_shown() {
        let on = |a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]| {
            a.iter().flatten().zip(b.iter().flatten()).all(|(a, b)| (a - b).abs() < 1e-9)
        };
        for (start, axis) in [
            ([30.0, 0.0, 0.0], 0),
            ([0.0, 0.0, 90.0], 2),
            ([20.0, 40.0, 60.0], 2),
            ([0.0, 40.0, 60.0], 1),
            ([30.0, 0.0, 60.0], 0),
            ([20.0, 90.0, 0.0], 0),
        ] {
            let (next, _) = turned_angles(start, axis, 0.4, false);
            let start_rad = start.map(f64::to_radians);
            let want = mul(&rotation(DIRS[axis], 0.4), &turn_of(start_rad, ANGLE_ORDER));
            assert!(
                on(&turn_of(next.map(f64::to_radians), ANGLE_ORDER), &want),
                "{start:?} ring {axis} -> {next:?}"
            );
        }
        // The default order is the Translate node's: Rz * Ry * Rx.
        let (rx, ry, rz) = (0.3, -0.7, 1.1);
        let zyx = mul(
            &mul(&rotation(DIRS[2], rz), &rotation(DIRS[1], ry)),
            &rotation(DIRS[0], rx),
        );
        assert!(on(&turn_of([rx, ry, rz], ANGLE_ORDER), &zyx));
    }

    /// In local space the drawn ring includes the node's own turn, so the drag
    /// turns about that intrinsic axis: the new turn is right-multiplied.
    #[test]
    fn a_local_ring_turns_about_its_own_axis() {
        let on = |a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]| {
            a.iter().flatten().zip(b.iter().flatten()).all(|(a, b)| (a - b).abs() < 1e-9)
        };
        for (start, axis) in [
            ([30.0, 0.0, 0.0], 0),
            ([20.0, 40.0, 60.0], 0),
            ([20.0, 40.0, 60.0], 1),
            ([20.0, 40.0, 60.0], 2),
        ] {
            let (next, _) = turned_angles(start, axis, 0.4, true);
            let want = mul(
                &turn_of(start.map(f64::to_radians), ANGLE_ORDER),
                &rotation(DIRS[axis], 0.4),
            );
            assert!(
                on(&turn_of(next.map(f64::to_radians), ANGLE_ORDER), &want),
                "{start:?} local ring {axis} -> {next:?}"
            );
        }
    }

    /// The plane squares stay inside the triad's arrow length in every view.
    #[test]
    fn plane_squares_stay_within_the_triad() {
        let rect = screen_rect();
        let triad = Triad { rotate: false, ..test_triad() };
        let limit = (0.45 * 2f64.sqrt() * LEN_PX) as f32 + 3.0;
        let mut worst = 0.0f32;
        for back in [
            Vec3::X,
            -Vec3::X,
            Vec3::Y,
            -Vec3::Y,
            Vec3::Z,
            -Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 2.0, 3.0),
        ] {
            let camera = Camera::default().looking_from(back);
            let proj = Proj { camera, rect, radius: 100.0, shift: [0.0, 0.0] };
            let l = mm_per_px(&proj, triad.center).expect("centre projects") * LEN_PX;
            let sh = shapes(&proj, &triad, l).expect("triad shapes");
            for (_, quad) in &sh.planes {
                for (&a, &b) in quad.edges() {
                    worst = worst.max(a.distance(sh.ball)).max(b.distance(sh.ball));
                }
            }
        }
        assert!(worst <= limit, "a square vertex {worst} px from the centre (limit {limit})");
    }

    /// A dragged angle stays within one turn, and a full turn reads 0.
    #[test]
    fn a_dragged_angle_stays_within_one_turn() {
        let tau = std::f64::consts::TAU;
        for (a, want) in [(tau, 0.0), (-tau, 0.0), (tau + 0.3, 0.3), (-tau - 0.3, -0.3), (1.0, 1.0), (-1.0, -1.0)] {
            assert!((within_turn(a) - want).abs() < 1e-12, "{a} -> {}", within_turn(a));
        }
        let (next, _) = turned_angles([0.0, 0.0, 0.0], 2, tau, false);
        assert_eq!(next[2], 0.0, "a full turn reads 0");
    }

    /// The second angle dragged reads what was dragged, whichever came first.
    /// Only going back to an angle with a later one on top starts it over.
    #[test]
    fn only_going_back_under_a_later_turn_starts_an_angle_over() {
        for (first, second) in [(0, 1), (1, 0), (2, 1), (1, 2), (0, 2), (2, 0)] {
            let mut a = [0.0; 3];
            a[first] = 1.5;
            let order = ring_order(a, ANGLE_ORDER, second).unwrap();
            a[second] = 0.4;
            assert!(ring_order(a, order, second).is_some(), "the last ring turned goes on adding");
            assert!(ring_order(a, order, first).is_none(), "{first} lies under {second}");
        }
    }

    /// A release ends the drag where the button was let go, not where the
    /// pointer got to in the same frame.
    #[test]
    fn a_release_uses_the_release_position() {
        let proj = test_proj();
        let l = arrow_length(&proj);
        let start = proj.to_screen(Vec3::X * (0.5 * l)).expect("start projects");
        let dragged = proj.to_screen(Vec3::X * (0.9 * l)).expect("dragged projects");
        let released = proj.to_screen(Vec3::X * (0.7 * l)).expect("release projects");
        let after = proj.to_screen(Vec3::X * (0.95 * l)).expect("after projects");
        let mut ctx = egui::Context::default();
        let mut gizmo = test_gizmo();
        let mut state = GizmoState::default();
        let press = frame(&ctx, vec![pointer_at(start), button_at(start, true)], &mut gizmo, &mut state);
        assert!(!press.started);
        let drag = frame(&ctx, vec![pointer_at(dragged)], &mut gizmo, &mut state);
        assert!(drag.started && drag.dragging && drag.changes.is_empty());
        let release = frame(
            &ctx,
            vec![button_at(released, false), pointer_at(after)],
            &mut gizmo,
            &mut state,
        );
        assert!(release.ended);
        assert_eq!(release.focus, None, "a real drag never focuses a field");
        close(value(&release, "x"), 0.2 * l);
    }

    /// Losing the pointer cancels the gesture: its return must not finish the
    /// drag from the stale grab point and jump the value.
    #[test]
    fn losing_the_pointer_cancels_the_drag() {
        let proj = test_proj();
        let l = arrow_length(&proj);
        let start = proj.to_screen(Vec3::X * (0.5 * l)).expect("start projects");
        let end = proj.to_screen(Vec3::X * (0.9 * l)).expect("end projects");
        let mut ctx = egui::Context::default();
        let mut gizmo = test_gizmo();
        let mut state = GizmoState::default();
        assert!(!frame(&ctx, vec![pointer_at(start), button_at(start, true)], &mut gizmo, &mut state).started);
        assert!(frame(&ctx, vec![pointer_at(end)], &mut gizmo, &mut state).started);
        let cancel = frame(&ctx, vec![egui::Event::PointerGone], &mut gizmo, &mut state);
        assert!(cancel.ended, "the preview is retired");
        assert!(cancel.changes.is_empty());
        let pointer_state = egui::Id::new(("solidcraft-transform-gizmo", gizmo.node_id.as_str()));
        assert!(
            ctx.data(|data| data.get_temp::<State>(pointer_state)).is_none(),
            "the lost pointer's drag state is dropped"
        );
        // The button is still down when the pointer comes back, as after
        // dragging outside the window.
        let back = frame(
            &ctx,
            vec![pointer_at(end), button_at(end, true)],
            &mut gizmo,
            &mut state,
        );
        assert!(back.changes.is_empty(), "no stale grab point writes on return");
        assert!(!back.dragging && !back.ended);
    }

    /// Translate along an axis: the app still receives absolute x/y/z fields.
    #[test]
    fn an_axis_drag_keeps_the_translate_mapping() {
        let proj = test_proj();
        let l = arrow_length(&proj);
        let start = proj.to_screen(Vec3::X * (0.5 * l)).expect("start projects");
        let end = proj.to_screen(Vec3::X * (0.9 * l)).expect("end projects");
        let mut ctx = egui::Context::default();
        let mut gizmo = test_gizmo();
        let mut state = GizmoState::default();
        let press = frame(&ctx, vec![pointer_at(start), button_at(start, true)], &mut gizmo, &mut state);
        assert!(!press.started);
        let moved = frame(&ctx, vec![pointer_at(end)], &mut gizmo, &mut state);
        assert!(moved.started && moved.changes.is_empty(), "the first drag frame only starts");
        let update = frame(&ctx, vec![button_at(end, false)], &mut gizmo, &mut state);
        assert!(update.ended && !update.dragging);
        assert_eq!(update.focus, None);
        close(value(&update, "x"), 0.4 * l);
        close(value(&update, "y"), 0.0);
        close(value(&update, "z"), 0.0);
        assert_eq!(update.changes.len(), 3, "Translate still receives all three fields");
    }

    /// A plane square: the app still receives the world move mapped through
    /// `world_to_local` as absolute x/y/z fields.
    #[test]
    fn a_plane_drag_keeps_the_translate_mapping() {
        let proj = test_proj();
        let l = arrow_length(&proj);
        let start_world = Vec3::X * (0.35 * l) + Vec3::Y * (0.35 * l);
        let end_world = start_world + Vec3::new(5.0, 3.0, 0.0);
        let start = proj.to_screen(start_world).expect("start projects");
        let end = proj.to_screen(end_world).expect("end projects");
        let mut ctx = egui::Context::default();
        let mut gizmo = test_gizmo();
        let mut state = GizmoState::default();
        let press = frame(&ctx, vec![pointer_at(start), button_at(start, true)], &mut gizmo, &mut state);
        assert!(!press.started);
        let moved = frame(&ctx, vec![pointer_at(end)], &mut gizmo, &mut state);
        assert!(moved.started && moved.changes.is_empty());
        let update = frame(&ctx, vec![button_at(end, false)], &mut gizmo, &mut state);
        assert_eq!(update.focus, None);
        close(value(&update, "x"), 5.0);
        close(value(&update, "y"), 3.0);
        close(value(&update, "z"), 0.0);
    }

    /// A ring drag: the app still receives absolute rotation fields, one field
    /// when the ring can add to its own angle.
    #[test]
    fn a_ring_drag_keeps_the_rotate_mapping() {
        let proj = test_proj();
        let triad = test_triad();
        let l = arrow_length(&proj);
        let grip = ring_grip(&proj, &triad, 2).expect("ring z has a free grip");
        let (e1, e2) = ring_basis(&triad.axes, 2);
        let at = |degrees: f64| {
            let th = degrees.to_radians();
            proj.to_screen(Vec3::ZERO + (e1 * th.cos() + e2 * th.sin()) * (RING * l))
                .expect("ring point projects")
        };
        let start = at(grip);
        let end = at(grip + 30.0);
        let mut ctx = egui::Context::default();
        let mut gizmo = test_gizmo();
        let mut state = GizmoState::default();
        let press = frame(&ctx, vec![pointer_at(start), button_at(start, true)], &mut gizmo, &mut state);
        assert!(!press.started);
        let moved = frame(&ctx, vec![pointer_at(end)], &mut gizmo, &mut state);
        assert!(moved.started && moved.changes.is_empty());
        let update = frame(&ctx, vec![button_at(end, false)], &mut gizmo, &mut state);
        assert_eq!(update.focus, None);
        assert_eq!(update.changes.len(), 1, "a free ring writes its own field: {:?}", update.changes);
        close(value(&update, "rz"), 30.0);
    }
}
