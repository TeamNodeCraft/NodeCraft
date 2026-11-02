//! The transform triad: X/Y/Z arrows, XY/YZ/XZ plane squares, X/Y/Z
//! rotation rings and a centre ball. Adapted from SolidCraft's
//! `crates/ui-egui/src/gizmo.rs` (MIT OR Apache-2.0).

use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, vec2};
use solidcraft_geom::{Ring, Vec3};
use solidcraft_render::Camera;

/// World ↔ screen mapping for the current viewport frame.
#[derive(Clone, Copy)]
struct Proj {
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
    pub local: bool,
    /// Linear inverse of the downstream chain, mapping world deltas back to
    /// this node's parent frame.
    pub world_to_local: [[f64; 3]; 3],
    /// World directions of the node's local X/Y/Z axes.
    pub axes: [[f64; 3]; 3],
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

/// Draw the SolidCraft triad and map its drag changes onto the Translate node.
pub fn show(
    ui: &mut egui::Ui,
    rect: Rect,
    camera: &Camera,
    shift: [f32; 2],
    radius: f64,
    gizmo: &mut Gizmo,
    state: &mut GizmoState,
) -> Vec<(String, f64)> {
    if state.node_id.as_deref() != Some(gizmo.node_id.as_str()) {
        state.node_id = Some(gizmo.node_id.clone());
        state.start_values = None;
        state.start_rotation = None;
        state.dragging = false;
    }
    let proj = Proj { camera: *camera, rect, radius, shift };
    let triad = Triad {
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
    };
    let id = egui::Id::new(("solidcraft-transform-gizmo", gizmo.node_id.as_str()));
    let painter = ui.painter().with_clip_rect(rect);
    let Some(drag) = solid_show(ui, &painter, &proj, id, &triad, 1.0) else {
        state.dragging = false;
        return Vec::new();
    };
    if drag.started || state.start_values.is_none() {
        state.start_values = Some([gizmo.x, gizmo.y, gizmo.z]);
        state.start_rotation = Some([gizmo.rx, gizmo.ry, gizmo.rz]);
    }
    state.dragging = !drag.done;
    let mut changes = Vec::new();
    let values = state.start_values.unwrap_or([gizmo.x, gizmo.y, gizmo.z]);
    let rotations = state.start_rotation.unwrap_or([gizmo.rx, gizmo.ry, gizmo.rz]);
    if !drag.started {
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
                let field = ["rx", "ry", "rz"][axis.min(2)];
                let index = axis.min(2);
                let value = rotations[index] + angle.to_degrees();
                match index {
                    0 => gizmo.rx = value,
                    1 => gizmo.ry = value,
                    _ => gizmo.rz = value,
                }
                changes.push((field.to_owned(), value));
            }
        }
    }
    if drag.done {
        state.start_values = None;
        state.start_rotation = None;
        state.dragging = false;
    }
    changes
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

/// Where the triad is and which handles it shows.
#[derive(Clone, Copy, Debug)]
pub struct Triad {
    pub center: Vec3,
    pub translate: bool,
    pub rotate: bool,
    pub axes: [Vec3; 3],
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
}

const DIRS: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
/// Arrow length on screen (px).
pub const LEN_PX: f64 = 110.0;

/// How near a handle a press still takes it (px): the widest, the centre ball.
pub const HIT_TOLERANCE: f32 = 9.0;

/// Ring radius as a share of the arrow length.
const RING: f64 = 0.62;

/// In-plane directions of the ring about an axis (right-handed: e1 × e2 = axis).
fn ring_basis(axes: &[Vec3; 3], i: usize) -> (Vec3, Vec3) {
    (axes[(i + 1) % 3], axes[(i + 2) % 3])
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
    s.rings.iter().find(|(_, pts)| pts.windows(2).any(|w| seg_dist(p, w[0], w[1]) < 5.0)).map(|(i, _)| Handle::Ring(*i))
}

/// Millimetres per screen pixel at a point.
fn mm_per_px(proj: &Proj, c: Vec3) -> Option<f64> {
    let (_, view) = proj.ray(proj.to_screen(c)?);
    let u = view.any_perp().normalized()?;
    let (a, b) = (proj.to_screen(c)?, proj.to_screen(c + u)?);
    let px = f64::from(a.distance(b));
    (px > 1e-9).then(|| 1.0 / px)
}

fn snap(v: f64, step: f64, free: bool) -> f64 {
    if free || step <= 0.0 { v } else { (v / step).round() * step }
}

/// Draw the triad and run a drag on it. `step` is the distance snap step.
fn solid_show(ui: &mut egui::Ui, painter: &egui::Painter, proj: &Proj, id: egui::Id, t: &Triad, step: f64) -> Option<Drag> {
    let palette = crate::theme::get();
    let accent = palette.color(palette.accent);
    let axis_colours = [
        palette.color(palette.axis_x),
        palette.color(palette.axis_y),
        palette.color(palette.axis_z),
    ];
    let l = mm_per_px(proj, t.center)? * LEN_PX;
    let sh = shapes(proj, t, l)?;
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
    let p = pointer?;
    active?;
    // A gizmo is a direct pointer tool, not a general overlay. Use the pointer
    // state directly instead of creating a drag-sensing widget: a widget kept
    // alive on hover steals primary drags from the viewport, while creating it
    // only on the press frame loses egui's drag-start transition.
    if state.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }
    let free = ui.input(|i| i.modifiers.alt || i.modifiers.command || i.modifiers.ctrl);
    let view = proj.ray(sh.ball).1;
    let plane_normal = |axes: &[Vec3; 3], h: Handle| match h {
        Handle::Plane(i) | Handle::Ring(i) => axes[i],
        _ => view,
    };
    let mut started = false;
    let mut st = match state {
        Some(s) => s,
        None => {
            if !primary_pressed {
                return None;
            }
            let h = hovered?;
            let from = ui.input(|i| i.pointer.press_origin()).unwrap_or(p);
            let c = t.center;
            let mut s = State { handle: h, center: c, axes: t.axes, grab: c, t0: 0.0, last: 0.0, turned: 0.0 };
            match h {
                Handle::Axis(i) => s.t0 = along(proj, from, c, s.axes[i])?,
                Handle::Ring(i) => {
                    let q = on_plane(proj, from, c, s.axes[i])? - c;
                    let (e1, e2) = ring_basis(&s.axes, i);
                    s.last = q.dot(e2).atan2(q.dot(e1));
                }
                _ => s.grab = on_plane(proj, from, c, plane_normal(&s.axes, h))?,
            }
            started = true;
            s
        }
    };
    let change = match st.handle {
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
            Change::Rotate { axis: i, angle: snap(st.turned, 5f64.to_radians(), free) }
        }
        h => {
            let d = on_plane(proj, p, st.center, plane_normal(&st.axes, h)).map(|q| q - st.grab).unwrap_or(Vec3::ZERO);
            Change::Translate(Vec3::new(snap(d.x, step, free), snap(d.y, step, free), snap(d.z, step, free)))
        }
    };
    let done = state.is_some() && !primary_down;
    if done {
        ui.data_mut(|d| d.remove::<State>(id));
    } else {
        ui.data_mut(|d| d.insert_temp(id, st));
    }
    Some(Drag { handle: st.handle, change, started, done })
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
}
