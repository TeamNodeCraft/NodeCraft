//! The top-right Axes view navigator: six signed world axes drawn as the
//! camera poses them, and a drag anywhere on it to orbit.
//! Adapted from SolidCraft's `crates/ui-kit/src/view_compass.rs`,
//! `crates/look/src/compass.rs`, the Axes branch of
//! `crates/ui-egui/src/viewport.rs` and the view direction maths of
//! `crates/engine/src/cmd/view_navigation.rs`, synced with SolidCraft c2ffd48e
//! (MIT OR Apache-2.0).
//!
//! NodeCraft's camera, placement, lens shift and command path stay ours: this
//! module draws and hit-tests one signed-axes layout, and returns the camera
//! change for the viewport to apply.

use egui::{Color32, FontId, Pos2, Rect, Stroke, pos2, vec2};
use solidcore_record::Vec3;
use solidcraft_geom::Vec3Geom;
use solidcraft_viewport::Camera;

/// How far a tip is from the centre, before [`ART_SCALE`].
pub(crate) const AXIS_LENGTH: f64 = 38.0;
/// The artwork sits inside the pointer reach so the compass keeps breathing
/// room in its circle.
pub(crate) const ART_SCALE: f64 = 0.84;
/// The compass circle's radius: anywhere inside drags the view.
pub(crate) const RADIUS: f32 = 49.5;
/// The pointer target around a compass end: larger than the drawn disc.
pub(crate) const AXIS_RADIUS: f32 = 15.0;
/// The reserved top-right box. SolidCraft's Axes navigator uses 116 points.
pub(crate) const AXIS_VIEW_SIZE: f32 = 116.0;
/// The compass sits a little higher in its reserved box.
pub(crate) const AXIS_CENTER_LIFT: f32 = 4.0;
/// NodeCraft's viewport edge padding.
pub(crate) const MARGIN: f32 = 14.0;
/// SolidCraft's compass lettering size.
const COMPASS_FONT: f32 = 13.5;

/// The six signed world axes, by name.
pub(crate) const AXES: [(&str, [f64; 3]); 6] = [
    ("x", [1.0, 0.0, 0.0]),
    ("-x", [-1.0, 0.0, 0.0]),
    ("y", [0.0, 1.0, 0.0]),
    ("-y", [0.0, -1.0, 0.0]),
    ("z", [0.0, 0.0, 1.0]),
    ("-z", [0.0, 0.0, -1.0]),
];

/// The camera's view axes in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ViewAxes {
    pub right: [f64; 3],
    pub up: [f64; 3],
    /// Toward the viewer.
    pub back: [f64; 3],
}

impl ViewAxes {
    /// World direction `d` as the camera poses it: its parts along the
    /// screen's right and down, and its depth toward the viewer.
    pub(crate) fn pose(&self, d: [f64; 3]) -> ([f64; 2], f64) {
        ([dot(d, self.right), -dot(d, self.up)], dot(d, self.back))
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The camera's view axes, as the compass takes them.
pub(crate) fn view_axes(cam: &Camera) -> ViewAxes {
    let (right, up, back) = cam.basis();
    ViewAxes { right: right.into(), up: up.into(), back: back.into() }
}

/// One signed axis as drawn: its name (`"x"`, `"-x"`, ...), world direction,
/// tip on screen and depth along the camera's back.
pub(crate) type Tip = (&'static str, [f64; 3], Pos2, f64);

/// What the user chose on the compass: a signed axis tip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ViewAction {
    /// Look along an axis: a tip was clicked.
    Snap { axis: &'static str, direction: [f64; 3] },
}

/// One frame's step of a drag on a view navigator.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct OrbitStep {
    pub dx: f64,
    pub dy: f64,
    /// Shift was held (SolidCraft's turntable snap hint).
    pub snap: bool,
}

/// Where a drag on the navigator stands between frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Drag {
    /// The pointer at the last step.
    pub at: Option<Pos2>,
    /// Where the drag started on the navigator, for the release warp.
    pub start: Option<Pos2>,
    /// A pending cursor warp (the window edge).
    pub warp: Option<Pos2>,
    /// The press passed the drag threshold and is turning the view. The
    /// cursor is hidden iff this is set; it is part of the one drag state.
    pub moving: bool,
}

/// One frame of a drag: whether it started/ended, whether it is turning the
/// view, and this frame's step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Orbit {
    pub grabbed: bool,
    pub released: bool,
    /// The drag ended without a release (the pointer was lost or the window
    /// lost focus).
    pub cancelled: bool,
    pub orbiting: bool,
    /// The drag was turning the view before this frame ended, so a hidden
    /// cursor must be shown again even if the release frame has no hover.
    pub restored: bool,
    pub step: Option<OrbitStep>,
}

/// Where the navigator is: the reserved box and the compass centre.
/// Drawing and hit-testing both go through this layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Layout {
    pub outer: Rect,
    pub centre: Pos2,
}

/// The navigator's top-right placement in `viewport`, keeping NodeCraft's
/// viewport margin and right-hand chrome inset.
pub(crate) fn layout(viewport: Rect, right_inset: f32) -> Layout {
    let outer = Rect::from_min_size(
        pos2(viewport.right() - AXIS_VIEW_SIZE - MARGIN - right_inset, viewport.top() + MARGIN),
        vec2(AXIS_VIEW_SIZE, AXIS_VIEW_SIZE),
    );
    let centre = outer.center() - vec2(0.0, AXIS_CENTER_LIFT);
    Layout { outer, centre }
}

/// The whole reserved box takes navigator input, as SolidCraft's Axes drag
/// start does; this is also what keeps a navigator press from picking the
/// model behind it.
pub(crate) fn contains(layout: &Layout, p: Pos2) -> bool {
    layout.outer.contains(p)
}

/// Where a tip is drawn from the centre, for a direction posed on screen as
/// `[right, down]`.
pub(crate) fn tip_offset([x, y]: [f64; 2]) -> egui::Vec2 {
    vec2((x * AXIS_LENGTH * ART_SCALE) as f32, (y * AXIS_LENGTH * ART_SCALE) as f32)
}

/// The six signed axes as `axes` poses them around `c`, back to front so
/// overlapping tips have one owner.
pub(crate) fn tips(axes: &ViewAxes, c: Pos2) -> Vec<Tip> {
    let mut tips: Vec<Tip> = AXES
        .into_iter()
        .map(|(name, d)| {
            let (screen, depth) = axes.pose(d);
            (name, d, c + tip_offset(screen), depth)
        })
        .collect();
    tips.sort_by(|a, b| a.3.total_cmp(&b.3));
    tips
}

/// The front-most tip within reach of `p`.
pub(crate) fn hit(tips: &[Tip], p: Pos2) -> Option<usize> {
    tips.iter()
        .enumerate()
        .rev()
        .find(|(_, (_, _, at, _))| at.distance(p) <= AXIS_RADIUS)
        .map(|(i, _)| i)
}

/// One frame of the compass.
#[derive(Clone, Debug)]
pub(crate) struct Compass {
    /// The area it may paint in.
    pub clip: Rect,
    pub centre: Pos2,
    pub axes: ViewAxes,
    /// The pointer, when it may act on the compass.
    pub hover: Option<Pos2>,
    /// A drag on it is turning the view (no tip lights while it does).
    pub orbiting: bool,
}

/// Draw the compass; a tip clicked this frame.
pub(crate) fn show(ui: &mut egui::Ui, compass: &Compass) -> Option<ViewAction> {
    let theme = crate::theme::get();
    let c = compass.centre;
    let tips = tips(&compass.axes, c);
    let hover = compass.hover;
    let hovered = if compass.orbiting { None } else { hover.and_then(|p| hit(&tips, p)) };
    let painter = ui.painter_at(compass.clip);
    let mut action = None;
    // The circle turns fully opaque under the pointer.
    let in_circle = hover.is_some_and(|p| p.distance(c) <= RADIUS);
    let fill = if in_circle || compass.orbiting { 220 } else { 90 };
    painter.circle(
        c,
        RADIUS,
        theme.with_alpha(theme.window, fill),
        Stroke::new(1.0, theme.with_alpha(theme.border, 191)),
    );
    let light = theme.panel[0] > 128;
    for (i, (name, d, at, depth)) in tips.iter().enumerate() {
        if i == 3 {
            // The origin dot sits behind the three front axes and over the
            // three back ones.
            painter.circle(
                c,
                3.6,
                theme.with_alpha(theme.text, 230),
                Stroke::new(1.4, theme.color(theme.panel)),
            );
        }
        axis_tip(&painter, c, name, *d, *at, *depth, hovered == Some(i), &theme, light);
    }
    if let Some(i) = hovered {
        if let Some((name, d, _, _)) = tips.get(i) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            if ui.input(|i| i.pointer.primary_clicked()) {
                action = Some(ViewAction::Snap { axis: name, direction: *d });
            }
        }
    }

    if compass.orbiting {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if in_circle && hovered.is_none() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }
    action
}

/// One signed axis end. Adapted from SolidCraft's `look` compass painter.
fn axis_tip(
    painter: &egui::Painter,
    c: Pos2,
    name: &str,
    d: [f64; 3],
    at: Pos2,
    depth: f64,
    hovered: bool,
    theme: &crate::theme::Theme,
    light: bool,
) {
    let col = if d[0] != 0.0 {
        theme.color(theme.axis_x)
    } else if d[1] != 0.0 {
        theme.color(theme.axis_y)
    } else {
        theme.color(theme.axis_z)
    };
    let positive = !name.starts_with('-');
    // Positive shafts are the full axis colour; negative ones are a dimmed
    // tint (and any end pointing away from the viewer dims further).
    let ink = if !positive || depth < -0.01 { col.gamma_multiply(0.55) } else { col };
    let radius = if positive { 11.0 } else { 4.8 };
    let direction = (at - c).normalized();
    let width = 3.6;
    // A negative shaft stops at the outer edge of its ring; a positive one
    // runs under its disc.
    let end = if positive { at - direction * radius } else { at - direction * (radius + 1.2) };
    painter.line_segment([c, end], Stroke::new(width, ink));
    // The round cap sits at the origin; the other end is covered by the disc
    // or ring.
    painter.circle_filled(c, width / 2.0, ink);
    if positive {
        painter.circle_filled(at, radius, col);
    }
    // Negative tips are hollow: neither the axis shaft nor a fill covers their
    // centres, so the translucent compass background shows through. A hovered
    // tip of either kind turns fully white.
    if hovered {
        painter.circle_filled(at, if positive { radius } else { radius + 1.2 }, Color32::WHITE);
        // On the light theme a white disc vanishes into the compass, so it
        // carries an outline in the axis colour, drawn inside the disc so it
        // does not grow.
        if light && positive {
            painter.circle_stroke(at, radius - 1.0, Stroke::new(2.0, col));
        }
    }
    if positive {
        let letter = if hovered { col } else { Color32::WHITE };
        let label = name.to_uppercase();
        let galley = painter.layout_no_wrap(label, FontId::proportional(COMPASS_FONT), letter);
        let origin = at - galley.size() / 2.0;
        let ppp = painter.ctx().pixels_per_point().max(0.5);
        let origin = pos2((origin.x * ppp).round() / ppp, (origin.y * ppp).round() / ppp);
        painter.galley(origin, galley, letter);
    }
    if !positive {
        painter.circle_stroke(
            at,
            radius,
            Stroke::new(
                2.4,
                if !hovered {
                    ink
                } else if light {
                    col
                } else {
                    Color32::WHITE
                },
            ),
        );
    }
}

/// Track a drag on the navigator. A press may start one where `starts_at`
/// says the navigator is; once the drag is decided each frame's motion is an
/// [`OrbitStep`].
pub(crate) fn orbit(
    ui: &egui::Ui,
    drag: &mut Drag,
    may_start: bool,
    starts_at: impl Fn(Pos2) -> bool,
) -> Orbit {
    let mut out = Orbit::default();
    let (pressed, down, decided, origin, pos, latest, shift, focused) = ui.input(|i| {
        let p = &i.pointer;
        (
            p.primary_pressed(),
            p.primary_down(),
            p.is_decidedly_dragging(),
            p.press_origin(),
            p.interact_pos(),
            p.latest_pos(),
            i.modifiers.shift,
            i.focused,
        )
    });
    if pressed && may_start {
        drag.at = origin.filter(|p| starts_at(*p));
        drag.start = drag.at;
        drag.warp = None;
        drag.moving = false;
        out.grabbed = drag.at.is_some();
    }
    // A lost pointer or a window that loses focus ends a held drag without a
    // release event. The pointer must become visible again; there is nowhere
    // trustworthy to warp it, so only the cursor is restored.
    let lost = drag.at.is_some() && (!focused || latest.is_none());
    if !down && drag.at.take().is_some() {
        let was_moving = std::mem::take(&mut drag.moving);
        let start = drag.start.take();
        drag.warp = None;
        out.released = true;
        out.restored = was_moving;
        // A press that only clicked a tip never moved, so it neither hid nor
        // warps the pointer. A real orbit drag is restored to the compass
        // point where it began.
        if was_moving && let Some(start) = start {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorPosition(start));
        }
    } else if lost && drag.at.take().is_some() {
        let was_moving = std::mem::take(&mut drag.moving);
        drag.start = None;
        drag.warp = None;
        out.cancelled = true;
        out.restored = was_moving;
    }
    let Some(last) = drag.at.filter(|_| decided) else {
        return out;
    };
    drag.moving = true;
    out.orbiting = true;
    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    if let Some(p) = pos {
        // While a warp is pending the pointer may still sit at the edge it
        // hit (the OS has not echoed the warp, or cannot), have landed at the
        // centre (the echo), or have moved past it (the echo plus a fast
        // flick): measure each step from whichever reference the pointer is
        // nearer to, so the view never jumps by the warp distance and a
        // platform that ignores warps keeps orbiting from the real pointer
        // instead of freezing.
        let from = match drag.warp {
            Some(w) if p.distance(w) <= p.distance(last) => {
                drag.warp = None;
                w
            }
            _ => last,
        };
        out.step = Some(OrbitStep {
            dx: f64::from(p.x - from.x),
            dy: f64::from(p.y - from.y),
            snap: shift,
        });
        drag.at = Some(p);
        // The OS stops feeding movement once the cursor is pinned at the
        // screen edge, so an orbit toward that edge would stall: warp the
        // cursor back to the window centre and measure the next steps from
        // there, keeping the orbit tracking while the button is held (#9943).
        let screen = ui.ctx().viewport_rect();
        let edge = 6.0;
        if drag.warp.is_none()
            && (p.x <= screen.min.x + edge
                || p.x >= screen.max.x - edge
                || p.y <= screen.min.y + edge
                || p.y >= screen.max.y - edge)
        {
            let home = screen.center();
            drag.warp = Some(home);
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorPosition(home));
        }
        ui.ctx().request_repaint();
    }
    out
}

/// The camera looking along a signed axis tip, keeping its target, distance
/// and up axis. This is the direction half of `view.navigate`.
pub(crate) fn snap(from: &Camera, direction: [f64; 3]) -> Camera {
    let d = Vec3::new(direction[0], direction[1], direction[2]);
    let frame = from.to_frame(d);
    // A tip that points at the viewer looks from the other side, as
    // `view.navigate` and the old compass both do.
    let dir = if from.back().dot(d) > 0.9999 { -d } else { d };
    let mut to = from.looking_from(dir);
    // At the poles the yaw is degenerate: keep a square heading, nearest to
    // the current one.
    if frame.z.abs() > 0.5 {
        to.yaw = nearest_quarter_turn(from.yaw);
    }
    to
}

/// At a halfway tie, prefer the square heading nearer the standard front
/// view.
pub(crate) fn nearest_quarter_turn(yaw: f64) -> f64 {
    if !yaw.is_finite() {
        return 0.0;
    }
    let q = std::f64::consts::FRAC_PI_2;
    let lo = (yaw / q).floor() * q;
    let hi = lo + q;
    let (dlo, dhi) = (yaw - lo, hi - yaw);
    if (dlo - dhi).abs() < 1e-6 {
        let off_front = |a: f64| {
            ((a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI)
                .abs()
        };
        if off_front(lo) <= off_front(hi) { lo } else { hi }
    } else if dlo < dhi {
        lo
    } else {
        hi
    }
}

/// The orbit and roll arrows' actions from `view.navigate`. NodeCraft's Axes
/// navigator has no arrow controls, so nothing dispatches these yet; the maths
/// is ported and tested so one can be added without re-deriving it.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NavigationAction {
    Left,
    Right,
    Up,
    Down,
    Clockwise,
    Counterclockwise,
}

/// A quarter-turn orbit or roll, as `view.navigate` does it.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn navigate(from: &Camera, action: NavigationAction) -> Camera {
    use std::f64::consts::{FRAC_PI_2, TAU};
    let (r, u, _b) = from.basis();
    match action {
        NavigationAction::Left => from.looking_from(-r),
        NavigationAction::Right => from.looking_from(r),
        NavigationAction::Up => from.looking_from(u),
        NavigationAction::Down => from.looking_from(-u),
        NavigationAction::Clockwise => {
            let mut to = *from;
            to.roll = (to.roll + FRAC_PI_2).rem_euclid(TAU);
            to
        }
        NavigationAction::Counterclockwise => {
            let mut to = *from;
            to.roll = (to.roll - FRAC_PI_2).rem_euclid(TAU);
            to
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, RawInput, pos2, vec2};
    use solidcraft_geom::Vec3Geom;

    const SCREEN: Rect = Rect { min: Pos2::ZERO, max: Pos2::new(400.0, 400.0) };

    const FRONT: ViewAxes =
        ViewAxes { right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0] };

    fn context() -> egui::Context {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        ctx
    }

    fn pointer_at(p: Pos2) -> Event {
        Event::PointerMoved(p)
    }

    fn button_at(p: Pos2, pressed: bool) -> Event {
        Event::PointerButton {
            pos: p,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        }
    }

    fn run_axes(
        ctx: &egui::Context,
        events: Vec<Event>,
        hover: Option<Pos2>,
        drag: &mut Drag,
        axes: ViewAxes,
    ) -> (Option<ViewAction>, Orbit) {
        let mut out = (None, Orbit::default());
        let mut output = ctx.run_ui(
            RawInput { screen_rect: Some(SCREEN), events, focused: true, ..Default::default() },
            |ui| {
                let orbit =
                    super::orbit(ui, drag, true, |p| p.distance(pos2(200.0, 200.0)) <= RADIUS);
                let compass = Compass {
                    clip: ui.max_rect(),
                    centre: pos2(200.0, 200.0),
                    axes,
                    hover,
                    orbiting: orbit.orbiting,
                };
                out = (show(ui, &compass), orbit);
            },
        );
        output.textures_delta.clear();
        out
    }

    fn run(
        ctx: &egui::Context,
        events: Vec<Event>,
        hover: Option<Pos2>,
        drag: &mut Drag,
    ) -> (Option<ViewAction>, Orbit) {
        run_axes(ctx, events, hover, drag, FRONT)
    }

    fn click_axes(ctx: &egui::Context, axes: ViewAxes, at: Pos2) -> Option<ViewAction> {
        let mut drag = Drag::default();
        run_axes(ctx, vec![pointer_at(at)], Some(at), &mut drag, axes);
        let pressed = run_axes(ctx, vec![button_at(at, true)], Some(at), &mut drag, axes).0;
        pressed.or(run_axes(ctx, vec![button_at(at, false)], Some(at), &mut drag, axes).0)
    }

    fn click(ctx: &egui::Context, at: Pos2) -> Option<ViewAction> {
        click_axes(ctx, FRONT, at)
    }

    fn tip_pos(axes: &ViewAxes, name: &str) -> Pos2 {
        tips(axes, pos2(200.0, 200.0))
            .into_iter()
            .find(|tip| tip.0 == name)
            .map(|tip| tip.2)
            .unwrap_or_else(|| panic!("no {name} tip"))
    }

    fn front_camera() -> Camera {
        Camera { yaw: 0.0, pitch: 0.0, roll: 0.0, ..Camera::default() }
    }

    fn assert_back(cam: &Camera, expected: Vec3) {
        assert!(cam.back().dist(expected) < 2e-5, "{:?} != {expected:?}", cam.back());
    }

    #[test]
    fn every_signed_tip_returns_its_view_and_lands_on_that_axis() {
        // Home's three-quarter view keeps all six ends apart on screen and
        // none of them pointing straight at the viewer.
        let from = Camera::default();
        let axes = view_axes(&from);
        for (name, direction) in AXES {
            let ctx = context();
            let at = tip_pos(&axes, name);
            let action = click_axes(&ctx, axes, at);
            assert_eq!(
                action,
                Some(ViewAction::Snap { axis: name, direction }),
                "{name} at {at:?}"
            );
            let to = snap(&from, direction);
            assert_back(&to, Vec3::new(direction[0], direction[1], direction[2]));
        }
    }

    #[test]
    fn the_compass_has_no_home_hit_area() {
        let ctx = context();
        let old_chip = pos2(200.0, 200.0) + vec2(-37.0, -37.0);
        assert_eq!(click(&ctx, old_chip), None, "the compass only returns axis tips now");
    }

    #[test]
    fn vertical_tips_keep_the_nearest_square_heading() {
        let from = Camera { yaw: 0.2, ..front_camera() };
        let top = snap(&from, [0.0, 0.0, 1.0]);
        assert!((top.yaw - 0.0).abs() < 1e-12, "top yaw {}", top.yaw);
        assert!((top.pitch - std::f64::consts::FRAC_PI_2).abs() < 2e-6);

        let from = Camera { yaw: 0.3, ..front_camera() };
        let top = snap(&from, [0.0, 0.0, 1.0]);
        assert!((top.yaw - nearest_quarter_turn(0.3)).abs() < 1e-12);

        let from = Camera { yaw: 3.0, ..front_camera() };
        let top = snap(&from, [0.0, 0.0, 1.0]);
        assert!((top.yaw - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn rapid_tip_clicks_start_from_the_previous_target() {
        // NodeCraft applies view changes at once, so the second click starts
        // from the camera the first click produced (SolidCraft uses a glide's
        // target here).
        let mut cam = snap(&Camera::default(), [1.0, 0.0, 0.0]);
        assert_back(&cam, Vec3::X);
        cam = snap(&cam, [0.0, 0.0, 1.0]);
        assert_back(&cam, Vec3::Z);
        cam = snap(&cam, [0.0, -1.0, 0.0]);
        assert_back(&cam, -Vec3::Y);
    }

    #[test]
    fn roll_steps_are_upstreams_quarter_turn() {
        let cam = front_camera();
        let clockwise = navigate(&cam, NavigationAction::Clockwise);
        assert!((clockwise.roll - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_back(&clockwise, -Vec3::Y);
        assert_eq!(clockwise.yaw, cam.yaw);
        assert_eq!(clockwise.pitch, cam.pitch);

        let ccw = navigate(&clockwise, NavigationAction::Counterclockwise);
        assert!(ccw.roll.abs() < 1e-12, "one step back to the start: {}", ccw.roll);
    }

    #[test]
    fn arrow_actions_turn_ninety_degrees() {
        let cam = front_camera();
        let (r, u, _b) = cam.basis();
        for (action, expected) in [
            (NavigationAction::Left, -r),
            (NavigationAction::Right, r),
            (NavigationAction::Up, u),
            (NavigationAction::Down, -u),
        ] {
            let to = navigate(&cam, action);
            assert_back(&to, expected);
            assert_eq!(
                (to.target, to.distance, to.fov, to.roll),
                (cam.target, cam.distance, cam.fov, cam.roll)
            );
        }
    }

    #[test]
    fn tips_are_drawn_where_the_hit_test_finds_them() {
        let front = tips(&FRONT, pos2(200.0, 200.0));
        let x = front.iter().find(|tip| tip.0 == "x").unwrap().2;
        assert_eq!(x, pos2(200.0 + (AXIS_LENGTH * ART_SCALE) as f32, 200.0), "X points right");
        let z = front.iter().find(|tip| tip.0 == "z").unwrap().2;
        assert_eq!(z, pos2(200.0, 200.0 - (AXIS_LENGTH * ART_SCALE) as f32), "Z points up");

        // A three-quarter view keeps the six ends apart, so each one's drawn
        // position is also the one the hit test reports.
        let from = Camera::default();
        let tips = tips(&view_axes(&from), pos2(200.0, 200.0));
        for (name, _, at, _) in &tips {
            let expected = tips.iter().position(|tip| tip.0 == *name).unwrap();
            assert_eq!(hit(&tips, *at), Some(expected), "{name}");
            assert!(hit(&tips, *at + vec2(AXIS_RADIUS - 0.1, 0.0)).is_some(), "{name}");
            assert_eq!(hit(&tips, *at + vec2(AXIS_RADIUS + 0.1, 0.0)), None, "{name}");
        }
    }

    #[test]
    fn the_front_most_tip_owns_an_overlap() {
        // Looking at the front, Y and -Y both project to the origin; -Y is
        // nearer the viewer and takes the hit.
        let tips = tips(&FRONT, pos2(200.0, 200.0));
        assert_eq!(tips[0].0, "y");
        assert_eq!(tips[5].0, "-y");
        assert_eq!(hit(&tips, pos2(200.0, 200.0)), Some(5));
    }

    #[test]
    fn a_drag_returns_orbit_steps_and_ends_on_release() {
        let ctx = context();
        let mut drag = Drag::default();
        let start = pos2(180.0, 215.0);
        run(&ctx, vec![pointer_at(start)], Some(start), &mut drag);
        let (_, pressed) = run(&ctx, vec![button_at(start, true)], Some(start), &mut drag);
        assert!(pressed.grabbed && pressed.step.is_none());
        assert_eq!(drag.start, Some(start));
        assert!(!drag.moving, "a press alone has not passed the drag threshold");
        let mut steps = Vec::new();
        for x in [190.0, 200.0, 230.0] {
            let at = pos2(x, 205.0);
            let (action, orbit) = run(&ctx, vec![pointer_at(at)], Some(at), &mut drag);
            assert_eq!(action, None, "dragging over a tip does not snap");
            steps.extend(orbit.step);
        }
        let travel = steps
            .iter()
            .fold((0.0, 0.0, false), |(x, y, held), s| (x + s.dx, y + s.dy, held || s.snap));
        assert!(!steps.is_empty());
        assert_eq!(travel, (50.0, -10.0, false), "the steps add up to the pointer's travel");
        let end = pos2(230.0, 205.0);
        let (_, released) = run(&ctx, vec![button_at(end, false)], Some(end), &mut drag);
        assert!(released.released && drag.at.is_none() && drag.warp.is_none());
    }

    #[test]
    fn orbit_keeps_tracking_after_the_cursor_hits_the_window_edge() {
        let ctx = context();
        let mut drag = Drag::default();
        let start = pos2(200.0, 200.0);
        run(&ctx, vec![pointer_at(start)], Some(start), &mut drag);
        run(&ctx, vec![button_at(start, true)], Some(start), &mut drag);
        let edge = SCREEN.max.x - 2.0;
        let (_, at_edge) =
            run(&ctx, vec![pointer_at(pos2(edge, start.y))], Some(pos2(edge, start.y)), &mut drag);
        assert!(at_edge.orbiting);
        assert_eq!(at_edge.step.map(|s| s.dx), Some(f64::from(edge - start.x)));
        let home = SCREEN.center();
        assert_eq!(drag.warp, Some(home), "the cursor was warped to the window centre");

        // The OS echoes the warp: the pointer reappears at the centre without
        // moving the view.
        let before = drag.at;
        let (_, echo) = run(&ctx, vec![pointer_at(home)], Some(home), &mut drag);
        assert_eq!(echo.step.map(|s| s.dx), Some(0.0));
        assert!(drag.warp.is_none(), "the echo clears the pending warp");
        assert_eq!(drag.at, Some(home));
        assert_ne!(before, drag.at);

        // The user keeps dragging right from the centre: the orbit tracks.
        let (_, moved) = run(
            &ctx,
            vec![pointer_at(home + vec2(30.0, 0.0))],
            Some(home + vec2(30.0, 0.0)),
            &mut drag,
        );
        assert_eq!(moved.step.map(|s| s.dx), Some(30.0));
        let (_, released) = run(&ctx, vec![button_at(home, false)], Some(home), &mut drag);
        assert!(released.released);
        assert!(drag.at.is_none() && drag.warp.is_none());
    }

    #[test]
    fn the_navigator_box_blocks_everything_in_it() {
        let layout = layout(SCREEN, 0.0);
        assert!(contains(&layout, layout.outer.min));
        assert!(contains(&layout, layout.centre));
        assert!(contains(&layout, layout.outer.max));
        assert!(!contains(&layout, layout.outer.min - vec2(1.0, 0.0)));
        assert!(!contains(&layout, layout.outer.max + vec2(1.0, 0.0)));
        // The compass is also exactly on the standard view's axis tips.
        for (_, direction) in AXES {
            let (screen, _) = FRONT.pose(direction);
            let tip = layout.centre + tip_offset(screen);
            assert!(contains(&layout, tip));
        }
    }
}
