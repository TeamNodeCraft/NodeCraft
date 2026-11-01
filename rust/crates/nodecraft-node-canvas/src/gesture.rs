//! Trackpad gesture routing shared by the node editor and 3D viewport.
//!
//! winit reports two-finger trackpad movement as pixel `MouseWheel` events and a
//! mouse wheel as line events. Pinch/ctrl-scroll is exposed separately by egui as
//! a zoom delta, so we can keep mouse wheels zooming while making two-finger XY
//! movement pan.

use egui::{Pos2, Vec2};

#[derive(Default, Clone, Debug)]
pub struct TrackpadGesture {
    /// Two-finger XY movement, in points.
    pub pan: Vec2,
    /// Mouse-wheel movement, in lines.
    pub wheel: Vec2,
    /// Pinch/ctrl-scroll zoom factor (1.0 = no zoom).
    pub zoom: f32,
    pub shift: bool,
    pub pointer: Option<Pos2>,
}

impl TrackpadGesture {
    pub fn read(ctx: &egui::Context) -> Self {
        ctx.input(|input| {
            let mut pan = Vec2::ZERO;
            let mut wheel = Vec2::ZERO;
            for event in &input.events {
                if let egui::Event::MouseWheel { unit, delta, .. } = event {
                    match unit {
                        egui::MouseWheelUnit::Point | egui::MouseWheelUnit::Page => pan += *delta,
                        egui::MouseWheelUnit::Line => wheel += *delta,
                    }
                }
            }
            Self {
                pan,
                wheel,
                zoom: input.zoom_delta(),
                shift: input.modifiers.shift,
                pointer: input.pointer.latest_pos(),
            }
        })
    }

    pub fn has_navigation(&self) -> bool {
        self.pan != Vec2::ZERO || self.wheel != Vec2::ZERO || (self.zoom - 1.0).abs() > f32::EPSILON
    }
}
