//! Continuous cable-routing factors.
//!
//! Cable length and bottom-to-top rise are normalised 0..1 factors rather than
//! hard direction booleans. Every routing consumer (tessellation, hit testing,
//! CPU fallback) uses this one function.

use egui::Pos2;

pub struct WireRouting {
    pub length_factor: f32,
    pub rise_factor: f32,
    pub control1: Pos2,
    pub control2: Pos2,
}

pub const MIN_HANDLE: f32 = 14.0;
pub const MAX_HANDLE: f32 = 140.0;
pub const MAX_UP_HANDLE: f32 = 220.0;
pub const LENGTH_SCALE: f32 = 360.0;
pub const UP_BEND_GAIN: f32 = 1.4;

pub fn route(x0: f32, y0: f32, x1: f32, y1: f32, scale: f32) -> WireRouting {
    let scale = scale.max(0.0001);
    let dx = x1 - x0;
    let dy = y1 - y0;
    let distance = dx.hypot(dy);
    let length_factor = (distance / (LENGTH_SCALE * scale)).clamp(0.0, 1.0);
    let rise_factor = ((y0 - y1).max(0.0) / distance.max(0.0001)).clamp(0.0, 1.0);
    let base = MIN_HANDLE + (MAX_HANDLE - MIN_HANDLE) * length_factor;
    let handle = (base * (1.0 + UP_BEND_GAIN * rise_factor) * scale)
        .clamp(MIN_HANDLE * scale, MAX_UP_HANDLE * scale);
    WireRouting {
        length_factor,
        rise_factor,
        control1: Pos2::new(x0, y0 + handle),
        control2: Pos2::new(x1, y1 - handle),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_factors_are_normalized_and_continuous() {
        let down = route(0.0, 0.0, 100.0, 300.0, 1.0);
        let up = route(0.0, 300.0, 100.0, 0.0, 1.0);
        for value in [down.length_factor, down.rise_factor, up.length_factor, up.rise_factor] {
            assert!((0.0..=1.0).contains(&value), "factor out of range: {value}");
        }
        assert!(up.rise_factor > down.rise_factor);
        assert!(up.control1.y > 300.0);
        assert!(up.control2.y < 0.0);
        assert!(down.control1.y > 0.0);
        assert!(down.control2.y < 300.0);
    }

    #[test]
    fn longer_and_more_vertical_cables_get_larger_handles() {
        let short = route(0.0, 0.0, 0.0, 20.0, 1.0);
        let long = route(0.0, 0.0, 0.0, 300.0, 1.0);
        assert!(long.length_factor > short.length_factor);
        assert!(long.control1.y > short.control1.y);
    }
}
