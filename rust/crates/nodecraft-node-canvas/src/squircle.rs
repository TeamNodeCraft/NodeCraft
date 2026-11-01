//! Continuous-curvature rounded rectangles (squircles) used by the app chrome.

use egui::{Color32, Pos2, Rect, Shape, Stroke, epaint::{PathShape, PathStroke}};

/// Append one quarter of a superellipse corner to `points`.
fn corner(center: Pos2, radius: f32, start: f32, points: &mut Vec<Pos2>) {
    const STEPS: usize = 12;
    let exponent = 2.0 / 4.0; // iOS-style squircle; closer to a circle as the exponent rises toward 1.
    for step in 0..=STEPS {
        let angle = start + std::f32::consts::FRAC_PI_2 * step as f32 / STEPS as f32;
        let cos = angle.cos();
        let sin = angle.sin();
        points.push(Pos2::new(
            center.x + radius * cos.signum() * cos.abs().powf(exponent),
            center.y + radius * sin.signum() * sin.abs().powf(exponent),
        ));
    }
}

/// Points for an axis-aligned squircle rounded rectangle.
pub fn points(rect: Rect, radius: f32) -> Vec<Pos2> {
    let radius = radius.max(0.0).min(rect.width() * 0.5).min(rect.height() * 0.5);
    if radius <= f32::EPSILON {
        return vec![rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom()];
    }
    // At half the short dimension the four corner centers collapse onto each
    // other (a tall, narrow lasso is the common case) and a corner-by-corner
    // path self-intersects into an X. Use one continuous superellipse instead.
    if radius * 2.0 >= rect.width().min(rect.height()) - 0.01 {
        const STEPS: usize = 64;
        let center = rect.center();
        let (a, b) = (rect.width() * 0.5, rect.height() * 0.5);
        let exponent = 2.0 / 4.0;
        return (0..STEPS)
            .map(|step| {
                let angle = std::f32::consts::TAU * step as f32 / STEPS as f32;
                let cos = angle.cos();
                let sin = angle.sin();
                Pos2::new(
                    center.x + a * cos.signum() * cos.abs().powf(exponent),
                    center.y + b * sin.signum() * sin.abs().powf(exponent),
                )
            })
            .collect();
    }
    let mut points = Vec::with_capacity(4 * 13 + 4);
    // Start at the top-right tangent and walk clockwise.
    let corners = [
        (Pos2::new(rect.max.x - radius, rect.min.y + radius), -std::f32::consts::FRAC_PI_2),
        (Pos2::new(rect.max.x - radius, rect.max.y - radius), 0.0),
        (Pos2::new(rect.min.x + radius, rect.max.y - radius), std::f32::consts::FRAC_PI_2),
        (Pos2::new(rect.min.x + radius, rect.min.y + radius), std::f32::consts::PI),
    ];
    for (center, start) in corners {
        corner(center, radius, start, &mut points);
    }
    points
}

pub fn shape(rect: Rect, radius: f32, fill: Color32, stroke: Stroke) -> Shape {
    Shape::Path(PathShape {
        points: points(rect, radius),
        closed: true,
        fill,
        stroke: PathStroke::new(stroke.width, stroke.color),
    })
}

/// Paint a texture clipped to a squircle. Used by the floating panel glass.
pub fn image(
    painter: &egui::Painter,
    texture: egui::TextureId,
    rect: Rect,
    uv: Rect,
    radius: f32,
) {
    let points = points(rect, radius);
    if points.len() < 3 {
        return;
    }
    let mut mesh = egui::Mesh::with_texture(texture);
    let centre = rect.center();
    let uv_at = |point: Pos2| {
        egui::pos2(
            uv.min.x + (point.x - rect.min.x) / rect.width().max(1.0) * uv.width(),
            uv.min.y + (point.y - rect.min.y) / rect.height().max(1.0) * uv.height(),
        )
    };
    mesh.vertices.push(egui::epaint::Vertex {
        pos: centre,
        uv: uv_at(centre),
        color: Color32::WHITE,
    });
    for point in &points {
        mesh.vertices.push(egui::epaint::Vertex {
            pos: *point,
            uv: uv_at(*point),
            color: Color32::WHITE,
        });
    }
    for index in 0..points.len() {
        let next = (index + 1) % points.len();
        mesh.add_triangle(0, index as u32 + 1, next as u32 + 1);
    }
    painter.add(Shape::mesh(mesh));
}
