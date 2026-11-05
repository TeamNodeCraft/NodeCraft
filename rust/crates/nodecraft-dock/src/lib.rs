//! Shared formatting and interaction for NodeCraft docks.
//!
//! The crate exposes three pieces the docks compose:
//!
//! - [`DockMetrics`] — geometry shared by every dock.
//! - [`DockStyle`] — theme colours and chrome.
//! - [`DockItem`] — one icon row.
//!
//! New dock features should extend these types rather than adding parallel
//! constants or one-off drawing paths.

use egui::{Color32, Painter, Rect, Response, Sense, Stroke, Ui, Vec2};
use nodecraft_node_canvas::squircle;

/// Which way a dock's rows run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockOrientation {
    Vertical,
    Horizontal,
}

/// Geometry shared by every dock.
#[derive(Clone, Copy, Debug)]
pub struct DockMetrics {
    pub orientation: DockOrientation,
    pub width: f32,
    pub icon_size: f32,
    pub row_padding: f32,
    pub gap: f32,
    pub padding: f32,
    /// Gap between a dock and the edge of its panel.
    pub edge_offset: f32,
    pub background_radius: f32,
    pub search_separator: f32,
    pub grip_height: f32,
}

impl DockMetrics {
    pub const DEFAULT: Self = Self {
        orientation: DockOrientation::Vertical,
        width: 44.0,
        icon_size: 19.0,
        row_padding: 8.0,
        gap: 10.0,
        padding: 8.0,
        edge_offset: 12.0,
        background_radius: 20.0,
        search_separator: 5.0,
        grip_height: 24.0,
    };

    /// Size of one square icon row.
    pub const fn row_size(self) -> f32 {
        self.icon_size + self.row_padding * 2.0
    }

    pub const fn with_orientation(mut self, orientation: DockOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Length of the rows themselves, without outer padding.
    pub fn content_length(self, rows: usize) -> f32 {
        if rows == 0 {
            return 0.0;
        }
        rows as f32 * self.row_size() + (rows.saturating_sub(1)) as f32 * self.gap
    }

    /// Content size for `rows`, without outer padding.
    pub fn content_size(self, rows: usize) -> Vec2 {
        let length = self.content_length(rows);
        match self.orientation {
            DockOrientation::Vertical => Vec2::new(self.width, length),
            DockOrientation::Horizontal => Vec2::new(length, self.width),
        }
    }

    /// Full dock size, including outer padding on both axes.
    pub fn dock_length(self, rows: usize) -> f32 {
        self.content_length(rows) + self.padding * 2.0
    }

    pub fn dock_size(self, rows: usize) -> Vec2 {
        let length = self.dock_length(rows);
        match self.orientation {
            DockOrientation::Vertical => Vec2::new(self.width, length),
            DockOrientation::Horizontal => Vec2::new(length, self.width),
        }
    }

    /// Offset from the dock background's corner to its content corner.
    /// Padding is applied along the dock axis only.
    pub fn content_offset(self) -> Vec2 {
        match self.orientation {
            DockOrientation::Vertical => Vec2::new(0.0, self.padding),
            DockOrientation::Horizontal => Vec2::new(self.padding, 0.0),
        }
    }

    fn row_allocation(self) -> Vec2 {
        let row = self.row_size();
        match self.orientation {
            DockOrientation::Vertical => Vec2::new(self.width, row),
            DockOrientation::Horizontal => Vec2::new(row, self.width),
        }
    }
}

/// Colours and chrome shared by every dock.
#[derive(Clone, Copy, Debug)]
pub struct DockStyle {
    pub background: Color32,
    pub border: Color32,
    pub accent: Color32,
    pub text: Color32,
    pub icon: Color32,
    pub icon_hover: Color32,
    pub shadow: Color32,
}


/// One icon row in a dock.
#[derive(Clone, Copy)]
pub struct DockItem<'a> {
    pub label: &'a str,
    pub icon_name: &'a str,
    pub icon: &'static [u8],
    pub active: bool,
}

impl<'a> DockItem<'a> {
    pub fn new(label: &'a str, icon_name: &'a str, icon: &'static [u8]) -> Self {
        Self {
            label,
            icon_name,
            icon,
            active: false,
        }
    }

    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }
}

/// Expand a rendered content rect into the dock background rect.
pub fn background_rect(content: Rect, metrics: DockMetrics) -> Rect {
    // Padding runs along the dock, not across its thickness: icon rows already
    // carry their own cross-axis padding.
    match metrics.orientation {
        DockOrientation::Vertical => Rect::from_min_max(
            egui::Pos2::new(content.left(), content.top() - metrics.padding),
            egui::Pos2::new(content.right(), content.bottom() + metrics.padding),
        ),
        DockOrientation::Horizontal => Rect::from_min_max(
            egui::Pos2::new(content.left() - metrics.padding, content.top()),
            egui::Pos2::new(content.right() + metrics.padding, content.bottom()),
        ),
    }
}

/// The square squircle used for hover and active rows.
pub fn highlight_rect(row: Rect, metrics: DockMetrics) -> Rect {
    Rect::from_center_size(row.center(), Vec2::splat(metrics.row_size()))
}

/// Paint the shared dock background, including its shadow and border.
pub fn paint_background(
    painter: &Painter,
    rect: Rect,
    metrics: DockMetrics,
    style: DockStyle,
) {
    painter.add(
        egui::epaint::Shadow {
            offset: [0, 0],
            blur: 32,
            spread: 0,
            color: style.shadow,
        }
        .as_shape(rect, egui::CornerRadius::same(metrics.background_radius as u8)),
    );
    painter.add(squircle::shape(
        rect,
        metrics.background_radius,
        style.background,
        egui::Stroke::NONE,
    ));
    painter.add(squircle::shape(
        rect,
        metrics.background_radius,
        Color32::TRANSPARENT,
        egui::Stroke::new(1.0, style.border),
    ));
}

/// Allocate and draw one dock row: highlight, themed icon and tooltip.
pub fn icon_button(
    ui: &mut Ui,
    metrics: DockMetrics,
    style: DockStyle,
    item: DockItem<'_>,
) -> Response {
    let (rect, response) = ui.allocate_exact_size(metrics.row_allocation(), Sense::click());
    if item.active || response.hovered() {
        let fill = if item.active {
            style.accent
        } else {
            style.border.gamma_multiply(0.75)
        };
        let painter = ui.painter();
        painter.add(squircle::shape(
            highlight_rect(rect, metrics),
            metrics.background_radius,
            fill,
            Stroke::NONE,
        ));
    }
    let tint = if item.active {
        style.text
    } else if response.hovered() {
        style.icon_hover
    } else {
        style.icon
    };
    let colour = format!("#{:02x}{:02x}{:02x}", tint.r(), tint.g(), tint.b());
    let svg = tint_svg(item.icon, &colour);
    let image = egui::Image::from_bytes(
        format!("bytes://tool-dock-{}.svg", item.icon_name),
        svg.into_bytes(),
    )
    .fit_to_exact_size(Vec2::splat(metrics.icon_size));
    let image_response = ui.put(
        Rect::from_center_size(rect.center(), Vec2::splat(metrics.icon_size)),
        image,
    );
    let response = response.union(image_response).on_hover_text(item.label);
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

fn tint_svg(source: &[u8], colour: &str) -> String {
    let mut svg = std::str::from_utf8(source)
        .unwrap_or_default()
        .replace("#212121", colour)
        .replace("#b4b4b4", colour)
        .replace("#d8d8dc", colour);
    if let Some(root_end) = svg.find('>') {
        let root = &svg[..root_end];
        if !root.contains("fill=") && !root.contains("stroke=") {
            svg.insert_str(root_end, &format!(" fill=\"{colour}\""));
        }
    }
    svg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_metrics_produce_a_square_highlight() {
        let metrics = DockMetrics::DEFAULT;
        assert_eq!(metrics.row_size(), 35.0);
        let row = Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(metrics.width, metrics.row_size()));
        let highlight = highlight_rect(row, metrics);
        let background = background_rect(row, metrics);
        assert_eq!(background.height(), row.height() + metrics.padding * 2.0);
        assert_eq!(metrics.edge_offset, 12.0);
        assert_eq!(highlight.width(), metrics.row_size());
        assert_eq!(highlight.height(), metrics.row_size());
    }

    #[test]
    fn horizontal_swaps_the_dock_axes() {
        let vertical = DockMetrics::DEFAULT;
        let horizontal = vertical.with_orientation(DockOrientation::Horizontal);
        let vertical_size = vertical.dock_size(3);
        let horizontal_size = horizontal.dock_size(3);
        assert_eq!(vertical_size.x, horizontal_size.y);
        assert_eq!(vertical_size.y, horizontal_size.x);
        assert_eq!(horizontal.content_length(3), 3.0 * 35.0 + 2.0 * 10.0);
        assert_eq!(horizontal.dock_length(3), 3.0 * 35.0 + 2.0 * 10.0 + 16.0);
    }

    #[test]
    fn dock_items_are_inactive_by_default() {
        let item = DockItem::new("Box", "box", &[]);
        assert!(!item.active);
        assert!(item.active(true).active);
    }
}
