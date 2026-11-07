//! One answer to "what is the pointer over?" for overlays that use raw input.
//!
//! egui widgets consume their own clicks, but the sketch editor reads pointer
//! positions directly. It needs a single viewport/blocker test instead of each
//! input branch re-deriving `interaction.contains(...) && !over_palette`.

use egui::{Pos2, Rect};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hover {
    /// Pointer is inside the viewport and outside every overlay/panel blocker.
    Viewport,
    /// Pointer is over an overlay, palette or viewport-chrome rectangle.
    Blocker,
    /// Pointer is outside the viewport.
    Outside,
}

pub struct HoverMap<'a> {
    viewport: Rect,
    blockers: &'a [Rect],
    pointer: Option<Pos2>,
}

impl<'a> HoverMap<'a> {
    pub fn new(viewport: Rect, blockers: &'a [Rect], pointer: Option<Pos2>) -> Self {
        Self { viewport, blockers, pointer }
    }

    pub fn hover(&self) -> Hover {
        let Some(pointer) = self.pointer else { return Hover::Outside };
        if !self.viewport.contains(pointer) {
            return Hover::Outside;
        }
        if self.blockers.iter().any(|rect| rect.contains(pointer)) {
            Hover::Blocker
        } else {
            Hover::Viewport
        }
    }


    pub fn over_viewport(&self) -> bool {
        self.hover() == Hover::Viewport
    }

    pub fn over_blocker(&self) -> bool {
        self.hover() == Hover::Blocker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hover_prioritises_blockers_over_the_viewport() {
        let viewport = Rect::from_min_size(Pos2::new(10.0, 20.0), egui::vec2(200.0, 100.0));
        let blocked = Rect::from_min_size(Pos2::new(40.0, 30.0), egui::vec2(20.0, 20.0));
        let blockers = [blocked];
        let map = |x, y| HoverMap::new(viewport, &blockers, Some(Pos2::new(x, y)));
        assert_eq!(map(100.0, 60.0).hover(), Hover::Viewport);
        assert_eq!(map(45.0, 35.0).hover(), Hover::Blocker);
        assert_eq!(map(5.0, 5.0).hover(), Hover::Outside);
        assert_eq!(HoverMap::new(viewport, &blockers, None).hover(), Hover::Outside);
    }
}
