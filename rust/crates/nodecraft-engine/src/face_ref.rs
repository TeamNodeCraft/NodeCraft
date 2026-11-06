//! Face reference parsing for `face:N` and `body:B:face:N` selection ids.

pub(crate) fn face_reference_body_and_index(reference: &str) -> Option<(Option<usize>, usize)> {
    // Viewer selection ids are namespaced by body: `body:0:face:3`; accept both
    // that and the plain `face:3` form.
    let mut body = None;
    let mut face = None;
    let mut parts = reference.split(':');
    while let Some(part) = parts.next() {
        match part {
            "body" => body = parts.next()?.parse().ok(),
            "face" => face = parts.next()?.parse().ok(),
            _ => {}
        }
    }
    Some((body, face?))
}

pub(crate) fn face_reference_index(reference: &str) -> Option<usize> {
    face_reference_body_and_index(reference).map(|(_, face)| face)
}

