//! Face reference parsing and resolution for `face:N` and `body:B:face:N`.
//!
//! One parser and one resolver serve every node that stores the face selector
//! list (`shell`, `offsetfaces`, `extrude`). A saved reference is a position in
//! the body's **record** face order, the same order the viewport picker and the
//! tessellator print as `face:N`. Parsing fails closed: a malformed entry, a
//! missing `id`, a body that is not in the Part, or a face that no body has is
//! an error naming the reference. Silently dropping one would turn a requested
//! opening into a closed feature of the wrong shape.

/// One parsed opening: an optional body scope and a record face index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FaceReference {
    /// `Some(b)` for `body:B:face:N`; `None` for an unscoped `face:N`.
    pub(crate) body: Option<usize>,
    /// The record face index on the body (or on each body for an unscoped one).
    pub(crate) face: usize,
}

impl FaceReference {
    /// The reference as the viewport and the inspector write it.
    pub(crate) fn label(self) -> String {
        match self.body {
            Some(body) => format!("body:{body}:face:{}", self.face),
            None => format!("face:{}", self.face),
        }
    }
}

impl std::fmt::Display for FaceReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Parse one saved id. The grammar is exact: `face:N` or `body:B:face:N`,
/// with `B` and `N` unsigned indices. Everything else is malformed, including
/// a `body:` prefix whose number does not parse: it must not be downgraded to
/// an unscoped face.
pub(crate) fn face_reference_body_and_index(reference: &str) -> Option<(Option<usize>, usize)> {
    let parts: Vec<&str> = reference.split(':').collect();
    match parts.as_slice() {
        ["face", face] => Some((None, face.parse().ok()?)),
        ["body", body, "face", face] => Some((Some(body.parse().ok()?), face.parse().ok()?)),
        _ => None,
    }
}

/// The saved `faces` selector list of a face feature, parsed and validated as
/// data. Absent or empty is an empty list (a closed feature for Shell, a
/// pass-through for Offset Faces).
///
/// Every entry must be an object with a non-empty string `id`. The error names
/// the bad entry or index so the user can reselect it.
pub(crate) fn parse_face_references(
    values: &serde_json::Value,
) -> Result<Vec<FaceReference>, String> {
    let Some(value) = values.get("faces") else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(entries) = value.as_array() else {
        return Err(
            "The faces value must be a list of face references. Reselect the faces to remove."
                .to_owned(),
        );
    };
    let mut references = Vec::with_capacity(entries.len());
    for (entry_index, entry) in entries.iter().enumerate() {
        let id = entry
            .get("id")
            .and_then(serde_json::Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                format!(
                    "Face selection entry {} is not a face reference object with a non-empty string id. Reselect the faces to remove.",
                    entry_index + 1
                )
            })?;
        let (body, face) = face_reference_body_and_index(id).ok_or_else(|| {
            format!(
                "Face reference {id:?} is not valid; expected face:N or body:B:face:N. Reselect the faces to remove."
            )
        })?;
        references.push(FaceReference { body, face });
    }
    Ok(references)
}

/// Resolve parsed references against the bodies they apply to.
///
/// `body_faces[b]` is body `b`'s record face count. The result is one opening
/// list per body, in Part body order, with repeats removed (the same face twice
/// is one opening).
///
/// An unscoped `face:N` applies to every body that has face `N`, as it has
/// since the multi-body shell landed and as the multi-body tests pin; it is an
/// error only when no body has it. A body-scoped `body:B:face:N` applies to
/// body `B` alone and is an error when `B` or face `N` does not exist. A
/// multi-body Part should use body-scoped ids, which the viewport writes.
pub(crate) fn resolve_face_references(
    references: &[FaceReference],
    body_faces: &[usize],
) -> Result<Vec<Vec<usize>>, String> {
    let mut openings = vec![Vec::new(); body_faces.len()];
    for reference in references {
        match reference.body {
            Some(body) => {
                if body >= body_faces.len() {
                    return Err(format!(
                        "Face {} names body {body}, but this Part has {}. Reselect the faces to remove.",
                        reference,
                        counted(body_faces.len(), "body", "bodies")
                    ));
                }
                let count = body_faces[body];
                if reference.face >= count {
                    return Err(format!(
                        "Face {} doesn't exist on body {body} (it has {}). Reselect the faces to remove.",
                        reference,
                        counted(count, "face", "faces")
                    ));
                }
                push_once(&mut openings[body], reference.face);
            }
            None => {
                let mut found = false;
                for (body, count) in body_faces.iter().enumerate() {
                    if reference.face < *count {
                        push_once(&mut openings[body], reference.face);
                        found = true;
                    }
                }
                if !found {
                    return Err(if body_faces.len() == 1 {
                        format!(
                            "Face {} doesn't exist on this body (it has {}). Reselect the faces to remove.",
                            reference,
                            counted(body_faces[0], "face", "faces")
                        )
                    } else {
                        format!(
                            "Face {} doesn't exist on any of the {} in this Part. Reselect the faces to remove.",
                            reference,
                            counted(body_faces.len(), "body", "bodies")
                        )
                    });
                }
            }
        }
    }
    Ok(openings)
}

fn push_once(list: &mut Vec<usize>, face: usize) {
    if !list.contains(&face) {
        list.push(face);
    }
}

/// `1 face` / `6 faces`, so a sentence does not say "1 faces".
fn counted(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 { format!("1 {singular}") } else { format!("{count} {plural}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn references(value: serde_json::Value) -> Result<Vec<FaceReference>, String> {
        parse_face_references(&serde_json::json!({ "faces": value }))
    }

    #[test]
    fn parses_both_reference_forms_exactly() {
        assert_eq!(face_reference_body_and_index("face:2"), Some((None, 2)));
        assert_eq!(face_reference_body_and_index("body:3:face:7"), Some((Some(3), 7)));
        for malformed in [
            "body:x:face:1",
            "body:face:1",
            "body:0:face:x",
            "face",
            "face:",
            ":face:1",
            "face:1:2",
            "body:0:face:1:2",
            "Body:0:face:1",
            "",
        ] {
            assert_eq!(
                face_reference_body_and_index(malformed),
                None,
                "{malformed:?} must not parse"
            );
        }
    }

    #[test]
    fn a_selector_entry_must_be_an_object_with_a_string_id() {
        assert!(references(serde_json::json!({})).is_err());
        assert!(references(serde_json::json!([{ "id": 5 }])).is_err());
        assert!(references(serde_json::json!([{}])).is_err());
        assert!(references(serde_json::json!([{ "id": "" }])).is_err());
        assert!(references(serde_json::json!([{ "id": "body:x:face:1" }])).is_err());
        assert!(references(serde_json::json!([{ "id": "face:0" }])).is_ok());
    }

    #[test]
    fn unscoped_faces_apply_to_every_body_that_has_them_and_are_errors_when_none_do() {
        let refs = references(serde_json::json!([{ "id": "face:1" }])).expect("parsed");
        assert_eq!(
            resolve_face_references(&refs, &[6, 6]).expect("resolved"),
            vec![vec![1], vec![1]]
        );
        let only_the_big_body =
            references(serde_json::json!([{ "id": "face:4" }])).expect("parsed");
        assert_eq!(
            resolve_face_references(&only_the_big_body, &[6, 3]).expect("resolved"),
            vec![vec![4], Vec::<usize>::new()]
        );
        let missing = references(serde_json::json!([{ "id": "face:99" }])).expect("parsed");
        let error = resolve_face_references(&missing, &[6, 6]).expect_err("no body has face 99");
        assert!(error.contains("face:99"), "{error}");
        assert!(error.contains("2 bodies"), "{error}");
        let single = resolve_face_references(&missing, &[6]).expect_err("no face 99");
        assert!(single.contains("this body"), "{single}");
    }

    #[test]
    fn body_scoped_faces_resolve_only_their_body_and_reject_unknown_bodies() {
        let refs = references(serde_json::json!([{ "id": "body:1:face:2" }])).expect("parsed");
        assert_eq!(
            resolve_face_references(&refs, &[6, 6]).expect("resolved"),
            vec![Vec::<usize>::new(), vec![2]]
        );
        let bad_body = references(serde_json::json!([{ "id": "body:9:face:2" }])).expect("parsed");
        let error = resolve_face_references(&bad_body, &[6, 6]).expect_err("bad body");
        assert!(error.contains("body:9:face:2"), "{error}");
        assert!(error.contains("2 bodies"), "{error}");
        let bad_face = references(serde_json::json!([{ "id": "body:1:face:9" }])).expect("parsed");
        let error = resolve_face_references(&bad_face, &[6, 6]).expect_err("bad face");
        assert!(error.contains("body:1:face:9"), "{error}");
        assert!(error.contains("6 faces"), "{error}");
    }

    #[test]
    fn repeated_references_resolve_once_per_body() {
        let refs = references(serde_json::json!([
            { "id": "face:0" },
            { "id": "face:0" },
            { "id": "body:0:face:0" }
        ]))
        .expect("parsed");
        assert_eq!(resolve_face_references(&refs, &[6]).expect("resolved"), vec![vec![0]]);
    }
}
