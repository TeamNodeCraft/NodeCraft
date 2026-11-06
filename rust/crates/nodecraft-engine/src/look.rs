//! Look-node colour attributes and display-mesh colour transfer.

/// Application-level colour attribute carried by a Look node. SolidCore
/// treats `SmColorAttribute` specially when regions merge through Booleans.
const LOOK_COLOR_ATTRIBUTE: u64 = 100; // solidcore `geom::attribute::ids::COLOR`

pub(crate) fn parse_look_color(values: &serde_json::Value) -> Result<[f64; 3], String> {
    let text = values
        .get("color")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("#7fa0cf");
    let hex = text.trim_start_matches('#');
    if hex.len() != 6 || !hex.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(format!("Look color must be #RRGGBB, got {text:?}"));
    }
    let channel = |range: std::ops::Range<usize>| {
        u8::from_str_radix(&hex[range], 16)
            .map(|value| f64::from(value) / 255.0)
            .map_err(|error| error.to_string())
    };
    Ok([channel(0..2)?, channel(2..4)?, channel(4..6)?])
}

pub(crate) fn apply_look_color(
    brep: &mut nodecraft_solidcore::solidcore::topology::Brep,
    color: [f64; 3],
) -> Result<(), String> {
    use nodecraft_solidcore::solidcore::error::EntityRef;
    use nodecraft_solidcore::solidcore::geom::attribute::{Attribute, Behavior};
    use nodecraft_solidcore::solidcore::math::Vec3;
    // Colour belongs to faces: a Boolean merges regions, so a region colour
    // would paint the whole result instead of only the Looked branch.
    let faces: Vec<_> = brep.faces().ids().map(EntityRef::Face).collect();
    for entity in faces {
        let old: Vec<_> = brep
            .entity_attributes(entity)
            .into_iter()
            .filter_map(|(handle, attribute)| {
                (attribute.attribute_id() == LOOK_COLOR_ATTRIBUTE).then_some(handle)
            })
            .collect();
        for handle in old {
            brep.remove_attribute(entity, handle)
                .map_err(|error| error.to_string())?;
        }
        brep.add_attribute(
            entity,
            Attribute::color(
                LOOK_COLOR_ATTRIBUTE,
                Vec3::new(color[0], color[1], color[2]),
                Behavior::Copy,
            ),
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Copy each B-rep face's colour attribute into the display mesh's per-face
/// colour channel. Faces without a colour keep the renderer default.
pub(crate) fn apply_brep_face_colors(
    brep: &nodecraft_solidcore::solidcore::topology::Brep,
    mesh: &mut nodecraft_protocol::Mesh,
) {
    use nodecraft_solidcore::solidcore::error::EntityRef;
    use nodecraft_solidcore::solidcore::geom::attribute::AttributeValue;
    // Display face groups use the position in `face_list`, not the arena id:
    // booleans and fillets leave stale arena slots, so those indices diverge.
    let mut by_face: std::collections::BTreeMap<usize, [f32; 3]> = std::collections::BTreeMap::new();
    for (position, &face) in brep.face_list().iter().enumerate() {
        let Some(attribute) = brep.find_attribute(EntityRef::Face(face), LOOK_COLOR_ATTRIBUTE) else {
            continue;
        };
        let (AttributeValue::Color(color) | AttributeValue::Vector3d(color)) = attribute.value() else {
            continue;
        };
        by_face.insert(position, [color.x as f32, color.y as f32, color.z as f32]);
    }
    if by_face.is_empty() {
        mesh.face_colors.clear();
        return;
    }
    mesh.face_colors = mesh
        .face_groups
        .iter()
        .map(|group| {
            group
                .rsplit_once("face:")
                .and_then(|(_, tail)| tail.parse::<usize>().ok())
                .and_then(|index| by_face.get(&index).copied())
                .unwrap_or([0.32, 0.34, 0.37])
        })
        .collect();
}

