//! STEP presentation color output (`STYLED_ITEM` + `COLOUR_RGB`).

use std::fs;

use crate::real;

fn entity_id(line: &str) -> Option<u32> {
    let rest = line.strip_prefix('#')?;
    let end = rest.find('=')?;
    rest[..end].trim().parse().ok()
}

/// Appends an RGB presentation style to the first solid in a STEP file.
pub fn set_color(path: &str, red: f64, green: f64, blue: f64) -> Result<(), String> {
    if [red, green, blue].iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err("color components must be between 0 and 1".into());
    }
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let mut max_id = 0u32;
    let mut solids = Vec::new();
    for line in text.lines() {
        if let Some(id) = entity_id(line) { max_id = max_id.max(id); }
        if line.contains("MANIFOLD_SOLID_BREP(") || line.contains("BREP_WITH_VOIDS(") {
            if let Some(id) = entity_id(line) { solids.push(id); }
        }
    }
    if solids.is_empty() { return Err("STEP file has no solid to style".into()); }
    let mut next = max_id;
    let mut add = |body: String| { next += 1; format!("#{next}={body};\n") };
    let colour = add(format!("COLOUR_RGB('',{},{},{})", real(red), real(green), real(blue)));
    let colour_ref = colour[1..colour.find('=').unwrap()].to_string();
    let fill_colour = add(format!("FILL_AREA_STYLE_COLOUR('',#{colour_ref})"));
    let fill_colour_id = fill_colour[1..fill_colour.find('=').unwrap()].to_string();
    let fill = add(format!("FILL_AREA_STYLE('',(#{fill_colour_id}))"));
    let fill_id = fill[1..fill.find('=').unwrap()].to_string();
    let surface = add(format!("SURFACE_STYLE_FILL_AREA(#{fill_id})"));
    let surface_id = surface[1..surface.find('=').unwrap()].to_string();
    let side = add(format!("SURFACE_SIDE_STYLE('',(#{surface_id}))"));
    let side_id = side[1..side.find('=').unwrap()].to_string();
    let usage = add(format!("SURFACE_STYLE_USAGE(.BOTH.,#{side_id})"));
    let usage_id = usage[1..usage.find('=').unwrap()].to_string();
    let assignment = add(format!("PRESENTATION_STYLE_ASSIGNMENT((#{usage_id}))"));
    let assignment_id = assignment[1..assignment.find('=').unwrap()].to_string();
    let styled: String = solids.iter().map(|solid| add(format!("STYLED_ITEM('',(#{assignment_id}),#{solid})"))).collect();
    let index = text.rfind("ENDSEC;").ok_or("STEP file has no closing ENDSEC")?;
    let mut out = String::with_capacity(text.len() + 512);
    out.push_str(&text[..index]);
    out.push_str(&colour); out.push_str(&fill_colour); out.push_str(&fill); out.push_str(&surface); out.push_str(&side); out.push_str(&usage); out.push_str(&assignment); out.push_str(&styled);
    out.push_str(&text[index..]);
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}


/// Renames the product and solid labels in a written STEP file.
pub fn set_product_name(path: &str, name: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let escaped = name.replace('\'', "''");
    let mut replaced_product = false;
    let mut out = String::new();
    for line in text.lines() {
        let mut line = line.to_string();
        if let Some(index) = line.find("=PRODUCT(") {
            let after = index + "=PRODUCT(".len();
            let (prefix, rest) = line.split_at(after);
            let mut parts = rest.splitn(3, ',');
            if let (Some(first), Some(second), Some(tail)) = (parts.next(), parts.next(), parts.next()) {
                if first.trim_start().starts_with('\'') && second.trim_start().starts_with('\'') {
                    line = format!("{prefix}'{escaped}','{escaped}',{tail}");
                    replaced_product = true;
                }
            }
        }
        for entity in ["ADVANCED_BREP_SHAPE_REPRESENTATION(", "MANIFOLD_SOLID_BREP(", "FACETED_BREP(", "BREP_WITH_VOIDS("] {
            if let Some(index) = line.find(entity) {
                let after = index + entity.len();
                let (prefix, rest) = line.split_at(after);
                if rest.starts_with('\'') {
                    let mut end = 1;
                    let bytes = rest.as_bytes();
                    while end < bytes.len() {
                        if bytes[end] == b'\'' {
                            if bytes.get(end + 1) == Some(&b'\'') { end += 2; continue; }
                            break;
                        }
                        end += 1;
                    }
                    if end < bytes.len() {
                        line = format!("{prefix}'{escaped}'{}", &rest[end + 1..]);
                    }
                }
            }
        }
        out.push_str(&line); out.push('\n');
    }
    if !replaced_product { return Err("STEP file has no PRODUCT to rename".into()); }
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}

/// Reads the first RGB presentation color assigned to a solid, if present.
pub fn read_color(path: &str) -> Result<Option<([f64; 3], u32)>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = crate::StepFile::parse(&text)?;
    fn color_of(file: &crate::StepFile, value: &crate::Value) -> Option<[f64; 3]> {
        match value {
            crate::Value::Ref(id) => {
                let entity = file.entities.get(id)?;
                if entity.name == "COLOUR_RGB" {
                    let mut color = [0.0; 3];
                    for (slot, arg) in color.iter_mut().zip(entity.args.iter().skip(1)) {
                        let crate::Value::Number(number) = arg else { return None };
                        *slot = *number;
                    }
                    Some(color)
                } else {
                    entity.args.iter().find_map(|arg| color_of(file, arg))
                }
            }
            crate::Value::List(values) | crate::Value::Typed(_, values) => values.iter().find_map(|value| color_of(file, value)),
            _ => None,
        }
    }
    for entity in file.entities.values() {
        if entity.name != "STYLED_ITEM" { continue; }
        let Some(target) = entity.args.get(2).and_then(|value| match value { crate::Value::Ref(id) => Some(*id), _ => None }) else { continue };
        if let Some(color) = entity.args.get(1).and_then(|value| color_of(&file, value)) { return Ok(Some((color, target))); }
    }
    Ok(None)
}


/// Assigns the first solid to a named presentation layer.
pub fn set_layer(path: &str, layer: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let mut solid = None;
    let mut max_id = 0u32;
    for line in text.lines() {
        if let Some(id) = entity_id(line) { max_id = max_id.max(id); }
        if solid.is_none() && line.contains("MANIFOLD_SOLID_BREP(") { solid = entity_id(line); }
    }
    let solid = solid.ok_or("STEP file has no solid to layer")?;
    let escaped = layer.replace('\'', "''");
    let assignment = format!("#{}=PRESENTATION_LAYER_ASSIGNMENT('{}','',(#{solid}));\n", max_id + 1, escaped);
    let index = text.rfind("ENDSEC;").ok_or("STEP file has no closing ENDSEC")?;
    let mut out = String::with_capacity(text.len() + assignment.len());
    out.push_str(&text[..index]); out.push_str(&assignment); out.push_str(&text[index..]);
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}


/// Reads every styled solid's RGB color.
pub fn read_colors(path: &str) -> Result<Vec<([f64; 3], u32)>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = crate::StepFile::parse(&text)?;
    fn color_of(file: &crate::StepFile, value: &crate::Value) -> Option<[f64; 3]> {
        match value {
            crate::Value::Ref(id) => {
                let entity = file.entities.get(id)?;
                if entity.name == "COLOUR_RGB" {
                    let mut color = [0.0; 3];
                    for (slot, arg) in color.iter_mut().zip(entity.args.iter().skip(1)) {
                        let crate::Value::Number(number) = arg else { return None };
                        *slot = *number;
                    }
                    Some(color)
                } else { entity.args.iter().find_map(|arg| color_of(file, arg)) }
            }
            crate::Value::List(values) | crate::Value::Typed(_, values) => values.iter().find_map(|value| color_of(file, value)),
            _ => None,
        }
    }
    let mut out = Vec::new();
    for entity in file.entities.values() {
        if entity.name != "STYLED_ITEM" { continue; }
        let Some(target) = entity.args.get(2).and_then(|value| match value { crate::Value::Ref(id) => Some(*id), _ => None }) else { continue };
        if let Some(color) = entity.args.get(1).and_then(|value| color_of(&file, value)) { out.push((color, target)); }
    }
    Ok(out)
}

/// Reads named presentation-layer assignments and their target entity ids.
pub fn read_layers(path: &str) -> Result<Vec<(String, Vec<u32>)>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = crate::StepFile::parse(&text)?;
    let mut out = Vec::new();
    for entity in file.entities.values() {
        if entity.name != "PRESENTATION_LAYER_ASSIGNMENT" { continue; }
        let Some(crate::Value::Str(name)) = entity.args.first() else { continue };
        let targets = match entity.args.get(2) {
            Some(crate::Value::List(values)) => values.iter().filter_map(|value| match value { crate::Value::Ref(id) => Some(*id), _ => None }).collect(),
            _ => Vec::new(),
        };
        out.push((name.clone(), targets));
    }
    Ok(out)
}


/// Reads the first PRODUCT name from a STEP file.
pub fn read_product_name(path: &str) -> Result<Option<String>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = crate::StepFile::parse(&text)?;
    for entity in file.entities.values() {
        if entity.name == "PRODUCT" {
            if let Some(crate::Value::Str(name)) = entity.args.first() { if !name.is_empty() { return Ok(Some(name.clone())); } }
        }
    }
    Ok(None)
}

/// Reads the labels of all solid B-reps.
pub fn read_solid_names(path: &str) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = crate::StepFile::parse(&text)?;
    let mut names = Vec::new();
    for entity in file.entities.values() {
        if matches!(entity.name.as_str(), "MANIFOLD_SOLID_BREP" | "BREP_WITH_VOIDS" | "FACETED_BREP") {
            if let Some(crate::Value::Str(name)) = entity.args.first() { names.push(name.clone()); }
        }
    }
    Ok(names)
}


/// Reads `MATERIAL_DESIGNATION` names from a STEP file.
pub fn read_materials(path: &str) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = crate::StepFile::parse(&text)?;
    let mut names = Vec::new();
    for entity in file.entities.values() {
        if entity.name == "MATERIAL_DESIGNATION" {
            if let Some(crate::Value::Str(name)) = entity.args.first() { if !name.is_empty() { names.push(name.clone()); } }
        }
    }
    Ok(names)
}


/// Appends a `MATERIAL_DESIGNATION` to the STEP file.
pub fn set_material(path: &str, material: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let max_id = text.lines().filter_map(entity_id).max().unwrap_or(0);
    let escaped = material.replace('\'', "''");
    let entity = format!("#{}=MATERIAL_DESIGNATION('{}','');\n", max_id + 1, escaped);
    let index = text.rfind("ENDSEC;").ok_or("STEP file has no closing ENDSEC")?;
    let mut out = String::with_capacity(text.len() + entity.len());
    out.push_str(&text[..index]); out.push_str(&entity); out.push_str(&text[index..]);
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}


/// Reads named `SURFACE_SIDE_STYLE` presentation styles.
pub fn read_style_names(path: &str) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = crate::StepFile::parse(&text)?;
    let mut names = Vec::new();
    for entity in file.entities.values() {
        if entity.name == "SURFACE_SIDE_STYLE" {
            if let Some(crate::Value::Str(name)) = entity.args.first() { if !name.is_empty() { names.push(name.clone()); } }
        }
    }
    Ok(names)
}


/// Reads `MATERIAL_PROPERTY` names and their material target ids.
pub fn read_material_properties(path: &str) -> Result<Vec<(String, Option<u32>)>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = crate::StepFile::parse(&text)?;
    let mut properties = Vec::new();
    for entity in file.entities.values() {
        if entity.name != "MATERIAL_PROPERTY" { continue; }
        let Some(crate::Value::Str(name)) = entity.args.first() else { continue };
        let target = match entity.args.get(2) { Some(crate::Value::Ref(id)) => Some(*id), _ => None };
        if !name.is_empty() { properties.push((name.clone(), target)); }
    }
    Ok(properties)
}


/// Reads material property names and their descriptive values, when present.
pub fn read_material_property_values(path: &str) -> Result<Vec<(String, Option<String>)>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = crate::StepFile::parse(&text)?;
    let mut out = Vec::new();
    for entity in file.entities.values() {
        if entity.name != "MATERIAL_PROPERTY" { continue; }
        let Some(crate::Value::Str(name)) = entity.args.first() else { continue };
        let value = entity.args.iter().filter_map(|arg| match arg {
            crate::Value::Ref(id) => file.entities.get(id),
            _ => None,
        }).find(|item| item.name == "DESCRIPTIVE_REPRESENTATION_ITEM")
            .and_then(|item| match item.args.get(1) { Some(crate::Value::Str(value)) => Some(value.clone()), _ => None });
        out.push((name.clone(), value));
    }
    Ok(out)
}


/// Appends a descriptive material property.
pub fn set_material_property(path: &str, name: &str, value: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let max_id = text.lines().filter_map(entity_id).max().unwrap_or(0);
    let escape = |value: &str| value.replace('\'', "''");
    let representation = max_id + 1;
    let property = max_id + 2;
    let entities = format!("#{representation}=DESCRIPTIVE_REPRESENTATION_ITEM('{}','{}');\n#{property}=MATERIAL_PROPERTY('{}','',#{representation});\n", escape(name), escape(value), escape(name));
    let index = text.rfind("ENDSEC;").ok_or("STEP file has no closing ENDSEC")?;
    let mut out = String::with_capacity(text.len() + entities.len());
    out.push_str(&text[..index]); out.push_str(&entities); out.push_str(&text[index..]);
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write_box;

    #[test]
    fn appends_rgb_style() {
        let path = std::env::temp_dir().join(format!("sn-color-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        set_color(path.to_str().unwrap(), 0.1, 0.5, 0.9).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("COLOUR_RGB('',0.1,0.5,0.9)"));
        assert!(text.contains("STYLED_ITEM("));
        assert!(set_color(path.to_str().unwrap(), 2.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn renames_arbitrary_product_name() {
        let path = std::env::temp_dir().join(format!("sn-rename-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap().replace("PRODUCT('solid','solid',", "PRODUCT('block','block',").replace("MANIFOLD_SOLID_BREP('solid'", "MANIFOLD_SOLID_BREP('block'").replace("ADVANCED_BREP_SHAPE_REPRESENTATION('solid'", "ADVANCED_BREP_SHAPE_REPRESENTATION('block'");
        std::fs::write(&path, text).unwrap();
        set_product_name(path.to_str().unwrap(), "Bracket").unwrap();
        assert_eq!(read_product_name(path.to_str().unwrap()).unwrap().as_deref(), Some("Bracket"));
        assert_eq!(read_solid_names(path.to_str().unwrap()).unwrap(), vec!["Bracket"]);
    }

    #[test]
    fn reads_material_properties() {
        let path = std::env::temp_dir().join(format!("sn-material-prop-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let index = text.rfind("ENDSEC;").unwrap();
        let mut styled = text.clone();
        styled.insert_str(index, "#998=MATERIAL_DESIGNATION('Steel','');\n#999=MATERIAL_PROPERTY('Density','',#998);\n");
        std::fs::write(&path, styled).unwrap();
        let properties = read_material_properties(path.to_str().unwrap()).unwrap();
        assert_eq!(properties.len(), 1);
        assert_eq!(properties[0].0, "Density");
        assert!(properties[0].1.is_some());
    }

    #[test]
    fn writes_material_property_values() {
        let path = std::env::temp_dir().join(format!("sn-material-write-value-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        set_material_property(path.to_str().unwrap(), "Density", "7.85").unwrap();
        assert_eq!(read_material_property_values(path.to_str().unwrap()).unwrap(), vec![("Density".to_string(), Some("7.85".to_string()))]);
    }

    #[test]
    fn reads_material_property_values() {
        let path = std::env::temp_dir().join(format!("sn-material-value-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let index = text.rfind("ENDSEC;").unwrap();
        let mut styled = text.clone();
        styled.insert_str(index, "#997=DESCRIPTIVE_REPRESENTATION_ITEM('Density','7.85');\n#999=MATERIAL_PROPERTY('Density','',#997);\n");
        std::fs::write(&path, styled).unwrap();
        assert_eq!(read_material_property_values(path.to_str().unwrap()).unwrap(), vec![("Density".to_string(), Some("7.85".to_string()))]);
    }

    #[test]
    fn reads_style_names() {
        let path = std::env::temp_dir().join(format!("sn-style-name-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let index = text.rfind("ENDSEC;").unwrap();
        let mut styled = text.clone();
        styled.insert_str(index, "#999=SURFACE_SIDE_STYLE('Machined',());\n");
        std::fs::write(&path, styled).unwrap();
        assert_eq!(read_style_names(path.to_str().unwrap()).unwrap(), vec!["Machined"]);
    }

    #[test]
    fn writes_material_designations() {
        let path = std::env::temp_dir().join(format!("sn-material-write-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        set_material(path.to_str().unwrap(), "Aluminium").unwrap();
        assert_eq!(read_materials(path.to_str().unwrap()).unwrap(), vec!["Aluminium"]);
    }

    #[test]
    fn reads_material_designations() {
        let path = std::env::temp_dir().join(format!("sn-material-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let index = text.rfind("ENDSEC;").unwrap();
        let mut with_material = text.clone();
        with_material.insert_str(index, "#999=MATERIAL_DESIGNATION('Steel','');\n");
        std::fs::write(&path, with_material).unwrap();
        assert_eq!(read_materials(path.to_str().unwrap()).unwrap(), vec!["Steel"]);
    }

    #[test]
    fn reads_product_and_solid_names() {
        let dir = std::env::temp_dir().join(format!("sn-names-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("named.step");
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        set_product_name(path.to_str().unwrap(), "Bracket").unwrap();
        assert_eq!(read_product_name(path.to_str().unwrap()).unwrap().as_deref(), Some("Bracket"));
        assert_eq!(read_solid_names(path.to_str().unwrap()).unwrap(), vec!["Bracket"]);
    }

    #[test]
    fn reads_all_colors_and_layers() {
        let dir = std::env::temp_dir().join(format!("sn-style-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.step"); let b = dir.join("b.step");
        write_box(a.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        write_box(b.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let merged = dir.join("merged.step");
        crate::merge_step_files(&[a.to_string_lossy().to_string(), b.to_string_lossy().to_string()], merged.to_str().unwrap()).unwrap();
        set_color(merged.to_str().unwrap(), 0.2, 0.4, 0.6).unwrap();
        set_layer(merged.to_str().unwrap(), "Parts").unwrap();
        assert_eq!(read_colors(merged.to_str().unwrap()).unwrap().len(), 2);
        let layers = read_layers(merged.to_str().unwrap()).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].0, "Parts");
        assert_eq!(layers[0].1.len(), 1);
    }

    #[test]
    fn styles_all_solids() {
        let path = std::env::temp_dir().join(format!("sn-color-all-{}.step", std::process::id()));
        crate::write_box_with_void(path.to_str().unwrap(), [10.0,10.0,10.0], [4.0,4.0,4.0]).unwrap();
        set_color(path.to_str().unwrap(), 0.2, 0.4, 0.6).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap().matches("STYLED_ITEM(").count(), 1);
    }

    #[test]
    fn assigns_presentation_layer() {
        let path = std::env::temp_dir().join(format!("sn-layer-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        set_layer(path.to_str().unwrap(), "Parts").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("PRESENTATION_LAYER_ASSIGNMENT('Parts','',(#"));
    }

    #[test]
    fn renames_product() {
        let path = std::env::temp_dir().join(format!("sn-name-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        set_product_name(path.to_str().unwrap(), "Bracket").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("PRODUCT('Bracket','Bracket'"));
        assert!(text.contains("ADVANCED_BREP_SHAPE_REPRESENTATION('Bracket'"));
    }

    #[test]
    fn reads_back_rgb_and_target() {
        let path = std::env::temp_dir().join(format!("sn-color-read-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        set_color(path.to_str().unwrap(), 0.1, 0.5, 0.9).unwrap();
        let (color, target) = read_color(path.to_str().unwrap()).unwrap().unwrap();
        assert!((color[0] - 0.1).abs() < 1e-9);
        assert!((color[1] - 0.5).abs() < 1e-9);
        assert!((color[2] - 0.9).abs() < 1e-9);
        assert!(target > 0);
    }
}
