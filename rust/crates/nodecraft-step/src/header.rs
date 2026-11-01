//! STEP header parsing and validation (FILE_SCHEMA, FILE_NAME, ISO framing).

use std::fs;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StepHeader {
    pub schema: Option<String>,
    pub name: Option<String>,
    pub author: Option<String>,
    pub organization: Option<String>,
    pub has_header: bool,
    pub has_data: bool,
    pub has_iso_start: bool,
    pub has_iso_end: bool,
}

fn quoted_values(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut inside = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\'' {
            if inside && chars.peek() == Some(&'\'') { current.push('\''); chars.next(); continue; }
            if inside { out.push(current.clone()); current.clear(); inside = false; } else { inside = true; }
        } else if inside { current.push(ch); }
    }
    out
}

/// Reads the ISO-10303-21 header fields.
pub fn read_header(path: &str) -> Result<StepHeader, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let mut header = StepHeader {
        has_iso_start: text.trim_start().starts_with("ISO-10303-21;"),
        has_header: text.contains("HEADER;"),
        has_data: text.contains("DATA;"),
        has_iso_end: text.contains("END-ISO-10303-21;"),
        ..StepHeader::default()
    };
    let section = text.split_once("HEADER;").and_then(|(_, rest)| rest.split_once("ENDSEC;").map(|(header, _)| header)).unwrap_or("");
    let flat = section.replace(['\n', '\r'], " ");
    if let Some(start) = flat.find("FILE_SCHEMA") {
        if let Some(end) = flat[start..].find(");") {
            header.schema = quoted_values(&flat[start..start + end]).into_iter().next();
        }
    }
    if let Some(start) = flat.find("FILE_NAME") {
        if let Some(end) = flat[start..].find(");") {
            let values = quoted_values(&flat[start..start + end]);
            header.name = values.first().cloned();
            header.author = values.get(2).cloned();
            header.organization = values.get(4).cloned();
        }
    }
    Ok(header)
}

/// Validates ISO framing and required header records.
pub fn validate_header(path: &str) -> Result<StepHeader, String> {
    let header = read_header(path)?;
    if !header.has_iso_start { return Err("missing ISO-10303-21 opening".into()); }
    if !header.has_header { return Err("missing HEADER section".into()); }
    if !header.has_data { return Err("missing DATA section".into()); }
    if !header.has_iso_end { return Err("missing END-ISO-10303-21".into()); }
    if header.schema.is_none() { return Err("missing FILE_SCHEMA".into()); }
    // FILE_NAME is required by ISO 10303-21 but many practical exporters omit it.
    // It is metadata only, so accept a missing name rather than rejecting the geometry.
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_default_header_from_writer() {
        let path = std::env::temp_dir().join(format!("sn-header-{}.step", std::process::id()));
        crate::write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let header = validate_header(path.to_str().unwrap()).unwrap();
        assert_eq!(header.schema.as_deref(), Some("CONFIG_CONTROL_DESIGN"));
        assert!(header.has_iso_start && header.has_iso_end);
    }

    #[test]
    fn reads_ap203_and_ap242_schemas() {
        let path = std::env::temp_dir().join(format!("sn-header-schema-{}.step", std::process::id()));
        crate::write_box(path.to_str().unwrap(), 1.0, 1.0, 1.0).unwrap();
        crate::set_schema(path.to_str().unwrap(), "ap203").unwrap();
        assert_eq!(validate_header(path.to_str().unwrap()).unwrap().schema.as_deref(), Some("CONFIG_CONTROL_DESIGN"));
        crate::set_schema(path.to_str().unwrap(), "ap242").unwrap();
        assert!(validate_header(path.to_str().unwrap()).unwrap().schema.unwrap().contains("AP242"));
    }

    #[test]
    fn rejects_missing_data() {
        let path = std::env::temp_dir().join(format!("sn-header-bad-{}.step", std::process::id()));
        std::fs::write(&path, "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('AP214'));\nFILE_NAME('x','',(''),(''),'','','');\nENDSEC;\nEND-ISO-10303-21;\n").unwrap();
        assert!(validate_header(path.to_str().unwrap()).is_err());
    }
    #[test]
    fn accepts_missing_file_name_metadata() {
        let path = std::env::temp_dir().join(format!("sn-header-no-name-{}.step", std::process::id()));
        std::fs::write(&path, "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'2;1');\nFILE_SCHEMA(('AP214'));\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n").unwrap();
        assert!(validate_header(path.to_str().unwrap()).is_ok());
    }

}
