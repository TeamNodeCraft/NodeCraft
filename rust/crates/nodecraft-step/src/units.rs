//! STEP length-unit conversion and file-header metadata.

use std::fs;

use crate::{Entity, StepFile, Value};

fn entity_id(line: &str) -> Option<u32> {
    let rest = line.strip_prefix('#')?;
    let end = rest.find('=')?;
    rest[..end].trim().parse().ok()
}

/// Supported STEP length units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthUnit { Millimetre, Centimetre, Metre, Kilometre, Micrometre, Inch }

impl LengthUnit {
    pub fn millimetres(self) -> f64 {
        match self { Self::Millimetre => 1.0, Self::Centimetre => 10.0, Self::Metre => 1000.0, Self::Kilometre => 1_000_000.0, Self::Micrometre => 0.001, Self::Inch => 25.4 }
    }
    fn si_tokens(self) -> &'static str {
        match self { Self::Millimetre => "SI_UNIT(.MILLI.,.METRE.)", Self::Centimetre => "SI_UNIT(.CENTI.,.METRE.)", Self::Metre => "SI_UNIT($,.METRE.)", Self::Kilometre => "SI_UNIT(.KILO.,.METRE.)", Self::Micrometre => "SI_UNIT(.MICRO.,.METRE.)", Self::Inch => "SI_UNIT($,.METRE.)" }
    }
}

fn split_top_level_args(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let bytes = input.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'\'' => in_string = !in_string,
            b'(' if !in_string => depth += 1,
            b')' if !in_string => depth = depth.saturating_sub(1),
            b',' if !in_string && depth == 0 => {
                args.push(input[start..index].trim().to_string());
                start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    args.push(input[start..].trim().to_string());
    args
}

/// Rewrites the numeric length-valued arguments of one analytic entity.
///
/// `indices` are argument positions (0 = entity name) that carry a length. The
/// placement reference and angle arguments are never scaled.
fn scale_entity_length_args(line: &str, entity: &str, indices: &[usize], factor: f64) -> Option<String> {
    let marker = format!("{entity}(");
    let start = line.find(&marker)? + marker.len();
    let bytes = line.as_bytes();
    let mut depth = 1usize;
    let mut in_string = false;
    let mut end = None;
    let mut index = start;
    while index < bytes.len() {
        match bytes[index] {
            b'\'' => in_string = !in_string,
            b'(' if !in_string => depth += 1,
            b')' if !in_string => {
                depth -= 1;
                if depth == 0 { end = Some(index); break; }
            }
            _ => {}
        }
        index += 1;
    }
    let end = end?;
    let mut args = split_top_level_args(&line[start..end]);
    let mut changed = false;
    for &position in indices {
        let Some(arg) = args.get_mut(position) else { continue };
        if let Ok(number) = arg.trim().parse::<f64>() {
            *arg = crate::real(number * factor);
            changed = true;
        }
    }
    changed.then(|| format!("{}{}{}", &line[..start], args.join(","), &line[end..]))
}

fn scale_analytic_entity_args(line: &str, factor: f64) -> Option<String> {
    let mut line = line.to_string();
    let mut changed = false;
    for (entity, indices) in [
        ("CIRCLE", &[2usize][..]),
        ("CYLINDRICAL_SURFACE", &[2usize][..]),
        ("SPHERICAL_SURFACE", &[2usize][..]),
        ("CONICAL_SURFACE", &[2usize][..]),
        ("TOROIDAL_SURFACE", &[2usize, 3usize][..]),
    ] {
        if let Some(scaled) = scale_entity_length_args(&line, entity, indices, factor) {
            line = scaled;
            changed = true;
        }
    }
    changed.then_some(line)
}

fn header_entity_span(text: &str, keyword: &str) -> Option<(usize, usize)> {
    let start = text.find(keyword)?;
    let mut open = start + keyword.len();
    while text.as_bytes().get(open) == Some(&b' ') { open += 1; }
    if text.as_bytes().get(open) != Some(&b'(') { return None; }
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut in_comment = false;
    let mut index = open;
    while index < bytes.len() {
        let current = bytes[index];
        let next = bytes.get(index + 1).copied();
        if in_comment {
            if current == b'*' && next == Some(b'/') { in_comment = false; index += 2; continue; }
        } else if in_string {
            if current == b'\'' { in_string = false; }
        } else if current == b'/' && next == Some(b'*') { in_comment = true; index += 2; continue; }
        else if current == b'\'' { in_string = true; }
        else if current == b'(' { depth += 1; }
        else if current == b')' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                let mut end = index + 1;
                while bytes.get(end) == Some(&b' ') { end += 1; }
                if bytes.get(end) == Some(&b';') { end += 1; }
                return Some((start, end));
            }
        }
        index += 1;
    }
    None
}

fn replace_header_entity(text: &str, keyword: &str, replacement: &str) -> Result<String, String> {
    let (start, end) = header_entity_span(text, keyword).ok_or_else(|| format!("STEP file has no {keyword} header"))?;
    let mut out = String::with_capacity(text.len() + replacement.len());
    out.push_str(&text[..start]);
    out.push_str(replacement);
    out.push_str(&text[end..]);
    Ok(out)
}


fn replace_length_unit_inch(text: &str, measure_id: u32, dimension_id: u32) -> Option<String> {
    let position = text.find("LENGTH_UNIT()")?;
    let start = text[..position].rfind('#')?;
    let end = position + text[position..].find(';')? + 1;
    let block = &text[start..end];
    if !block.contains("SI_UNIT") || block.contains("ANGLE") { return None; }
    let id_end = block.find('=')?;
    let id = block[1..id_end].trim();
    let replacement = format!("#{id}=(CONVERSION_BASED_UNIT('INCH',#{measure_id}) LENGTH_UNIT() NAMED_UNIT(#{dimension_id}));");
    let mut out = String::with_capacity(text.len() + replacement.len());
    out.push_str(&text[..start]); out.push_str(&replacement); out.push_str(&text[end..]);
    Some(out)
}


fn replace_inch_unit_with_millimetres(text: &str) -> Option<String> {
    let position = text.find("CONVERSION_BASED_UNIT('INCH'")?;
    let start = text[..position].rfind('#')?;
    let end = position + text[position..].find(';')? + 1;
    let block = &text[start..end];
    let id_end = block.find('=')?;
    let id = block[1..id_end].trim();
    let replacement = format!("#{id}=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));");
    let mut out = String::with_capacity(text.len() + replacement.len());
    out.push_str(&text[..start]); out.push_str(&replacement); out.push_str(&text[end..]);
    Some(out)
}

/// Converts a written STEP file between length units and rewrites its SI unit entity.
pub fn convert_step_units(path: &str, from: LengthUnit, to: LengthUnit) -> Result<(), String> {
    if from == to { return Ok(()); }

    let factor = from.millimetres() / to.millimetres();
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let max_id = text.lines().filter_map(entity_id).max().unwrap_or(0);
    let measure_id = max_id + 1;
    let millimetre_id = max_id + 2;
    let dim_id = max_id + 3;
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let mut line = line.to_string();
        if let Some(start) = line.find("CARTESIAN_POINT('',(") {
            let prefix = &line[..start + "CARTESIAN_POINT('',(".len()];
            if let Some(end) = line[start..].find("))") {
                let tuple = &line[start + "CARTESIAN_POINT('',(".len()..start + end];
                let scaled = tuple.split(',').map(|v| v.trim().parse::<f64>().map(|n| crate::real(n * factor)).unwrap_or_else(|_| v.trim().into())).collect::<Vec<_>>().join(",");
                line = format!("{prefix}{scaled})){}", &line[start + end + 2..]);
            }
        }
        if let Some(changed) = scale_analytic_entity_args(&line, factor) { line = changed; }
        if line.contains("LENGTH_UNIT()") || line.contains("SI_UNIT(") {
            if to != LengthUnit::Inch {
                for unit in [LengthUnit::Millimetre, LengthUnit::Centimetre, LengthUnit::Metre, LengthUnit::Kilometre, LengthUnit::Micrometre, LengthUnit::Inch] {
                    let token = unit.si_tokens();
                    if line.contains(token) { line = line.replace(token, to.si_tokens()); }
                }
                if from == LengthUnit::Inch {
                    line = line.replace("SI_UNIT($,.METRE.)", to.si_tokens());
                }
            }
        }
        out.push_str(&line); out.push('\n');
    }
    if from == LengthUnit::Inch && to != LengthUnit::Inch {
        out = replace_inch_unit_with_millimetres(&out).ok_or("STEP file has no conversion-based inch unit")?;
    }
    if to == LengthUnit::Inch {
        let replaced = replace_length_unit_inch(&out, measure_id, dim_id).ok_or("STEP file has no length-unit entity to convert to inch")?;
        out = replaced;
        let extra = format!("#{measure_id}=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#{millimetre_id});\n#{millimetre_id}=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));\n#{dim_id}=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n");
        if let Some(index) = out.rfind("ENDSEC;") { out.insert_str(index, &extra); }
    }
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}


/// Rewrites FILE_SCHEMA for a supported application protocol.
pub fn set_schema(path: &str, schema: &str) -> Result<(), String> {
    let identifier = match schema.to_ascii_lowercase().as_str() {
        "ap203" | "203" | "config_control_design" => "CONFIG_CONTROL_DESIGN",
        "ap214" | "214" | "automotive_design" => "AUTOMOTIVE_DESIGN",
        "ap242" | "242" => "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF",
        other => return Err(format!("unsupported STEP schema {other:?}; use ap203, ap214 or ap242")),
    };
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let line = format!("FILE_SCHEMA(('{identifier}'));");
    let out = replace_header_entity(&text, "FILE_SCHEMA", &line)?;
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}

/// Rewrites the FILE_NAME header fields.
pub fn set_header(path: &str, name: &str, author: &str, organization: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let escape = |value: &str| value.replace('\'', "''");
    let line = format!("FILE_NAME('{}','',('{}'),(''),'{}','nodecraft-step','');", escape(name), escape(author), escape(organization));
    let out = replace_header_entity(&text, "FILE_NAME", &line)?;
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}


/// Reads the STEP uncertainty/tolerance measure (millimetres), if present.
pub fn read_tolerance(path: &str) -> Result<Option<f64>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    for line in text.lines() {
        if !line.contains("UNCERTAINTY_MEASURE_WITH_UNIT") { continue; }
        if let Some(start) = line.find("LENGTH_MEASURE(") {
            let value = &line[start + "LENGTH_MEASURE(".len()..];
            let value = value.split(')').next().unwrap_or("");
            if let Ok(number) = value.trim().parse::<f64>() { return Ok(Some(number)); }
        }
    }
    Ok(None)
}


/// Where a resolved length scale came from.
#[derive(Debug, Clone, PartialEq)]
pub enum UnitSource {
    /// A metric `SI_UNIT(prefix, .METRE.)`, e.g. millimetres or plain metres.
    Si { prefix: Option<String> },
    /// A `CONVERSION_BASED_UNIT(name, LENGTH_MEASURE_WITH_UNIT(factor, base_unit))`.
    Conversion {
        name: String,
        factor: f64,
        /// Millimetres per `base_unit`.
        base_millimetres: f64,
    },
    /// No usable declaration: the caller should assume millimetres (and warn).
    AssumedMillimetres,
}

/// A resolved STEP length unit, in millimetres per model unit, plus its provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct LengthResolution {
    pub millimetres_per_unit: f64,
    pub source: UnitSource,
    /// Non-fatal note: an assumption or a fallback was used.
    pub warning: Option<String>,
}

impl LengthResolution {
    /// The safe fallback used when a file declares no usable length unit: millimetres.
    pub fn assumed_millimetres() -> Self {
        Self {
            millimetres_per_unit: 1.0,
            source: UnitSource::AssumedMillimetres,
            warning: Some("no length unit declared; assuming millimetres".to_string()),
        }
    }
}

/// Part 21 short entity name -> long name for the unit entities we resolve.
fn canonical_unit_name(name: &str) -> &str {
    match name {
        "GUAC" => "GLOBAL_UNIT_ASSIGNED_CONTEXT",
        "CNBSUN" => "CONVERSION_BASED_UNIT",
        "LNGUNT" => "LENGTH_UNIT",
        "SUNT" => "SI_UNIT",
        "NMDUNT" => "NAMED_UNIT",
        other => other,
    }
}

fn unit_has_member(entity: &Entity, name: &str) -> bool {
    entity
        .complex_members()
        .iter()
        .any(|member| canonical_unit_name(member) == name)
}

/// Arguments of a typed call inside a (possibly complex) entity instance.
///
/// The Part 21 parser stores `(A(x) B(y))` as `Typed("A") List([x]) Typed("B") …`,
/// so a type token's arguments are the following list value.
fn unit_typed_call<'a>(entity: &'a Entity, name: &str) -> Option<&'a [Value]> {
    for (index, arg) in entity.args.iter().enumerate() {
        let Value::Typed(kind, values) = arg else { continue };
        if canonical_unit_name(kind) != name {
            continue;
        }
        if !values.is_empty() {
            return Some(values.as_slice());
        }
        if let Some(Value::List(values)) = entity.args.get(index + 1) {
            return Some(values.as_slice());
        }
    }
    None
}

fn unit_enum_arg(value: &Value) -> Option<&str> {
    match value {
        Value::Enum(text) => Some(text.as_str()),
        _ => None,
    }
}

fn si_prefix_millimetres(prefix: Option<&str>) -> Option<f64> {
    Some(match prefix {
        // `$` means plain metre (1 model unit = 1 m = 1000 mm).
        None => 1000.0,
        Some(prefix) => match prefix.to_ascii_uppercase().as_str() {
            "MILLI" => 1.0,
            "CENTI" => 10.0,
            "DECI" => 100.0,
            "DECA" => 10_000.0,
            "KILO" => 1_000_000.0,
            "MICRO" => 0.001,
            "NANO" => 0.000_001,
            _ => return None,
        },
    })
}

/// Resolves one unit entity (`SI_UNIT` or `CONVERSION_BASED_UNIT`).
fn resolve_unit_entity(file: &StepFile, entity: &Entity, depth: usize) -> Option<LengthResolution> {
    if depth > 8 {
        return None;
    }
    if let Some(values) = unit_typed_call(entity, "SI_UNIT") {
        let prefix = match values.first()? {
            Value::Null => None,
            other => Some(unit_enum_arg(other)?),
        };
        if values.get(1).and_then(unit_enum_arg) != Some("METRE") {
            return None;
        }
        let millimetres = si_prefix_millimetres(prefix)?;
        return Some(LengthResolution {
            millimetres_per_unit: millimetres,
            source: UnitSource::Si { prefix: prefix.map(str::to_string) },
            warning: None,
        });
    }
    if let Some(values) = unit_typed_call(entity, "CONVERSION_BASED_UNIT") {
        let name = values.first().and_then(|value| match value {
            Value::Str(name) => Some(name.clone()),
            _ => None,
        })?;
        let measure_ref = values.iter().skip(1).find_map(|value| match value {
            Value::Ref(id) => Some(*id),
            _ => None,
        })?;
        let measure = file.entities.get(&measure_ref)?;
        let factor = unit_typed_call(measure, "LENGTH_MEASURE")
            .and_then(|args| args.first())
            .and_then(|value| match value {
                Value::Number(number) => Some(*number),
                _ => None,
            });
        let base_ref = measure.args.iter().find_map(|value| match value {
            Value::Ref(id) => Some(*id),
            _ => None,
        });
        if let (Some(factor), Some(base_ref)) = (factor, base_ref) {
            if factor.is_finite() && factor > 0.0 {
                if let Some(base) = file.entities.get(&base_ref).and_then(|base| resolve_unit_entity(file, base, depth + 1)) {
                    let millimetres = factor * base.millimetres_per_unit;
                    return Some(LengthResolution {
                        millimetres_per_unit: millimetres,
                        source: UnitSource::Conversion { name, factor, base_millimetres: base.millimetres_per_unit },
                        warning: None,
                    });
                }
            }
        }
        // Base unit unresolvable: use the conversion name as a documented fallback.
        let millimetres = match name.to_ascii_uppercase().as_str() {
            "INCH" => 25.4,
            "FOOT" => 304.8,
            "METRE" | "METER" => 1000.0,
            "MILLIMETRE" | "MILLIMETER" => 1.0,
            "CENTIMETRE" | "CENTIMETER" => 10.0,
            "MICROMETRE" | "MICROMETER" => 0.001,
            "KILOMETRE" | "KILOMETER" => 1_000_000.0,
            _ => return None,
        };
        return Some(LengthResolution {
            millimetres_per_unit: millimetres,
            source: UnitSource::Conversion { name: name.clone(), factor: millimetres, base_millimetres: 1.0 },
            warning: Some(format!("conversion unit {name:?} has no resolvable base unit; scale taken from its name")),
        });
    }
    None
}

fn unit_context_entities(file: &StepFile) -> Vec<&Entity> {
    file.entities
        .values()
        .filter(|entity| unit_has_member(entity, "GLOBAL_UNIT_ASSIGNED_CONTEXT"))
        .collect()
}

fn context_unit_count(file: &StepFile) -> Option<usize> {
    let context = unit_context_entities(file).into_iter().next()?;
    let values = unit_typed_call(context, "GLOBAL_UNIT_ASSIGNED_CONTEXT")?;
    match values.first()? {
        Value::List(units) => Some(units.len()),
        _ => None,
    }
}

/// Resolves the file's length unit from a parsed STEP file.
///
/// The `GLOBAL_UNIT_ASSIGNED_CONTEXT` is authoritative: its length unit is resolved
/// through `SI_UNIT` or recursively through `CONVERSION_BASED_UNIT` →
/// `LENGTH_MEASURE_WITH_UNIT(value, base_unit)`. Only when no context unit is
/// resolvable does it fall back to any `LENGTH_UNIT` entity in the file (with a
/// warning), and then to millimetres with an explicit warning.
pub fn resolve_length_unit_from_file(file: &StepFile) -> LengthResolution {
    let contexts = unit_context_entities(file);
    let mut unit_refs: Vec<u32> = Vec::new();
    for context in &contexts {
        let Some(values) = unit_typed_call(context, "GLOBAL_UNIT_ASSIGNED_CONTEXT") else { continue };
        let Some(Value::List(units)) = values.first() else { continue };
        for value in units {
            if let Value::Ref(id) = value {
                unit_refs.push(*id);
            }
        }
    }
    for id in &unit_refs {
        let Some(entity) = file.entities.get(id) else { continue };
        if !unit_has_member(entity, "LENGTH_UNIT") {
            continue;
        }
        if let Some(resolution) = resolve_unit_entity(file, entity, 0) {
            return resolution;
        }
    }
    // Some writers omit the length unit from the context list but keep the entity.
    for entity in file.entities.values() {
        if !unit_has_member(entity, "LENGTH_UNIT") {
            continue;
        }
        if let Some(resolution) = resolve_unit_entity(file, entity, 0) {
            // The unit is declared explicitly, just not linked from a context; use it
            // without an assumption warning.
            return resolution;
        }
    }
    let mut assumed = LengthResolution::assumed_millimetres();
    if !contexts.is_empty() {
        assumed.warning = Some(
            "GLOBAL_UNIT_ASSIGNED_CONTEXT has no resolvable length unit; assuming millimetres"
                .to_string(),
        );
    }
    assumed
}

/// Padding-tolerant text fallback for unit declarations (used only when a file's
/// unit entities cannot be resolved structurally).
///
/// Returns `(millimetres_per_unit, description)`.
pub fn length_scale_from_text(text: &str) -> Option<(f64, String)> {
    let compact: String = text.chars().filter(|character| !character.is_whitespace()).collect();
    if compact.contains("CONVERSION_BASED_UNIT('INCH'") || compact.contains(".INCH.") {
        return Some((25.4, "conversion inch".to_string()));
    }
    if compact.contains("CONVERSION_BASED_UNIT('METRE'") || compact.contains("CONVERSION_BASED_UNIT('METER'") {
        return Some((1000.0, "conversion metre".to_string()));
    }
    for (token, millimetres) in [
        (".MILLI.,.METRE.", 1.0),
        (".CENTI.,.METRE.", 10.0),
        (".MICRO.,.METRE.", 0.001),
        (".KILO.,.METRE.", 1_000_000.0),
        ("SI_UNIT($,.METRE.)", 1000.0),
    ] {
        if compact.contains(token) {
            return Some((millimetres, token.to_string()));
        }
    }
    None
}

/// Resolves the length unit of a STEP file on disk (millimetres per model unit).
///
/// A file that declares no usable length unit yields the millimetre assumption with
/// `LengthResolution::warning` set; this function does not reject such files.
pub fn resolve_length_unit(path: &str) -> Result<LengthResolution, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let mut resolution = match StepFile::parse(&text) {
        Ok(file) => resolve_length_unit_from_file(&file),
        Err(error) => {
            let mut assumed = LengthResolution::assumed_millimetres();
            assumed.warning = Some(format!("STEP parse failed ({error}); assuming millimetres"));
            assumed
        }
    };
    if matches!(resolution.source, UnitSource::AssumedMillimetres) {
        if let Some((millimetres, description)) = length_scale_from_text(&text) {
            resolution = LengthResolution {
                millimetres_per_unit: millimetres,
                source: UnitSource::AssumedMillimetres,
                warning: Some(format!(
                    "length unit recovered from the file text ({description}); assuming {millimetres} mm per unit"
                )),
            };
        }
    }
    Ok(resolution)
}

fn length_unit_from_resolution(resolution: &LengthResolution) -> Result<LengthUnit, String> {
    let from_millimetres = |millimetres: f64| -> Option<LengthUnit> {
        const TOLERANCE: f64 = 1e-9;
        [
            (1.0, LengthUnit::Millimetre),
            (10.0, LengthUnit::Centimetre),
            (1000.0, LengthUnit::Metre),
            (1_000_000.0, LengthUnit::Kilometre),
            (0.001, LengthUnit::Micrometre),
            (25.4, LengthUnit::Inch),
        ]
        .into_iter()
        .find(|(scale, _)| (millimetres - scale).abs() <= TOLERANCE * scale.abs().max(1.0))
        .map(|(_, unit)| unit)
    };
    match &resolution.source {
        UnitSource::AssumedMillimetres => Ok(LengthUnit::Millimetre),
        UnitSource::Si { prefix } => match prefix.as_deref() {
            None => Ok(LengthUnit::Metre),
            Some(prefix) => match prefix.to_ascii_uppercase().as_str() {
                "MILLI" => Ok(LengthUnit::Millimetre),
                "CENTI" => Ok(LengthUnit::Centimetre),
                "MICRO" => Ok(LengthUnit::Micrometre),
                "KILO" => Ok(LengthUnit::Kilometre),
                _ => Err(format!("unsupported SI length prefix {prefix:?}")),
            },
        },
        UnitSource::Conversion { name, .. } => match name.to_ascii_uppercase().as_str() {
            "INCH" => Ok(LengthUnit::Inch),
            "METRE" | "METER" => Ok(LengthUnit::Metre),
            "MILLIMETRE" | "MILLIMETER" => Ok(LengthUnit::Millimetre),
            "CENTIMETRE" | "CENTIMETER" => Ok(LengthUnit::Centimetre),
            "MICROMETRE" | "MICROMETER" => Ok(LengthUnit::Micrometre),
            "KILOMETRE" | "KILOMETER" => Ok(LengthUnit::Kilometre),
            _ => from_millimetres(resolution.millimetres_per_unit).ok_or_else(|| {
                format!(
                    "unsupported length unit conversion {name:?} ({} mm per unit); use resolve_length_unit for the scale",
                    resolution.millimetres_per_unit
                )
            }),
        },
    }
}

/// Detects the file's length unit, including conversion-based units.
///
/// Missing units are treated as millimetres (safe STEP convention); callers that need
/// the assumption warning should use [`resolve_length_unit`].
pub fn read_length_unit(path: &str) -> Result<LengthUnit, String> {
    let resolution = resolve_length_unit(path)?;
    length_unit_from_resolution(&resolution)
}

/// Representation-context unit summary.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitContext {
    pub length_unit: LengthUnit,
    /// Millimetres per model unit, resolved through conversion-based units too.
    pub millimetres_per_unit: f64,
    pub unit_count: usize,
    /// Provenance of the resolved scale.
    pub source: UnitSource,
    /// Non-fatal note when an assumption/fallback was used.
    pub warning: Option<String>,
}

/// Reads the `GLOBAL_UNIT_ASSIGNED_CONTEXT` unit list alongside the resolved unit.
pub fn read_unit_context(path: &str) -> Result<Option<UnitContext>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text).map_err(|error| format!("{path}: {error}"))?;
    if unit_context_entities(&file).is_empty() {
        return Ok(None);
    }
    let resolution = resolve_length_unit_from_file(&file);
    let length_unit = length_unit_from_resolution(&resolution)?;
    Ok(Some(UnitContext {
        length_unit,
        millimetres_per_unit: resolution.millimetres_per_unit,
        unit_count: context_unit_count(&file).unwrap_or(0),
        source: resolution.source,
        warning: resolution.warning,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_step_bounds, write_box};

    #[test]
    fn converts_box_to_centimetres_and_back() {
        let path = std::env::temp_dir().join(format!("sn-units-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        convert_step_units(path.to_str().unwrap(), LengthUnit::Millimetre, LengthUnit::Centimetre).unwrap();
        let (min, max) = read_step_bounds(path.to_str().unwrap()).unwrap();
        assert_eq!(min, [-5.0, -2.0, -10.0]);
        assert_eq!(max, [5.0, 2.0, 10.0]);
        assert!(std::fs::read_to_string(&path).unwrap().contains("SI_UNIT(.CENTI.,.METRE.)"));
    }

    #[test]
    fn reads_writer_length_unit() {
        let path = std::env::temp_dir().join(format!("sn-unit-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 1.0, 1.0, 1.0).unwrap();
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Millimetre);
        convert_step_units(path.to_str().unwrap(), LengthUnit::Millimetre, LengthUnit::Inch).unwrap();
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Inch);
        let (min, max) = crate::read_step_bounds(path.to_str().unwrap()).unwrap();
        assert!((min[0] + 0.5).abs() < 1e-9 && (max[0] - 0.5).abs() < 1e-9, "{min:?} {max:?}");
    }

    #[test]
    fn reads_representation_unit_context() {
        let path = std::env::temp_dir().join(format!("sn-unit-context-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 1.0, 1.0, 1.0).unwrap();
        let context = read_unit_context(path.to_str().unwrap()).unwrap().expect("context");
        assert_eq!(context.length_unit, LengthUnit::Millimetre);
        assert_eq!(context.unit_count, 3);
    }

    #[test]
    fn rewrites_multiline_header_entities() {
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/Sample AP203 Data Files/block.stp");
        if !source.exists() { return; }
        let path = std::env::temp_dir().join(format!("sn-multiline-{}.step", std::process::id()));
        std::fs::copy(&source, &path).unwrap();
        set_header(path.to_str().unwrap(), "Renamed", "Alex", "NodeCraft").unwrap();
        set_schema(path.to_str().unwrap(), "ap242").unwrap();
        let header = crate::read_header(path.to_str().unwrap()).unwrap();
        assert_eq!(header.name.as_deref(), Some("Renamed"));
        assert_eq!(header.schema.as_deref(), Some("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF"));
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Millimetre);
    }

    #[test]
    fn converts_inch_back_to_millimetres() {
        let path = std::env::temp_dir().join(format!("sn-inch-roundtrip-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 4.0, 6.0).unwrap();
        let (before_min, before_max) = crate::read_step_bounds(path.to_str().unwrap()).unwrap();
        convert_step_units(path.to_str().unwrap(), LengthUnit::Millimetre, LengthUnit::Inch).unwrap();
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Inch);
        convert_step_units(path.to_str().unwrap(), LengthUnit::Inch, LengthUnit::Millimetre).unwrap();
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Millimetre);
        let (after_min, after_max) = crate::read_step_bounds(path.to_str().unwrap()).unwrap();
        for axis in 0..3 {
            assert!((before_min[axis]-after_min[axis]).abs() < 1e-6, "{before_min:?} {after_min:?}");
            assert!((before_max[axis]-after_max[axis]).abs() < 1e-6, "{before_max:?} {after_max:?}");
        }
    }

    #[test]
    fn detects_kilometres() {
        let path = std::env::temp_dir().join(format!("sn-km-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 1.0, 1.0, 1.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap().replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT(.KILO.,.METRE.)");
        std::fs::write(&path, text).unwrap();
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Kilometre);
        let (min, _) = crate::read_step_bounds(path.to_str().unwrap()).unwrap();
        assert!((min[0] - -500_000.0).abs() < 1e-3, "{min:?}");
    }

    #[test]
    fn reads_writer_tolerance() {
        let path = std::env::temp_dir().join(format!("sn-tol-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 1.0, 1.0, 1.0).unwrap();
        assert_eq!(read_tolerance(path.to_str().unwrap()).unwrap(), Some(1e-6));
    }

    #[test]
    fn sets_schema_identifiers() {
        let path = std::env::temp_dir().join(format!("sn-schema-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 1.0, 1.0, 1.0).unwrap();
        set_schema(path.to_str().unwrap(), "ap203").unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));"));
        set_schema(path.to_str().unwrap(), "ap242").unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF"));
        assert!(set_schema(path.to_str().unwrap(), "ap999").is_err());
    }

    #[test]
    fn sets_header_metadata() {
        let path = std::env::temp_dir().join(format!("sn-header-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 1.0, 1.0, 1.0).unwrap();
        set_header(path.to_str().unwrap(), "part", "Alex", "NodeCraft").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("FILE_NAME('part','',('Alex'),(''),'NodeCraft','nodecraft-step','');"));
    }
    #[test]
    fn reads_unit_context_with_whitespace_before_the_list() {
        let path = std::env::temp_dir().join(format!("sn-unit-context-space-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 1.0, 1.0, 1.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap().replace("GLOBAL_UNIT_ASSIGNED_CONTEXT((", "GLOBAL_UNIT_ASSIGNED_CONTEXT ((");
        std::fs::write(&path, text).unwrap();
        let context = read_unit_context(path.to_str().unwrap()).unwrap().expect("context");
        assert_eq!(context.length_unit, LengthUnit::Millimetre);
        assert_eq!(context.unit_count, 3);
    }

    fn entity_args(text: &str, name: &str) -> Vec<String> {
        let line = text.lines().find(|line| line.contains(&format!("{name}("))).expect("entity line");
        let start = line.find(&format!("{name}(")).unwrap() + name.len() + 1;
        let end = line.rfind(')').expect("entity close");
        split_top_level_args(&line[start..end])
    }

    #[test]
    fn converting_units_scales_analytic_lengths_not_placement_refs() {
        let path = std::env::temp_dir().join(format!("sn-unit-analytic-{}.step", std::process::id()));
        crate::write_cylinder(path.to_str().unwrap(), 5.0, 10.0).unwrap();
        let before = std::fs::read_to_string(&path).unwrap();
        for name in ["CIRCLE", "CYLINDRICAL_SURFACE"] {
            let args = entity_args(&before, name);
            assert!(args[2].parse::<f64>().is_ok(), "{name} radius arg {args:?}");
        }
        let circle_ref = entity_args(&before, "CIRCLE")[1].clone();
        let surface_ref = entity_args(&before, "CYLINDRICAL_SURFACE")[1].clone();

        convert_step_units(path.to_str().unwrap(), LengthUnit::Millimetre, LengthUnit::Inch).unwrap();
        let inch = std::fs::read_to_string(&path).unwrap();
        assert_eq!(entity_args(&inch, "CIRCLE")[1], circle_ref, "circle placement ref must not be scaled");
        assert_eq!(entity_args(&inch, "CYLINDRICAL_SURFACE")[1], surface_ref, "surface placement ref must not be scaled");
        let circle_radius: f64 = entity_args(&inch, "CIRCLE")[2].parse().unwrap();
        let surface_radius: f64 = entity_args(&inch, "CYLINDRICAL_SURFACE")[2].parse().unwrap();
        assert!((circle_radius - 5.0 / 25.4).abs() < 1e-9, "circle radius {circle_radius}");
        assert!((surface_radius - 5.0 / 25.4).abs() < 1e-9, "surface radius {surface_radius}");

        convert_step_units(path.to_str().unwrap(), LengthUnit::Inch, LengthUnit::Millimetre).unwrap();
        let back = std::fs::read_to_string(&path).unwrap();
        let circle_radius: f64 = entity_args(&back, "CIRCLE")[2].parse().unwrap();
        let surface_radius: f64 = entity_args(&back, "CYLINDRICAL_SURFACE")[2].parse().unwrap();
        assert!((circle_radius - 5.0).abs() < 1e-9, "round-trip circle radius {circle_radius}");
        assert!((surface_radius - 5.0).abs() < 1e-9, "round-trip surface radius {surface_radius}");
    }

    fn fixture_path(relative: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests").join(relative)
    }

    #[test]
    fn resolves_conversion_based_metre_abc_fixtures() {
        for name in ["abc-0000", "abc-0006", "abc-0008", "abc-0035"] {
            let path = std::path::Path::new("/tmp/truck-resources/step").join(format!("{name}.step"));
            if !path.exists() {
                eprintln!("skipping {name}: missing {}", path.display());
                continue;
            }
            let resolution = resolve_length_unit(path.to_str().unwrap()).unwrap();
            assert!((resolution.millimetres_per_unit - 1000.0).abs() < 1e-9, "{name}: {resolution:?}");
            match &resolution.source {
                UnitSource::Conversion { name: unit, factor, base_millimetres } => {
                    assert_eq!(unit, "METRE", "{name}: unit name");
                    assert!((factor - 1.0).abs() < 1e-12, "{name}: factor");
                    assert!((base_millimetres - 1000.0).abs() < 1e-9, "{name}: base");
                }
                other => panic!("{name}: expected a conversion-based unit, got {other:?}"),
            }
            assert!(resolution.warning.is_none(), "{name}: {resolution:?}");
            assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Metre, "{name}");
        }
    }

    #[test]
    fn resolves_doghouse_inch_conversion() {
        let path = fixture_path("Sample AP203 Data Files/doghouse.stp");
        if !path.exists() {
            return;
        }
        let resolution = resolve_length_unit(path.to_str().unwrap()).unwrap();
        assert!((resolution.millimetres_per_unit - 25.4).abs() < 1e-9, "{resolution:?}");
        assert!(resolution.warning.is_none(), "{resolution:?}");
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Inch);
    }

    #[test]
    fn resolves_occt_cube_si_millimetres() {
        let path = std::path::Path::new("/tmp/truck-resources/step/occt-cube.step");
        if !path.exists() {
            return;
        }
        let resolution = resolve_length_unit(path.to_str().unwrap()).unwrap();
        assert!((resolution.millimetres_per_unit - 1.0).abs() < 1e-12, "{resolution:?}");
        match &resolution.source {
            UnitSource::Si { prefix: Some(prefix) } => assert_eq!(prefix, "MILLI", "{resolution:?}"),
            other => panic!("expected SI millimetre, got {other:?}"),
        }
        assert!(resolution.warning.is_none(), "{resolution:?}");
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Millimetre);
    }

    #[test]
    fn assumes_millimetres_with_warning_when_units_are_missing() {
        let path = std::env::temp_dir().join(format!("sn-unitless-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let stripped: String = text
            .lines()
            .filter(|line| {
                !line.contains("LENGTH_UNIT")
                    && !line.contains("SI_UNIT")
                    && !line.contains("GLOBAL_UNIT_ASSIGNED_CONTEXT")
                    && !line.contains("UNCERTAINTY")
                    && !line.contains("GEOMETRIC_REPRESENTATION_CONTEXT")
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, stripped).unwrap();
        let resolution = resolve_length_unit(path.to_str().unwrap()).unwrap();
        assert!((resolution.millimetres_per_unit - 1.0).abs() < 1e-12, "{resolution:?}");
        assert!(matches!(resolution.source, UnitSource::AssumedMillimetres), "{resolution:?}");
        let warning = resolution.warning.expect("missing units must warn");
        assert!(warning.contains("assuming millimetres"), "{warning}");
        // The legacy API does not reject; it returns the millimetre assumption.
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Millimetre);
    }

    #[test]
    fn resolves_conversion_chain_to_metre_from_written_box() {
        let path = std::env::temp_dir().join(format!("sn-conversion-metre-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let replaced = text.replace(
            "(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))",
            "( CONVERSION_BASED_UNIT('METRE', #900) LENGTH_UNIT() NAMED_UNIT(*))",
        );
        assert!(replaced.contains("CONVERSION_BASED_UNIT('METRE'"), "writer unit entity not found");
        let extra = "#900=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0),#901);\n#901=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));\n";
        let position = replaced.rfind("ENDSEC;").unwrap();
        let mut with_conversion = replaced;
        with_conversion.insert_str(position, extra);
        std::fs::write(&path, with_conversion).unwrap();
        let resolution = resolve_length_unit(path.to_str().unwrap()).unwrap();
        assert!((resolution.millimetres_per_unit - 1000.0).abs() < 1e-9, "{resolution:?}");
        assert!(matches!(resolution.source, UnitSource::Conversion { .. }), "{resolution:?}");
        assert!(resolution.warning.is_none(), "{resolution:?}");
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Metre);
    }

    #[test]
    fn resolves_padded_metre_token() {
        let path = std::env::temp_dir().join(format!("sn-padded-metre-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap().replace(
            "SI_UNIT(.MILLI.,.METRE.)",
            "SI_UNIT( $, .METRE. )",
        );
        std::fs::write(&path, text).unwrap();
        let resolution = resolve_length_unit(path.to_str().unwrap()).unwrap();
        assert!((resolution.millimetres_per_unit - 1000.0).abs() < 1e-9, "{resolution:?}");
        assert_eq!(read_length_unit(path.to_str().unwrap()).unwrap(), LengthUnit::Metre);
    }

    #[test]
    fn text_fallback_is_padding_tolerant() {
        assert_eq!(length_scale_from_text("SI_UNIT( $, .METRE. )").map(|(scale, _)| scale), Some(1000.0));
        assert_eq!(length_scale_from_text("SI_UNIT( .MILLI., .METRE. )").map(|(scale, _)| scale), Some(1.0));
        assert_eq!(length_scale_from_text("CONVERSION_BASED_UNIT( 'INCH', #7 )").map(|(scale, _)| scale), Some(25.4));
    }

}
