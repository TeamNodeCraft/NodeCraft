//! STEP `COMPOSITE_CURVE` / `COMPOSITE_CURVE_SEGMENT` parsing.

use nodecraft_step::{StepFile, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct CompositeSegment {
    pub segment_id: u32,
    pub same_sense: bool,
    pub parent_curve: u32,
    pub transition: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompositeCurve {
    pub id: u32,
    pub segments: Vec<CompositeSegment>,
    pub self_intersect: bool,
}

/// Parses a `COMPOSITE_CURVE` entity by id.
pub fn parse_composite_curve(file: &StepFile, id: u32) -> Result<CompositeCurve, String> {
    let entity = file.entities.get(&id).ok_or_else(|| format!("#{id}: no such entity"))?;
    if entity.name != "COMPOSITE_CURVE" { return Err(format!("#{id}: expected COMPOSITE_CURVE, got {}", entity.name)); }
    let segments = match entity.args.get(1) {
        Some(Value::List(values)) => values.iter().filter_map(|value| match value {
            Value::Ref(segment_id) => parse_segment(file, *segment_id).ok(),
            _ => None,
        }).collect(),
        _ => Vec::new(),
    };
    Ok(CompositeCurve { id, segments, self_intersect: matches!(entity.args.get(2), Some(Value::Enum(flag)) if flag == "T") })
}

fn parse_segment(file: &StepFile, id: u32) -> Result<CompositeSegment, String> {
    let entity = file.entities.get(&id).ok_or_else(|| format!("#{id}: no such segment"))?;
    if entity.name != "COMPOSITE_CURVE_SEGMENT" { return Err(format!("#{id}: expected COMPOSITE_CURVE_SEGMENT")); }
    let same_sense = matches!(entity.args.get(1), Some(Value::Enum(flag)) if flag == "T");
    let parent_curve = match entity.args.get(2) { Some(Value::Ref(curve)) => *curve, _ => return Err(format!("#{id}: missing parent curve")) };
    let transition = match entity.args.get(3) { Some(Value::Enum(value)) => value.clone(), _ => "CONTINUOUS".into() };
    Ok(CompositeSegment { segment_id: id, same_sense, parent_curve, transition })
}


impl CompositeCurve {
    /// Samples the composite by sampling each segment's parent B-spline curve.
    /// Segment orientation is honoured. Non-B-spline parents are reported, not guessed.
    pub fn sample(&self, file: &StepFile, segments: usize) -> Result<Vec<[f64; 3]>, String> {
        if segments == 0 { return Err("sample count must be positive".into()); }
        let mut points: Vec<[f64; 3]> = Vec::new();
        for segment in &self.segments {
            let curve = crate::parse_curve(file, segment.parent_curve)
                .map_err(|error| format!("composite segment #{}: {error}", segment.segment_id))?;
            let mut sampled = curve.sample(segments)?;
            if !segment.same_sense { sampled.reverse(); }
            if let Some(last) = points.last().copied() {
                if sampled.first().copied() == Some(last) { sampled.remove(0); }
            }
            points.extend(sampled);
        }
        Ok(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "DATA;\n#1=LINE('',#2,#3);\n#2=CARTESIAN_POINT('',(0.,0.,0.));\n#3=VECTOR('',#4,1.);\n#4=DIRECTION('',(1.,0.,0.));\n#5=COMPOSITE_CURVE_SEGMENT('',.T.,#1,.CONTINUOUS.);\n#6=COMPOSITE_CURVE('',(#5),.F.);\nENDSEC;";

    #[test]
    fn parses_composite_segment() {
        let file = StepFile::parse(TEXT).unwrap();
        let curve = parse_composite_curve(&file, 6).unwrap();
        assert!(!curve.self_intersect);
        assert_eq!(curve.segments.len(), 1);
        assert_eq!(curve.segments[0].parent_curve, 1);
        assert!(curve.segments[0].same_sense);
        assert_eq!(curve.segments[0].transition, "CONTINUOUS");
    }

    #[test]
    fn samples_composite_curve() {
        let text = "DATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(1.,0.,0.));\n#10=B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#2),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);\n#20=COMPOSITE_CURVE_SEGMENT('',.T.,#10,.CONTINUOUS.);\n#21=COMPOSITE_CURVE('',(#20),.F.);\nENDSEC;";
        let file = nodecraft_step::StepFile::parse(text).unwrap();
        let curve = parse_composite_curve(&file, 21).unwrap();
        let points = curve.sample(&file, 2).unwrap();
        assert_eq!(points.len(), 3);
        assert_eq!(points[0], [0.0,0.0,0.0]);
        assert_eq!(points[2], [1.0,0.0,0.0]);
    }

    #[test]
    fn rejects_wrong_entity_kind() {
        let file = StepFile::parse(TEXT).unwrap();
        assert!(parse_composite_curve(&file, 1).is_err());
    }
}
