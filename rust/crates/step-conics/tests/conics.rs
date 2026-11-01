//! Integration tests for the conic STEP reader/writer, using inline STEP text.

use step_conics::{
    AngleUnit, Axis2Placement3d, Conic, ConicDomain, ConicError, ConicLimit, ConicScene,
    ConicWriter, Ellipse, Hyperbola, MasterRepresentation, Parabola, PlacementError, Trim,
    write_conics, validate_trim,
};

/// Wraps `body` in a minimal ISO-10303-21 file with a valid header.
fn step_file(body: &str) -> String {
    format!(
        "ISO-10303-21;\n\
         HEADER;\n\
         FILE_DESCRIPTION((''),'2;1');\n\
         FILE_NAME('','',(''),(''),'step-conics-test','step-conics-test','');\n\
         FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n\
         ENDSEC;\n\
         DATA;\n\
         {body}\
         ENDSEC;\n\
         END-ISO-10303-21;\n"
    )
}

fn parse(body: &str) -> ConicScene {
    ConicScene::parse(&step_file(body)).expect("inline STEP parses")
}

fn ellipse() -> Conic {
    Conic::Ellipse(Ellipse {
        placement: Axis2Placement3d::with_axes([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        semi_axis_1: 10.0,
        semi_axis_2: 5.0,
    })
}

fn hyperbola() -> Conic {
    Conic::Hyperbola(Hyperbola {
        placement: Axis2Placement3d::new([0.0, 0.0, 0.0]),
        semi_axis: 4.0,
        semi_imag_axis: 3.0,
    })
}

fn parabola() -> Conic {
    Conic::Parabola(Parabola {
        placement: Axis2Placement3d::with_axis([0.0, 0.0, 5.0], [0.0, 1.0, 0.0]),
        focal_distance: 2.5,
    })
}

#[test]
fn parses_all_three_conics_with_placements() {
    let scene = parse(
        "\
#1=CARTESIAN_POINT('',(1.,2.,3.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=ELLIPSE('',#4,10.,5.);
#6=CARTESIAN_POINT('',(0.,0.,0.));
#7=AXIS2_PLACEMENT_3D('',#6,$,$);
#8=HYPERBOLA('',#7,4.,3.);
#9=PARABOLA('',#7,2.5);
",
    );

    assert_eq!(scene.conics.len(), 3);
    assert_eq!(scene.placements.len(), 2, "#7 is shared by #8 and #9");
    assert_eq!(scene.trimmed_curves.len(), 0);

    assert_eq!(
        scene.conic(5),
        Some(&Conic::Ellipse(Ellipse {
            placement: Axis2Placement3d::with_axes(
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ),
            semi_axis_1: 10.0,
            semi_axis_2: 5.0,
        }))
    );
    assert_eq!(
        scene.conic(8),
        Some(&Conic::Hyperbola(Hyperbola {
            placement: Axis2Placement3d::new([0.0, 0.0, 0.0]),
            semi_axis: 4.0,
            semi_imag_axis: 3.0,
        }))
    );
    assert_eq!(
        scene.conic(9),
        Some(&Conic::Parabola(Parabola {
            placement: Axis2Placement3d::new([0.0, 0.0, 0.0]),
            focal_distance: 2.5,
        }))
    );
    assert_eq!(
        scene.placements.get(&4),
        Some(&Axis2Placement3d::with_axes(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ))
    );

    assert_eq!(scene.conic(5).map(Conic::name), Some("ELLIPSE"));
    assert_eq!(scene.conic(8).map(Conic::name), Some("HYPERBOLA"));
    assert_eq!(scene.conic(9).map(Conic::name), Some("PARABOLA"));
}

#[test]
fn placement_defaults_resolve_to_orthonormal_frames() {
    // Default axis +Z and reference +X.
    let frame = Axis2Placement3d::new([1.0, 2.0, 3.0]).resolve().unwrap();
    assert_eq!(frame.origin, [1.0, 2.0, 3.0]);
    assert_eq!(frame.x, [1.0, 0.0, 0.0]);
    assert_eq!(frame.y, [0.0, 1.0, 0.0]);
    assert_eq!(frame.z, [0.0, 0.0, 1.0]);

    // Default reference when the axis is +X is +Y (otherwise +X would be parallel).
    let frame = Axis2Placement3d::with_axis([0.0; 3], [1.0, 0.0, 0.0])
        .resolve()
        .unwrap();
    assert_eq!(frame.x, [0.0, 1.0, 0.0]);
    assert_eq!(frame.y, [0.0, 0.0, 1.0]);
    assert_eq!(frame.z, [1.0, 0.0, 0.0]);

    // Axis +Y: x = +X, y = z × x = (0,0,-1).
    let frame = Axis2Placement3d::with_axis([0.0; 3], [0.0, 1.0, 0.0])
        .resolve()
        .unwrap();
    assert_eq!(frame.x, [1.0, 0.0, 0.0]);
    assert_eq!(frame.y, [0.0, 0.0, -1.0]);
    assert_eq!(frame.z, [0.0, 1.0, 0.0]);

    // An explicit non-orthogonal reference is orthogonalized against the axis.
    let frame = Axis2Placement3d::with_axes([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0])
        .resolve()
        .unwrap();
    assert_eq!(frame.x, [1.0, 0.0, 0.0]);
    assert_eq!(frame.y, [0.0, 1.0, 0.0]);
    assert_eq!(frame.z, [0.0, 0.0, 1.0]);

    assert!(matches!(
        Axis2Placement3d::with_axis([0.0; 3], [0.0, 0.0, 0.0]).resolve(),
        Err(PlacementError::ZeroAxis)
    ));
    assert!(matches!(
        Axis2Placement3d::with_axes([0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]).resolve(),
        Err(PlacementError::ZeroRefDirection)
    ));
    assert!(matches!(
        Axis2Placement3d::with_axes([0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]).resolve(),
        Err(PlacementError::RefDirectionParallelToAxis)
    ));
    assert!(matches!(
        Axis2Placement3d::with_axis([0.0; 3], [f64::NAN, 0.0, 1.0]).resolve(),
        Err(PlacementError::NonFinite("axis"))
    ));
}

#[test]
fn parses_step_defaults_as_none_axes() {
    let scene = parse(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#3=ELLIPSE('',#2,10.,5.);
",
    );
    let placement = *scene.conic(3).unwrap().placement();
    assert_eq!(placement.axis, None);
    assert_eq!(placement.ref_direction, None);
    assert_eq!(scene.placements.get(&2), Some(&placement));
}

#[test]
fn writer_round_trips_all_three_conics() {
    let conics = vec![ellipse(), hyperbola(), parabola()];
    let text = write_conics(&conics).expect("write conics");
    assert!(text.starts_with("ISO-10303-21;\nHEADER;"));
    assert!(text.contains("ELLIPSE('',#"));
    assert!(text.contains("HYPERBOLA('',#"));
    assert!(text.contains("PARABOLA('',#"));
    assert!(text.contains("AXIS2_PLACEMENT_3D('',#"));
    assert!(text.contains("$,$)"), "missing default placement axes");

    let scene = ConicScene::parse(&text).expect("parse written conics");
    let parsed: Vec<Conic> = scene.conics.values().copied().collect();
    assert_eq!(parsed, conics, "write -> parse must preserve the conics");

    // The writer is deterministic.
    assert_eq!(write_conics(&conics).unwrap(), text);
}

#[test]
fn writer_entity_strings_are_step_entities() {
    assert_eq!(
        ellipse().step_entity(7).unwrap(),
        "ELLIPSE('',#7,10.0,5.0)"
    );
    assert_eq!(
        Trim::new(0.0, 1.5).step_trimmed_curve(5).unwrap(),
        "TRIMMED_CURVE('',#5,(PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.5)),.T.,.PARAMETER.)"
    );
    assert_eq!(
        Trim::with_sense(0.0, 1.5, false)
            .step_trimmed_curve(5)
            .unwrap(),
        "TRIMMED_CURVE('',#5,(PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.5)),.F.,.PARAMETER.)"
    );

    let mut writer = ConicWriter::new();
    let placement = writer
        .push_placement(&axis_placement())
        .expect("placement entity");
    assert_eq!(placement, 4, "point, axis, reference, placement");
    let conic = writer.push_conic(&ellipse()).expect("conic entity");
    assert_eq!(conic, 9, "placement 4-8 then the conic");
    assert_eq!(writer.next_id(), 10);
    assert_eq!(writer.entities().len(), 9);
    assert_eq!(
        writer.entities()[0],
        "#1=CARTESIAN_POINT('',(1.0,2.0,3.0));"
    );
    assert_eq!(
        writer.entities()[3],
        "#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);"
    );
    assert_eq!(writer.entities()[8], "#9=ELLIPSE('',#8,10.0,5.0);");
    assert!(writer.finish().contains("#9=ELLIPSE('',#8,10.0,5.0);\n"));
}

fn axis_placement() -> Axis2Placement3d {
    Axis2Placement3d::with_axes([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
}

#[test]
fn writer_round_trips_a_trimmed_conic() {
    let trim = Trim::with_sense(-0.5, 1.5, false);
    let mut writer = ConicWriter::new();
    let trimmed_id = writer
        .push_trimmed_conic(&ellipse(), &trim)
        .expect("trimmed conic");
    assert_eq!(
        writer.entities().last().map(String::as_str),
        Some(
            "#6=TRIMMED_CURVE('',#5,(PARAMETER_VALUE(-0.5)),(PARAMETER_VALUE(1.5)),.F.,.PARAMETER.);"
        )
    );

    let scene = ConicScene::parse(&writer.finish()).expect("parse trimmed conic");
    assert_eq!(scene.conics.len(), 1);
    let trimmed = scene.trimmed_curves.get(&trimmed_id).expect("trimmed curve");
    assert_eq!(trimmed.basis, 5);
    assert_eq!(trimmed.trim, trim);
    assert_eq!(trimmed.master, MasterRepresentation::Parameter);
    assert_eq!(scene.trim_for(5), Some(trimmed));
}

#[test]
fn parses_parameter_trim_with_cartesian_point_alongside() {
    let scene = parse(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#3=ELLIPSE('',#2,10.,5.);
#4=CARTESIAN_POINT('',(10.,0.,0.));
#5=TRIMMED_CURVE('',#3,(#4,PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.5)),.T.,.PARAMETER.);
",
    );
    let trimmed = scene.trimmed_curves.get(&5).expect("trimmed curve #5");
    assert_eq!(trimmed.basis, 3);
    assert_eq!(trimmed.trim, Trim::new(0.0, 1.5));
    assert_eq!(trimmed.master, MasterRepresentation::Parameter);
    assert_eq!(scene.trim_for(3), Some(trimmed));
}

#[test]
fn unsupported_trims_are_explicit() {
    // Cartesian master representation.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#3=ELLIPSE('',#2,10.,5.);
#4=TRIMMED_CURVE('',#3,(PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.)),.T.,.CARTESIAN.);
",
    ))
    .expect_err("cartesian master must be rejected");
    assert!(
        matches!(err, ConicError::UnsupportedTrim { id: 4, .. }),
        "got {err}"
    );

    // A parameter-master trim that only has cartesian points.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#3=ELLIPSE('',#2,10.,5.);
#4=CARTESIAN_POINT('',(10.,0.,0.));
#5=TRIMMED_CURVE('',#3,(#4),(#4),.T.,.PARAMETER.);
",
    ))
    .expect_err("cartesian-only trim must be rejected");
    assert!(
        matches!(err, ConicError::UnsupportedTrim { id: 5, .. }),
        "got {err}"
    );

    // More than one parameter value in one trim end.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#3=ELLIPSE('',#2,10.,5.);
#4=TRIMMED_CURVE('',#3,(PARAMETER_VALUE(0.),PARAMETER_VALUE(0.5)),(PARAMETER_VALUE(1.)),.T.,.PARAMETER.);
",
    ))
    .expect_err("multi-point trim must be rejected");
    assert!(
        matches!(err, ConicError::UnsupportedTrim { id: 4, .. }),
        "got {err}"
    );

    // Non-finite trim parameter.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#3=ELLIPSE('',#2,10.,5.);
#4=TRIMMED_CURVE('',#3,(PARAMETER_VALUE(NaN)),(PARAMETER_VALUE(1.)),.T.,.PARAMETER.);
",
    ))
    .expect_err("non-finite trim must be rejected");
    assert!(
        matches!(err, ConicError::NonFiniteTrim { id: 4, .. }),
        "got {err}"
    );
}

#[test]
fn non_conic_trimmed_curves_are_ignored() {
    let scene = parse(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(1.,0.,0.));
#3=VECTOR('',#2,1.);
#4=LINE('',#1,#3);
#5=TRIMMED_CURVE('',#4,(PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.)),.T.,.PARAMETER.);
",
    );
    assert!(scene.conics.is_empty());
    assert!(scene.trimmed_curves.is_empty(), "LINE trim is not a conic trim");
}

#[test]
fn trim_limits_are_explicit() {
    let ellipse = ellipse();
    assert_eq!(ellipse.domain(), ConicDomain::PeriodicEllipse);
    assert_eq!(
        ConicDomain::PeriodicEllipse.period(AngleUnit::Radians),
        Some(step_conics::ELLIPSE_PERIOD_RADIANS)
    );
    assert_eq!(
        ConicDomain::PeriodicEllipse.period(AngleUnit::Degrees),
        Some(step_conics::ELLIPSE_PERIOD_DEGREES)
    );
    assert_eq!(ConicDomain::HyperbolaBranch.period(AngleUnit::Degrees), None);

    // Radians: an ordinary arc and the full period.
    let range = validate_trim(&ellipse, &Trim::new(0.0, 1.5), AngleUnit::Radians).unwrap();
    assert_eq!((range.start, range.end, range.span), (0.0, 1.5, 1.5));
    assert!(!range.full_period);
    let range = validate_trim(
        &ellipse,
        &Trim::new(0.0, step_conics::ELLIPSE_PERIOD_RADIANS),
        AngleUnit::Radians,
    )
    .unwrap();
    assert!(range.full_period);

    // Degrees: the same full period expressed in degrees.
    let range = validate_trim(
        &ellipse,
        &Trim::new(0.0, step_conics::ELLIPSE_PERIOD_DEGREES),
        AngleUnit::Degrees,
    )
    .unwrap();
    assert!(range.full_period);

    // Span beyond one period, reversed and empty intervals are rejected.
    let err = validate_trim(&ellipse, &Trim::new(0.0, 7.0), AngleUnit::Radians).unwrap_err();
    assert!(matches!(
        err,
        ConicLimit::TrimSpanExceedsPeriod { span, period }
            if span == 7.0 && period == step_conics::ELLIPSE_PERIOD_RADIANS
    ));
    let err = validate_trim(&ellipse, &Trim::new(360.0, 0.0), AngleUnit::Degrees).unwrap_err();
    assert!(matches!(
        err,
        ConicLimit::TrimReversed { trim_1: 360.0, trim_2: 0.0 }
    ));
    let err = validate_trim(&ellipse, &Trim::new(2.0, 2.0), AngleUnit::Degrees).unwrap_err();
    assert_eq!(err, ConicLimit::EmptyTrim);
    let err = validate_trim(&ellipse, &Trim::new(f64::NAN, 1.0), AngleUnit::Radians).unwrap_err();
    assert!(matches!(err, ConicLimit::NonFiniteTrim { value } if value.is_nan()));

    // sense_agreement does not change the parameter interval.
    let range =
        validate_trim(&ellipse, &Trim::with_sense(0.0, 1.5, false), AngleUnit::Radians).unwrap();
    assert_eq!(range.span, 1.5);
}

#[test]
fn hyperbola_and_parabola_are_open_branches() {
    let hyperbola = hyperbola();
    assert_eq!(hyperbola.domain(), ConicDomain::HyperbolaBranch);
    assert_eq!(hyperbola.domain().period(AngleUnit::Radians), None);
    let range = validate_trim(
        &hyperbola,
        &Trim::new(-1.0e6, 1.0e6),
        AngleUnit::Radians,
    )
    .unwrap();
    assert_eq!(range.span, 2.0e6);
    assert!(!range.full_period);
    assert!(matches!(
        validate_trim(&hyperbola, &Trim::new(2.0, 1.0), AngleUnit::Radians),
        Err(ConicLimit::TrimReversed { .. })
    ));

    let parabola = parabola();
    assert_eq!(parabola.domain(), ConicDomain::ParabolaBranch);
    assert_eq!(parabola.domain().period(AngleUnit::Degrees), None);
    let range = validate_trim(&parabola, &Trim::new(-10.0, 10.0), AngleUnit::Radians).unwrap();
    assert_eq!(range.span, 20.0);
    assert!(!range.full_period);
}

#[test]
fn malformed_entities_report_specific_errors() {
    // Missing placement.
    let err = ConicScene::parse(&step_file("#5=ELLIPSE('',#99,10.,5.);"))
        .expect_err("missing placement");
    assert_eq!(err, ConicError::MissingEntity { id: 99 });

    // Wrong placement entity type.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#5=ELLIPSE('',#1,10.,5.);
",
    ))
    .expect_err("wrong placement type");
    assert!(
        matches!(
            err,
            ConicError::WrongEntity {
                id: 1,
                expected: "AXIS2_PLACEMENT_3D",
                ..
            }
        ),
        "got {err}"
    );

    // Wrong argument count.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#5=ELLIPSE('',#2,10.);
",
    ))
    .expect_err("missing ellipse argument");
    assert!(
        matches!(
            err,
            ConicError::BadArgumentCount {
                id: 5,
                expected: 4,
                found: 3,
                ..
            }
        ),
        "got {err}"
    );

    // Non-positive semi-axis.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#5=ELLIPSE('',#2,-10.,5.);
",
    ))
    .expect_err("negative semi-axis");
    assert!(
        matches!(
            err,
            ConicError::NonPositive {
                id: 5,
                ref field,
                value: -10.0,
                ..
            } if field == "semi_axis_1"
        ),
        "got {err}"
    );

    // Non-positive focal distance on a parabola.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#3=PARABOLA('',#2,0.);
",
    ))
    .expect_err("zero focal distance");
    assert!(
        matches!(err, ConicError::NonPositive { id: 3, .. }),
        "got {err}"
    );
}

#[test]
fn degenerate_placements_are_errors() {
    // Reference direction parallel to the axis.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=AXIS2_PLACEMENT_3D('',#1,#2,#2);
#4=ELLIPSE('',#3,10.,5.);
",
    ))
    .expect_err("parallel reference direction");
    assert!(
        matches!(err, ConicError::InvalidPlacement { id: 3, .. }),
        "got {err}"
    );

    // Zero axis.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,0.));
#3=AXIS2_PLACEMENT_3D('',#1,#2,$);
#4=ELLIPSE('',#3,10.,5.);
",
    ))
    .expect_err("zero placement axis");
    assert!(
        matches!(err, ConicError::InvalidPlacement { id: 3, .. }),
        "got {err}"
    );

    // A 2D placement location is not a 3D placement.
    let err = ConicScene::parse(&step_file(
        "\
#1=CARTESIAN_POINT('',(0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#3=ELLIPSE('',#2,10.,5.);
",
    ))
    .expect_err("2D location");
    assert!(
        matches!(err, ConicError::BadValue { id: 1, .. }),
        "got {err}"
    );
}

#[test]
fn writer_rejects_invalid_geometry_without_partial_entities() {
    let bad_axis = Conic::Ellipse(Ellipse {
        placement: Axis2Placement3d::new([0.0; 3]),
        semi_axis_1: f64::NAN,
        semi_axis_2: 5.0,
    });
    assert!(matches!(
        write_conics(&[bad_axis]),
        Err(ConicError::InvalidGeometry { entity: "ELLIPSE", .. })
    ));

    let bad_placement = Conic::Parabola(Parabola {
        placement: Axis2Placement3d::with_axis([0.0; 3], [0.0, 0.0, 0.0]),
        focal_distance: 1.0,
    });
    assert!(matches!(
        write_conics(&[bad_placement]),
        Err(ConicError::InvalidGeometry {
            entity: "AXIS2_PLACEMENT_3D",
            ..
        })
    ));

    let mut writer = ConicWriter::new();
    assert!(writer.push_conic(&bad_axis).is_err());
    assert!(writer.entities().is_empty(), "failed push added entities");
    assert!(writer.finish().contains("DATA;\nENDSEC;"));

    assert!(
        writer
            .push_trimmed_conic(&ellipse(), &Trim::new(f64::INFINITY, 1.0))
            .is_err()
    );
    assert!(writer.entities().is_empty(), "failed trim added entities");
}
