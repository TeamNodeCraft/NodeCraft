//! STEP product/assembly structure parser.
//!
//! Reads the AP203/AP214/AP242 product graph with the generic `nodecraft_step::StepFile`
//! parser and turns it into a neutral tree:
//!
//! ```text
//! PRODUCT / PRODUCT_DEFINITION / PRODUCT_DEFINITION_SHAPE / SHAPE_REPRESENTATION
//! NEXT_ASSEMBLY_USAGE_OCCURRENCE            (parent -> child, one per occurrence)
//! CONTEXT_DEPENDENT_SHAPE_REPRESENTATION    (NAUO -> representation relationship)
//! REPRESENTATION_RELATIONSHIP(+_WITH_TRANSFORMATION)
//! ITEM_DEFINED_TRANSFORMATION / CARTESIAN_TRANSFORMATION_OPERATOR_3D
//! AXIS2_PLACEMENT_3D                        (local frames)
//! ```
//!
//! # Transform direction
//!
//! Exporters disagree on which representation is the parent in a
//! `REPRESENTATION_RELATIONSHIP`, so the product structure decides: the occurrence's
//! parent/child product definitions (from `NEXT_ASSEMBLY_USAGE_OCCURRENCE`) select the
//! relationship side, with the CAx-IF reading (`rep_1` = child, `rep_2` = parent) as the
//! fallback. The local transform is `placement(parent item) * placement(child item)^-1`,
//! i.e. it maps child representation coordinates into the parent's, matching smlibpy's
//! `step_reader.py`.
//!
//! World transforms are accumulated down the tree: `world = parent_world * local`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use nodecraft_step::{Entity, StepFile, Value};

use crate::model::{Assembly, Definition, Node, Product, ProductDefinitionShape, Representation};
use crate::{AssemblyError, Transform};

/// Parses an assembly from ISO-10303-21 text.
pub fn parse_assembly(text: &str) -> Result<Assembly, AssemblyError> {
    let file = StepFile::parse(text).map_err(AssemblyError::Step)?;
    parse_assembly_from_step_file(&file)
}

/// Parses an assembly from a STEP file on disk.
pub fn parse_assembly_file(path: impl AsRef<Path>) -> Result<Assembly, AssemblyError> {
    let text = std::fs::read_to_string(path)?;
    parse_assembly(&text)
}

/// Parses an assembly from an already-parsed `StepFile`.
pub fn parse_assembly_from_step_file(file: &StepFile) -> Result<Assembly, AssemblyError> {
    Parser::new(file).run()
}

// ---------------------------------------------------------------------------
// Intermediate records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct ProductRecord {
    product_id: String,
    name: String,
    description: String,
}

#[derive(Debug, Clone, Default)]
struct DefinitionRecord {
    product: Option<u32>,
    representation: Option<u32>,
}

#[derive(Debug, Clone, Default)]
struct PdsRecord {
    name: String,
    description: String,
    definition: Option<u32>,
}

#[derive(Debug, Clone, Default)]
struct NauoRecord {
    occurrence_id: String,
    name: String,
    parent: Option<u32>,
    child: Option<u32>,
}

#[derive(Debug, Clone, Copy, Default)]
struct RelationshipRecord {
    rep1: Option<u32>,
    rep2: Option<u32>,
    operator: Option<u32>,
}

#[derive(Debug, Clone)]
struct Occurrence {
    nauo: u32,
    occurrence_id: String,
    occurrence_name: Option<String>,
    placement_name: Option<String>,
    parent_definition: u32,
    child_definition: u32,
    local_transform: Transform,
}

struct Parser<'a> {
    file: &'a StepFile,
    warnings: Vec<String>,
    products: BTreeMap<u32, ProductRecord>,
    formations: BTreeMap<u32, u32>,
    definitions: BTreeMap<u32, DefinitionRecord>,
    representations: BTreeMap<u32, Representation>,
    representation_items: HashMap<u32, HashSet<u32>>,
    product_definition_shapes: BTreeMap<u32, PdsRecord>,
    nauos: BTreeMap<u32, NauoRecord>,
    relationships: BTreeMap<u32, RelationshipRecord>,
}

impl<'a> Parser<'a> {
    fn new(file: &'a StepFile) -> Self {
        Self {
            file,
            warnings: Vec::new(),
            products: BTreeMap::new(),
            formations: BTreeMap::new(),
            definitions: BTreeMap::new(),
            representations: BTreeMap::new(),
            representation_items: HashMap::new(),
            product_definition_shapes: BTreeMap::new(),
            nauos: BTreeMap::new(),
            relationships: BTreeMap::new(),
        }
    }

    fn run(&mut self) -> Result<Assembly, AssemblyError> {
        self.collect_products();
        self.collect_formations();
        self.collect_definitions();
        self.collect_representations();
        self.collect_product_definition_shapes();
        self.collect_shape_definition_representations();
        self.collect_nauos();
        self.collect_relationships();

        if self.products.is_empty() {
            return Err(AssemblyError::Malformed(
                "STEP file has no PRODUCT entities".into(),
            ));
        }
        if self.definitions.is_empty() {
            return Err(AssemblyError::Malformed(
                "STEP file has no PRODUCT_DEFINITION entities".into(),
            ));
        }

        let occurrences = self.collect_occurrences()?;
        let roots = self.build_tree(&occurrences)?;

        let mut products: Vec<Product> = self
            .products
            .iter()
            .map(|(id, record)| Product {
                entity_id: *id,
                product_id: record.product_id.clone(),
                name: record.name.clone(),
                description: record.description.clone(),
            })
            .collect();
        products.sort_by_key(|product| product.entity_id);
        let mut definitions: Vec<Definition> = self
            .definitions
            .iter()
            .map(|(id, record)| Definition {
                entity_id: *id,
                product_entity: record.product,
                product_name: self.product_name(record.product),
                representation: record.representation,
            })
            .collect();
        definitions.sort_by_key(|definition| definition.entity_id);
        let mut representations: Vec<Representation> =
            self.representations.values().cloned().collect();
        representations.sort_by_key(|representation| representation.entity_id);
        let product_definition_shapes: Vec<ProductDefinitionShape> = self
            .product_definition_shapes
            .iter()
            .map(|(id, record)| ProductDefinitionShape {
                entity_id: *id,
                name: record.name.clone(),
                description: record.description.clone(),
                defined: record.definition,
            })
            .collect();

        let warnings = std::mem::take(&mut self.warnings);
        Ok(Assembly {
            roots,
            products,
            definitions,
            representations,
            product_definition_shapes,
            warnings,
        })
    }

    // -- collection ---------------------------------------------------------

    fn collect_products(&mut self) {
        for (&id, entity) in &self.file.entities {
            if !type_names(entity).contains(&"PRODUCT") {
                continue;
            }
            let product_id = non_blank(arg_string(&entity.args, 0)).unwrap_or_default();
            let name = non_blank(arg_string(&entity.args, 1))
                .or_else(|| non_blank(arg_string(&entity.args, 0)))
                .unwrap_or_default();
            let description = arg_string(&entity.args, 2).unwrap_or_default().trim().to_string();
            self.products.insert(
                id,
                ProductRecord {
                    product_id,
                    name,
                    description,
                },
            );
        }
    }

    fn collect_formations(&mut self) {
        for (&id, entity) in &self.file.entities {
            let is_formation = type_names(entity)
                .iter()
                .any(|name| name.starts_with("PRODUCT_DEFINITION_FORMATION"));
            if !is_formation {
                continue;
            }
            if let Some(product) = arg_ref(&entity.args, 2) {
                self.formations.insert(id, product);
            }
        }
    }

    fn collect_definitions(&mut self) {
        for (&id, entity) in &self.file.entities {
            if !type_names(entity).contains(&"PRODUCT_DEFINITION") {
                continue;
            }
            let formation = arg_ref(&entity.args, 2);
            let product = formation.and_then(|formation| self.formations.get(&formation).copied());
            self.definitions.insert(id, DefinitionRecord { product, representation: None });
        }
    }

    fn collect_representations(&mut self) {
        for (&id, entity) in &self.file.entities {
            let components = components(entity);
            for (name, args) in &components {
                if !name.ends_with("REPRESENTATION") {
                    continue;
                }
                let Some(Value::List(items)) = args.get(1) else { continue };
                let items: Vec<u32> = items
                    .iter()
                    .filter_map(|value| match value {
                        Value::Ref(item) => Some(*item),
                        _ => None,
                    })
                    .collect();
                let representation = Representation {
                    entity_id: id,
                    name: non_blank(arg_string(args, 0)).unwrap_or_default(),
                    kind: (*name).to_string(),
                    items: items.clone(),
                };
                self.representation_items.insert(id, items.into_iter().collect());
                self.representations.insert(id, representation);
                break;
            }
        }
    }

    fn collect_product_definition_shapes(&mut self) {
        for (&id, entity) in &self.file.entities {
            if !type_names(entity).contains(&"PRODUCT_DEFINITION_SHAPE") {
                continue;
            }
            let args = component_args(entity, "PRODUCT_DEFINITION_SHAPE")
                .unwrap_or(entity.args.as_slice());
            self.product_definition_shapes.insert(
                id,
                PdsRecord {
                    name: non_blank(arg_string(args, 0)).unwrap_or_default(),
                    description: arg_string(args, 1).unwrap_or_default().trim().to_string(),
                    definition: arg_ref(args, 2),
                },
            );
        }
    }

    fn collect_shape_definition_representations(&mut self) {
        for entity in self.file.entities.values() {
            if !type_names(entity).contains(&"SHAPE_DEFINITION_REPRESENTATION") {
                continue;
            }
            let args = component_args(entity, "SHAPE_DEFINITION_REPRESENTATION")
                .unwrap_or(entity.args.as_slice());
            let (Some(shape), Some(representation)) = (arg_ref(args, 0), arg_ref(args, 1)) else {
                continue;
            };
            let Some(pds) = self.product_definition_shapes.get(&shape) else {
                continue;
            };
            let Some(definition) = pds.definition else { continue };
            if let Some(record) = self.definitions.get_mut(&definition) {
                record.representation = Some(representation);
            }
            // Also accept the shape itself as a definition-less link (products whose PDS
            // points at a PRODUCT_DEFINITION_REFERENCE are rare; ignore silently).
        }
    }

    fn collect_nauos(&mut self) {
        for (&id, entity) in &self.file.entities {
            let names = type_names(entity);
            let is_usage = names.iter().any(|name| {
                name.contains("ASSEMBLY_USAGE_OCCURRENCE")
                    || *name == "ASSEMBLY_COMPONENT_USAGE"
                    || *name == "PRODUCT_DEFINITION_USAGE"
            });
            if !is_usage {
                continue;
            }
            let args = names
                .iter()
                .find_map(|name| component_args(entity, name))
                .unwrap_or(entity.args.as_slice());
            let parent = arg_ref(args, 3);
            let child = arg_ref(args, 4);
            if parent.is_none() || child.is_none() {
                self.warnings.push(format!(
                    "usage occurrence #{id} lacks a parent/child product definition; ignored"
                ));
                continue;
            }
            self.nauos.insert(
                id,
                NauoRecord {
                    occurrence_id: arg_string(args, 0).unwrap_or_default().trim().to_string(),
                    name: non_blank(arg_string(args, 1)).unwrap_or_default(),
                    parent,
                    child,
                },
            );
        }
    }

    fn collect_relationships(&mut self) {
        for (&id, entity) in &self.file.entities {
            let components = components(entity);
            let Some((_, relationship)) = components
                .iter()
                .find(|(name, _)| *name == "REPRESENTATION_RELATIONSHIP")
            else {
                continue;
            };
            let operator = components
                .iter()
                .find(|(name, _)| *name == "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
                .and_then(|(_, args)| arg_ref(args, 0));
            self.relationships.insert(
                id,
                RelationshipRecord {
                    rep1: arg_ref(relationship, 2),
                    rep2: arg_ref(relationship, 3),
                    operator,
                },
            );
        }
    }

    // -- occurrences and tree ----------------------------------------------

    fn collect_occurrences(&mut self) -> Result<Vec<Occurrence>, AssemblyError> {
        // NAUO entity -> its PRODUCT_DEFINITION_SHAPE (first one wins).
        let mut shape_of_usage: BTreeMap<u32, u32> = BTreeMap::new();
        for (&shape_id, pds) in &self.product_definition_shapes {
            let Some(definition) = pds.definition else { continue };
            if self.nauos.contains_key(&definition) {
                shape_of_usage.entry(definition).or_insert(shape_id);
            }
        }
        // shape -> relationship via CONTEXT_DEPENDENT_SHAPE_REPRESENTATION.
        let mut relationship_of_shape: BTreeMap<u32, u32> = BTreeMap::new();
        for entity in self.file.entities.values() {
            if !type_names(entity).contains(&"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION") {
                continue;
            }
            let args = component_args(entity, "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION")
                .unwrap_or(entity.args.as_slice());
            let (Some(relationship), Some(shape)) = (arg_ref(args, 0), arg_ref(args, 1)) else {
                continue;
            };
            if self.product_definition_shapes.contains_key(&shape) {
                relationship_of_shape.insert(shape, relationship);
            }
        }

        let nauos: Vec<(u32, NauoRecord)> = self
            .nauos
            .iter()
            .map(|(id, record)| (*id, record.clone()))
            .collect();
        let mut occurrences = Vec::with_capacity(nauos.len());
        for (nauo_id, nauo) in nauos {
            let (Some(parent), Some(child)) = (nauo.parent, nauo.child) else {
                continue;
            };
            if !self.definitions.contains_key(&parent) || !self.definitions.contains_key(&child) {
                return Err(AssemblyError::Malformed(format!(
                    "usage occurrence #{nauo_id} references a missing product definition \
                     (parent {parent}, child {child})"
                )));
            }
            let shape_id = shape_of_usage.get(&nauo_id).copied();
            let placement_name = shape_id
                .and_then(|shape| self.product_definition_shapes.get(&shape))
                .map(|pds| pds.name.clone())
                .filter(|name| !name.trim().is_empty());
            let relationship_id = shape_id.and_then(|shape| relationship_of_shape.get(&shape).copied());
            let relationship = relationship_id.and_then(|id| self.relationships.get(&id).copied());
            let local_transform = self.occurrence_transform(relationship, parent, child, nauo_id)?;
            occurrences.push(Occurrence {
                nauo: nauo_id,
                occurrence_id: nauo.occurrence_id.clone(),
                occurrence_name: non_blank(Some(nauo.name.as_str())),
                placement_name,
                parent_definition: parent,
                child_definition: child,
                local_transform,
            });
        }
        Ok(occurrences)
    }

    fn occurrence_transform(
        &mut self,
        relationship: Option<RelationshipRecord>,
        parent_definition: u32,
        child_definition: u32,
        nauo_id: u32,
    ) -> Result<Transform, AssemblyError> {
        let Some(relationship) = relationship else {
            self.warnings.push(format!(
                "usage occurrence #{nauo_id} has no placement relationship; local transform is identity"
            ));
            return Ok(Transform::IDENTITY);
        };
        let parent_representation = self
            .definitions
            .get(&parent_definition)
            .and_then(|record| record.representation);
        let child_representation = self
            .definitions
            .get(&child_definition)
            .and_then(|record| record.representation);
        // CAx-IF default: rep_1 is the child, rep_2 the parent. If the representations
        // match the product structure the other way round, flip.
        let (mut child_side, mut parent_side) = (relationship.rep1, relationship.rep2);
        if let (Some(parent_rep), Some(child_rep)) = (parent_representation, child_representation) {
            if child_side == Some(parent_rep) || parent_side == Some(child_rep) {
                child_side = relationship.rep2;
                parent_side = relationship.rep1;
            }
        }
        match relationship.operator {
            Some(operator) => {
                self.operator_transform(operator, child_side, parent_side, nauo_id)
            }
            None => Ok(Transform::IDENTITY),
        }
    }

    fn operator_transform(
        &mut self,
        operator: u32,
        child_representation: Option<u32>,
        parent_representation: Option<u32>,
        nauo_id: u32,
    ) -> Result<Transform, AssemblyError> {
        let entity = self.file.entities.get(&operator).ok_or_else(|| {
            AssemblyError::Malformed(format!(
                "usage occurrence #{nauo_id} references missing transformation operator #{operator}"
            ))
        })?;
        let names = type_names(entity);
        if names.contains(&"ITEM_DEFINED_TRANSFORMATION") {
            let args = component_args(entity, "ITEM_DEFINED_TRANSFORMATION")
                .unwrap_or(entity.args.as_slice());
            let item1 = arg_ref(args, 2);
            let item2 = arg_ref(args, 3);
            let (parent_item, child_item) = self.item_roles(
                item1,
                item2,
                child_representation,
                parent_representation,
            );
            let Some(parent_item) = parent_item else {
                return Err(AssemblyError::Malformed(format!(
                    "ITEM_DEFINED_TRANSFORMATION #{operator} has no parent placement item"
                )));
            };
            let Some(child_item) = child_item else {
                return Err(AssemblyError::Malformed(format!(
                    "ITEM_DEFINED_TRANSFORMATION #{operator} has no child placement item"
                )));
            };
            let parent_placement = self.placement(parent_item)?;
            let child_placement = self.placement(child_item)?;
            let child_inverse = child_placement.inverse().ok_or_else(|| {
                AssemblyError::Malformed(format!(
                    "ITEM_DEFINED_TRANSFORMATION #{operator} has a singular child placement"
                ))
            })?;
            return Ok(parent_placement.multiply(&child_inverse));
        }
        if names.contains(&"CARTESIAN_TRANSFORMATION_OPERATOR_3D") {
            let args = component_args(entity, "CARTESIAN_TRANSFORMATION_OPERATOR_3D")
                .unwrap_or(entity.args.as_slice());
            let x = self.direction_arg(args, 2)?;
            let y = self.direction_arg(args, 3)?;
            let z = self.direction_arg(args, 4)?;
            let origin = self.point_arg(args, 5)?.unwrap_or([0.0, 0.0, 0.0]);
            let scale = arg_number(args, 6).unwrap_or(1.0);
            return Ok(Transform::from_columns(
                [x[0] * scale, x[1] * scale, x[2] * scale],
                [y[0] * scale, y[1] * scale, y[2] * scale],
                [z[0] * scale, z[1] * scale, z[2] * scale],
                origin,
            ));
        }
        if names.contains(&"CARTESIAN_TRANSFORMATION_OPERATOR") {
            let args = component_args(entity, "CARTESIAN_TRANSFORMATION_OPERATOR")
                .unwrap_or(entity.args.as_slice());
            let x = self.direction_arg(args, 2)?;
            let y = self.direction_arg(args, 3)?;
            let origin = self.point_arg(args, 4)?.unwrap_or([0.0, 0.0, 0.0]);
            let scale = arg_number(args, 5).unwrap_or(1.0);
            let z = cross(x, y);
            return Ok(Transform::from_columns(
                [x[0] * scale, x[1] * scale, x[2] * scale],
                [y[0] * scale, y[1] * scale, y[2] * scale],
                [z[0] * scale, z[1] * scale, z[2] * scale],
                origin,
            ));
        }
        self.warnings.push(format!(
            "usage occurrence #{nauo_id} uses unsupported transformation operator #{operator} \
             ({}); local transform is identity",
            names.join("+")
        ));
        Ok(Transform::IDENTITY)
    }

    /// Which of `item_1`/`item_2` is the parent-side and which the child-side placement.
    fn item_roles(
        &self,
        item1: Option<u32>,
        item2: Option<u32>,
        child_representation: Option<u32>,
        parent_representation: Option<u32>,
    ) -> (Option<u32>, Option<u32>) {
        let parent_items = parent_representation
            .and_then(|representation| self.representation_items.get(&representation));
        let child_items = child_representation
            .and_then(|representation| self.representation_items.get(&representation));
        let in_parent = |item: Option<u32>| {
            item.is_some_and(|item| parent_items.is_some_and(|items| items.contains(&item)))
        };
        let in_child = |item: Option<u32>| {
            item.is_some_and(|item| child_items.is_some_and(|items| items.contains(&item)))
        };
        if in_parent(item2) && in_child(item1) {
            (item2, item1)
        } else if in_parent(item1) || in_child(item2) {
            (item1, item2)
        } else {
            // CAx-IF convention: item 2 is in the parent, item 1 in the child.
            (item2, item1)
        }
    }

    fn placement(&self, item: u32) -> Result<Transform, AssemblyError> {
        let entity = self.file.entities.get(&item).ok_or_else(|| {
            AssemblyError::Malformed(format!("placement item #{item} does not exist"))
        })?;
        let names = type_names(entity);
        let args = names
            .iter()
            .find_map(|name| component_args(entity, name))
            .unwrap_or(entity.args.as_slice());
        if !names.contains(&"AXIS2_PLACEMENT_3D") {
            return Err(AssemblyError::Malformed(format!(
                "transformation item #{item} is not an AXIS2_PLACEMENT_3D ({})",
                names.join("+")
            )));
        }
        let origin = self.point_arg(args, 1)?.unwrap_or([0.0, 0.0, 0.0]);
        let axis = self.direction_arg(args, 2).unwrap_or([0.0, 0.0, 1.0]);
        let reference = self.direction_arg(args, 3).unwrap_or([1.0, 0.0, 0.0]);
        Transform::from_placement(origin, axis, reference)
    }

    fn point_arg(
        &self,
        args: &[Value],
        index: usize,
    ) -> Result<Option<[f64; 3]>, AssemblyError> {
        let Some(point) = arg_ref(args, index) else { return Ok(None) };
        let entity = self.file.entities.get(&point).ok_or_else(|| {
            AssemblyError::Malformed(format!("CARTESIAN_POINT #{point} does not exist"))
        })?;
        if !type_names(entity).contains(&"CARTESIAN_POINT") {
            return Err(AssemblyError::Malformed(format!(
                "#{point} is not a CARTESIAN_POINT"
            )));
        }
        let Some(Value::List(values)) = entity.args.get(1) else {
            return Err(AssemblyError::Malformed(format!(
                "CARTESIAN_POINT #{point} has no coordinate list"
            )));
        };
        Ok(Some(coordinates(values)))
    }

    fn direction_arg(
        &self,
        args: &[Value],
        index: usize,
    ) -> Result<[f64; 3], AssemblyError> {
        let Some(direction) = arg_ref(args, index) else {
            return Ok([0.0, 0.0, 1.0]);
        };
        let entity = self.file.entities.get(&direction).ok_or_else(|| {
            AssemblyError::Malformed(format!("DIRECTION #{direction} does not exist"))
        })?;
        if !type_names(entity).contains(&"DIRECTION") {
            return Err(AssemblyError::Malformed(format!("#{direction} is not a DIRECTION")));
        }
        let Some(Value::List(values)) = entity.args.get(1) else {
            return Err(AssemblyError::Malformed(format!(
                "DIRECTION #{direction} has no direction ratios"
            )));
        };
        Ok(coordinates(values))
    }

    // -- tree ---------------------------------------------------------------

    fn build_tree(&mut self, occurrences: &[Occurrence]) -> Result<Vec<Node>, AssemblyError> {
        let mut parent_definitions: BTreeSet<u32> = BTreeSet::new();
        let mut child_definitions: BTreeSet<u32> = BTreeSet::new();
        let mut by_parent: BTreeMap<u32, Vec<&Occurrence>> = BTreeMap::new();
        for occurrence in occurrences {
            parent_definitions.insert(occurrence.parent_definition);
            child_definitions.insert(occurrence.child_definition);
            by_parent
                .entry(occurrence.parent_definition)
                .or_default()
                .push(occurrence);
        }
        for children in by_parent.values_mut() {
            children.sort_by_key(|occurrence| occurrence.nauo);
        }

        let mut root_definitions: Vec<u32> = parent_definitions
            .difference(&child_definitions)
            .copied()
            .collect();
        if root_definitions.is_empty() {
            if !occurrences.is_empty() {
                return Err(AssemblyError::Malformed(
                    "assembly structure is cyclic: every parent is also a child".into(),
                ));
            }
            // No occurrences at all: every definition with a shape representation is a
            // top-level part; fall back to every definition when none has a shape.
            root_definitions = self
                .definitions
                .iter()
                .filter(|(_, record)| record.representation.is_some())
                .map(|(id, _)| *id)
                .collect();
            if root_definitions.is_empty() {
                root_definitions = self.definitions.keys().copied().collect();
            }
        }

        let mut roots = Vec::with_capacity(root_definitions.len());
        for definition in root_definitions {
            let mut path = BTreeSet::new();
            roots.push(self.build_node(
                definition,
                None,
                Transform::IDENTITY,
                Transform::IDENTITY,
                &by_parent,
                &mut path,
            )?);
        }
        Ok(roots)
    }

    fn build_node(
        &mut self,
        definition: u32,
        occurrence: Option<&Occurrence>,
        local_transform: Transform,
        parent_world: Transform,
        by_parent: &BTreeMap<u32, Vec<&Occurrence>>,
        path: &mut BTreeSet<u32>,
    ) -> Result<Node, AssemblyError> {
        if !path.insert(definition) {
            return Err(AssemblyError::Malformed(format!(
                "assembly structure is cyclic at product definition #{definition}"
            )));
        }
        let product_name = self
            .definitions
            .get(&definition)
            .map(|record| self.product_name(record.product))
            .unwrap_or_default();
        let representation_id = self
            .definitions
            .get(&definition)
            .and_then(|record| record.representation);
        let world_transform = parent_world.multiply(&local_transform);
        let mut children = Vec::new();
        if let Some(nested) = by_parent.get(&definition) {
            for nested_occurrence in nested {
                children.push(self.build_node(
                    nested_occurrence.child_definition,
                    Some(nested_occurrence),
                    nested_occurrence.local_transform,
                    world_transform,
                    by_parent,
                    path,
                )?);
            }
        }
        path.remove(&definition);

        let (name, occurrence_id, occurrence_name, placement_name) = match occurrence {
            Some(occurrence) => {
                let name = occurrence
                    .placement_name
                    .clone()
                    .or_else(|| occurrence.occurrence_name.clone())
                    .filter(|name| !name.trim().is_empty())
                    .unwrap_or_else(|| product_name.clone());
                (
                    name,
                    Some(occurrence.occurrence_id.clone()),
                    occurrence.occurrence_name.clone(),
                    occurrence.placement_name.clone(),
                )
            }
            None => (product_name.clone(), None, None, None),
        };
        Ok(Node {
            name,
            product_name,
            occurrence_id,
            occurrence_name,
            placement_name,
            definition_id: Some(definition),
            representation_id,
            local_transform,
            world_transform,
            children,
        })
    }

    fn product_name(&self, product: Option<u32>) -> String {
        product
            .and_then(|product| self.products.get(&product))
            .map(|record| record.name.clone())
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// STEP value helpers
// ---------------------------------------------------------------------------

/// Entity type names: the plain entity name, or every typed component of a complex
/// instance (`#n=(A(...) B(...) C());`).
fn type_names(entity: &Entity) -> Vec<&str> {
    if !entity.name.is_empty() {
        return vec![entity.name.as_str()];
    }
    components(entity).into_iter().map(|(name, _)| name).collect()
}

/// The components of an entity: `(type name, arguments)` pairs. A plain entity is one
/// component; a complex instance contributes one per typed part.
fn components(entity: &Entity) -> Vec<(&str, &[Value])> {
    if !entity.name.is_empty() {
        return vec![(entity.name.as_str(), entity.args.as_slice())];
    }
    let mut out = Vec::new();
    let mut index = 0;
    while index < entity.args.len() {
        let Value::Typed(name, inline) = &entity.args[index] else {
            index += 1;
            continue;
        };
        if !inline.is_empty() {
            out.push((name.as_str(), inline.as_slice()));
            index += 1;
            continue;
        }
        if let Some(Value::List(values)) = entity.args.get(index + 1) {
            out.push((name.as_str(), values.as_slice()));
            index += 2;
        } else {
            out.push((name.as_str(), &[]));
            index += 1;
        }
    }
    out
}

/// Arguments of one named component of a (possibly complex) entity.
fn component_args<'b>(entity: &'b Entity, name: &str) -> Option<&'b [Value]> {
    components(entity)
        .into_iter()
        .find(|(component, _)| *component == name)
        .map(|(_, args)| args)
}

fn arg_ref(args: &[Value], index: usize) -> Option<u32> {
    match args.get(index) {
        Some(Value::Ref(id)) => Some(*id),
        _ => None,
    }
}

fn arg_string<'a>(args: &'a [Value], index: usize) -> Option<&'a str> {
    match args.get(index) {
        Some(Value::Str(text)) => Some(text.as_str()),
        _ => None,
    }
}

fn arg_number(args: &[Value], index: usize) -> Option<f64> {
    match args.get(index) {
        Some(Value::Number(number)) if number.is_finite() => Some(*number),
        _ => None,
    }
}

fn non_blank(text: Option<&str>) -> Option<String> {
    text.map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

/// `(x, y, z)` from a coordinate/ratio list, padding missing components with 0.
fn coordinates(values: &[Value]) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for (slot, value) in out.iter_mut().zip(values) {
        if let Value::Number(number) = value {
            if number.is_finite() {
                *slot = *number;
            }
        }
    }
    out
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
