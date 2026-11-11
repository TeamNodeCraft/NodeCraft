//! Document tabs, open/save lifecycle and native file dialogs.

use super::*;

/// Tabs must sit above title-bar drag handling but below menu/popup layers.
/// egui paints layers in `Order` order, then by creation order within an order;
/// using Foreground here let later-created tabs paint over the File menu.
const TAB_LAYER_ORDER: egui::Order = super::CONTENT_OVERLAY_ORDER;

#[derive(Clone)]
pub(super) struct TabDrag {
    pub(super) id: u64,
    pub(super) grab_offset: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TabViewMode {
    Split,
    NodesOnly,
    ViewportOnly,
}

#[derive(Clone)]
pub(super) struct DocumentTab {
    pub(super) id: u64,
    pub(super) document_id: String,
    pub(super) path: Option<PathBuf>,
    pub(super) selected_nodes: HashSet<String>,
    pub(super) selected_connections: HashSet<ConnectionKey>,
    pub(super) edit_target: Option<String>,
    pub(super) viewer_breakpoint: Option<String>,
    pub(super) view_mode: TabViewMode,
    pub(super) node_width: f32,
    pub(super) inspector_width: f32,
    pub(super) mesh: Option<Mesh>,
    pub(super) display: Option<(
        std::sync::Arc<nodecraft_display::service::DisplayFrame>,
        String,
        bool,
    )>,
    pub(super) status: String,
    pub(super) error: Option<String>,
    pub(super) evaluation_ms: f64,
    pub(super) cache_hits: u32,
    pub(super) cache_misses: u32,
    pub(super) view: Option<crate::viewport::ViewSnapshot>,
}


impl NodeCraftApp {
    pub(super) fn open(&mut self, path: PathBuf) {
        self.layout_animation = None;
        self.node_position_animation = None;
        let started = Instant::now();
        let document: GraphDocument = match nodecraft_session::store::load(&path) {
            Ok(document) => document,
            Err(error) => {
                self.error = Some(error);
                self.status = "Open failed".to_owned();
                return;
            }
        };
        let document_id = document_id_for_path(&path);
        let canonical = PathBuf::from(&document_id);
        let document_id = self
            .documents
            .id_for_path(&canonical)
            .map(str::to_owned)
            .unwrap_or(document_id);
        if let Some(index) = self.tabs.iter().position(|tab| tab.document_id == document_id) {
            self.remember_document(&path);
            self.activate_tab(index);
            return;
        }
        let session = Session::new(document);
        self.remember_document(&path);
        self.open_session(
            session,
            Some(path),
            format!("Opened in {:.1} ms", started.elapsed().as_secs_f64() * 1000.0),
        );
    }

    pub(super) fn remember_document(&mut self, path: &Path) {
        let absolute = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.recent_files.retain(|recent| recent != &absolute);
        self.recent_files.insert(0, absolute);
        self.recent_files.truncate(10);
    }

    pub(super) fn clear_recent_documents(&mut self) {
        self.recent_files.clear();
    }

    pub(super) fn enter_group(&mut self, group_id: &str) {
        let Some(parent_session) = &self.session else { return };
        let Some(node) = parent_session
            .document()
            .nodes
            .iter()
            .find(|node| node.id == group_id && node.node_type == "group")
            .cloned()
        else {
            return;
        };
        let Some(parent_document_id) = self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone()) else { return };
        // Double-clicking a graph that is already open focuses its existing
        // editing tab instead of creating a second session for the same body.
        if let Some(index) = self.tabs.iter().position(|tab| {
            self.group_tabs.get(&tab.id).is_some_and(|meta| {
                meta.view.document_id == parent_document_id && meta.group_id == group_id
            })
        }) {
            self.activate_tab(index);
            return;
        }
        let Ok(payload) = read_group(&node) else {
            self.error = Some(format!("Graph {group_id} has invalid body data"));
            return;
        };
        let child_document = GraphDocument {
            schema_version: 1,
            id: format!("{group_id}::group"),
            root: nodecraft_session::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes: payload.nodes,
                connections: payload.connections,
                interface: nodecraft_session::GraphInterface::default(),
            },
            definitions: Vec::new(),
        };
        let tab_id = self.next_tab_id;
        let source_tab = self.tabs.get(self.active_tab).cloned();
        let source_title = source_tab
            .as_ref()
            .map(|tab| self.document_title(tab))
            .unwrap_or_else(|| "Graph".to_owned());
        let inputs = payload.inputs;
        let outputs = payload.outputs;
        let collapsed = payload.collapsed;
        self.open_session(
            Session::new(child_document),
            None,
            format!("Editing graph {group_id}"),
        );
        self.group_tabs.insert(
            tab_id,
            GroupTab {
                view: ViewKey::root(parent_document_id),
                instance_path: vec![group_id.to_owned()],
                breadcrumb: vec![
                    BreadcrumbEntry { label: source_title, definition_id: None },
                    BreadcrumbEntry { label: payload.name.clone(), definition_id: None },
                ],
                group_id: group_id.to_owned(),
                inputs,
                outputs,
                collapsed,
            },
        );
    }

    pub(super) fn enter_graph(&mut self, graph_id: &str) {
        let Some(active_session) = &self.session else { return };
        let Some(node) = active_session
            .document()
            .nodes
            .iter()
            .find(|node| node.id == graph_id && node.node_type == "graph")
            .cloned()
        else {
            return;
        };
        let Some(reference) = GraphReference::from_node(&node) else {
            self.error = Some(format!("Graph {graph_id} has no definition"));
            return;
        };
        let instance_base = self
            .tabs
            .get(self.active_tab)
            .and_then(|tab| self.group_tabs.get(&tab.id))
            .map(|meta| meta.instance_path.clone())
            .unwrap_or_default();
        let (source_document_id, definition, breadcrumb_base) = match reference {
            GraphReference::Local(definition_id) => {
                let active_document_id = self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone());
                let source_document_id = active_document_id
                    .as_ref()
                    .and_then(|id| {
                        self.tabs
                            .iter()
                            .find(|tab| &tab.document_id == id)
                            .and_then(|tab| self.group_tabs.get(&tab.id))
                            .map(|meta| meta.view.document_id.clone())
                    })
                    .or(active_document_id)
                    .unwrap_or_default();
                let parent_document = if self
                    .tabs
                    .get(self.active_tab)
                    .is_some_and(|tab| tab.document_id == source_document_id)
                {
                    self.session.as_ref().map(Session::document)
                } else {
                    self.documents.session(&source_document_id).map(Session::document)
                };
                let Some(definition) = parent_document
                    .and_then(|document| document.definition(&definition_id))
                    .cloned()
                else {
                    self.error = Some(format!("Missing graph definition {definition_id}"));
                    return;
                };
                let breadcrumb_base = self
                    .tabs
                    .get(self.active_tab)
                    .and_then(|tab| self.group_tabs.get(&tab.id))
                    .map(|meta| meta.breadcrumb.clone())
                    .or_else(|| {
                        self.tabs
                            .get(self.active_tab)
                            .map(|tab| {
                                vec![BreadcrumbEntry {
                                    label: self.document_title(tab),
                                    definition_id: None,
                                }]
                            })
                    })
                    .unwrap_or_default();
                (source_document_id, definition, breadcrumb_base)
            }
            GraphReference::External { path, definition } => {
                self.open(PathBuf::from(path));
                if definition == "root" {
                    return;
                }
                let Some(source_document_id) =
                    self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone())
                else {
                    return;
                };
                let Some(definition) = self
                    .session
                    .as_ref()
                    .and_then(|session| session.document().definition(&definition))
                    .cloned()
                else {
                    self.error = Some(format!("Missing external graph definition {definition}"));
                    return;
                };
                let breadcrumb_base = self
                    .tabs
                    .get(self.active_tab)
                    .map(|tab| {
                        vec![BreadcrumbEntry {
                            label: self.document_title(tab),
                            definition_id: None,
                        }]
                    })
                    .unwrap_or_default();
                (source_document_id, definition, breadcrumb_base)
            }
        };
        let definition_id = definition.id.clone();
        let projection_definitions = self
            .documents
            .session(&source_document_id)
            .map(|session| session.document().definitions.clone())
            .unwrap_or_default();
        if let Some(index) = self.tabs.iter().position(|tab| {
            self.group_tabs.get(&tab.id).is_some_and(|meta| {
                meta.view.document_id == source_document_id
                    && meta.view.definition_id.as_deref() == Some(definition_id.as_str())
            })
        }) {
            self.activate_tab(index);
            return;
        }
        let child_document = GraphDocument {
            schema_version: 1,
            id: format!("{definition_id}::graph"),
            root: nodecraft_session::GraphDefinition {
                id: definition.id.clone(),
                name: definition.name.clone(),
                kind: definition.kind.clone(),
                interface: definition.interface.clone(),
                nodes: definition.nodes.clone(),
                connections: definition.connections.clone(),
            },
            definitions: projection_definitions,
        };
        let tab_id = self.next_tab_id;
        self.open_session(
            Session::new(child_document),
            None,
            format!("Editing graph {}", definition.name),
        );
        self.group_tabs.insert(
            tab_id,
            GroupTab {
                view: ViewKey::definition(source_document_id, definition_id.clone()),
                instance_path: instance_base
                    .into_iter()
                    .chain(std::iter::once(graph_id.to_owned()))
                    .collect(),
                breadcrumb: breadcrumb_base
                    .into_iter()
                    .chain(std::iter::once(BreadcrumbEntry {
                        label: definition.name.clone(),
                        definition_id: Some(definition_id.clone()),
                    }))
                    .collect(),
                group_id: graph_id.to_owned(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                collapsed: false,
            },
        );
        // The active projection lives only in `self.session`; it is not a
        // document session in the store. Activation rebuilds it from the parent.
        if let Some(tab) = self.tabs.get(self.active_tab) {
            self.documents.remove(&tab.document_id);
        }
    }

    pub(super) fn sync_group_tab(&mut self, tab_id: u64) {
        let Some(meta) = self.group_tabs.get(&tab_id).cloned() else { return };
        let Some(tab) = self.tabs.iter().find(|tab| tab.id == tab_id) else { return };
        let Some(child) = self.documents.session(&tab.document_id) else { return };
        if meta.view.definition_id.is_some() {
            // Definition tabs edit the parent session directly; the child document
            // is a projection and has no unsynced edits to flush.
            return;
        }
        let name = self
            .documents
            .session(&meta.view.document_id)
            .and_then(|parent| parent.document().nodes.iter().find(|node| node.id == meta.group_id))
            .and_then(|node| node.values.get("name").and_then(serde_json::Value::as_str))
            .unwrap_or_default()
            .to_owned();
        let payload = GroupPayload {
            collapsed: meta.collapsed,
            name,
            nodes: child.document().nodes.clone(),
            connections: child.document().connections.clone(),
            inputs: meta.inputs.clone(),
            outputs: meta.outputs.clone(),
        };
        let Ok(values) = group_values(&payload) else { return };
        let Some(mut parent) = self.documents.session(&meta.view.document_id).cloned() else { return };
        let fields = ["nodes", "connections", "inputs", "outputs", "collapsed"];
        let commands: Vec<Command> = fields
            .iter()
            .filter_map(|field| {
                let value = values.get(*field)?.clone();
                Some(Command::SetValue { node: meta.group_id.clone(), field: (*field).to_owned(), value })
            })
            .collect();
        if parent.apply_commands(&commands).is_ok() {
            let _ = self.documents.set_session(meta.view.document_id.clone(), parent);
        }
    }

    pub(super) fn save_active_tab(&mut self) {
        self.save_active_tab_impl(false);
    }

    pub(super) fn save_active_tab_impl(&mut self, move_mesh: bool) {
        let Some(session) = self.session.clone() else { return };
        let Some(document_id) = self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone()) else { return };
        let definition_view = self
            .tabs
            .get(self.active_tab)
            .and_then(|tab| self.group_tabs.get(&tab.id))
            .is_some_and(|meta| meta.view.definition_id.is_some());
        if definition_view {
            // The projection is rebuilt from the parent definition on activation;
            // keeping it in DocumentStore would make it a second document session.
            self.documents.remove(&document_id);
        } else {
            if session.revision() != self.active_session_revision {
                let _ = self.documents.set_session(document_id.clone(), session);
                self.active_session_revision = self.session.as_ref().map_or(0, Session::revision);
            }
            if let Some(path) = &self.path {
                self.documents.set_path(&document_id, path);
            }
        }
        let mesh = if move_mesh { self.mesh.take() } else { self.mesh.clone() };
        let display = if move_mesh { self.last_applied_display.take() } else { self.last_applied_display.clone() };
        let Some(tab) = self.tabs.get_mut(self.active_tab) else { return };
        tab.path = self.path.clone();
        tab.selected_nodes = self.selected_nodes.clone();
        tab.selected_connections = self.selected_connections.clone();
        tab.edit_target = self.edit_target.clone();
        tab.viewer_breakpoint = self.viewer_breakpoint.clone();
        tab.mesh = mesh;
        tab.display = display;
        tab.status = self.status.clone();
        tab.error = self.error.clone();
        tab.evaluation_ms = self.evaluation_ms;
        tab.cache_hits = self.cache_hits;
        tab.cache_misses = self.cache_misses;
        tab.view = self.viewport.view_snapshot();
        tab.node_width = self.node_editor.panel_width();
        tab.inspector_width = self.inspector_width;
    }

    /// Open another tab over the same document session, with its own layout,
    /// selection, camera and history scope.
    pub(super) fn duplicate_active_tab(&mut self) {
        let Some(source) = self.tabs.get(self.active_tab).cloned() else { return };
        let group_meta = self.group_tabs.get(&source.id).cloned();
        let Some(session) = self.session.clone() else { return };
        self.save_active_tab_impl(false);
        let tab_id = self.next_tab_id;
        self.next_tab_id += 1;
        let document_id = if group_meta.is_some() {
            format!("{}::view{}", source.document_id, tab_id)
        } else {
            source.document_id.clone()
        };
        let tab = DocumentTab {
            id: tab_id,
            document_id,
            path: source.path.clone(),
            selected_nodes: HashSet::new(),
            selected_connections: HashSet::new(),
            edit_target: None,
            viewer_breakpoint: source.viewer_breakpoint.clone(),
            view_mode: source.view_mode,
            node_width: source.node_width,
            inspector_width: source.inspector_width,
            mesh: None,
            display: None,
            status: "Duplicated view".to_owned(),
            error: None,
            evaluation_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            view: source.view,
        };
        self.tabs.push(tab);
        if let Some(meta) = group_meta {
            self.group_tabs.insert(tab_id, meta);
        }
        self.active_tab = self.tabs.len().saturating_sub(1);
        self.install_session(Some(session.clone()));
        self.active_session_revision = session.revision();
        self.path = source.path.clone();
        self.selected_nodes.clear();
        self.selected_connections.clear();
        self.edit_target = None;
        self.viewer_breakpoint = source.viewer_breakpoint.clone();
        self.mesh = None;
        self.last_eval_key = None;
        if let Some(view) = source.view {
            self.viewport.restore_view(view);
            self.camera_initialized = true;
        }
        self.evaluate();
    }

    pub(super) fn save_document(&mut self, ctx: &egui::Context, save_as: bool) {
        if self.session.is_none() {
            self.error = Some("No document loaded to save".to_owned());
            self.status = "Save failed".to_owned();
            return;
        }
        if save_as || self.path.is_none() {
            let suggested_name = self
                .path
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                .unwrap_or("untitled.nodecraft")
                .to_owned();
            self.start_dialog(
                ctx,
                DialogAction::SaveDocument,
                FileDialogKind::SaveDocument { suggested_name },
            );
            return;
        }
        let path = self.path.clone().unwrap_or_default();
        self.finish_save_document(path);
    }

    pub(super) fn finish_save_document(&mut self, path: PathBuf) {
        let path = normalize_document_path(path);
        let Some(session) = self.session.clone() else {
            self.error = Some("No document loaded to save".to_owned());
            self.status = "Save failed".to_owned();
            return;
        };
        if let Err(error) = nodecraft_session::store::save_atomic(&path, session.document()) {
            self.error = Some(error);
            self.status = "Save failed".to_owned();
            return;
        }
        self.path = Some(path.clone());
        self.remember_document(&path);
        if let Some(tab) = self.tabs.get(self.active_tab) {
            let document_id = tab.document_id.clone();
            let _ = self.documents.set_session(document_id.clone(), session);
            self.documents.set_path(&document_id, &path);
        }
        self.save_active_tab();
        self.status = format!("Saved {}", path.display());
        self.error = None;
    }

    /// Start a native dialog off the event thread. Ignored while another dialog
    /// is already open.
    pub(super) fn start_dialog(&mut self, ctx: &egui::Context, action: DialogAction, kind: FileDialogKind) {
        if self.pending_dialog.is_some() {
            return;
        }
        self.pending_dialog = Some(PendingDialogState {
            action,
            dialog: PendingFileDialog::spawn(kind),
            status_before: self.status.clone(),
        });
        self.status = "Choose a file…".to_owned();
        ctx.request_repaint();
    }

    /// Fetch a finished dialog result and apply its action. Called once per frame.
    pub(super) fn poll_dialog(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending_dialog.as_mut() else { return };
        let Some(result) = pending.dialog.try_finish() else {
            // Keep frames coming while the modal panel is up.
            ctx.request_repaint_after(Duration::from_millis(50));
            return;
        };
        let Some(state) = self.pending_dialog.take() else { return };
        self.status = state.status_before;
        match (state.action, result) {
            (DialogAction::SaveDocument, FileDialogResult::Chosen(path)) => {
                self.finish_save_document(path);
            }
            (DialogAction::OpenDocument, FileDialogResult::Chosen(path)) => {
                self.open(path);
            }
            (
                DialogAction::SetNodePath { node, field, open_linked_part },
                FileDialogResult::Chosen(path),
            ) => {
                let coalesce = if open_linked_part {
                    "linked-part-path"
                } else if field == "directory" {
                    "step-directory"
                } else {
                    "step-path"
                };
                let command = Command::SetValue {
                    node: node.clone(),
                    field: field.to_owned(),
                    value: serde_json::Value::String(path.display().to_string()),
                };
                if self.apply_commands(&[command], Some(coalesce)) {
                    if open_linked_part {
                        self.open_linked_part_document(&node);
                    }
                    self.evaluate();
                }
            }
            (_, FileDialogResult::Cancelled) => {}
        }
        ctx.request_repaint();
    }

    pub(super) fn restore_active_tab(&mut self) {
        self.pending_dirty.clear();
        self.eval_progress = None;
        let Some(tab) = self.tabs.get_mut(self.active_tab) else { return };
        let document_id = tab.document_id.clone();
        let path = tab.path.clone();
        let selected_nodes = std::mem::take(&mut tab.selected_nodes);
        let selected_connections = std::mem::take(&mut tab.selected_connections);
        let edit_target = tab.edit_target.take();
        let viewer_breakpoint = tab.viewer_breakpoint.take();
        let mesh = tab.mesh.take();
        let display = tab.display.take();
        let status = tab.status.clone();
        let error = tab.error.clone();
        let evaluation_ms = tab.evaluation_ms;
        let cache_hits = tab.cache_hits;
        let cache_misses = tab.cache_misses;
        let view = tab.view;
        let node_width = tab.node_width;
        let inspector_width = tab.inspector_width;
        let definition_view = self
            .tabs
            .get(self.active_tab)
            .and_then(|tab| self.group_tabs.get(&tab.id))
            .and_then(|meta| meta.view.definition_id.clone());
        let session = if let Some(definition_id) = definition_view {
            let Some(meta) = self
                .tabs
                .get(self.active_tab)
                .and_then(|tab| self.group_tabs.get(&tab.id))
                .cloned()
            else {
                self.install_session(None);
                return;
            };
            let Some((definition, definitions)) = self
                .documents
                .session(&meta.view.document_id)
                .and_then(|parent| {
                    parent
                        .document()
                        .definition(&definition_id)
                        .cloned()
                        .map(|definition| (definition, parent.document().definitions.clone()))
                })
            else {
                self.install_session(None);
                return;
            };
            Some(Session::new(GraphDocument {
                schema_version: 1,
                id: format!("{definition_id}::graph"),
                root: nodecraft_session::GraphDefinition {
                    id: definition.id.clone(),
                    name: definition.name.clone(),
                    kind: definition.kind.clone(),
                    interface: definition.interface.clone(),
                    nodes: definition.nodes.clone(),
                    connections: definition.connections.clone(),
                },
                definitions,
            }))
        } else {
            self.documents.session(&document_id).cloned()
        };
        let Some(session) = session else {
            self.install_session(None);
            return;
        };
        self.active_session_revision = session.revision();
        self.install_session(Some(session));
        self.node_editor.set_panel_width(node_width);
        if inspector_width > 0.0 {
            self.inspector_width = inspector_width;
        }
        self.path = path;
        self.selected_nodes = selected_nodes;
        self.selected_connections = selected_connections;
        self.edit_target = edit_target;
        self.viewer_breakpoint = viewer_breakpoint;
        self.mesh = mesh;
        self.status = status;
        self.error = error;
        self.evaluation_ms = evaluation_ms;
        self.cache_hits = cache_hits;
        self.cache_misses = cache_misses;
        if let Some(view) = view {
            self.preview_detail = view.quality_detail;
            self.viewport.restore_view(view);
            self.camera_initialized = true;
        } else {
            self.camera_initialized = false;
            self.viewport.set_mesh(self.mesh.as_ref());
        }
        if let Some(path) = self.path.clone() {
            self.remember_document(&path);
        }
        self.node_editor.reset_view();
        self.eval_pending = None;
        self.layout_animation = None;
        self.node_position_animation = None;
        if let Some((frame, target, feature_ghost)) = display {
            self.applying_deferred_display = true;
            self.apply_display_frame(&frame, &target, feature_ghost, true);
            self.applying_deferred_display = false;
        } else if self.mesh.is_some() {
            self.viewport
                .set_mesh_preserving_camera(self.mesh.as_ref());
        }
        self.evaluate();
    }

    /// The graph node that owns a group tab's body. Used as the inspector
    /// fallback when the body has no selected node or cable.
    pub(super) fn group_owner_node(&self, tab_id: u64) -> Option<(String, GraphNode)> {
        let meta = self.group_tabs.get(&tab_id)?;
        let document = self.documents.session(&meta.view.document_id)?.document();
        let node = document.nodes.iter().find(|node| node.id == meta.group_id)?.clone();
        Some((meta.view.document_id.clone(), node))
    }

    pub(super) fn document_title(&self, tab: &DocumentTab) -> String {
        if let Some(meta) = self.group_tabs.get(&tab.id) {
            let parent_document = if self
                .tabs
                .get(self.active_tab)
                .is_some_and(|active| active.document_id == meta.view.document_id)
            {
                self.session.as_ref().map(Session::document)
            } else {
                self.documents
                    .session(&meta.view.document_id)
                    .map(Session::document)
            };
            if let Some(definition_id) = meta.view.definition_id.as_deref() {
                let name = parent_document
                    .and_then(|document| document.definition(definition_id))
                    .map(|definition| definition.name.clone())
                    .filter(|name| !name.is_empty());
                return name.unwrap_or_else(|| format!("Graph {definition_id}"));
            }
            let name = parent_document
                .and_then(|document| document.nodes.iter().find(|node| node.id == meta.group_id))
                .and_then(|node| node.values.get("name").and_then(serde_json::Value::as_str))
                .filter(|name| !name.is_empty());
            return name
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Graph {}", meta.group_id));
        }
        tab.path
            .as_ref()
            .and_then(|path| path.file_stem().and_then(|name| name.to_str()))
            .unwrap_or("Untitled")
            .to_owned()
    }

    fn graph_tab_icon(&mut self, ctx: &egui::Context, theme: &crate::theme::Theme) -> egui::TextureHandle {
        let colour = theme.color(theme.text);
        if let Some((cached, texture)) = &self.graph_tab_icon
            && *cached == colour
        {
            return texture.clone();
        }
        let hex = format!("#{:02x}{:02x}{:02x}", colour.r(), colour.g(), colour.b());
        let svg = std::str::from_utf8(include_bytes!(
            "../../../../../vendor/iconify-design/nodes/share-network.svg"
        ))
        .unwrap_or_default()
        .replace("currentColor", &hex);
        let image = egui_extras::image::load_svg_bytes_with_size(
            svg.as_bytes(),
            egui::SizeHint::Size {
                width: 64,
                height: 64,
                maintain_aspect_ratio: true,
            },
            &Default::default(),
        )
        .unwrap_or_else(|_| egui::ColorImage::example());
        let texture = ctx.load_texture("graph-tab-icon", image, egui::TextureOptions::LINEAR);
        self.graph_tab_icon = Some((colour, texture.clone()));
        texture
    }

    pub(super) fn document_tabs(
        &mut self,
        ui: &mut egui::Ui,
        theme: &crate::theme::Theme,
    ) -> (Option<usize>, Option<usize>, Option<(usize, usize)>) {
        struct TabMeta {
            id: u64,
            title: String,
            width: f32,
            active: bool,
            graph: bool,
        }
        let meta: Vec<TabMeta> = self
            .tabs
            .iter()
            .enumerate()
            .map(|(index, tab)| {
                let title = self.document_title(tab);
                let graph = self
                    .group_tabs
                    .get(&tab.id)
                    .is_some_and(|meta| meta.view.definition_id.is_some());
                let width = ui
                    .painter()
                    .layout_no_wrap(title.clone(), egui::FontId::proportional(13.0), egui::Color32::WHITE)
                    .size()
                    .x
                    + if graph { 60.0 } else { 42.0 };
                TabMeta { id: tab.id, title, width, active: index == self.active_tab, graph }
            })
            .collect();
        if meta.is_empty() {
            self.tab_drag = None;
            self.tab_strip = None;
            return (None, None, None);
        }
        let gap = 4.0;
        // Keep the strip at the real tab layout width. The dragged tab ghost
        // follows the pointer and must not push the "+" button to the right.
        let total = meta.iter().map(|tab| tab.width + gap).sum::<f32>();
        let (strip, _) = ui.allocate_exact_size(egui::vec2(total, 30.0), egui::Sense::hover());
        self.tab_strip = Some(strip);
        // Tabs live above the title-bar drag interaction but below menu and
        // popup layers. Using Foreground here made tabs paint over an open File
        // menu because both layers had the same order and the tab layer was
        // created later.
        let mut tab_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(strip)
                .layer_id(egui::LayerId::new(
                    TAB_LAYER_ORDER,
                    egui::Id::new("document-tabs-layer"),
                )),
        );
        let ui = &mut tab_ui;
        let start = strip.min.x;
        let dragging_id = self.tab_drag.as_ref().map(|drag| drag.id);
        let drag_index = dragging_id.and_then(|id| meta.iter().position(|tab| tab.id == id));
        let pointer = ui.ctx().pointer_latest_pos();
        let dragged_centre = match (drag_index, self.tab_drag.as_ref(), pointer) {
            (Some(index), Some(drag), Some(pointer)) => {
                Some(pointer.x - drag.grab_offset + meta[index].width * 0.5)
            }
            _ => None,
        };
        let insertion = drag_index
            .map(|index| {
                let mut count = 0;
                let mut x = start;
                for (other, tab) in meta.iter().enumerate() {
                    if other == index {
                        continue;
                    }
                    if dragged_centre.is_some_and(|centre| centre < x + tab.width * 0.5) {
                        break;
                    }
                    x += tab.width + gap;
                    count += 1;
                }
                count
            })
            .unwrap_or(0);

        let mut targets = vec![start; meta.len()];
        if let (Some(index), Some(_)) = (drag_index, dragging_id) {
            let dragged_width = meta[index].width + gap;
            let mut x = start;
            let mut seen = 0;
            for (slot, tab) in meta.iter().enumerate() {
                if slot == index {
                    continue;
                }
                if seen == insertion {
                    x += dragged_width;
                }
                targets[slot] = x;
                x += tab.width + gap;
                seen += 1;
            }
        } else {
            let mut x = start;
            for (slot, tab) in meta.iter().enumerate() {
                targets[slot] = x;
                x += tab.width + gap;
            }
        }

        let dt = ui.input(|input| input.stable_dt).clamp(0.0, 0.1);
        let blend = 1.0 - (-18.0 * dt).exp();
        let mut animating = false;
        let mut drag = self.tab_drag.clone();
        for (index, tab) in meta.iter().enumerate() {
            let current = self.tab_positions.entry(tab.id).or_insert(targets[index]);
            if drag.as_ref().is_some_and(|drag| drag.id == tab.id) {
                if let (Some(pointer), Some(drag)) = (pointer, drag.as_ref()) {
                    *current = pointer.x - drag.grab_offset;
                }
            } else {
                let delta = targets[index] - *current;
                if delta.abs() < 0.35 {
                    *current = targets[index];
                } else {
                    *current += delta * blend;
                    animating = true;
                }
            }
        }
        if animating && self.tab_drag.is_none() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(16));
        }

        let graph_tab_icon = if meta.iter().any(|tab| tab.graph) {
            Some(self.graph_tab_icon(ui.ctx(), theme))
        } else {
            None
        };
        let mut requested_tab = None;
        let mut requested_close = None;
        let mut requested_move = None;
        let reorderable = meta.len() > 1;
        let pressed = ui.input(|input| input.pointer.primary_pressed());
        for (index, tab) in meta.iter().enumerate() {
            if dragging_id == Some(tab.id) {
                continue;
            }
            let x = *self.tab_positions.get(&tab.id).unwrap_or(&targets[index]);
            let rect = egui::Rect::from_min_size(egui::pos2(x, strip.min.y), egui::vec2(tab.width, 30.0));
            let fill = theme.color(if tab.active { theme.control } else { theme.window });
            let response = ui.interact(
                rect,
                egui::Id::new(("document-tab", tab.id)),
                egui::Sense::click_and_drag(),
            );
            ui.painter().add(ui_shape::shape(rect, fill, egui::Stroke::NONE));
            if tab.graph
                && let Some(icon) = graph_tab_icon.as_ref()
            {
                let icon_rect = egui::Rect::from_center_size(
                    egui::Pos2::new(rect.left() + 18.0, rect.center().y),
                    egui::Vec2::splat(14.0),
                );
                ui.painter().image(
                    icon.id(),
                    icon_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
            let galley = ui.painter().layout_no_wrap(
                tab.title.clone(),
                egui::FontId::proportional(13.0),
                theme.color(if tab.active { theme.text } else { theme.muted }),
            );
            let text_colour = theme.color(if tab.active { theme.text } else { theme.muted });
            let text_offset = if tab.graph { egui::vec2(10.0, 0.0) } else { egui::Vec2::ZERO };
            ui.painter().galley(rect.center() - galley.size() * 0.5 + text_offset, galley, text_colour);
            let mut close_hovered = false;
            // Only the active tab exposes its close button.
            if tab.active {
                let close_rect = egui::Rect::from_center_size(
                    egui::Pos2::new(rect.right() - 14.0, rect.center().y),
                    egui::Vec2::splat(20.0),
                );
                let close = ui.interact(
                    close_rect,
                    egui::Id::new(("close-document-tab", tab.id)),
                    egui::Sense::click(),
                );
                close_hovered = close.hovered();
                if close_hovered {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                if response.hovered() || close.hovered() {
                    let centre = close_rect.center();
                    let arm = 4.0;
                    let stroke = egui::Stroke::new(
                        1.4,
                        if close.hovered() { theme.color(theme.text) } else { theme.color(theme.muted) },
                    );
                    ui.painter().line_segment([centre + egui::vec2(-arm, -arm), centre + egui::vec2(arm, arm)], stroke);
                    ui.painter().line_segment([centre + egui::vec2(arm, -arm), centre + egui::vec2(-arm, arm)], stroke);
                }
                if close.clicked() {
                    requested_close = Some(index);
                }
            }
            if reorderable
                && response.drag_started()
                && let Some(pointer) = pointer
            {
                drag = Some(TabDrag { id: tab.id, grab_offset: pointer.x - x });
            } else if !reorderable
                && cfg!(target_os = "macos")
                && response.drag_started()
            {
                // A lone tab has no reorder destination; its drag is a window
                // drag. Start it directly so the foreground tab layer cannot
                // swallow the gesture before the title-bar interaction sees it.
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
            if requested_close == Some(index) {
                drag = None;
            } else if pressed && response.hovered() && !close_hovered {
                // Chrome-style selection happens on press, so a drag begins
                // from an already-active tab.
                requested_tab = Some(index);
            }
        }
        self.tab_drag = drag;
        if let Some(drag) = &self.tab_drag
            && !ui.input(|input| input.pointer.primary_down())
            && let Some(from) = self.tabs.iter().position(|tab| tab.id == drag.id)
            && let Some(drag_index) = meta.iter().position(|tab| tab.id == drag.id)
        {
            let mut k = 0;
            let mut x = start;
            let centre = meta[drag_index].width * 0.5
                + pointer.map_or(0.0, |pointer| pointer.x - drag.grab_offset);
            for (index, tab) in meta.iter().enumerate() {
                if index == drag_index {
                    continue;
                }
                if centre < x + tab.width * 0.5 {
                    break;
                }
                x += tab.width + gap;
                k += 1;
            }
            let to = if k <= from { k } else { k + 1 };
            requested_move = Some((from, to));
            self.tab_drag = None;
        }
        if self.tab_drag.is_some() {
            ui.ctx().request_repaint();
        }

        if let Some(drag) = &self.tab_drag
            && let Some(index) = meta.iter().position(|tab| tab.id == drag.id)
            && let Some(pointer) = pointer
        {
            let tab = &meta[index];
            let x = pointer.x - drag.grab_offset;
            let rect = egui::Rect::from_min_size(
                egui::pos2(x, strip.min.y),
                egui::vec2(tab.width, 30.0),
            );
            ui.painter().rect_filled(
                rect.translate(egui::vec2(0.0, 3.0)),
                egui::CornerRadius::same(8),
                egui::Color32::from_black_alpha(90),
            );
            ui.painter().add(ui_shape::shape(
                rect,
                theme.color(theme.control),
                egui::Stroke::new(1.0, theme.with_alpha(theme.border_strong, 220)),
            ));
            if tab.graph
                && let Some(icon) = graph_tab_icon.as_ref()
            {
                let icon_rect = egui::Rect::from_center_size(
                    egui::Pos2::new(rect.left() + 18.0, rect.center().y),
                    egui::Vec2::splat(14.0),
                );
                ui.painter().image(
                    icon.id(),
                    icon_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
            let galley = ui.painter().layout_no_wrap(
                tab.title.clone(),
                egui::FontId::proportional(13.0),
                theme.color(theme.text),
            );
            let text_offset = if tab.graph { egui::vec2(10.0, 0.0) } else { egui::Vec2::ZERO };
            ui.painter().galley(rect.center() - galley.size() * 0.5 + text_offset, galley, theme.color(theme.text));
        }

        (requested_tab, requested_close, requested_move)
    }

    pub(super) fn move_tab(&mut self, from: usize, to: usize) {
        let Some((insert, active)) =
            reorder_tab_index(self.tabs.len(), from, to, self.active_tab)
        else {
            return;
        };
        let tab = self.tabs.remove(from);
        self.tabs.insert(insert, tab);
        self.active_tab = active;
    }

    pub(super) fn close_tab(&mut self, index: usize) {
        if let Some(tab) = self.tabs.get(index) {
            let tab_id = tab.id;
            self.sync_group_tab(tab_id);
            self.group_tabs.remove(&tab_id);
        }
        if index >= self.tabs.len() {
            return;
        }
        self.save_active_tab();
        let closing_active = index == self.active_tab;
        let removed = self.tabs.remove(index);
        if !self.tabs.iter().any(|tab| tab.document_id == removed.document_id) {
            self.documents.remove(&removed.document_id);
        }
        if self.tabs.is_empty() {
            self.install_session(None);
            self.path = None;
            self.selected_nodes.clear();
            self.selected_connections.clear();
            self.viewer_breakpoint = None;
            self.new_blank();
            return;
        }
        if closing_active {
            self.active_tab = index.min(self.tabs.len() - 1);
            self.restore_active_tab();
            if self.mesh.is_none() && self.session.is_some() {
                self.evaluate();
            }
        } else if index < self.active_tab {
            self.active_tab -= 1;
        }
    }

    /// Replace the graph in the active tab without opening another tab.
    pub(super) fn replace_active_session(&mut self, session: Session, status: String) {
        let selected_nodes: HashSet<String> = initial_viewer_target(session.document())
            .into_iter()
            .collect();
        let viewer_breakpoint = selected_nodes.iter().next().cloned();
        self.install_session(Some(session.clone()));
        self.pending_dirty.clear();
        self.eval_progress = None;
        self.path = None;
        self.selected_nodes = selected_nodes.clone();
        self.selected_connections.clear();
        self.edit_target = None;
        self.viewer_breakpoint = viewer_breakpoint.clone();
        self.mesh = None;
        self.camera_initialized = false;
        self.viewport.set_mesh(None);
        self.node_editor.reset_interaction();
        self.node_editor.reset_view();
        self.error = None;
        self.status = status;
        self.eval_pending = None;
        self.layout_animation = None;
        self.node_position_animation = None;
        if let Some(tab) = self.tabs.get_mut(self.active_tab) {
            let document_id = tab.document_id.clone();
            let _ = self.documents.set_session(document_id.clone(), session.clone());
            self.documents.clear_path(&document_id);
            tab.path = None;
            tab.selected_nodes = selected_nodes;
            tab.selected_connections.clear();
            tab.viewer_breakpoint = viewer_breakpoint;
            tab.mesh = None;
            tab.status = self.status.clone();
            tab.error = None;
            tab.evaluation_ms = 0.0;
            tab.cache_hits = 0;
            tab.cache_misses = 0;
        } else {
            self.open_session(self.session.clone().expect("session was just set"), None, self.status.clone());
            return;
        }
        self.evaluate();
    }

    pub(super) fn open_linked_part_document(&mut self, node_id: &str) {
        let Some(session) = &self.session else { return };
        let Some(node) = session.document().nodes.iter().find(|node| node.id == node_id) else { return };
        let path = node.values.get("path").and_then(serde_json::Value::as_str).unwrap_or_default().to_owned();
        if path.is_empty() {
            self.error = Some("Linked part has no .nodecraft path".to_owned());
            return;
        }
        let path = normalize_document_path(PathBuf::from(path));
        let document_id = document_id_for_path(&path);
        if let Some(index) = self.tabs.iter().position(|tab| tab.document_id == document_id) {
            self.activate_tab(index);
            return;
        }
        match nodecraft_session::store::load(&path) {
            Ok(document) => {
                self.remember_document(&path);
                self.open_session(
                    Session::new(document),
                    Some(path.clone()),
                    format!("Opened linked document {}", path.display()),
                );
            }
            Err(error) => {
                self.error = Some(format!("Linked document: {error}"));
                self.status = "Linked document failed to load".to_owned();
            }
        }
    }

    /// Install the active session. Every install is a new evaluation identity.
    pub(super) fn install_session(&mut self, session: Option<Session>) {
        self.session_generation = self.session_generation.wrapping_add(1);
        self.reset_display_pipeline();
        self.session = session;
    }

    /// One choke point for every open/restore/replace/close/duplicate. Display
    /// frames and plans belong to the outgoing document; none of their state may
    /// leak into the next tab.
    fn reset_display_pipeline(&mut self) {
        self.pending_display = None;
        self.quality_display = None;
        self.display_snapshot = None;
        self.last_exact = None;
        self.displayed_target = None;
        self.edit_preview = None;
        self.edit_preview_target = None;
        self.comparison_pair = None;
        self.difference_stats = None;
        self.difference_error = None;
        self.display_error = None;
        self.gizmo_drag = None;
        self.gizmo_dragging = false;
        self.deferred_display = None;
        self.applying_deferred_display = false;
        self.last_applied_display = None;
        self.mesh = None;
        self.viewport.clear_document_display();
    }

    pub(super) fn open_session(&mut self, session: Session, path: Option<PathBuf>, status: String) {
        self.save_active_tab();
        self.close_tool_dock();
        self.tool_menu_rect = None;
        self.node_editor.reset_interaction();
        let id = self.next_tab_id;
        self.next_tab_id = self.next_tab_id.wrapping_add(1);
        let document_id = match &path {
            Some(path) => document_id_for_path(path),
            None => {
                let id = self.next_document_id;
                self.next_document_id = self.next_document_id.wrapping_add(1);
                format!("untitled-{id}")
            }
        };
        let _ = self.documents.set_session(document_id.clone(), session.clone());
        if let Some(path) = &path {
            self.documents.set_path(&document_id, path);
        }
        let selected_nodes: HashSet<String> = initial_viewer_target(session.document())
            .into_iter()
            .collect();
        let viewer_breakpoint = selected_nodes.iter().next().cloned();
        let tab = DocumentTab {
            id,
            document_id,
            path: path.clone(),
            selected_nodes: selected_nodes.clone(),
            selected_connections: HashSet::new(),
            edit_target: None,
            viewer_breakpoint: viewer_breakpoint.clone(),
            view_mode: TabViewMode::Split,
            node_width: 0.0,
            inspector_width: 0.0,
            mesh: None,
            display: None,
            status: status.clone(),
            error: None,
            evaluation_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            view: None,
        };
        self.tabs.push(tab);
        self.active_tab = self.tabs.len().saturating_sub(1);
        self.active_session_revision = session.revision();
        self.install_session(Some(session));
        self.pending_dirty.clear();
        self.eval_progress = None;
        self.path = path;
        self.selected_nodes = selected_nodes;
        self.selected_connections.clear();
        self.viewer_breakpoint = viewer_breakpoint;
        self.mesh = None;
        self.camera_initialized = false;
        self.viewport.clear_document_display();
        self.node_editor.reset_view();
        self.error = None;
        self.status = status;
        self.eval_pending = None;
        self.layout_animation = None;
        self.node_position_animation = None;
        self.evaluate();
    }

    pub(super) fn activate_tab(&mut self, index: usize) {
        if index >= self.tabs.len() || index == self.active_tab {
            return;
        }
        self.save_active_tab_impl(true);
        if let Some(tab) = self.tabs.get(self.active_tab) {
            self.sync_group_tab(tab.id);
        }
        self.active_tab = index;
        self.restore_active_tab();
        if self.mesh.is_none() && self.session.is_some() {
            self.evaluate();
        }
    }

}


#[cfg(test)]
mod layer_tests {
    use super::*;

    #[test]
    fn content_overlays_paint_below_menu_popups() {
        // Tabs stay below the node canvas/menu path; the breadcrumb is on the
        // canvas order, and the File menu is raised to Tooltip above it.
        assert!(TAB_LAYER_ORDER < egui::Order::Foreground);
        assert!(super::super::BREADCRUMB_LAYER_ORDER < egui::Order::Tooltip);
    }
}
