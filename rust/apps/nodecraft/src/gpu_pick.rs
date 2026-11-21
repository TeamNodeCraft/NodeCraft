//! CPU-side helpers for the GPU edge-pick ID pass.

/// A ghost edge segment as a POD instance for the ribbon vertex shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PickSegment {
    pub a: [f32; 3],
    pub id: u32,
    pub b: [f32; 3],
    /// The same [`nodecraft_layers::PlanEdgeClass`] bits the visible ribbon
    /// carries, so the pick mask follows the draw mask.
    pub class: u32,
}

const _: () = assert!(std::mem::size_of::<PickSegment>() == 32);

/// Face ids occupy the high half of the pick ID space; edge ids stay below it.
/// A pixel therefore identifies its kind without a second channel.
pub(crate) const FACE_BASE: u32 = 1 << 31;
/// Face picks read the pixel under the cursor; edges keep their wider
/// screen-space tolerance.
pub(crate) const FACE_PICK_RADIUS_PX: f32 = 1.0;
const FACE_MULTIPLE: u32 = 1 << 30;
const FACE_BODY_SHIFT: u32 = 20;
const FACE_BODY_MASK: u32 = 0x3ff;
const FACE_MASK: u32 = (1 << 20) - 1;

/// Encode the B-rep face of one plan instance into the pick ID space. Body is
/// the plan instance index (the same index `plan_face_ids` uses for its
/// `body:N:` prefix); `multiple` is whether the display has more than one body.
pub(crate) fn face_pick_id(body: u32, face: u32, multiple: bool) -> u32 {
    if body > FACE_BODY_MASK || face > FACE_MASK {
        return 0;
    }
    FACE_BASE | if multiple { FACE_MULTIPLE } else { 0 } | (body << FACE_BODY_SHIFT) | face
}

pub(crate) fn is_face_pick(id: u32) -> bool {
    id & FACE_BASE != 0
}

/// The face id string exactly as `plan_face_ids` builds it.
pub(crate) fn resolve_face_id(id: u32) -> Option<String> {
    if !is_face_pick(id) {
        return None;
    }
    let body = (id >> FACE_BODY_SHIFT) & FACE_BODY_MASK;
    let face = id & FACE_MASK;
    Some(if id & FACE_MULTIPLE != 0 {
        format!("body:{body}:face:{face}")
    } else {
        format!("face:{face}")
    })
}

/// Crop matrix that maps the viewport region centred on the **continuous**
/// cursor position `(px, py)` to the whole pick target. All sizes are physical
/// pixels; do not round the cursor to a pixel centre.
pub(crate) fn crop_matrix(px: f32, py: f32, width: f32, height: f32, pick: f32) -> [[f32; 4]; 4] {
    let ndc_x = 2.0 * px / width - 1.0;
    let ndc_y = 1.0 - 2.0 * py / height;
    let sx = width / pick;
    let sy = height / pick;
    glam::Mat4::from_cols_array_2d(&[
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-sx * ndc_x, -sy * ndc_y, 0.0, 1.0],
    ])
    .to_cols_array_2d()
}

/// What a pick request is looking for. The pass draws both edges and faces into
/// one target; the request filters the nearest ID to its own kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PickKind {
    Edges,
    Faces,
}

impl PickKind {
    fn accepts(self, id: u32) -> bool {
        match self {
            Self::Edges => id != 0 && !is_face_pick(id),
            Self::Faces => is_face_pick(id),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PickRequest {
    pub id: u64,
    /// Physical pixels relative to the viewport top-left, unrounded.
    pub px: [f32; 2],
    /// Pick radius in physical pixels (`7 * pixels_per_point`).
    pub radius_px: f32,
    pub ghost_key: u64,
    pub commit: bool,
    pub kind: PickKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PickOutcome {
    Hover(u32),
    Commit(u32),
    StaleCommit,
    Ignore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PickResult {
    pub id: u64,
    pub ghost_key: u64,
    pub commit: bool,
    pub index: u32,
}

/// Everything the render callback needs for one pick frame.
#[derive(Clone)]
pub(crate) struct PickFrame {
    pub request: Option<PickRequest>,
    pub ghost_key: u64,
    /// The active edge set: the edit ghost when one is shown, otherwise the
    /// displayed body's edges.
    pub segments: std::sync::Arc<[PickSegment]>,
    /// Whether `segments` belong to the edit ghost and therefore use
    /// `ghost_matrix` rather than the frame's preview matrix.
    pub ghost_edges: bool,
    /// Preview matrix for the drawn plan triangles (face ids). The object mesh
    /// path carries its preview in the instance buffer instead, so the frame
    /// sets identity for it.
    pub face_matrix: [[f32; 4]; 4],
    pub ghost_matrix: [[f32; 4]; 4],
    pub size_px: [f32; 2],
    /// Standard edge width in physical pixels. The pick ribbon uses the same
    /// value as the drawn ribbon, so drawn and pickable bands agree.
    pub edge_width_px: f32,
    /// The drawn ribbon's seam mask. Hidden seams are not pickable.
    pub show_seam_edges: bool,
}

/// True cursor is the corner between cells 15 and 16; the marked cell is cosmetic.
pub(crate) fn format_grid(ids: &[u32], pick: usize, header: &str) -> String {
    fn digit(id: u32) -> char {
        match id {
            0 => '.',
            1..=9 => char::from(b'0' + id as u8),
            10..=35 => char::from(b'a' + (id - 10) as u8),
            _ => '#',
        }
    }
    let centre = pick / 2;
    let mut text = format!("{header}\n");
    for y in 0..pick {
        for x in 0..pick {
            let id = ids.get(y * pick + x).copied().unwrap_or(0);
            let mut c = digit(id);
            if x == centre && y == centre {
                if id == 0 {
                    c = '+';
                } else {
                    c = c.to_ascii_uppercase();
                }
            }
            text.push(c);
        }
        text.push('\n');
    }
    text
}

/// Nearest accepted pick id inside `radius` pixels of the target centre.
fn nearest_accepting(ids: &[u32], pick: usize, radius: f32, accept: impl Fn(u32) -> bool) -> u32 {
    let centre = (pick as f32 - 1.0) * 0.5;
    let radius2 = radius * radius;
    let mut best = 0u32;
    let mut best2 = radius2;
    for (index, id) in ids.iter().copied().enumerate() {
        if id == 0 || index >= pick * pick || !accept(id) {
            continue;
        }
        let x = (index % pick) as f32 - centre;
        let y = (index / pick) as f32 - centre;
        let distance2 = x * x + y * y;
        if distance2 <= best2 {
            best = id;
            best2 = distance2;
        }
    }
    best
}

/// Nearest non-zero pick id inside `radius` pixels of the target centre.
pub(crate) fn nearest_pick(ids: &[u32], pick: usize, radius: f32) -> u32 {
    nearest_accepting(ids, pick, radius, |id| id != 0)
}

pub(crate) fn nearest_pick_of_kind(ids: &[u32], pick: usize, radius: f32, kind: PickKind) -> u32 {
    nearest_accepting(ids, pick, radius, |id| kind.accepts(id))
}

pub(crate) fn accept_pick(
    result: &PickResult,
    ghost_key: u64,
    pending_click: Option<u64>,
) -> PickOutcome {
    if result.ghost_key != ghost_key {
        return if result.commit && Some(result.id) == pending_click {
            PickOutcome::StaleCommit
        } else {
            PickOutcome::Ignore
        };
    }
    if result.commit {
        if Some(result.id) == pending_click {
            PickOutcome::Commit(result.index)
        } else {
            PickOutcome::Ignore
        }
    } else {
        PickOutcome::Hover(result.index)
    }
}

/// What a finished edge-pick result should do next. There is deliberately no
/// multi-body branch: every body's edges go through the same GPU pass and the
/// resolved id keeps its `body:N:` prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EdgeClickAction {
    Ignore,
    Apply(u32),
    CheckBody,
    Retry,
}

pub(crate) fn edge_click_action(outcome: PickOutcome, index: u32) -> EdgeClickAction {
    match outcome {
        PickOutcome::Commit(_) if index == 0 => EdgeClickAction::CheckBody,
        PickOutcome::Commit(_) => EdgeClickAction::Apply(index),
        PickOutcome::StaleCommit => EdgeClickAction::Retry,
        PickOutcome::Hover(_) | PickOutcome::Ignore => EdgeClickAction::Ignore,
    }
}

/// Decode an R32Uint readback with a row stride in bytes.
pub(crate) fn decode_rows(bytes: &[u8], pick: usize, row_stride: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(pick * pick);
    for row in 0..pick {
        let start = row * row_stride;
        let Some(row_bytes) = bytes.get(start..start + pick * 4) else { break };
        ids.extend(
            row_bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]])),
        );
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(matrix: [[f32; 4]; 4], point: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0; 4];
        for row in 0..4 {
            out[row] = (0..4).map(|column| matrix[column][row] * point[column]).sum();
        }
        out
    }

    #[test]
    fn crop_matrix_centres_the_pick_point() {
        let width = 1600.0;
        let height = 900.0;
        let (px, py) = (640.0, 300.0);
        let crop = crop_matrix(px, py, width, height, 32.0);
        let ndc = [2.0 * px / width - 1.0, 1.0 - 2.0 * py / height, 0.0, 1.0];
        let clipped = project(crop, ndc);
        assert!((clipped[0] / clipped[3]).abs() < 1.0e-6);
        assert!((clipped[1] / clipped[3]).abs() < 1.0e-6);
    }

    #[test]
    fn crop_matrix_handles_perspective_w() {
        let width = 1600.0;
        let height = 900.0;
        let (px, py) = (640.0, 300.0);
        let crop = crop_matrix(px, py, width, height, 32.0);
        let w = 3.0;
        let ndc = [(2.0 * px / width - 1.0) * w, (1.0 - 2.0 * py / height) * w, 0.0, w];
        let clipped = project(crop, ndc);
        assert!((clipped[0] / clipped[3]).abs() < 1.0e-6);
        assert!((clipped[1] / clipped[3]).abs() < 1.0e-6);
    }

    #[test]
    fn nearest_pick_prefers_the_closest_id_inside_the_radius() {
        let mut ids = vec![0; 32 * 32];
        ids[16 * 32 + 18] = 3;
        ids[16 * 32 + 23] = 5;
        assert_eq!(nearest_pick(&ids, 32, 7.0), 3);
        ids[16 * 32 + 18] = 0;
        assert_eq!(nearest_pick(&ids, 32, 7.0), 0);
    }

    #[test]
    fn stale_and_unmatched_results_are_rejected() {
        let result = PickResult { id: 3, ghost_key: 7, commit: true, index: 4 };
        assert_eq!(accept_pick(&result, 7, Some(3)), PickOutcome::Commit(4));
        assert_eq!(accept_pick(&result, 7, Some(9)), PickOutcome::Ignore);
        assert_eq!(accept_pick(&result, 8, Some(3)), PickOutcome::StaleCommit);
        assert_eq!(accept_pick(&result, 8, Some(9)), PickOutcome::Ignore);
    }

    #[test]
    fn edge_click_actions_apply_any_body_and_ignore_an_older_result() {
        let multi_body = PickResult { id: 2, ghost_key: 7, commit: true, index: 6 };
        assert_eq!(accept_pick(&multi_body, 7, Some(2)), PickOutcome::Commit(6));
        assert_eq!(edge_click_action(PickOutcome::Commit(6), 6), EdgeClickAction::Apply(6));
        // The click has its own request id; a result for the earlier pick is
        // not allowed to resolve it.
        let older = PickResult { id: 1, ghost_key: 7, commit: true, index: 3 };
        assert_eq!(accept_pick(&older, 7, Some(2)), PickOutcome::Ignore);
        // A miss asks the face pass whether the body was hit at all.
        assert_eq!(edge_click_action(PickOutcome::Commit(0), 0), EdgeClickAction::CheckBody);
        assert_eq!(edge_click_action(PickOutcome::StaleCommit, 0), EdgeClickAction::Retry);
    }

    #[test]
    fn grid_formatter_marks_centre_and_base36() {
        let ids = vec![0u32; 32 * 32];
        let text = format_grid(&ids, 32, "header");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "header");
        assert_eq!(lines[17].chars().nth(16), Some('+'));
        let mut ids = vec![0u32; 32 * 32];
        ids[16 * 32 + 10] = 40;
        let text = format_grid(&ids, 32, "x");
        assert_eq!(text.lines().nth(17).and_then(|line| line.chars().nth(10)), Some('#'));
    }

    fn request(id: u64, commit: bool) -> PickRequest {
        PickRequest {
            id,
            px: [16.0, 16.0],
            radius_px: 7.0,
            ghost_key: 1,
            commit,
            kind: PickKind::Edges,
        }
    }

    fn result(id: u64, commit: bool, index: u32) -> PickResult {
        PickResult { id, ghost_key: 1, commit, index }
    }

    #[test]
    fn ring_state_machine_reuses_slots_only_after_free() {
        let mut ring = Ring::default();
        let slot = ring.free_slot().expect("free slot");
        ring.copied(slot, request(1, false));
        assert_eq!(ring.start_mapping(), vec![slot]);
        assert!(ring.in_flight());
        assert_eq!(ring.finish(slot, 0), None);
        assert!(ring.in_flight());
        assert_eq!(ring.finish(slot, 1).map(|r| r.id), Some(1));
        assert!(!ring.in_flight());
        assert_eq!(ring.free_slot(), Some(slot));
    }

    #[test]
    fn ring_keeps_a_request_when_all_slots_are_busy() {
        let mut ring = Ring::default();
        for slot in 0..3 {
            ring.copied(slot, request(slot as u64 + 1, false));
        }
        assert!(ring.free_slot().is_none());
        assert!(ring.in_flight());
    }

    #[test]
    fn ring_slot_reuse_two_cycles() {
        let mut ring = Ring::default();
        let status = std::sync::atomic::AtomicU8::new(0);
        let slot = ring.free_slot().expect("slot");
        ring.copied(slot, request(1, false));
        assert_eq!(ring.start_mapping(), vec![slot]);
        status.store(1, std::sync::atomic::Ordering::Release);
        assert_eq!(ring.finish(slot, 1).map(|r| r.id), Some(1));
        ring.copied(slot, request(2, false));
        assert_eq!(ring.start_mapping(), vec![slot]);
        assert_eq!(ring.finish(slot, 0), None);
        assert!(ring.in_flight());
    }

    #[test]
    fn finish_on_copied_is_noop() {
        let mut ring = Ring::default();
        ring.copied(0, request(1, false));
        assert!(ring.finish(0, 1).is_none());
        assert!(matches!(ring.phases[0], SlotPhase::Copied(_)));
    }

    #[test]
    fn ring_start_mapping_only_once() {
        let mut ring = Ring::default();
        ring.copied(0, request(1, false));
        assert_eq!(ring.start_mapping(), vec![0]);
        assert!(ring.start_mapping().is_empty());
    }

    #[test]
    fn merge_newer_commit_wins() {
        assert_eq!(merge_result(Some(result(3, true, 1)), result(5, true, 2)).id, 5);
    }

    #[test]
    fn merge_into_none() {
        assert_eq!(merge_result(None, result(4, false, 7)).id, 4);
    }

    #[test]
    fn take_decision_handles_stale_and_busy() {
        assert_eq!(take_decision(Some(&request(1, false)), 1, true), TakeDecision::Take);
        assert_eq!(take_decision(Some(&request(1, false)), 1, false), TakeDecision::Keep);
        assert_eq!(take_decision(Some(&request(1, false)), 2, true), TakeDecision::Drop);
        assert_eq!(take_decision(None, 1, true), TakeDecision::Keep);
    }

    #[test]
    fn ring_error_frees_the_slot() {
        let mut ring = Ring::default();
        ring.copied(0, request(1, false));
        ring.start_mapping();
        assert!(ring.finish(0, 2).is_none());
        assert!(!ring.in_flight());
    }

    #[test]
    fn commit_beats_a_newer_hover() {
        let held = Some(result(3, true, 2));
        let merged = merge_result(held, result(9, false, 3));
        assert!(merged.commit);
        assert_eq!(merged.index, 2);
        let merged = merge_result(Some(result(1, false, 1)), result(2, false, 2));
        assert_eq!(merged.id, 2);
    }

    #[test]
    fn grid_formatter_line_and_centre() {
        let mut ids = vec![0u32; 32 * 32];
        for x in 0..32 {
            ids[16 * 32 + x] = 3;
        }
        let text = format_grid(&ids, 32, "x");
        let line = text.lines().nth(17).expect("line");
        assert_eq!(line.chars().filter(|c| *c == '3').count(), 32);
        assert_eq!(line.chars().nth(16), Some('3'));
    }

    #[test]
    fn readback_decodes_padded_rows() {
        let mut bytes = vec![0u8; 256 * 4];
        bytes[0..4].copy_from_slice(&7u32.to_le_bytes());
        bytes[256 + 8..256 + 12].copy_from_slice(&9u32.to_le_bytes());
        let ids = decode_rows(&bytes, 4, 256);
        assert_eq!(ids[0], 7);
        assert_eq!(ids[6], 9);
    }

    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: true,
                compatible_surface: None,
                ..Default::default()
            }))
            .ok()
        })?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn segment(a: [f32; 3], b: [f32; 3], id: u32) -> PickSegment {
        PickSegment { a, id, b, class: 0 }
    }

    fn run_pick(
        picker: &mut PickRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &PickUniforms,
        frame: &PickFrame,
        request: PickRequest,
    ) -> Option<PickResult> {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test") });
        picker.encode(device, queue, &mut encoder, uniforms, frame, request, |_, _, _| {}, None);
        queue.submit([encoder.finish()]);
        for _ in 0..200 {
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            picker.service(device);
            if let Some(result) = picker.take_result() {
                return Some(result);
            }
        }
        None
    }

    #[test]
    fn gpu_pick_sequential_requests_reuse_slots() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let mut picker = PickRenderer::install(&device, wgpu::TextureFormat::Depth32Float, true);
        let segments: std::sync::Arc<[PickSegment]> =
            std::sync::Arc::from(vec![segment([-0.4, 0.2, 0.0], [0.4, 0.2, 0.0], 1)]);
        let frame = PickFrame {
            request: None,
            ghost_key: 10,
            segments,
            ghost_edges: false,
            face_matrix: crate::plan_math::identity_f32(),
            ghost_matrix: crate::plan_math::identity_f32(),
            size_px: [32.0, 32.0],
            edge_width_px: 1.0,
            show_seam_edges: false,
        };
        let uniforms = PickUniforms {
            view_proj: crate::plan_math::identity_f32(),
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: crate::plan_math::identity_f32(),
        };
        for id in 1..=4 {
            let request = PickRequest {
                id,
                px: [16.0, 16.0],
                radius_px: 7.0,
                ghost_key: 10,
                commit: false,
                kind: PickKind::Edges,
            };
            let result =
                run_pick(&mut picker, &device, &queue, &uniforms, &frame, request).expect("result");
            assert_eq!(result.id, id);
            assert_eq!(result.index, 1);
        }
    }

    struct TestMesh {
        vertices: wgpu::Buffer,
        indices: wgpu::Buffer,
        instance: wgpu::Buffer,
        count: u32,
    }

    fn test_quad(device: &wgpu::Device, queue: &wgpu::Queue, z: f32) -> TestMesh {
        let stride = solidcraft_viewport_gpu::VERTEX_STRIDE as usize;
        let positions = [[-1.0, -1.0, z], [1.0, -1.0, z], [1.0, 1.0, z], [-1.0, 1.0, z]];
        let mut bytes = vec![0u8; stride * positions.len()];
        for (index, position) in positions.iter().enumerate() {
            bytes[index * stride..index * stride + 12]
                .copy_from_slice(bytemuck::cast_slice(&position[..]));
        }
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-quad-vertices"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertices, 0, &bytes);
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-quad-indices"),
            size: 6 * 4,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&indices, 0, bytemuck::cast_slice(&[0u32, 1, 2, 0, 2, 3]));
        let instance =
            crate::gpu_viewport::GpuInstance::from_model(crate::plan_math::identity_f32(), 1.0);
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-quad-instance"),
            size: std::mem::size_of_val(&instance) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buffer, 0, bytemuck::bytes_of(&instance));
        TestMesh { vertices, indices, instance: instance_buffer, count: 6 }
    }

    fn run_pick_with_faces(
        picker: &mut PickRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &PickUniforms,
        frame: &PickFrame,
        request: PickRequest,
        faces: impl FnOnce(&mut wgpu::RenderPass<'_>, &PickPasses<'_>, &PickBinds<'_>),
    ) -> Option<PickResult> {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test") });
        picker.encode(device, queue, &mut encoder, uniforms, frame, request, faces, None);
        queue.submit([encoder.finish()]);
        for _ in 0..200 {
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            picker.service(device);
            if let Some(result) = picker.take_result() {
                return Some(result);
            }
        }
        None
    }

    fn draw_mesh(
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        bind: &wgpu::BindGroup,
        mesh: &TestMesh,
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, mesh.instance.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.count, 0, 0..1);
    }

    #[test]
    fn gpu_pick_diagonal_draws_both_triangles() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let mut picker = PickRenderer::install(&device, wgpu::TextureFormat::Depth32Float, true);
        let segments: std::sync::Arc<[PickSegment]> =
            std::sync::Arc::from(vec![segment([-0.8, -0.8, 0.5], [0.8, 0.8, 0.5], 1)]);
        let frame = PickFrame {
            request: None,
            ghost_key: 11,
            segments,
            ghost_edges: false,
            face_matrix: crate::plan_math::identity_f32(),
            ghost_matrix: crate::plan_math::identity_f32(),
            size_px: [32.0, 32.0],
            edge_width_px: 3.0,
            show_seam_edges: false,
        };
        let uniforms = PickUniforms {
            view_proj: crate::plan_math::identity_f32(),
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: crate::plan_math::identity_f32(),
        };
        let request = PickRequest {
            id: 1,
            px: [16.0, 16.0],
            radius_px: 7.0,
            ghost_key: 11,
            commit: false,
            kind: PickKind::Edges,
        };
        let result = run_pick_with_faces(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            request,
            |_, _, _| {},
        );
        assert_eq!(result.expect("result").index, 1);
        let grid = picker.last_grid().expect("grid");
        assert!(grid.iter().filter(|id| **id != 0).count() >= 64);
    }

    fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        glam::Mat4::perspective_rh(fov_y, aspect, near, far).to_cols_array_2d()
    }

    fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        glam::Mat4::look_at_rh(eye.into(), target.into(), up.into()).to_cols_array_2d()
    }

    fn project_px(vp: [[f32; 4]; 4], p: [f32; 3], size: [f32; 2]) -> [f32; 2] {
        let clip = crate::plan_math::multiply_f32(vp, crate::plan_math::identity_f32());
        let _ = clip;
        let mut out = [0.0; 4];
        for row in 0..4 {
            out[row] = (0..4)
                .map(|column| vp[column][row] * if column == 3 { 1.0 } else { p[column] })
                .sum();
        }
        let ndc = [out[0] / out[3], out[1] / out[3]];
        [(ndc[0] * 0.5 + 0.5) * size[0], (1.0 - (ndc[1] * 0.5 + 0.5)) * size[1]]
    }

    fn test_box(device: &wgpu::Device, queue: &wgpu::Queue) -> TestMesh {
        let corners: [[f32; 3]; 8] = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let tris = [
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7,
            6, 3, 0, 4, 3, 4, 7,
        ];
        let stride = solidcraft_viewport_gpu::VERTEX_STRIDE as usize;
        let mut bytes = vec![0u8; stride * corners.len()];
        for (index, position) in corners.iter().enumerate() {
            bytes[index * stride..index * stride + 12]
                .copy_from_slice(bytemuck::cast_slice(&position[..]));
        }
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-box-vertices"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertices, 0, &bytes);
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-box-indices"),
            size: tris.len() as u64 * 4,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&indices, 0, bytemuck::cast_slice(&tris));
        let instance =
            crate::gpu_viewport::GpuInstance::from_model(crate::plan_math::identity_f32(), 1.0);
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-box-instance"),
            size: std::mem::size_of_val(&instance) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buffer, 0, bytemuck::bytes_of(&instance));
        TestMesh { vertices, indices, instance: instance_buffer, count: tris.len() as u32 }
    }

    #[test]
    fn gpu_pick_box_ghost_perspective() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let mut picker = PickRenderer::install(&device, wgpu::TextureFormat::Depth32Float, true);
        let corners = [
            [-1.0f32, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let edges = [
            (0u32, 1u32),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        let segments: std::sync::Arc<[PickSegment]> = std::sync::Arc::from(
            edges
                .iter()
                .enumerate()
                .map(|(index, (a, b))| {
                    segment(corners[*a as usize], corners[*b as usize], index as u32 + 1)
                })
                .collect::<Vec<_>>(),
        );
        let size = [512.0f32, 512.0];
        let vp = solidcraft_viewport_gpu::view::gpu_view_proj(
            crate::plan_math::multiply_f32(
                perspective_rh(0.8, 1.0, 0.1, 100.0),
                look_at_rh([3.0, 3.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ),
            true,
        );
        let frame = PickFrame {
            request: None,
            ghost_key: 20,
            segments,
            ghost_edges: false,
            face_matrix: crate::plan_math::identity_f32(),
            ghost_matrix: crate::plan_math::identity_f32(),
            size_px: size,
            edge_width_px: 1.0,
            show_seam_edges: false,
        };
        let uniforms = PickUniforms {
            view_proj: vp,
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: crate::plan_math::identity_f32(),
        };
        let mesh = test_box(&device, &queue);
        let mut pick_at = |picker: &mut PickRenderer, id: u64, point: [f32; 3]| {
            let px = project_px(vp, point, size);
            let request = PickRequest {
                id,
                px,
                radius_px: 7.0,
                ghost_key: 20,
                commit: false,
                kind: PickKind::Edges,
            };
            run_pick_with_faces(
                picker,
                &device,
                &queue,
                &uniforms,
                &frame,
                request,
                |pass, passes, binds| draw_mesh(pass, passes.face, binds.main, &mesh),
            )
            .map(|result| result.index)
            .unwrap_or(0)
        };
        let front = pick_at(&mut picker, 1, [1.0, 1.0, 1.0]);
        assert!([6, 7, 11].contains(&front), "front corner edge id {front}");
        let visible = pick_at(&mut picker, 2, [1.0, 1.0, 0.0]);
        assert_eq!(visible, 11, "visible edge id {visible}");
        let hidden = pick_at(&mut picker, 3, [-1.0, -1.0, 0.0]);
        assert_eq!(hidden, 0, "hidden back edge should not pick");
        let grazing_vp = solidcraft_viewport_gpu::view::gpu_view_proj(
            crate::plan_math::multiply_f32(
                perspective_rh(0.8, 1.0, 0.1, 100.0),
                look_at_rh([3.0, 0.15, 0.4], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ),
            true,
        );
        let px = project_px(grazing_vp, [1.0, 0.0, 1.0], size);
        let grazing_frame = PickFrame {
            request: None,
            ghost_key: 21,
            segments: frame.segments.clone(),
            ghost_edges: false,
            face_matrix: crate::plan_math::identity_f32(),
            ghost_matrix: crate::plan_math::identity_f32(),
            size_px: size,
            edge_width_px: 1.0,
            show_seam_edges: false,
        };
        let grazing_uniforms = PickUniforms { view_proj: grazing_vp, ..uniforms };
        let request = PickRequest {
            id: 4,
            px,
            radius_px: 7.0,
            ghost_key: 21,
            commit: false,
            kind: PickKind::Edges,
        };
        let grazing = run_pick_with_faces(
            &mut picker,
            &device,
            &queue,
            &grazing_uniforms,
            &grazing_frame,
            request,
            |pass, passes, binds| draw_mesh(pass, passes.face, binds.main, &mesh),
        )
        .map(|r| r.index)
        .unwrap_or(0);
        assert_eq!(grazing, 6, "grazing edge id {grazing}");
    }

    #[test]
    fn gpu_pick_face_depth_occludes_and_bias_keeps_on_surface_edges() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let mut picker = PickRenderer::install(&device, wgpu::TextureFormat::Depth32Float, true);
        let segments: std::sync::Arc<[PickSegment]> =
            std::sync::Arc::from(vec![segment([-0.8, 0.1, 0.5], [0.8, 0.1, 0.5], 1)]);
        let frame = PickFrame {
            request: None,
            ghost_key: 12,
            segments,
            ghost_edges: false,
            face_matrix: crate::plan_math::identity_f32(),
            ghost_matrix: crate::plan_math::identity_f32(),
            size_px: [32.0, 32.0],
            edge_width_px: 1.0,
            show_seam_edges: false,
        };
        let uniforms = PickUniforms {
            view_proj: crate::plan_math::identity_f32(),
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: crate::plan_math::identity_f32(),
        };
        let request = PickRequest {
            id: 1,
            px: [16.0, 16.0],
            radius_px: 7.0,
            ghost_key: 12,
            commit: false,
            kind: PickKind::Edges,
        };
        let near = test_quad(&device, &queue, 0.8);
        let result = run_pick_with_faces(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            request,
            move |pass, passes, binds| draw_mesh(pass, passes.face, binds.main, &near),
        );
        assert_eq!(result.expect("hidden result").index, 0);
        let far = test_quad(&device, &queue, 0.2);
        let result = run_pick_with_faces(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            request,
            move |pass, passes, binds| draw_mesh(pass, passes.face, binds.main, &far),
        );
        assert_eq!(result.expect("visible result").index, 1);
        let on = test_quad(&device, &queue, 0.5);
        let result = run_pick_with_faces(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            request,
            move |pass, passes, binds| draw_mesh(pass, passes.face, binds.main, &on),
        );
        assert_eq!(result.expect("on-surface result").index, 1);
    }

    struct TestFaceMesh {
        vertices: wgpu::Buffer,
        face_ids: wgpu::Buffer,
        indices: wgpu::Buffer,
        instance: wgpu::Buffer,
        bodies: wgpu::Buffer,
        count: u32,
    }

    /// A box with per-face vertices so each face id reaches its own triangles.
    fn face_box(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: [[f32; 4]; 4],
        body: u32,
        multiple: bool,
    ) -> TestFaceMesh {
        let corners: [[f32; 3]; 8] = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ];
        let faces: [[usize; 4]; 6] = [
            [4, 5, 6, 7], // +z
            [1, 0, 3, 2], // -z
            [5, 1, 2, 6], // +x
            [0, 4, 7, 3], // -x
            [3, 7, 6, 2], // +y
            [0, 1, 5, 4], // -y
        ];
        let stride = solidcraft_viewport_gpu::VERTEX_STRIDE as usize;
        let mut vertices = vec![0u8; stride * faces.len() * 4];
        let mut ids = Vec::with_capacity(faces.len() * 4);
        let mut indices = Vec::new();
        for (face, quad) in faces.iter().enumerate() {
            let base = (face * 4) as u32;
            for (slot, corner) in quad.iter().enumerate() {
                let offset = face * 4 + slot;
                vertices[offset * stride..offset * stride + 12]
                    .copy_from_slice(bytemuck::cast_slice(&corners[*corner][..]));
                ids.push(face_pick_id(0, face as u32, multiple));
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-face-vertices"),
            size: vertices.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, &vertices);
        let face_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-face-ids"),
            size: bytemuck::cast_slice::<u32, u8>(&ids).len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&face_buffer, 0, bytemuck::cast_slice(&ids));
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-face-indices"),
            size: bytemuck::cast_slice::<u32, u8>(&indices).len() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&indices));
        let instance = crate::gpu_viewport::GpuInstance::from_model(model, 1.0);
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-face-instance"),
            size: std::mem::size_of_val(&instance) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buffer, 0, bytemuck::bytes_of(&instance));
        let body_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-face-bodies"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&body_buffer, 0, bytemuck::cast_slice(&[body]));
        TestFaceMesh {
            vertices: vertex_buffer,
            face_ids: face_buffer,
            indices: index_buffer,
            instance: instance_buffer,
            bodies: body_buffer,
            count: indices.len() as u32,
        }
    }

    fn draw_face_mesh(
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        bind: &wgpu::BindGroup,
        mesh: &TestFaceMesh,
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, mesh.face_ids.slice(..));
        pass.set_vertex_buffer(2, mesh.instance.slice(..));
        pass.set_vertex_buffer(3, mesh.bodies.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.count, 0, 0..1);
    }

    #[test]
    fn gpu_pick_box_face_ids_follow_instance_and_preview_matrices() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let mut picker = PickRenderer::install(&device, wgpu::TextureFormat::Depth32Float, true);
        let size = [128.0f32, 128.0];
        let view = look_at_rh([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = solidcraft_viewport_gpu::view::gpu_view_proj(
            crate::plan_math::multiply_f32(perspective_rh(0.8, 1.0, 0.1, 100.0), view),
            true,
        );
        let frame = PickFrame {
            request: None,
            ghost_key: 30,
            segments: std::sync::Arc::from(Vec::new()),
            ghost_edges: false,
            face_matrix: crate::plan_math::identity_f32(),
            ghost_matrix: crate::plan_math::identity_f32(),
            size_px: size,
            edge_width_px: 1.0,
            show_seam_edges: false,
        };
        let uniforms = PickUniforms {
            view_proj: vp,
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: crate::plan_math::identity_f32(),
        };
        let mut model = crate::plan_math::identity_f32();
        model[3] = [0.3, 0.0, 0.0, 1.0];
        let mesh = face_box(&device, &queue, model, 0, false);
        let front = face_pick_id(0, 0, false);
        let mut pick = |picker: &mut PickRenderer, id: u64, point: [f32; 3], frame: &PickFrame| {
            let px = project_px(vp, point, size);
            let request = PickRequest {
                id,
                px,
                radius_px: 3.0,
                ghost_key: 30,
                commit: false,
                kind: PickKind::Faces,
            };
            let result = run_pick_with_faces(
                picker,
                &device,
                &queue,
                &uniforms,
                frame,
                request,
                |pass, passes, binds| draw_face_mesh(pass, passes.face_id, binds.main, &mesh),
            )
            .expect("face pick result");
            result.index
        };
        // The instance transform moved the box to x=0.3; the +z face is on top.
        assert_eq!(pick(&mut picker, 1, [0.3, 0.0, 0.5], &frame), front);
        assert_eq!(resolve_face_id(front).as_deref(), Some("face:0"));
        // A preview matrix moves the same instance further without a new mesh.
        let mut preview = frame.clone();
        preview.face_matrix = crate::plan_math::identity_f32();
        preview.face_matrix[3] = [0.2, 0.0, 0.0, 1.0];
        assert_eq!(pick(&mut picker, 2, [0.5, 0.0, 0.5], &preview), front);
        // Off the body: the target clears to zero.
        assert_eq!(pick(&mut picker, 3, [1.2, 0.8, 0.5], &frame), 0);
    }

    #[test]
    fn gpu_pick_multi_body_edge_click_resolves_its_body_prefix() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let edge = |id: &str, y: f64| crate::adapter::DisplayEdge {
            id: id.into(),
            points: vec![
                solidcore_record::Vec3::new(-0.4, y, 0.0),
                solidcore_record::Vec3::new(0.4, y, 0.0),
            ],
            midpoint: [0.0, y, 0.0],
            class: nodecraft_layers::PlanEdgeClass::NORMAL,
        };
        let data = crate::gpu_viewport::build_ghost_pick_data(&[
            edge("body:0:edge:0", 0.2),
            edge("body:1:edge:0", 0.75),
        ]);
        assert!(data.multi_body, "two body prefixes");
        let mut picker = PickRenderer::install(&device, wgpu::TextureFormat::Depth32Float, true);
        let size = [32.0f32, 32.0];
        let vp = crate::plan_math::identity_f32();
        let frame = PickFrame {
            request: None,
            ghost_key: 31,
            segments: data.segments.clone(),
            ghost_edges: false,
            face_matrix: vp,
            ghost_matrix: vp,
            size_px: size,
            edge_width_px: 1.0,
            show_seam_edges: false,
        };
        let uniforms = PickUniforms {
            view_proj: vp,
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: vp,
        };
        let px = project_px(vp, [0.0, 0.75, 0.0], size);
        let request = PickRequest {
            id: 1,
            px,
            radius_px: 3.0,
            ghost_key: 31,
            commit: true,
            kind: PickKind::Edges,
        };
        let result = run_pick(&mut picker, &device, &queue, &uniforms, &frame, request)
            .expect("pick result");
        let (edge_id, _) = crate::gpu_viewport::resolve_pick(&data, result.index).expect("edge id");
        assert_eq!(edge_id, "body:1:edge:0");
        assert_eq!(
            edge_click_action(PickOutcome::Commit(result.index), result.index),
            EdgeClickAction::Apply(result.index)
        );
    }

    #[test]
    fn face_pick_ids_keep_the_plan_face_strings() {
        assert_eq!(resolve_face_id(face_pick_id(0, 7, false)).as_deref(), Some("face:7"));
        assert_eq!(resolve_face_id(face_pick_id(2, 7, true)).as_deref(), Some("body:2:face:7"));
        assert!(is_face_pick(face_pick_id(0, 0, false)));
        assert!(!is_face_pick(1));
        assert_eq!(face_pick_id(0, 1 << 20, false), 0);
    }

    #[test]
    fn gpu_pick_nearest_edge_headless() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let mut picker = PickRenderer::install(&device, wgpu::TextureFormat::Depth32Float, true);
        let segments: std::sync::Arc<[PickSegment]> = std::sync::Arc::from(vec![
            segment([-0.4, 0.2, 0.0], [0.4, 0.2, 0.0], 1),
            segment([-0.4, 0.75, 0.0], [0.4, 0.75, 0.0], 2),
        ]);
        let frame = PickFrame {
            request: None,
            ghost_key: 9,
            segments,
            ghost_edges: false,
            face_matrix: crate::plan_math::identity_f32(),
            ghost_matrix: crate::plan_math::identity_f32(),
            size_px: [32.0, 32.0],
            edge_width_px: 1.0,
            show_seam_edges: false,
        };
        let request = PickRequest {
            id: 1,
            px: [16.0, 16.0],
            radius_px: 7.0,
            ghost_key: 9,
            commit: false,
            kind: PickKind::Edges,
        };
        let uniforms = PickUniforms {
            view_proj: crate::plan_math::identity_f32(),
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: crate::plan_math::identity_f32(),
        };
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test") });
        picker.encode(
            &device,
            &queue,
            &mut encoder,
            &uniforms,
            &frame,
            request,
            |_, _, _| {},
            None,
        );
        queue.submit([encoder.finish()]);
        let mut result = None;
        for _ in 0..200 {
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            picker.service(&device);
            result = picker.take_result().or(result);
            if result.is_some() {
                break;
            }
        }
        let result = result.expect("pick result");
        assert_eq!(result.index, 1, "nearest edge should be id 1");
    }

    fn box_plan(size: f64) -> std::sync::Arc<nodecraft_layers::tess::TessPlan> {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives as api;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut brep = api::create_box(
            &mut kernel,
            Point3::new(-size * 0.5, -size * 0.5, -size * 0.5),
            size,
            size,
            size,
        )
        .expect("box");
        nodecraft_layers::tessellate_kernel_preview(&mut kernel, &mut brep, 2)
            .expect("box preview plan")
            .plan
    }

    /// The plan-local edge pick path: the same local edge buffer and instance
    /// matrices as the drawn ribbon, read through the shared vertex function.
    #[test]
    fn gpu_pick_plan_local_edges_resolve_through_the_ribbon() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = solidcraft_viewport_gpu::TessGpu::new("test", device.clone(), queue.clone())
            .expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut plan = crate::gpu_plan::PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        plan.set_plans(
            vec![nodecraft_display::PlanInstance {
                target: "edges".into(),
                plan: box_plan(1.0),
                face_colors: Vec::new(),
                body: 0,
                placement: nodecraft_display::Placement::identity(),
                edge_ids: std::sync::Arc::from(vec![Some("edge:front".to_owned())]),
                edge_classes: std::sync::Arc::from(Vec::new()),
            }],
            true,
            None,
            1,
        );
        plan.prepare_chunks(&device, &queue);
        let data = plan.edge_pick_data().clone();
        assert!(data.edges.len() > 1, "plan edges have a pick table");
        let (_, midpoint) = data.edges[1].clone();
        let size = [512.0f32, 512.0];
        let vp = solidcraft_viewport_gpu::view::gpu_view_proj(
            crate::plan_math::multiply_f32(
                perspective_rh(0.8, 1.0, 0.1, 100.0),
                look_at_rh([3.0, 3.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ),
            true,
        );
        let frame = PickFrame {
            request: None,
            ghost_key: 40,
            segments: std::sync::Arc::from(Vec::new()),
            ghost_edges: false,
            face_matrix: crate::plan_math::identity_f32(),
            ghost_matrix: crate::plan_math::identity_f32(),
            size_px: size,
            edge_width_px: 3.0,
            show_seam_edges: false,
        };
        let uniforms = PickUniforms {
            view_proj: vp,
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: crate::plan_math::identity_f32(),
        };
        let px = project_px(vp, [midpoint[0] as f32, midpoint[1] as f32, midpoint[2] as f32], size);
        let request = PickRequest {
            id: 1,
            px,
            radius_px: 7.0,
            ghost_key: 40,
            commit: false,
            kind: PickKind::Edges,
        };
        let mut picker = PickRenderer::install(&device, wgpu::TextureFormat::Depth32Float, true);
        let result = run_pick_with_faces(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            request,
            |pass, passes, binds| {
                plan.paint_edge_ids(pass, passes.plan_edge, binds.plan_edge);
            },
        )
        .expect("plan edge pick result");
        assert!(result.index > 0, "an edge id was written");
        let (id, _) =
            crate::gpu_viewport::resolve_pick(&data, result.index).expect("edge id resolves");
        assert!(id.starts_with("edge:") || id.starts_with("body:"), "{id}");
    }

    /// A seam is not pickable while hidden and is pickable when shown: the pick
    /// shader and the drawn ribbon share the class and the uniform mask.
    #[test]
    fn a_hidden_seam_is_not_pickable_until_the_setting_shows_it() {
        use nodecraft_solidcore::solidcore::{math::Point3, primitive::api_primitives as api};
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut brep = api::create_cone(&mut kernel, Point3::new(0.0, 0.0, -20.0), 20.0, 0.0, 40.0)
            .expect("cone");
        let display = nodecraft_layers::tessellate_kernel_preview(&mut kernel, &mut brep, 2)
            .expect("cone preview plan");
        let seam_index = display
            .edge_classes
            .iter()
            .position(|class| class.is_seam())
            .expect("the cone has a seam");
        let seam = display.plan.edges[seam_index].clone();
        let midpoint = seam[seam.len() / 2];
        let tess = solidcraft_viewport_gpu::TessGpu::new("test", device.clone(), queue.clone())
            .expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut plan = crate::gpu_plan::PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        plan.set_plans(
            vec![nodecraft_display::PlanInstance {
                target: "seams".into(),
                plan: std::sync::Arc::clone(&display.plan),
                face_colors: Vec::new(),
                body: 0,
                placement: nodecraft_display::Placement::identity(),
                edge_ids: std::sync::Arc::from(Vec::new()),
                edge_classes: std::sync::Arc::from(display.edge_classes.clone()),
            }],
            true,
            None,
            1,
        );
        plan.prepare_chunks(&device, &queue);
        let size = [512.0f32, 512.0];
        let vp = solidcraft_viewport_gpu::view::gpu_view_proj(
            crate::plan_math::multiply_f32(
                perspective_rh(0.8, 1.0, 0.1, 300.0),
                look_at_rh([70.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ),
            true,
        );
        let uniforms = PickUniforms {
            view_proj: vp,
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: crate::plan_math::identity_f32(),
        };
        let px = project_px(vp, [midpoint.x as f32, midpoint.y as f32, midpoint.z as f32], size);
        let mut frame = PickFrame {
            request: None,
            ghost_key: 50,
            segments: std::sync::Arc::from(Vec::new()),
            ghost_edges: false,
            face_matrix: crate::plan_math::identity_f32(),
            ghost_matrix: crate::plan_math::identity_f32(),
            size_px: size,
            edge_width_px: 4.0,
            show_seam_edges: false,
        };
        let request = PickRequest {
            id: 1,
            px,
            radius_px: 7.0,
            ghost_key: 50,
            commit: false,
            kind: PickKind::Edges,
        };
        let mut picker = PickRenderer::install(&device, wgpu::TextureFormat::Depth32Float, true);
        let hidden = run_pick_with_faces(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            request,
            |pass, passes, binds| {
                plan.paint_edge_ids(pass, passes.plan_edge, binds.plan_edge);
            },
        );
        assert!(hidden.is_none_or(|result| result.index == 0), "a hidden seam has no pick id",);
        frame.show_seam_edges = true;
        plan.set_show_seam_edges(true);
        let shown = run_pick_with_faces(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            request,
            |pass, passes, binds| {
                plan.paint_edge_ids(pass, passes.plan_edge, binds.plan_edge);
            },
        )
        .expect("the shown seam is pickable");
        assert!(shown.index > 0, "the shown seam writes its edge id");
    }
}

use egui_wgpu::wgpu;

use crate::edge_ribbon::{
    EDGE_RIBBON_WGSL, EdgeUniforms, edge_instance_layout, edge_vertex_layout,
};
use solidcraft_viewport_gpu::view::{FACE_BIAS, depth_bias};

const PICK: u32 = 32;
const FACE_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    back: vec4<f32>,
    eye: vec4<f32>,
    params: vec4<f32>,
    extra_model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
struct In {
    @location(0) position: vec3<f32>,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
};
@vertex
fn vs(in: In) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    return u.view_proj * u.extra_model * model * vec4<f32>(in.position, 1.0);
}
@fragment
fn fs() -> @location(0) u32 { return 0u; }
"#;
const FACE_ID_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    back: vec4<f32>,
    eye: vec4<f32>,
    params: vec4<f32>,
    extra_model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
struct In {
    @location(0) position: vec3<f32>,
    @location(2) id: u32,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
};
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
};
@vertex
fn vs(in: In) -> Out {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    var out: Out;
    out.position = u.view_proj * u.extra_model * model * vec4<f32>(in.position, 1.0);
    out.id = in.id;
    return out;
}
@fragment
fn fs(in: Out) -> @location(0) u32 { return in.id; }
"#;
/// Plan faces share one instanced draw per plan, so the per-vertex face id
/// carries the B-rep face and the multi-body flag, while the body index is a
/// per-instance attribute built from the draw group's body list.
const PLAN_FACE_ID_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    back: vec4<f32>,
    eye: vec4<f32>,
    params: vec4<f32>,
    extra_model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
struct In {
    @location(0) position: vec3<f32>,
    @location(2) id: u32,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(12) body: u32,
};
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
};
@vertex
fn vs(in: In) -> Out {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    var out: Out;
    out.position = u.view_proj * u.extra_model * model * vec4<f32>(in.position, 1.0);
    // The per-vertex id already has FACE_BASE and the multi-body flag set; the
    // body occupies the same bits `face_pick_id` encodes on the CPU. Bodies
    // past that field are not pickable, exactly as the CPU encoder refuses them.
    if (in.body > 1023u) {
        out.id = 0u;
    } else {
        out.id = in.id | (in.body << 20u);
    }
    return out;
}
@fragment
fn fs(in: Out) -> @location(0) u32 { return in.id; }
"#;
const EDGE_SHADER_BODY: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    back: vec4<f32>,
    eye: vec4<f32>,
    params: vec4<f32>,
    extra_model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
const PICK: f32 = 32.0;
struct In {
    @location(0) a: vec3<f32>,
    @location(1) id: u32,
    @location(2) b: vec3<f32>,
    @location(3) edge_class: u32,
};
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
};
@vertex
fn vs(in: In, @builtin(vertex_index) vi: u32) -> Out {
    var out: Out;
    // u.params.y is the same seam setting the visible ribbon's mask carries.
    let seam_mask = select(0u, EDGE_CLASS_SEAM, u.params.y > 0.5);
    if (edge_class_hidden(in.edge_class, seam_mask)) {
        out.position = vec4<f32>(0.0);
        out.id = 0u;
        return out;
    }
    let ca = u.view_proj * u.extra_model * vec4<f32>(in.a, 1.0);
    let cb = u.view_proj * u.extra_model * vec4<f32>(in.b, 1.0);
    let ribbon = edge_ribbon(ca, cb, vec2<f32>(PICK, PICK), u.params.x, vi);
    out.position = ribbon.position;
    out.id = in.id;
    return out;
}
@fragment
fn fs(in: Out) -> @location(0) u32 { return in.id; }
"#;

/// Plan-local edge pick shader. It shares `EDGE_RIBBON_WGSL` with the visible
/// pipeline, so the pick band is the same shape and width as the drawn band.
const PLAN_EDGE_SHADER_BODY: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    extra_model: mat4x4<f32>,
    screen: vec4<f32>,
    colour: vec4<f32>,
    mask: vec4<u32>,
};
@group(0) @binding(0) var<uniform> u: U;
struct In {
    @location(0) a: vec3<f32>,
    @location(1) b: vec3<f32>,
    @location(2) edge: u32,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(12) edge_class: u32,
    @location(13) edge_base: u32,
};
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
};
@vertex
fn vs(in: In, @builtin(vertex_index) vi: u32) -> Out {
    var out: Out;
    // Same class/mask rule as the visible ribbon: a hidden seam has no pick id.
    if (edge_class_hidden(in.edge_class, u.mask.x)) {
        out.position = vec4<f32>(0.0);
        out.id = 0u;
        return out;
    }
    let model = u.extra_model * mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let ca = u.view_proj * model * vec4<f32>(in.a, 1.0);
    let cb = u.view_proj * model * vec4<f32>(in.b, 1.0);
    let ribbon = edge_ribbon(ca, cb, u.screen.xy, u.screen.z, vi);
    out.position = ribbon.position;
    out.id = in.edge_base + in.edge;
    return out;
}
@fragment
fn fs(in: Out) -> @location(0) u32 { return in.id; }
"#;

fn edge_base_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset: 0,
        shader_location: 13,
    }];
    wgpu::VertexBufferLayout {
        array_stride: 4,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &ATTRS,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SlotPhase {
    Free,
    Copied(PickRequest),
    Mapping(PickRequest),
}

pub(crate) struct Ring {
    pub phases: [SlotPhase; 3],
}

impl Default for Ring {
    fn default() -> Self {
        Self { phases: [SlotPhase::Free, SlotPhase::Free, SlotPhase::Free] }
    }
}

impl Ring {
    pub(crate) fn free_slot(&self) -> Option<usize> {
        self.phases.iter().position(|phase| matches!(phase, SlotPhase::Free))
    }

    pub(crate) fn copied(&mut self, slot: usize, request: PickRequest) {
        debug_assert!(matches!(self.phases[slot], SlotPhase::Free));
        self.phases[slot] = SlotPhase::Copied(request);
    }

    pub(crate) fn start_mapping(&mut self) -> Vec<usize> {
        let mut slots = Vec::new();
        for (index, phase) in self.phases.iter_mut().enumerate() {
            if let SlotPhase::Copied(request) = phase {
                *phase = SlotPhase::Mapping(*request);
                slots.push(index);
            }
        }
        slots
    }

    pub(crate) fn finish(&mut self, slot: usize, status: u8) -> Option<PickRequest> {
        let SlotPhase::Mapping(request) = self.phases[slot] else {
            return None;
        };
        match status {
            1 => {
                self.phases[slot] = SlotPhase::Free;
                Some(request)
            }
            2 => {
                self.phases[slot] = SlotPhase::Free;
                None
            }
            _ => None,
        }
    }

    pub(crate) fn in_flight(&self) -> bool {
        self.phases.iter().any(|phase| !matches!(phase, SlotPhase::Free))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TakeDecision {
    Take,
    Keep,
    Drop,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PickUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub back: [f32; 4],
    pub eye: [f32; 4],
    pub params: [f32; 4],
    pub extra_model: [[f32; 4]; 4],
}

pub(crate) fn take_decision(
    request: Option<&PickRequest>,
    ghost_key: u64,
    free_slot: bool,
) -> TakeDecision {
    let Some(request) = request else { return TakeDecision::Keep };
    if request.ghost_key != ghost_key {
        return TakeDecision::Drop;
    }
    if free_slot { TakeDecision::Take } else { TakeDecision::Keep }
}

pub(crate) fn merge_result(held: Option<PickResult>, new: PickResult) -> PickResult {
    match held {
        Some(old) if old.commit && !new.commit => old,
        Some(old) if old.id > new.id && !new.commit => old,
        _ => new,
    }
}

pub(crate) struct PickPasses<'a> {
    pub face: &'a wgpu::RenderPipeline,
    pub face_id: &'a wgpu::RenderPipeline,
    /// Face ids from the object mesh's position/face-id/instance buffers. The
    /// same shader as `face_id`, with the object vertex layout.
    pub object_face_id: &'a wgpu::RenderPipeline,
    pub ghost_depth: &'a wgpu::RenderPipeline,
    /// Plan-local edge ids through the shared ribbon function.
    pub plan_edge: &'a wgpu::RenderPipeline,
}

pub(crate) struct PickBinds<'a> {
    pub main: &'a wgpu::BindGroup,
    pub ghost: &'a wgpu::BindGroup,
    /// Uniform bind for `PickPasses::plan_edge`.
    pub plan_edge: &'a wgpu::BindGroup,
}

pub(crate) struct PickRenderer {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    reversed: bool,
    depth_view: wgpu::TextureView,
    face_pipeline: wgpu::RenderPipeline,
    face_id_pipeline: wgpu::RenderPipeline,
    object_face_id_pipeline: wgpu::RenderPipeline,
    ghost_depth_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    plan_edge_pipeline: wgpu::RenderPipeline,
    plan_edge_uniforms: wgpu::Buffer,
    plan_edge_bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    ghost_uniforms: wgpu::Buffer,
    ghost_bind_group: wgpu::BindGroup,
    segments: wgpu::Buffer,
    segment_capacity: u32,
    uploaded_key: u64,
    buffers: [wgpu::Buffer; 3],
    status: [std::sync::Arc<std::sync::atomic::AtomicU8>; 3],
    ring: Ring,
    result: Option<PickResult>,
    last_grid: Option<Vec<u32>>,
}

impl PickRenderer {
    pub(crate) fn install(
        device: &wgpu::Device,
        depth_format: wgpu::TextureFormat,
        reversed: bool,
    ) -> Self {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick-color"),
            size: wgpu::Extent3d { width: PICK, height: PICK, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick-depth"),
            size: wgpu::Extent3d { width: PICK, height: PICK, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick-uniforms"),
            size: std::mem::size_of::<crate::gpu_viewport::ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick-bind-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick-bind-group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let ghost_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick-ghost-uniforms"),
            size: std::mem::size_of::<crate::gpu_viewport::ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ghost_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick-ghost-bind-group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ghost_uniforms.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pick-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let face_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pick-face"),
            source: wgpu::ShaderSource::Wgsl(FACE_SHADER.into()),
        });
        let face_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pick-face-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &face_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(crate::gpu_plan::plan_vertex_layout()),
                    Some(crate::gpu_viewport::object_instance_layout()),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &face_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(if reversed {
                    wgpu::CompareFunction::GreaterEqual
                } else {
                    wgpu::CompareFunction::LessEqual
                }),
                stencil: wgpu::StencilState::default(),
                bias: depth_bias(reversed, FACE_BIAS),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let face_id_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pick-face-id"),
            source: wgpu::ShaderSource::Wgsl(FACE_ID_SHADER.into()),
        });
        let plan_face_id_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pick-plan-face-id"),
            source: wgpu::ShaderSource::Wgsl(PLAN_FACE_ID_SHADER.into()),
        });
        let face_id_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pick-face-id-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &plan_face_id_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(crate::gpu_plan::plan_vertex_layout()),
                    Some(crate::gpu_plan::plan_face_layout()),
                    Some(crate::gpu_viewport::object_instance_layout()),
                    Some(crate::gpu_plan::plan_body_layout()),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &plan_face_id_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(if reversed {
                    wgpu::CompareFunction::GreaterEqual
                } else {
                    wgpu::CompareFunction::LessEqual
                }),
                stencil: wgpu::StencilState::default(),
                bias: depth_bias(reversed, FACE_BIAS),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        // The object renderer stores position/normal/colour interleaved, so its
        // face-id draw needs the position-only object layout and a parallel
        // per-vertex face-id buffer. Objects bake the body into the vertex id,
        // so they use the original shader without a per-instance body buffer.
        let object_face_id_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pick-object-face-id-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &face_id_module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[
                        Some(crate::gpu_viewport::object_pick_vertex_layout()),
                        Some(crate::gpu_plan::plan_face_layout()),
                        Some(crate::gpu_viewport::object_instance_layout()),
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &face_id_module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(if reversed {
                        wgpu::CompareFunction::GreaterEqual
                    } else {
                        wgpu::CompareFunction::LessEqual
                    }),
                    stencil: wgpu::StencilState::default(),
                    bias: depth_bias(reversed, FACE_BIAS),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        // The edit ghost writes depth for edge occlusion but must not overwrite
        // the face ids already drawn underneath its transparent surface.
        let ghost_depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pick-ghost-depth-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &face_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(crate::gpu_plan::plan_vertex_layout()),
                    Some(crate::gpu_viewport::object_instance_layout()),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &face_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(if reversed {
                    wgpu::CompareFunction::GreaterEqual
                } else {
                    wgpu::CompareFunction::LessEqual
                }),
                stencil: wgpu::StencilState::default(),
                bias: depth_bias(reversed, FACE_BIAS),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let edge_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pick-edge"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{EDGE_RIBBON_WGSL}\n{EDGE_SHADER_BODY}").into(),
            ),
        });
        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pick-edge-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &edge_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 16,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 28,
                            shader_location: 3,
                        },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(if reversed {
                    wgpu::CompareFunction::GreaterEqual
                } else {
                    wgpu::CompareFunction::LessEqual
                }),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let plan_edge_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pick-plan-edge"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{EDGE_RIBBON_WGSL}\n{PLAN_EDGE_SHADER_BODY}").into(),
            ),
        });
        let plan_edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pick-plan-edge-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &plan_edge_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(edge_vertex_layout()),
                    Some(edge_instance_layout()),
                    Some(edge_base_layout()),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &plan_edge_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(if reversed {
                    wgpu::CompareFunction::GreaterEqual
                } else {
                    wgpu::CompareFunction::LessEqual
                }),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let plan_edge_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick-plan-edge-uniforms"),
            size: std::mem::size_of::<EdgeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plan_edge_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick-plan-edge-bind-group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: plan_edge_uniforms.as_entire_binding(),
            }],
        });
        let segments = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick-segments"),
            size: 1,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffers = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pick-readback"),
                size: 256 * PICK as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let status =
            std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)));
        Self {
            color_view: color.create_view(&wgpu::TextureViewDescriptor::default()),
            color,
            reversed,
            depth_view: depth.create_view(&wgpu::TextureViewDescriptor::default()),
            face_pipeline,
            face_id_pipeline,
            object_face_id_pipeline,
            ghost_depth_pipeline,
            edge_pipeline,
            plan_edge_pipeline,
            plan_edge_uniforms,
            plan_edge_bind_group,
            uniforms,
            bind_group,
            ghost_uniforms,
            ghost_bind_group,
            segments,
            segment_capacity: 0,
            uploaded_key: 0,
            buffers,
            status,
            ring: Ring::default(),
            result: None,
            last_grid: None,
        }
    }

    pub(crate) fn last_grid(&self) -> Option<&[u32]> {
        self.last_grid.as_deref()
    }

    pub(crate) fn take_result(&mut self) -> Option<PickResult> {
        self.result.take()
    }

    pub(crate) fn has_free_slot(&self) -> bool {
        self.ring.free_slot().is_some()
    }

    pub(crate) fn in_flight(&self) -> bool {
        self.ring.in_flight()
    }

    pub(crate) fn service(&mut self, device: &wgpu::Device) {
        use std::sync::atomic::Ordering;
        for slot in self.ring.start_mapping() {
            self.status[slot].store(0, Ordering::Release);
            let buffer = self.buffers[slot].slice(..);
            let status = std::sync::Arc::clone(&self.status[slot]);
            buffer.map_async(wgpu::MapMode::Read, move |result| {
                status.store(if result.is_ok() { 1 } else { 2 }, Ordering::Release);
            });
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let mut best: Option<PickResult> = None;
        for slot in 0..self.ring.phases.len() {
            let status = self.status[slot].load(Ordering::Acquire);
            if status == 0 {
                continue;
            }
            let Some(req) = self.ring.finish(slot, status) else { continue };
            if status != 1 {
                continue;
            }
            let ids = {
                let Ok(view) = self.buffers[slot].slice(..).get_mapped_range() else {
                    self.buffers[slot].unmap();
                    continue;
                };
                decode_rows(&view, PICK as usize, 256)
            };
            self.buffers[slot].unmap();
            let index = nearest_pick_of_kind(&ids, PICK as usize, req.radius_px, req.kind);
            self.last_grid = Some(ids.clone());
            let grid = format_grid(
                &ids,
                PICK as usize,
                &format!("pick id={} index={index} radius={}", req.id, req.radius_px),
            );
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!("{grid}");
            }
            if req.commit
                && let Ok(path) = std::env::var("NODECRAFT_PICK_DUMP")
                && let Ok(mut file) =
                    std::fs::OpenOptions::new().create(true).append(true).open(path)
            {
                use std::io::Write as _;
                let _ = writeln!(file, "{grid}\n");
            }
            let result =
                PickResult { id: req.id, ghost_key: req.ghost_key, commit: req.commit, index };
            best = Some(match best {
                Some(old) => merge_result(Some(old), result),
                None => result,
            });
        }
        if let Some(result) = best {
            self.result = Some(merge_result(self.result.take(), result));
        }
    }

    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        uniforms: &PickUniforms,
        frame: &PickFrame,
        request: PickRequest,
        faces: impl FnOnce(&mut wgpu::RenderPass<'_>, &PickPasses<'_>, &PickBinds<'_>),
        mut profiler: Option<&mut crate::gpu_profiler::GpuProfilerSlot>,
    ) {
        if request.ghost_key != frame.ghost_key {
            return;
        }
        let Some(slot_index) = self.ring.free_slot() else { return };
        if self.uploaded_key != frame.ghost_key {
            let bytes: &[u8] = bytemuck::cast_slice(frame.segments.as_ref());
            if frame.segments.len() as u32 > self.segment_capacity {
                self.segment_capacity = frame.segments.len().max(1) as u32;
                self.segments = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pick-segments"),
                    size: u64::from(self.segment_capacity) * 32,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            if !bytes.is_empty() {
                queue.write_buffer(&self.segments, 0, bytes);
            }
            self.uploaded_key = frame.ghost_key;
        }
        let crop = crop_matrix(
            request.px[0],
            request.px[1],
            frame.size_px[0],
            frame.size_px[1],
            PICK as f32,
        );
        let view_proj = crate::plan_math::multiply_f32(crop, uniforms.view_proj);
        let mut main = PickUniforms { view_proj, extra_model: frame.face_matrix, ..*uniforms };
        main.params =
            [frame.edge_width_px, if frame.show_seam_edges { 1.0 } else { 0.0 }, 0.0, 0.0];
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&main));
        let mut ghost = PickUniforms { view_proj, extra_model: frame.ghost_matrix, ..*uniforms };
        ghost.params = main.params;
        queue.write_buffer(&self.ghost_uniforms, 0, bytemuck::bytes_of(&ghost));
        let plan_edge = EdgeUniforms::new(
            view_proj,
            frame.face_matrix,
            [PICK as f32, PICK as f32],
            frame.edge_width_px,
            [0.0; 4],
            true,
        )
        .show_seams(frame.show_seam_edges);
        queue.write_buffer(&self.plan_edge_uniforms, 0, bytemuck::bytes_of(&plan_edge));
        let passes = PickPasses {
            face: &self.face_pipeline,
            face_id: &self.face_id_pipeline,
            object_face_id: &self.object_face_id_pipeline,
            ghost_depth: &self.ghost_depth_pipeline,
            plan_edge: &self.plan_edge_pipeline,
        };
        let binds = PickBinds {
            main: &self.bind_group,
            ghost: &self.ghost_bind_group,
            plan_edge: &self.plan_edge_bind_group,
        };
        crate::gpu_profiler::render_pass(
            profiler.as_deref_mut(),
            crate::gpu_profiler::pass::PICK,
            encoder,
            wgpu::RenderPassDescriptor {
                label: Some(crate::gpu_profiler::pass::PICK),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if self.reversed { 0.0 } else { 1.0 }),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            },
            |pass| {
                pass.set_bind_group(0, &self.bind_group, &[]);
                faces(pass, &passes, &binds);
                if !frame.segments.is_empty() {
                    pass.set_pipeline(&self.edge_pipeline);
                    let edge_bind =
                        if frame.ghost_edges { &self.ghost_bind_group } else { &self.bind_group };
                    pass.set_bind_group(0, edge_bind, &[]);
                    pass.set_vertex_buffer(0, self.segments.slice(..));
                    pass.draw(0..6, 0..frame.segments.len() as u32);
                }
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffers[slot_index],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(PICK),
                },
            },
            wgpu::Extent3d { width: PICK, height: PICK, depth_or_array_layers: 1 },
        );
        self.ring.copied(slot_index, request);
    }
}
