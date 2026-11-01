//! Grid tessellation of one curved face, clipped to its UV trim loops.
//!
//! The primitive curved faces of the kernel are untrimmed: their loops lie on the UV
//! domain boundary and the whole grid is filled. A face imported from STEP is usually
//! trimmed somewhere in the interior, and keeping only cells whose four corners are
//! inside drops every cell the boundary passes through, which loses area and leaves the
//! display mesh open.
//!
//! This module fills the whole domain grid and cuts the cells the loop crosses:
//!
//! * a cell no loop touches is wholly in or out, decided once per grid row by crossings;
//! * a cell a loop cuts is split at the grid lines, and the pieces of loop inside the
//!   cell are joined by the cell's own corners going anticlockwise, giving the face's
//!   polygon in that cell, which is ear-clipped.
//!
//! Boundary points keep the loop's own 3D points (crossings interpolate along the loop's
//! 3D chord), so two faces meshed from the same sampled edge meet along the same points.
//! The caller supplies the grid lines and evaluates the surface; this module only works
//! in (u, v) and keeps a 3D point per vertex.

use std::collections::BTreeMap;

use solidcore::math::Vec3;
use crate::nurbs_grid_mesh::LoopPoint;

/// The mesh of one face in its own vertex array.
#[derive(Debug, Clone, Default)]
pub(crate) struct GridMesh {
    pub positions: Vec<[f64; 3]>,
    /// Each vertex's (u, v), for evaluating the surface normal.
    pub uvs: Vec<[f64; 2]>,
    /// Triangles, anticlockwise in (u, v).
    pub triangles: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, Copy)]
struct Bp {
    u: f64,
    v: f64,
    id: u32,
    border: bool,
}

const NONE: u32 = u32::MAX;

struct Verts {
    pos: Vec<[f64; 3]>,
    uv: Vec<[f64; 2]>,
    grid_id: Vec<u32>,
    n_v: usize,
}

impl Verts {
    fn add(&mut self, p: [f64; 3], uv: [f64; 2]) -> Result<u32, String> {
        let id = u32::try_from(self.pos.len()).map_err(|_| "trim mesh too large")?;
        self.pos.push(p);
        self.uv.push(uv);
        Ok(id)
    }

    fn grid(&mut self, i: usize, j: usize, u_lines: &[f64], v_lines: &[f64]) -> Result<u32, String> {
        let k = i * self.n_v + j;
        let existing = *self.grid_id.get(k).ok_or("grid index out of range")?;
        if existing != NONE {
            return Ok(existing);
        }
        let id = self.add([f64::NAN; 3], [at(u_lines, i)?, at(v_lines, j)?])?;
        *self.grid_id.get_mut(k).ok_or("grid index out of range")? = id;
        Ok(id)
    }
}

fn at(xs: &[f64], i: usize) -> Result<f64, String> {
    xs.get(i).copied().ok_or_else(|| "grid line index out of range".into())
}

/// The largest index with `lines[i] <= t`, or 0.
fn line_index(lines: &[f64], t: f64) -> usize {
    lines.partition_point(|&x| x <= t).saturating_sub(1)
}

/// Mesh the part of the surface inside `loops`.
///
/// The grid covers `u_lines` x `v_lines` over the face's UV domain. Loops are closed
/// (their last point is not a copy of the first) and must be oriented so the face is on
/// the left: the largest loop anticlockwise and the rest clockwise in (u, v). The caller
/// orients them; this module does not guess a handedness.
///
/// `eval` evaluates the surface at a grid (u, v); boundary vertices keep the loop's own
/// 3D point and crossings interpolate along the loop's 3D chord.
pub(crate) fn mesh_trimmed_grid(
    loops: &[Vec<LoopPoint>],
    u_lines: &mut Vec<f64>,
    v_lines: &mut Vec<f64>,
    eval: impl Fn(f64, f64) -> Result<[f64; 3], String>,
) -> Result<GridMesh, String> {
    let mut loops: Vec<Vec<LoopPoint>> = loops.iter().filter(|l| l.len() >= 3).cloned().collect();
    if loops.is_empty() {
        return Err("trimmed face has no usable loop".into());
    }
    if loops
        .iter()
        .flatten()
        .any(|p| !(p.u.is_finite() && p.v.is_finite()))
    {
        return Err("loop point (u, v) is not finite".into());
    }
    if u_lines.len() < 2 || v_lines.len() < 2 {
        return Err("trimmed face has fewer than two grid lines".into());
    }
    let eps_u = 1e-9 * (at(u_lines, u_lines.len() - 1)? - at(u_lines, 0)?);
    let eps_v = 1e-9 * (at(v_lines, v_lines.len() - 1)? - at(v_lines, 0)?);

    // No loop point may sit on a grid line: nudge it into a cell by a hair (its 3D point is the
    // edge's and does not move). A loop that then crosses no line, a hole smaller than a cell,
    // gets a line of its own through its middle, and the nudge runs again.
    let mut settled = false;
    for _ in 0..=loops.len() {
        for p in loops.iter_mut().flatten() {
            p.u = nudge(u_lines, p.u, eps_u)?;
            p.v = nudge(v_lines, p.v, eps_v)?;
        }
        let mut added = false;
        for l in &loops {
            let (lu0, lu1) = l.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
                (a.min(p.u), b.max(p.u))
            });
            let (lv0, lv1) = l.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
                (a.min(p.v), b.max(p.v))
            });
            let crosses = |lines: &[f64], lo: f64, hi: f64| lines.iter().any(|&x| lo < x && x < hi);
            if crosses(u_lines, lu0, lu1) || crosses(v_lines, lv0, lv1) {
                continue;
            }
            // Through the longer extent, relative to the cell, so the line has room.
            if (lu1 - lu0) / eps_u >= (lv1 - lv0) / eps_v {
                insert_sorted(u_lines, (lu0 + lu1) / 2.0);
            } else {
                insert_sorted(v_lines, (lv0 + lv1) / 2.0);
            }
            added = true;
        }
        if !added {
            settled = true;
            break;
        }
    }
    if !settled {
        return Err("a trim loop stays inside one grid cell".into());
    }
    let (n_u, n_v) = (u_lines.len(), v_lines.len());

    let mut verts = Verts {
        pos: Vec::new(),
        uv: Vec::new(),
        grid_id: vec![NONE; n_u * n_v],
        n_v,
    };

    // Split every loop segment where it crosses a grid line. A crossing's (u, v) is snapped to
    // its line; its 3D point is interpolated along the segment's 3D chord.
    let mut split: Vec<Vec<Bp>> = Vec::with_capacity(loops.len());
    let mut cuts: Vec<(f64, Option<f64>, Option<f64>)> = Vec::new();
    for l in &loops {
        let mut out = Vec::with_capacity(l.len() * 2);
        for (a, b) in l.iter().zip(l.iter().cycle().skip(1)) {
            out.push(Bp {
                u: a.u,
                v: a.v,
                id: verts.add(a.p, [a.u, a.v])?,
                border: false,
            });
            if (a.u - b.u).abs() < 1e-15 && (a.v - b.v).abs() < 1e-15 {
                continue;
            }
            cuts.clear();
            let (ia, ib) = (line_index(u_lines, a.u), line_index(u_lines, b.u));
            for i in ia.min(ib) + 1..=ia.max(ib) {
                let x = at(u_lines, i)?;
                cuts.push(((x - a.u) / (b.u - a.u), Some(x), None));
            }
            let (ja, jb) = (line_index(v_lines, a.v), line_index(v_lines, b.v));
            for j in ja.min(jb) + 1..=ja.max(jb) {
                let y = at(v_lines, j)?;
                cuts.push(((y - a.v) / (b.v - a.v), None, Some(y)));
            }
            cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
            for &(t, on_u, on_v) in &cuts {
                if !(t.is_finite() && t > 0.0 && t < 1.0) {
                    continue;
                }
                let u = on_u.unwrap_or(a.u + t * (b.u - a.u));
                let v = on_v.unwrap_or(a.v + t * (b.v - a.v));
                // Crossing a grid point exactly makes one point, not two.
                if out.last().is_some_and(|q: &Bp| {
                    q.border && (q.u - u).abs() < eps_u && (q.v - v).abs() < eps_v
                }) {
                    continue;
                }
                let p = [
                    a.p[0] + t * (b.p[0] - a.p[0]),
                    a.p[1] + t * (b.p[1] - a.p[1]),
                    a.p[2] + t * (b.p[2] - a.p[2]),
                ];
                out.push(Bp {
                    u,
                    v,
                    id: verts.add(p, [u, v])?,
                    border: true,
                });
            }
        }
        split.push(out);
    }

    // Chains, filed by the cell they run through.
    let mut cut_cells: BTreeMap<usize, Vec<Chain>> = BTreeMap::new();
    for l in &split {
        let start = l
            .iter()
            .position(|p| p.border)
            .ok_or("a trim loop crosses no grid line")?;
        let mut cur: Vec<Bp> = vec![*l.get(start).ok_or("empty split loop")?];
        for p in l.iter().cycle().skip(start + 1).take(l.len()) {
            cur.push(*p);
            if p.border {
                let (a, b) = match cur.as_slice() {
                    [a, b, ..] => (*a, *b),
                    _ => return Err("trim chain has fewer than two points".into()),
                };
                let i = (n_u - 2).min(line_index(u_lines, (a.u + b.u) / 2.0));
                let j = (n_v - 2).min(line_index(v_lines, (a.v + b.v) / 2.0));
                let pts = std::mem::replace(&mut cur, vec![*p]);
                cut_cells
                    .entry(i * n_v + j)
                    .or_default()
                    .push(Chain { pts, used: false });
            }
        }
    }

    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut tri = |a: u32, b: u32, c: u32| tris.push([a, b, c]);

    // Whole cells, in or out by crossings of each row's center line. Every split segment lies in
    // one row, so each row tests only its own.
    let mut is_cut = vec![false; n_u * n_v];
    for &key in cut_cells.keys() {
        *is_cut.get_mut(key).ok_or("cut cell index out of range")? = true;
    }
    let mut rows: Vec<Vec<(Bp, Bp)>> = vec![Vec::new(); n_v - 1];
    for l in &split {
        for (a, b) in l.iter().zip(l.iter().cycle().skip(1)) {
            let j = (n_v - 2).min(line_index(v_lines, (a.v + b.v) / 2.0));
            rows.get_mut(j).ok_or("grid row out of range")?.push((*a, *b));
        }
    }
    let mut xs: Vec<f64> = Vec::new();
    for (j, row) in rows.iter().enumerate() {
        let vc = (at(v_lines, j)? + at(v_lines, j + 1)?) / 2.0;
        xs.clear();
        xs.extend(
            row.iter()
                .filter(|(a, b)| (a.v <= vc) != (b.v <= vc))
                .map(|(a, b)| a.u + (vc - a.v) / (b.v - a.v) * (b.u - a.u)),
        );
        xs.sort_by(f64::total_cmp);
        let mut crossed = 0;
        for i in 0..n_u - 1 {
            let uc = (at(u_lines, i)? + at(u_lines, i + 1)?) / 2.0;
            while xs.get(crossed).is_some_and(|&x| x < uc) {
                crossed += 1;
            }
            if crossed % 2 == 0 || *is_cut.get(i * n_v + j).ok_or("cell index out of range")? {
                continue;
            }
            let a = verts.grid(i, j, u_lines, v_lines)?;
            let b = verts.grid(i + 1, j, u_lines, v_lines)?;
            let c = verts.grid(i + 1, j + 1, u_lines, v_lines)?;
            let d = verts.grid(i, j + 1, u_lines, v_lines)?;
            tri(a, b, c);
            tri(a, c, d);
        }
    }

    // Cut cells: chains joined anticlockwise round the cell by its corners, then ear-clipped.
    let mut poly: Vec<Bp> = Vec::new();
    for (&key, chains) in &mut cut_cells {
        let (i, j) = (key / n_v, key % n_v);
        let (u0, u1, v0, v1) = (
            at(u_lines, i)?,
            at(u_lines, i + 1)?,
            at(v_lines, j)?,
            at(v_lines, j + 1)?,
        );
        // Position round the cell boundary, 0 to 4 anticlockwise from (u0, v0); corner k is at k.
        let around = |p: &Bp| {
            let (d_b, d_r, d_t, d_l) = (
                (p.v - v0).abs(),
                (p.u - u1).abs(),
                (p.v - v1).abs(),
                (p.u - u0).abs(),
            );
            let m = d_b.min(d_r).min(d_t).min(d_l);
            if m == d_b {
                (p.u - u0) / (u1 - u0)
            } else if m == d_r {
                1.0 + (p.v - v0) / (v1 - v0)
            } else if m == d_t {
                2.0 + (u1 - p.u) / (u1 - u0)
            } else {
                3.0 + (v1 - p.v) / (v1 - v0)
            }
        };
        let corners = [
            (u0, v0, i, j),
            (u1, v0, i + 1, j),
            (u1, v1, i + 1, j + 1),
            (u0, v1, i, j + 1),
        ];
        for first in 0..chains.len() {
            if chains.get(first).ok_or("chain index out of range")?.used {
                continue;
            }
            // The polygon: a chain from its entry to its exit, then the cell's corners met going
            // anticlockwise from that exit to the nearest entry, then that chain, until back round.
            poly.clear();
            let mut cur = first;
            let mut closed = false;
            for _ in 0..=chains.len() {
                let exit = {
                    let c = chains.get_mut(cur).ok_or("chain index out of range")?;
                    c.used = true;
                    poly.extend_from_slice(&c.pts);
                    around(c.pts.last().ok_or("empty trim chain")?)
                };
                let mut next = None;
                let mut best = f64::INFINITY;
                for (ci, c) in chains.iter().enumerate() {
                    let d =
                        (around(c.pts.first().ok_or("empty trim chain")?) - exit).rem_euclid(4.0);
                    if d > 1e-12 && d < best {
                        best = d;
                        next = Some(ci);
                    }
                }
                let Some(next) = next else { break };
                let mut between: Vec<(f64, usize)> = (0..4)
                    .map(|k| ((k as f64 - exit).rem_euclid(4.0), k))
                    .filter(|&(d, _)| d > 1e-12 && d < best - 1e-12)
                    .collect();
                between.sort_by(|a, b| a.0.total_cmp(&b.0));
                for (_, k) in between {
                    let &(cu, cv, ci, cj) = corners.get(k).ok_or("cell corner out of range")?;
                    poly.push(Bp {
                        u: cu,
                        v: cv,
                        id: verts.grid(ci, cj, u_lines, v_lines)?,
                        border: true,
                    });
                }
                if next == first {
                    closed = true;
                    break;
                }
                if chains.get(next).ok_or("chain index out of range")?.used {
                    break;
                }
                cur = next;
            }
            if !closed {
                return Err(format!("trim chains do not close in cell ({i}, {j})"));
            }
            ear_clip(&poly, &mut tri);
        }
    }

    // Grid points are evaluated last, only those used.
    for (k, &id) in verts.grid_id.iter().enumerate() {
        if id == NONE {
            continue;
        }
        let (i, j) = (k / n_v, k % n_v);
        let p = eval(at(u_lines, i)?, at(v_lines, j)?)?;
        *verts.pos.get_mut(id as usize).ok_or("grid vertex out of range")? = p;
    }

    // A cell on a collapsed edge (a pole) has two corners at one point, so one of its triangles
    // has no area; so does a fan over a straight run of boundary. Neither covers anything.
    let positions = &verts.pos;
    tris.retain(|t| {
        let [a, b, c] = t.map(|k| positions.get(k as usize).copied().unwrap_or([f64::NAN; 3]));
        let u = Vec3::new(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        let v = Vec3::new(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
        let cross = u.cross(v);
        let longest = u.length().max(v.length()).max(
            Vec3::new(c[0] - b[0], c[1] - b[1], c[2] - b[2]).length(),
        );
        cross.length() > 1e-12 * longest * longest
    });

    Ok(GridMesh {
        positions: verts.pos,
        uvs: verts.uv,
        triangles: tris,
    })
}

struct Chain {
    pts: Vec<Bp>,
    used: bool,
}

fn insert_sorted(lines: &mut Vec<f64>, x: f64) {
    let at = lines.partition_point(|&y| y < x);
    lines.insert(at, x);
}

/// `t` moved off any grid line within `eps`, into the cell beside it (inward at the last line).
fn nudge(lines: &[f64], t: f64, eps: f64) -> Result<f64, String> {
    let i = line_index(lines, t);
    let x = at(lines, i)?;
    if (t - x).abs() < eps {
        return Ok(if i + 1 == lines.len() {
            x - 2.0 * eps
        } else {
            x + 2.0 * eps
        });
    }
    if let Some(&y) = lines.get(i + 1) {
        if (y - t).abs() < eps {
            return Ok(y - 2.0 * eps);
        }
    }
    Ok(t)
}

fn cross(a: &Bp, b: &Bp, c: &Bp) -> f64 {
    (b.u - a.u) * (c.v - a.v) - (b.v - a.v) * (c.u - a.u)
}

/// Triangulate one simple polygon, anticlockwise in (u, v), by ear clipping.
fn ear_clip(poly: &[Bp], tri: &mut impl FnMut(u32, u32, u32)) {
    let Some(last) = poly.len().checked_sub(1) else {
        return;
    };
    // Drop consecutive duplicates; they make zero-area ears that block the scan.
    let mut ring: Vec<Bp> = poly
        .iter()
        .zip(poly.iter().cycle().skip(last))
        .filter(|(p, q)| p.id != q.id && (p.u != q.u || p.v != q.v))
        .map(|(p, _)| *p)
        .collect();
    let mut guard = 0;
    while ring.len() > 3 && guard < 10_000 {
        guard += 1;
        let m = ring.len();
        let mut ear = None;
        for k in 0..m {
            let (Some(a), Some(b), Some(c)) = (
                ring.get((k + m - 1) % m),
                ring.get(k),
                ring.get((k + 1) % m),
            ) else {
                return;
            };
            if cross(a, b, c) <= 0.0 {
                continue;
            }
            let blocked = ring.iter().enumerate().any(|(n, p)| {
                n != (k + m - 1) % m
                    && n != k
                    && n != (k + 1) % m
                    && cross(a, b, p) > 0.0
                    && cross(b, c, p) > 0.0
                    && cross(c, a, p) > 0.0
            });
            if !blocked {
                ear = Some((k, [a.id, b.id, c.id]));
                break;
            }
        }
        let Some((k, [a, b, c])) = ear else { break };
        tri(a, b, c);
        ring.remove(k);
    }
    // What ear clipping could not settle (collinear runs along the cell border) is fanned.
    if let Some((first, rest)) = ring.split_first() {
        for w in rest.windows(2) {
            if let [b, c] = w {
                tri(first.id, b.id, c.id);
            }
        }
    }
}
