//! The ops session exists to get the thread pool's parallelism back: each worker runs ops
//! on its own kernel, with no process-wide lock. These tests time 16 operations on 1 and
//! on 4 threads and require a real speedup, so a lock creeping back into the session
//! path fails the suite rather than quietly serialising every node.

use std::sync::Arc;

use nodecraft_ops::OpsSession;
use solidcore::kernel::Kernel;
use solidcore::math::Point3;
use solidcore::primitive::api_primitives::create_box;
use solidcore::topology::Brep;

fn worker_kernel() -> Kernel {
    nodecraft_kernel_profile::kernel()
}

fn box_at(kernel: &mut Kernel, x: f64) -> Brep {
    create_box(kernel, Point3::new(x, 0.0, 0.0), 20.0, 20.0, 20.0).expect("box")
}

/// Wall time for `threads` workers to run `16 / threads` jobs each, each worker reusing its
/// own kernel as the pool does.
fn wall_seconds(threads: usize, job: impl Fn(&mut Kernel, usize) + Send + Sync + 'static) -> f64 {
    let job = Arc::new(job);
    let started = std::time::Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|thread| {
            let job = Arc::clone(&job);
            std::thread::spawn(move || {
                let mut kernel = worker_kernel();
                for index in 0..16 / threads {
                    job(&mut kernel, thread * (16 / threads) + index);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("worker thread");
    }
    started.elapsed().as_secs_f64()
}

#[test]
#[ignore = "timing: needs a quiet machine; run with --ignored --release"]
fn sixteen_shells_speed_up_on_four_threads() {
    let job = |kernel: &mut Kernel, _: usize| {
        let brep = box_at(kernel, 0.0);
        OpsSession::new(kernel).shell(&brep, &[], 1.0).expect("shell");
    };
    let serial = wall_seconds(1, job);
    let parallel = wall_seconds(4, job);
    eprintln!(
        "16 shells: serial {serial:.3}s parallel {parallel:.3}s speedup {:.2}",
        serial / parallel
    );
    assert!(serial / parallel > 2.0, "shells speedup {:.2}", serial / parallel);
}

#[test]
#[ignore = "timing: needs a quiet machine; run with --ignored --release"]
fn sixteen_booleans_speed_up_on_four_threads() {
    let job = |kernel: &mut Kernel, index: usize| {
        let a = box_at(kernel, 0.0);
        // Overlapping apart enough that each job is a real merge, not a coincident no-op.
        let b = box_at(kernel, 10.0 + index as f64 * 0.001);
        OpsSession::new(kernel)
            .boolean(
                &nodecraft_ops::SolidBody::new(a.clone_standing_for_self().expect("copy a")),
                &nodecraft_ops::SolidBody::new(b.clone_standing_for_self().expect("copy b")),
                nodecraft_ops::BooleanOp::Union,
            )
            .expect("union");
    };
    let serial = wall_seconds(1, job);
    let parallel = wall_seconds(4, job);
    eprintln!(
        "16 booleans: serial {serial:.3}s parallel {parallel:.3}s speedup {:.2}",
        serial / parallel
    );
    assert!(serial / parallel > 2.0, "booleans speedup {:.2}", serial / parallel);
}
