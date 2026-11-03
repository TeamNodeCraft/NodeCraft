//! Dependency-level parallel scheduler for owned node tasks.
//!
//! Callers build tasks after collecting their inputs; a task owns all data it
//! mutates and returns a typed result. The scheduler runs all tasks in a level
//! together on the kernel pool, then advances to dependents.

use crate::kernel_pool::{KernelPool, RevisionToken};
use nodecraft_solidcore::solidcore::kernel::Kernel;
use std::collections::{BTreeMap, BTreeSet};

pub type TaskJob<T> = Box<dyn FnOnce(&mut Kernel) -> Result<T, String> + Send + 'static>;
pub type TaskBuilder<T> = Box<
    dyn FnOnce(&BTreeMap<String, Result<T, String>>) -> Result<TaskJob<T>, String>
        + Send
        + 'static,
>;

pub struct ScheduledTask<T> {
    pub id: String,
    pub dependencies: Vec<String>,
    pub seed: u64,
    /// Builds the owned kernel job after all dependencies have results. This is
    /// where typed inputs are collected from the flat-graph dependency map.
    pub build: TaskBuilder<T>,
}

impl<T> ScheduledTask<T> {
    pub fn new(
        id: impl Into<String>,
        dependencies: Vec<String>,
        seed: u64,
        build: impl FnOnce(&BTreeMap<String, Result<T, String>>) -> Result<TaskJob<T>, String>
            + Send
            + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            dependencies,
            seed,
            build: Box::new(build),
        }
    }

    /// A task whose job needs no dependency values.
    pub fn leaf(
        id: impl Into<String>,
        seed: u64,
        job: impl FnOnce(&mut Kernel) -> Result<T, String> + Send + 'static,
    ) -> Self {
        let mut job = Some(job);
        Self::new(id, Vec::new(), seed, move |_| {
            Ok(Box::new(job.take().expect("leaf job used once")))
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ScheduleError {
    DuplicateId(String),
    MissingDependency { task: String, dependency: String },
    Cycle,
    WorkerStopped,
    Task(String),
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "duplicate scheduled task {id:?}"),
            Self::MissingDependency { task, dependency } => {
                write!(f, "task {task:?} depends on missing task {dependency:?}")
            }
            Self::Cycle => f.write_str("scheduled task graph has a cycle"),
            Self::WorkerStopped => f.write_str("kernel worker stopped during scheduling"),
            Self::Task(error) => write!(f, "scheduled task failed: {error}"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Run tasks in dependency levels. Tasks in one level are submitted together;
/// their results are collected before dependents start.
pub fn run_levels<T: Send + 'static>(
    pool: &KernelPool,
    tasks: Vec<ScheduledTask<T>>,
    token: RevisionToken,
) -> Result<BTreeMap<String, Result<T, String>>, ScheduleError> {
    let mut by_id = BTreeMap::new();
    for task in tasks {
        let id = task.id.clone();
        if by_id.insert(id.clone(), task).is_some() {
            return Err(ScheduleError::DuplicateId(id));
        }
    }
    let ids: BTreeSet<String> = by_id.keys().cloned().collect();
    for task in by_id.values() {
        for dependency in &task.dependencies {
            if !ids.contains(dependency) {
                return Err(ScheduleError::MissingDependency {
                    task: task.id.clone(),
                    dependency: dependency.clone(),
                });
            }
        }
    }
    let levels = dependency_levels(&by_id)?;
    let mut results = BTreeMap::new();
    for level in levels {
        let mut running = Vec::new();
        for id in level {
            let task = by_id.remove(&id).expect("level task exists");
            let job = match (task.build)(&results) {
                Ok(job) => job,
                Err(error) => {
                    results.insert(id, Err(error));
                    continue;
                }
            };
            let receiver = pool.submit(task.seed, token.clone(), job);
            running.push((id, receiver));
        }
        for (id, receiver) in running {
            let result = receiver.recv().map_err(|_| ScheduleError::WorkerStopped)?;
            let result = result.map_err(ScheduleError::Task)?;
            results.insert(id, result);
        }
    }
    Ok(results)
}

fn dependency_levels<T>(tasks: &BTreeMap<String, ScheduledTask<T>>) -> Result<Vec<Vec<String>>, ScheduleError> {
    fn depth<T>(
        id: &str,
        tasks: &BTreeMap<String, ScheduledTask<T>>,
        memo: &mut BTreeMap<String, usize>,
        visiting: &mut BTreeSet<String>,
    ) -> Result<usize, ScheduleError> {
        if let Some(depth) = memo.get(id) {
            return Ok(*depth);
        }
        if !visiting.insert(id.to_owned()) {
            return Err(ScheduleError::Cycle);
        }
        let mut task_depth = 0;
        if let Some(task) = tasks.get(id) {
            for dependency in &task.dependencies {
                task_depth = task_depth.max(1 + depth(dependency, tasks, memo, visiting)?);
            }
        }
        visiting.remove(id);
        memo.insert(id.to_owned(), task_depth);
        Ok(task_depth)
    }

    let mut memo = BTreeMap::new();
    let mut levels: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for id in tasks.keys() {
        let level = depth(id, tasks, &mut memo, &mut BTreeSet::new())?;
        levels.entry(level).or_default().push(id.clone());
    }
    Ok(levels.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn dependents_run_after_all_dependencies() {
        let pool = KernelPool::new(4);
        let order = Arc::new(Mutex::new(Vec::new()));
        let a_order = order.clone();
        let b_order = order.clone();
        let c_order = order.clone();
        let tasks = vec![
            ScheduledTask::leaf("a", 1, move |_| {
                a_order.lock().unwrap().push("a");
                Ok(1u32)
            }),
            ScheduledTask::leaf("b", 2, move |_| {
                b_order.lock().unwrap().push("b");
                Ok(2u32)
            }),
            ScheduledTask::new("c", vec!["a".into(), "b".into()], 3, move |deps| {
                let a = deps.get("a").and_then(|result| result.as_ref().ok()).copied();
                let b = deps.get("b").and_then(|result| result.as_ref().ok()).copied();
                let c_order = c_order.clone();
                Ok(Box::new(move |_| {
                    c_order.lock().unwrap().push("c");
                    Ok(a.unwrap_or_default() + b.unwrap_or_default() + 1)
                }))
            }),
        ];
        let results = run_levels(&pool, tasks, RevisionToken::new()).expect("schedule");
        assert_eq!(results["a"], Ok(1));
        assert_eq!(results["b"], Ok(2));
        assert_eq!(results["c"], Ok(4));
        assert_eq!(order.lock().unwrap().last().copied(), Some("c"));
    }

    #[test]
    fn cycles_and_missing_dependencies_are_rejected() {
        let pool = KernelPool::new(1);
        let cycle = vec![
            ScheduledTask::new("a", vec!["b".into()], 1, |_| Ok(Box::new(|_| Ok(1u32)))),
            ScheduledTask::new("b", vec!["a".into()], 2, |_| Ok(Box::new(|_| Ok(2u32)))),
        ];
        assert_eq!(run_levels(&pool, cycle, RevisionToken::new()), Err(ScheduleError::Cycle));
        let missing = vec![ScheduledTask::new("a", vec!["nope".into()], 1, |_| Ok(Box::new(|_| Ok(1u32))))];
        assert!(matches!(
            run_levels(&pool, missing, RevisionToken::new()),
            Err(ScheduleError::MissingDependency { .. })
        ));
    }
}
