//! Off-thread CAD source work: hashing, ingest records and (later) watching.
//!
//! The UI thread never reads or hashes a source file. One service owns one
//! worker thread for the whole app; importers register paths with it instead of
//! each node spawning its own watcher.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use nodecraft_engine::import_io;
use nodecraft_protocol::IngestRecord;
use notify::{Config as NotifyConfig, PollWatcher, RecommendedWatcher, RecursiveMode, Watcher};

/// What the worker should do with a source path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceAction {
    Ingest,
    Hash,
    Watch,
    SaveCopy,
}

/// Why a content hash is being computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashPurpose {
    /// Background check; reuse a cached hash while size/mtime are unchanged.
    Check,
    /// Explicit Reload: always read and hash the file.
    Reload,
}

/// One watched CAD source. The app replaces the whole list when the active
/// document, watch switches or paths change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WatchTarget {
    pub document_id: String,
    pub node_id: String,
    pub path: PathBuf,
    /// Last imported/ingested content hash; the first watched change compares
    /// against it.
    pub content_hash: Option<String>,
}

impl SourceAction {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ingest => "embedded copy",
            Self::Hash => "hash",
            Self::Watch => "watch",
            Self::SaveCopy => "save copy",
        }
    }
}

enum Request {
    Ingest {
        id: u64,
        document_id: String,
        node_id: String,
        path: PathBuf,
    },
    Hash {
        id: u64,
        document_id: String,
        node_id: String,
        path: PathBuf,
        purpose: HashPurpose,
    },
    SaveCopy {
        id: u64,
        document_id: String,
        node_id: String,
        record: IngestRecord,
        destination: PathBuf,
    },
}

/// A finished background source operation.
#[derive(Clone, Debug)]
pub enum Event {
    Ingested {
        id: u64,
        document_id: String,
        node_id: String,
        record: IngestRecord,
    },
    Hashed {
        id: u64,
        document_id: String,
        node_id: String,
        content_hash: String,
        purpose: HashPurpose,
    },
    Failed {
        id: u64,
        document_id: String,
        node_id: String,
        action: SourceAction,
        error: String,
    },
    /// A watched file became stable with a different content hash.
    Watched {
        document_id: String,
        node_id: String,
        content_hash: String,
    },
    /// The embedded copy's bytes were written to a chosen file.
    CopySaved {
        id: u64,
        document_id: String,
        node_id: String,
        path: PathBuf,
    },
}

impl Event {
    #[must_use]
    pub fn request_id(&self) -> u64 {
        match self {
            Self::Ingested { id, .. }
            | Self::Hashed { id, .. }
            | Self::Failed { id, .. }
            | Self::CopySaved { id, .. } => *id,
            Self::Watched { .. } => 0,
        }
    }

    #[must_use]
    pub fn action(&self) -> SourceAction {
        match self {
            Self::Ingested { .. } => SourceAction::Ingest,
            Self::Hashed { .. } => SourceAction::Hash,
            Self::Failed { action, .. } => *action,
            Self::Watched { .. } => SourceAction::Watch,
            Self::CopySaved { .. } => SourceAction::SaveCopy,
        }
    }
}

/// Last imported content hash per importer plus the set of nodes whose source
/// differs from it. One place decides when the "Source changed" badge appears
/// and when a successful reload clears it.
#[derive(Clone, Debug, Default)]
pub struct ChangeTracker {
    hashes: HashMap<(String, String), String>,
    changed: HashMap<String, std::collections::HashSet<String>>,
}

impl ChangeTracker {
    fn key(document_id: &str, node_id: &str) -> (String, String) {
        (document_id.to_owned(), node_id.to_owned())
    }

    /// Record the hash the document last imported, without flagging a change.
    pub fn seed(&mut self, document_id: &str, node_id: &str, content_hash: &str) {
        self.hashes
            .entry(Self::key(document_id, node_id))
            .or_insert_with(|| content_hash.to_owned());
    }

    #[must_use]
    pub fn has_baseline(&self, document_id: &str, node_id: &str) -> bool {
        self.hashes.contains_key(&Self::key(document_id, node_id))
    }

    /// Apply a background check. Returns true when this is a change against the
    /// recorded baseline; the first observation is a baseline, not a change.
    pub fn observe(&mut self, document_id: &str, node_id: &str, content_hash: &str) -> bool {
        let key = Self::key(document_id, node_id);
        let previous = self.hashes.insert(key, content_hash.to_owned());
        let differs = previous.as_deref().is_some_and(|previous| previous != content_hash);
        if differs {
            self.changed.entry(document_id.to_owned()).or_default().insert(node_id.to_owned());
        } else {
            self.clear(document_id, node_id);
        }
        differs
    }

    /// Record a successful reload/ingest, so the same content no longer shows
    /// the badge.
    pub fn reloaded(&mut self, document_id: &str, node_id: &str, content_hash: &str) {
        self.hashes.insert(Self::key(document_id, node_id), content_hash.to_owned());
        self.clear(document_id, node_id);
    }

    /// Forget a node's baseline after its path or ingest setting changed.
    pub fn forget(&mut self, document_id: &str, node_id: &str) {
        self.hashes.remove(&Self::key(document_id, node_id));
        self.clear(document_id, node_id);
    }

    fn clear(&mut self, document_id: &str, node_id: &str) {
        if let Some(changed) = self.changed.get_mut(document_id) {
            changed.remove(node_id);
        }
    }

    #[must_use]
    pub fn hash(&self, document_id: &str, node_id: &str) -> Option<&str> {
        self.hashes.get(&Self::key(document_id, node_id)).map(String::as_str)
    }

    #[must_use]
    pub fn is_changed(&self, document_id: &str, node_id: &str) -> bool {
        self.changed.get(document_id).is_some_and(|changed| changed.contains(node_id))
    }

    #[must_use]
    pub fn changed(&self, document_id: &str) -> Option<&std::collections::HashSet<String>> {
        self.changed.get(document_id)
    }
}

fn fingerprint(path: &Path) -> Option<(u64, u128)> {
    let metadata = std::fs::metadata(path).ok()?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    Some((metadata.len(), modified))
}

/// One app-wide source worker.
pub struct SourceService {
    tx: Sender<Request>,
    watch_tx: Sender<Vec<WatchTarget>>,
    rx: Receiver<Event>,
    next_id: u64,
}

impl Default for SourceService {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceService {
    #[must_use]
    pub fn new() -> Self {
        let (tx, requests) = channel::<Request>();
        let (events, rx) = channel::<Event>();
        let (watch_tx, watch_rx) = channel::<Vec<WatchTarget>>();
        let watch_events = events.clone();
        std::thread::Builder::new()
            .name("nodecraft-watch".into())
            .spawn(move || run_watch_thread(watch_rx, watch_events))
            .expect("spawn nodecraft-watch worker");
        std::thread::Builder::new()
            .name("nodecraft-source".into())
            .stack_size(4 * 1024 * 1024)
            .spawn(move || {
                while let Ok(request) = requests.recv() {
                    match request {
                        Request::Ingest { id, document_id, node_id, path } => {
                            match import_io::ingest_record_from_file(&path) {
                                Ok(record) => {
                                    let _ = events.send(Event::Ingested {
                                        id,
                                        document_id,
                                        node_id,
                                        record,
                                    });
                                }
                                Err(error) => {
                                    let _ = events.send(Event::Failed {
                                        id,
                                        document_id,
                                        node_id,
                                        action: SourceAction::Ingest,
                                        error,
                                    });
                                }
                            }
                        }
                        Request::Hash { id, document_id, node_id, path, purpose } => {
                            // One shared fingerprint cache lives in import_io:
                            // Check reuses it, and an explicit Reload invalidates
                            // this file through the same cache rather than
                            // taking a second, uncached path.
                            if purpose == HashPurpose::Reload {
                                import_io::invalidate_file_hash(&path);
                            }
                            match import_io::hash_file(&path) {
                                Ok(content_hash) => {
                                    let _ = events.send(Event::Hashed {
                                        id,
                                        document_id,
                                        node_id,
                                        content_hash,
                                        purpose,
                                    });
                                }
                                Err(error) => {
                                    let _ = events.send(Event::Failed {
                                        id,
                                        document_id,
                                        node_id,
                                        action: SourceAction::Hash,
                                        error,
                                    });
                                }
                            }
                        }
                        Request::SaveCopy { id, document_id, node_id, record, destination } => {
                            match import_io::save_ingest_copy(&record, &destination) {
                                Ok(()) => {
                                    let _ = events.send(Event::CopySaved {
                                        id,
                                        document_id,
                                        node_id,
                                        path: destination,
                                    });
                                }
                                Err(error) => {
                                    let _ = events.send(Event::Failed {
                                        id,
                                        document_id,
                                        node_id,
                                        action: SourceAction::SaveCopy,
                                        error,
                                    });
                                }
                            }
                        }
                    }
                }
            })
            .expect("spawn nodecraft-source worker");
        Self { tx, watch_tx, rx, next_id: 1 }
    }

    /// Queue an ingest-record build and return its request id.
    pub fn ingest(&mut self, document_id: &str, node_id: &str, path: &Path) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let _ = self.tx.send(Request::Ingest {
            id,
            document_id: document_id.to_owned(),
            node_id: node_id.to_owned(),
            path: path.to_path_buf(),
        });
        id
    }

    /// Queue a content hash and return its request id.
    pub fn hash(&mut self, document_id: &str, node_id: &str, path: &Path) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let _ = self.tx.send(Request::Hash {
            id,
            document_id: document_id.to_owned(),
            node_id: node_id.to_owned(),
            path: path.to_path_buf(),
            purpose: HashPurpose::Check,
        });
        id
    }

    /// Force a content hash for an explicit Reload.
    pub fn hash_reload(&mut self, document_id: &str, node_id: &str, path: &Path) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let _ = self.tx.send(Request::Hash {
            id,
            document_id: document_id.to_owned(),
            node_id: node_id.to_owned(),
            path: path.to_path_buf(),
            purpose: HashPurpose::Reload,
        });
        id
    }

    /// Write one embedded record's original bytes to a chosen file, off the UI
    /// thread. Returns the request id.
    pub fn save_copy(
        &mut self,
        document_id: &str,
        node_id: &str,
        record: &IngestRecord,
        destination: &Path,
    ) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let _ = self.tx.send(Request::SaveCopy {
            id,
            document_id: document_id.to_owned(),
            node_id: node_id.to_owned(),
            record: record.clone(),
            destination: destination.to_path_buf(),
        });
        id
    }

    /// Replace the watched set with the active document's watch-on imports.
    /// Replacing also unregisters every stale path.
    pub fn set_watches(&mut self, targets: Vec<WatchTarget>) {
        let _ = self.watch_tx.send(targets);
    }

    #[must_use]
    pub fn try_recv(&self) -> Option<Event> {
        self.rx.try_recv().ok()
    }
}

/// Whether this path is likely on a network or cloud-sync volume, where the
/// native OS watcher is unreliable and the polling backend is the safe choice.
#[must_use]
pub fn prefer_polling(path: &Path) -> bool {
    let text = path.to_string_lossy().to_lowercase();
    text.starts_with("/volumes/")
        || text.starts_with("/network/")
        || text.contains("/dropbox/")
        || text.contains("/onedrive/")
        || text.contains("google drive")
        || text.contains("icloud")
        || text.contains("library/mobile documents")
}

struct WatchEntry {
    document_id: String,
    node_id: String,
    path: PathBuf,
    last_hash: Option<String>,
    due: Option<Instant>,
    last_meta: Option<(u64, u128)>,
    stable: u8,
}

struct WatchManager {
    entries: Vec<WatchEntry>,
    recommended: Option<RecommendedWatcher>,
    poll: Option<PollWatcher>,
    recommended_dirs: HashSet<PathBuf>,
    poll_dirs: HashSet<PathBuf>,
    force_poll: bool,
    touched: Sender<PathBuf>,
    events: Sender<Event>,
}

impl WatchManager {
    fn new(touched: Sender<PathBuf>, events: Sender<Event>) -> Self {
        Self {
            entries: Vec::new(),
            recommended: None,
            poll: None,
            recommended_dirs: HashSet::new(),
            poll_dirs: HashSet::new(),
            force_poll: false,
            touched,
            events,
        }
    }

    fn set_targets(&mut self, targets: Vec<WatchTarget>) {
        self.entries = targets
            .into_iter()
            .map(|target| {
                // Seed an initial baseline off the UI thread when the document
                // has not recorded one yet.
                let last_hash =
                    target.content_hash.or_else(|| import_io::hash_file(&target.path).ok());
                WatchEntry {
                    document_id: target.document_id,
                    node_id: target.node_id,
                    path: target.path,
                    last_hash,
                    due: None,
                    last_meta: None,
                    stable: 0,
                }
            })
            .collect();
        self.rebuild_watches();
    }

    fn ensure_watchers(&mut self) {
        if self.recommended.is_none() && !self.force_poll {
            let touched = self.touched.clone();
            match notify::recommended_watcher(move |result: notify::Result<notify::Event>| {
                if let Ok(event) = result {
                    for path in event.paths {
                        let _ = touched.send(path);
                    }
                }
            }) {
                Ok(watcher) => self.recommended = Some(watcher),
                Err(_) => self.force_poll = true,
            }
        }
        if self.poll.is_none() {
            let touched = self.touched.clone();
            let config = NotifyConfig::default().with_poll_interval(Duration::from_secs(2));
            if let Ok(watcher) = PollWatcher::new(
                move |result: notify::Result<notify::Event>| {
                    if let Ok(event) = result {
                        for path in event.paths {
                            let _ = touched.send(path);
                        }
                    }
                },
                config,
            ) {
                self.poll = Some(watcher);
            }
        }
    }

    fn rebuild_watches(&mut self) {
        self.ensure_watchers();
        if self.force_poll {
            self.recommended = None;
        }
        let mut recommended: HashSet<PathBuf> = HashSet::new();
        let mut poll: HashSet<PathBuf> = HashSet::new();
        for entry in &self.entries {
            let Some(directory) = entry.path.parent() else { continue };
            if self.force_poll || prefer_polling(&entry.path) {
                poll.insert(directory.to_path_buf());
            } else {
                recommended.insert(directory.to_path_buf());
            }
        }
        // Unregister paths that are no longer watched before adding new ones.
        if let Some(watcher) = self.recommended.as_mut() {
            for directory in self.recommended_dirs.difference(&recommended) {
                let _ = watcher.unwatch(directory);
            }
        }
        if let Some(watcher) = self.poll.as_mut() {
            for directory in self.poll_dirs.difference(&poll) {
                let _ = watcher.unwatch(directory);
            }
        }
        let mut failed_recommended = false;
        if let Some(watcher) = self.recommended.as_mut() {
            for directory in recommended.difference(&self.recommended_dirs) {
                if watcher.watch(directory, RecursiveMode::NonRecursive).is_err() {
                    failed_recommended = true;
                    break;
                }
            }
        }
        if failed_recommended {
            // The OS watcher refused the volume: fall back to polling every
            // target, including the paths already registered above.
            self.force_poll = true;
            self.recommended = None;
            recommended.clear();
            for entry in &self.entries {
                if let Some(directory) = entry.path.parent() {
                    poll.insert(directory.to_path_buf());
                }
            }
        }
        if let Some(watcher) = self.poll.as_mut() {
            for directory in poll.difference(&self.poll_dirs) {
                let _ = watcher.watch(directory, RecursiveMode::NonRecursive);
            }
        }
        self.recommended_dirs = recommended;
        self.poll_dirs = poll;
    }

    fn touch(&mut self, path: &Path) {
        let now = Instant::now();
        for entry in &mut self.entries {
            let matches = entry.path == path
                || (entry.path.parent() == path.parent()
                    && entry.path.file_name() == path.file_name());
            if matches {
                entry.due = Some(now + Duration::from_millis(500));
                entry.stable = 0;
                entry.last_meta = None;
            }
        }
    }

    fn process_due(&mut self) {
        let now = Instant::now();
        for entry in &mut self.entries {
            let Some(due) = entry.due else { continue };
            if due > now {
                continue;
            }
            let Some(meta) = fingerprint(&entry.path) else {
                entry.due = None;
                continue;
            };
            if entry.last_meta == Some(meta) {
                entry.stable = entry.stable.saturating_add(1);
            } else {
                entry.last_meta = Some(meta);
                entry.stable = 1;
            }
            if entry.stable < 2 {
                entry.due = Some(now + Duration::from_millis(200));
                continue;
            }
            entry.due = None;
            let Ok(content_hash) = import_io::hash_file(&entry.path) else { continue };
            if entry.last_hash.as_deref() == Some(content_hash.as_str()) {
                continue;
            }
            entry.last_hash = Some(content_hash.clone());
            let _ = self.events.send(Event::Watched {
                document_id: entry.document_id.clone(),
                node_id: entry.node_id.clone(),
                content_hash,
            });
        }
    }
}

fn run_watch_thread(commands: Receiver<Vec<WatchTarget>>, events: Sender<Event>) {
    let (touched_tx, touched_rx) = channel::<PathBuf>();
    let mut manager = WatchManager::new(touched_tx, events);
    let mut commands_open = true;
    loop {
        if commands_open {
            while let Ok(targets) = commands.try_recv() {
                manager.set_targets(targets);
            }
        }
        match touched_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(path) => manager.touch(&path),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
        // A command may have arrived while waiting; drain it next iteration.
        commands_open = match commands.try_recv() {
            Ok(targets) => {
                manager.set_targets(targets);
                true
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => true,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => false,
        };
        manager.process_due();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "sn-source-service-{}-{}-{name}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("dir");
        let path = directory.join(name);
        std::fs::write(&path, bytes).expect("write");
        path
    }

    fn wait(service: &SourceService, id: u64) -> Event {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if let Some(event) = service.try_recv()
                && event.request_id() == id
            {
                return event;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("source service did not answer request {id}");
    }

    #[test]
    fn ingest_event_carries_a_decodable_record() {
        let path = temp_file("part.step", b"source bytes");
        let mut service = SourceService::new();
        let id = service.ingest("doc", "node", &path);
        match wait(&service, id) {
            Event::Ingested { record, .. } => {
                assert_eq!(
                    import_io::decode_ingest_record(&record).expect("decode"),
                    b"source bytes"
                );
                assert_eq!(record.file_name, "part.step");
            }
            other => panic!("unexpected event {other:?}"),
        }
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn change_tracker_flags_only_a_difference_from_the_baseline() {
        let mut tracker = ChangeTracker::default();
        assert!(!tracker.has_baseline("doc", "s"));
        assert!(!tracker.observe("doc", "s", "aaa"), "first observation is the baseline");
        assert!(!tracker.is_changed("doc", "s"));
        assert!(tracker.observe("doc", "s", "bbb"), "different content is a change");
        assert!(tracker.is_changed("doc", "s"));
        assert!(tracker.changed("doc").is_some_and(|changed| changed.contains("s")));
        tracker.reloaded("doc", "s", "bbb");
        assert!(!tracker.is_changed("doc", "s"));
        assert!(!tracker.observe("doc", "s", "bbb"), "reloaded content is the new baseline");
        tracker.forget("doc", "s");
        assert!(!tracker.has_baseline("doc", "s"));
    }

    #[test]
    fn locating_an_identical_source_relinks_silently_but_different_content_is_a_change() {
        // Locate… seeds the source baseline from the embedded copy's hash, then
        // the shared hasher classifies the chosen file through ChangeTracker.
        let record = import_io::ingest_record("bracket.step", b"embedded bytes").expect("record");
        let identical = temp_file("bracket.step", b"embedded bytes");
        let different = temp_file("other.step", b"different bytes");

        let mut tracker = ChangeTracker::default();
        tracker.seed("doc", "node", &record.content_hash);
        let same_hash = import_io::hash_file(&identical).expect("hash identical");
        assert!(!tracker.observe("doc", "node", &same_hash));
        assert!(!tracker.is_changed("doc", "node"), "identical content re-links silently");

        let mut tracker = ChangeTracker::default();
        tracker.seed("doc", "node", &record.content_hash);
        let different_hash = import_io::hash_file(&different).expect("hash different");
        assert!(tracker.observe("doc", "node", &different_hash));
        assert!(tracker.is_changed("doc", "node"), "different content needs a Reload");

        let _ = std::fs::remove_dir_all(identical.parent().unwrap());
        let _ = std::fs::remove_dir_all(different.parent().unwrap());
    }

    #[test]
    fn watcher_prefers_polling_on_network_and_cloud_paths() {
        assert!(prefer_polling(Path::new("/Volumes/Team/model.step")));
        assert!(prefer_polling(Path::new("/Users/me/Dropbox/model.step")));
        assert!(prefer_polling(Path::new("/Users/me/Library/Mobile Documents/model.step")));
        assert!(!prefer_polling(Path::new("/Users/me/project/model.step")));
    }

    fn watched_manager(path: &Path, content_hash: &str, events: Sender<Event>) -> WatchManager {
        let (touched, _touched_rx) = channel();
        WatchManager {
            entries: vec![WatchEntry {
                document_id: "doc".into(),
                node_id: "s".into(),
                path: path.to_path_buf(),
                last_hash: Some(content_hash.into()),
                due: None,
                last_meta: None,
                stable: 0,
            }],
            recommended: None,
            poll: None,
            recommended_dirs: HashSet::new(),
            poll_dirs: HashSet::new(),
            force_poll: true,
            touched,
            events,
        }
    }

    #[test]
    fn watcher_debounces_then_requires_two_stable_checks_before_hashing() {
        let path = temp_file("watched.step", b"first");
        let old_hash = import_io::content_hash(b"old");
        let (events, received) = channel();
        let mut manager = watched_manager(&path, &old_hash, events);
        manager.touch(&path);
        let due = manager.entries[0].due.expect("debounce timer");
        assert!(
            due.duration_since(Instant::now()) >= Duration::from_millis(400),
            "debounce is about half a second"
        );
        // Rewrite the file, then force the debounce timer twice: the first
        // check records metadata, the second sees it stable and hashes.
        std::fs::write(&path, b"second").expect("rewrite");
        manager.entries[0].due = Some(Instant::now());
        manager.process_due();
        assert!(received.try_recv().is_err(), "one check is not stable yet");
        manager.entries[0].due = Some(Instant::now());
        manager.process_due();
        match received.recv_timeout(Duration::from_secs(1)).expect("watched event") {
            Event::Watched { content_hash, .. } => {
                assert_eq!(content_hash, import_io::content_hash(b"second"));
            }
            other => panic!("unexpected event {other:?}"),
        }
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn watcher_does_not_emit_when_stable_content_matches_the_baseline() {
        let path = temp_file("same.step", b"same");
        let hash = import_io::content_hash(b"same");
        let (events, received) = channel();
        let mut manager = watched_manager(&path, &hash, events);
        manager.touch(&path);
        for _ in 0..2 {
            manager.entries[0].due = Some(Instant::now());
            manager.process_due();
        }
        assert!(received.try_recv().is_err(), "same bytes must not reload");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn hash_event_is_the_shared_content_hash() {
        let path = temp_file("part.step", b"source bytes");
        let mut service = SourceService::new();
        let id = service.hash("doc", "node", &path);
        match wait(&service, id) {
            Event::Hashed { content_hash, .. } => {
                assert_eq!(content_hash, import_io::content_hash(b"source bytes"));
            }
            other => panic!("unexpected event {other:?}"),
        }
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
