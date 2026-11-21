//! Shared CAD source resolution, content hashing and ingested-record payloads.
//!
//! This is the one module that decides which bytes an importer reads:
//!
//! 1. the stored path resolved against the document folder,
//! 2. the alternate form kept beside it (for an absolute path, its
//!    document-relative form),
//! 3. the ingested copy embedded in the document.
//!
//! The importer node never inspects the filesystem; it asks
//! [`crate::nodes::PartEval::step_import`], which resolves through here. The
//! inspector and the app's source watcher call the same functions so the badge,
//! the displayed path and evaluation cannot disagree.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use base64::Engine as _;
use nodecraft_protocol::IngestRecord;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Version of the ingested-record payload encoding.
pub const INGEST_SCHEMA_VERSION: u32 = 1;
/// Payload codec name written into every record.
pub const INGEST_COMPRESSION: &str = "flate2+base64";
/// Import cache format/version. Bump when the importer or cadmpeg changes in a
/// way that can change the converted bodies.
pub const IMPORT_FORMAT_VERSION: &str = concat!("nodecraft-import-v1/cadmpeg-0.6.7");

/// Values an importer stores that are not contract fields.
pub const PATH_KEY: &str = "path";
pub const PATH_MODE_KEY: &str = "pathMode";
pub const INGEST_KEY: &str = "ingest";
pub const WATCH_KEY: &str = "watch";
pub const CONTENT_HASH_KEY: &str = "contentHash";
pub const INGEST_HASH_KEY: &str = "ingestHash";
pub const RELATIVE_PATH_KEY: &str = "relativePath";
pub const PATH_WARNING_KEY: &str = "pathWarning";

/// The document context an evaluation needs to resolve importer sources.
#[derive(Clone, Debug, Default)]
pub struct ImportContext {
    /// Folder containing the `.nodecraft` document. `None` for an unsaved
    /// document; relative paths are then resolved from the process directory.
    pub document_directory: Option<PathBuf>,
    /// Top-level ingested source records from the document.
    pub ingested_sources: Vec<IngestRecord>,
    /// Runtime content hash of every importer source, keyed by node id. This is
    /// the ChangeTracker's last imported content, not a document field: it is
    /// folded into the node's cache identity so a changed file invalidates the
    /// importer and its descendants through normal content-hash propagation.
    pub source_hashes: BTreeMap<String, String>,
}

impl ImportContext {
    #[must_use]
    pub fn new(document_directory: Option<PathBuf>, ingested_sources: Vec<IngestRecord>) -> Self {
        Self { document_directory, ingested_sources, source_hashes: BTreeMap::new() }
    }

    #[must_use]
    pub fn with_directory(document_directory: Option<PathBuf>) -> Self {
        Self { document_directory, ingested_sources: Vec::new(), source_hashes: BTreeMap::new() }
    }

    #[must_use]
    pub fn with_source_hashes(mut self, source_hashes: BTreeMap<String, String>) -> Self {
        self.source_hashes = source_hashes;
        self
    }
}

/// The current instant as a UTC ISO-8601 string, e.g.
/// `2026-11-16T09:30:00Z`. This is deliberately dependency-free: the document
/// stores the string verbatim and the inspector renders only its date part.
#[must_use]
pub fn now_utc_iso8601() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let days = (seconds / 86_400) as i64;
    let remainder = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = remainder / 3_600;
    let minute = (remainder % 3_600) / 60;
    let second = remainder % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = (z - era * 146_097) as u64;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_prime + 2) / 5 + 1) as u32;
    let month = if month_prime < 10 { month_prime + 3 } else { month_prime - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// sha256 of the original bytes, lowercase hex.
#[must_use]
pub fn content_hash(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Identity of a file's bytes for the shared hash cache. `size` and `mtime`
/// are the portable parts; inode/device catch replacements that preserve both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileFingerprint {
    size: u64,
    modified_ns: u128,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    device: u64,
}

impl FileFingerprint {
    fn read(path: &Path) -> Result<Self, String> {
        let metadata =
            std::fs::metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Self::from_metadata(&metadata))
    }

    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        let modified_ns = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt as _;
        Self {
            size: metadata.len(),
            modified_ns,
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            device: metadata.dev(),
        }
    }
}

struct CachedHash {
    fingerprint: FileFingerprint,
    hash: String,
    /// Invalidation generation this entry belongs to. An explicit Reload bumps
    /// the generation so an entry inserted by a racing reader cannot satisfy
    /// the forced read with pre-invalidation bytes.
    generation: u64,
}

/// The one fingerprint → content-hash cache. Both importer evaluation and the
/// app's source service resolve through [`hash_file`], so the engine never
/// re-reads an unchanged source and the app keeps no second copy.
#[derive(Default)]
struct HashCache {
    by_path: HashMap<PathBuf, CachedHash>,
    generations: HashMap<PathBuf, u64>,
    /// Full byte reads per path. Test instrumentation; never evicted so a test
    /// can read a stable count for its own temporary file.
    reads: HashMap<PathBuf, u64>,
}

fn hash_cache() -> &'static Mutex<HashCache> {
    static CACHE: OnceLock<Mutex<HashCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashCache::default()))
}

fn lock_hash_cache() -> std::sync::MutexGuard<'static, HashCache> {
    hash_cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Canonical identity of a source file. A file that cannot be canonicalised
/// (for example because it is missing) falls back to the path as given; the
/// following metadata/read reports the real error.
fn cache_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn generation(cache: &HashCache, key: &Path) -> u64 {
    cache.generations.get(key).copied().unwrap_or(0)
}

/// sha256 of a file's bytes, read off the caller's thread by construction and
/// reused while the file's fingerprint (canonical path + size + mtime +
/// inode/device) is unchanged. Call [`invalidate_file_hash`] first to force a
/// fresh read, as an explicit Reload does.
pub fn hash_file(path: &Path) -> Result<String, String> {
    let key = cache_key(path);
    let before = FileFingerprint::read(path)?;
    let (cached, observed_generation) = {
        let cache = lock_hash_cache();
        let generation = generation(&cache, &key);
        let cached = cache
            .by_path
            .get(&key)
            .filter(|cached| cached.fingerprint == before && cached.generation == generation)
            .map(|cached| cached.hash.clone());
        (cached, generation)
    };
    if let Some(hash) = cached {
        return Ok(hash);
    }
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let hash = content_hash(&bytes);
    store_hash(&key, before, path, &hash, observed_generation);
    Ok(hash)
}

/// Forget a file's cached hash, so the next [`hash_file`] reads it again. An
/// explicit Reload uses this instead of a separate uncached path.
pub fn invalidate_file_hash(path: &Path) {
    let key = cache_key(path);
    let mut cache = lock_hash_cache();
    let next = generation(&cache, &key).wrapping_add(1);
    cache.by_path.remove(&key);
    cache.generations.insert(key.clone(), next);
    if key != path {
        let next = generation(&cache, path).wrapping_add(1);
        cache.by_path.remove(path);
        cache.generations.insert(path.to_path_buf(), next);
    }
}

/// Number of full byte reads [`hash_file`] performed for `path` in this
/// process. Test instrumentation for the shared fingerprint cache.
#[doc(hidden)]
#[must_use]
pub fn file_hash_reads(path: &Path) -> u64 {
    let key = cache_key(path);
    lock_hash_cache().reads.get(&key).copied().unwrap_or(0)
}

/// Record one full read. The cache entry is only replaced when the file did not
/// change while it was being read and no explicit invalidation raced the read;
/// in either case the bytes' hash is still returned to this caller, and the
/// next lookup reads again.
fn store_hash(
    key: &Path,
    before: FileFingerprint,
    path: &Path,
    hash: &str,
    observed_generation: u64,
) {
    let after = FileFingerprint::read(path).ok();
    let mut cache = lock_hash_cache();
    *cache.reads.entry(key.to_path_buf()).or_insert(0) += 1;
    if after == Some(before) && generation(&cache, key) == observed_generation {
        cache.by_path.insert(
            key.to_path_buf(),
            CachedHash {
                fingerprint: before,
                hash: hash.to_owned(),
                generation: observed_generation,
            },
        );
    }
}

/// Whether a stored import path is a full path on every operating system.
///
/// Documents move between platforms, so the host's path parser alone is not
/// enough: on Windows `/models/x.step` is rooted but not absolute, on
/// macOS/Linux `\models\x.step` and `C:\models\x.step` are ordinary relative
/// names to `Path`. A full path is anchored to something outside the document
/// folder in all of those cases, so it must never be joined to that folder.
#[must_use]
pub fn is_full_path(stored: &str) -> bool {
    let path = Path::new(stored);
    if path.is_absolute() || path.has_root() {
        return true;
    }
    // `Path` only treats a leading `\` as a root on Windows. Check it
    // explicitly so a Windows path stored in a document opened on macOS is
    // still full.
    if stored.starts_with(['/', '\\']) {
        return true;
    }
    // Windows drive prefix (`C:\models\x.step`, `C:/models/x.step`). Off
    // Windows `Path` sees a drive letter as part of a relative file name.
    let bytes = stored.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// `relative` or `absolute`.
///
/// An explicit `pathMode` wins. A document saved before the field existed has no
/// `pathMode`, so derive the mode from the stored `path`: a full path is
/// `absolute`, a relative path is `relative`, and an empty path keeps the
/// contract default, `relative`.
#[must_use]
pub fn path_mode(values: &Value) -> &'static str {
    if let Some(mode) = values.get(PATH_MODE_KEY).and_then(Value::as_str) {
        return if mode == "absolute" { "absolute" } else { "relative" };
    }
    match values.get(PATH_KEY).and_then(Value::as_str) {
        Some(path) if is_full_path(path) => "absolute",
        _ => "relative",
    }
}

/// Whether the stored path is already in the mode requested by `pathMode`.
#[must_use]
pub fn path_is_absolute(stored: &str) -> bool {
    is_full_path(stored)
}

/// The primary absolute path candidate for a node, in the stored mode's form.
#[must_use]
pub fn primary_path(values: &Value, document_directory: Option<&Path>) -> Option<PathBuf> {
    let stored = values.get(PATH_KEY)?.as_str()?;
    if stored.is_empty() {
        return None;
    }
    Some(resolve_stored(stored, document_directory))
}

fn resolve_stored(stored: &str, document_directory: Option<&Path>) -> PathBuf {
    let path = Path::new(stored);
    if is_full_path(stored) {
        path.to_path_buf()
    } else if let Some(directory) = document_directory {
        directory.join(path)
    } else {
        path.to_path_buf()
    }
}

/// The alternate candidate kept beside the primary path.
///
/// Absolute mode stores `path` verbatim and a document-relative `relativePath`
/// for the same file, so moving the project folder still finds the source. A
/// relative mode path has no useful second form beyond the document folder it
/// was just resolved against, so it has no alternate.
#[must_use]
pub fn alternate_path(values: &Value, document_directory: Option<&Path>) -> Option<PathBuf> {
    let relative = values.get(RELATIVE_PATH_KEY)?.as_str()?;
    if relative.is_empty() {
        return None;
    }
    let candidate = resolve_stored(relative, document_directory);
    let primary = primary_path(values, document_directory)?;
    (candidate != primary).then_some(candidate)
}

/// Why the resolved source was chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportSourceKind {
    /// The stored path (resolved against the document folder).
    Path,
    /// The alternate form stored beside it.
    Alternate,
    /// The compressed copy embedded in the document.
    Ingested,
}

impl ImportSourceKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Path => "path",
            Self::Alternate => "alternate",
            Self::Ingested => "ingested",
        }
    }
}

/// The concrete bytes an importer will read.
#[derive(Clone, Debug)]
pub struct ResolvedImport {
    pub kind: ImportSourceKind,
    /// The file to import, when the source is on disk.
    pub path: Option<PathBuf>,
    /// Decoded bytes, when the source is the ingested copy.
    pub bytes: Option<Vec<u8>>,
    /// sha256 of the source bytes.
    pub content_hash: String,
    /// A user-facing warning, e.g. that the ingested copy was used.
    pub warning: Option<String>,
}

/// The embedded copy's identity, without its payload bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddedCopyStatus {
    pub file_name: String,
    pub byte_length: u64,
    /// UTC ISO-8601 instant, when the record has one.
    pub embedded_at: Option<String>,
}

/// What the inspector should say about one importer's source, resolved without
/// reading or hashing a file. This mirrors [`resolve_import`]'s candidate order
/// and record lookup through the same helpers, so the UI cannot disagree with
/// evaluation over which source is used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImportStatus {
    /// A file on disk (the stored path or its alternate) is available.
    Available { kind: ImportSourceKind, path: PathBuf },
    /// Neither file candidate exists, but the document has the embedded copy.
    Ingested { missing: PathBuf, copy: EmbeddedCopyStatus },
    /// Neither a file nor an embedded copy is available.
    Missing { path: PathBuf },
}

/// The ingest record an importer would fall back to, if any. No hashing or file
/// access: just the node's recorded hash and the document's record list.
#[must_use]
pub fn ingest_record_for<'a>(
    values: &Value,
    ingested_sources: &'a [IngestRecord],
    ingest_enabled: bool,
) -> Option<&'a IngestRecord> {
    if !ingest_enabled {
        return None;
    }
    let hash = values
        .get(INGEST_HASH_KEY)
        .and_then(Value::as_str)
        .filter(|hash| !hash.is_empty())
        .or_else(|| values.get(CONTENT_HASH_KEY).and_then(Value::as_str))?;
    ingested_sources.iter().find(|record| record.content_hash == hash)
}

/// Resolve the inspector status without hashing. `None` means the importer has
/// no path, so its source controls are disabled and no status line is shown.
#[must_use]
pub fn import_status(
    values: &Value,
    document_directory: Option<&Path>,
    ingested_sources: &[IngestRecord],
    ingest_enabled: bool,
) -> Option<ImportStatus> {
    let stored = values.get(PATH_KEY).and_then(Value::as_str).unwrap_or("");
    if stored.is_empty() {
        return None;
    }
    let primary = resolve_stored(stored, document_directory);
    for (kind, candidate) in [
        (ImportSourceKind::Path, Some(primary.clone())),
        (ImportSourceKind::Alternate, alternate_path(values, document_directory)),
    ] {
        if let Some(candidate) = candidate
            && candidate.is_file()
        {
            return Some(ImportStatus::Available { kind, path: candidate });
        }
    }
    Some(match ingest_record_for(values, ingested_sources, ingest_enabled) {
        Some(record) => ImportStatus::Ingested {
            missing: primary,
            copy: EmbeddedCopyStatus {
                file_name: record.file_name.clone(),
                byte_length: record.byte_length,
                embedded_at: record.embedded_at.clone(),
            },
        },
        None => ImportStatus::Missing { path: primary },
    })
}

/// Resolve the source for one importer.
///
/// Fallback order is the stored mode's path, the alternate form, then the
/// embedded record. The record is looked up by `ingestHash`; a node that has no
/// `ingestHash` cannot fall back to a record even when ingestion is enabled.
///
/// `None` means the importer has no stored path yet: it is not configured, so
/// it produces an empty Part instead of an error. This is the same neutral
/// state [`import_status`] reports for an empty path.
///
/// # Errors
/// A stored path exists but no candidate can be read.
pub fn resolve_import(
    values: &Value,
    document_directory: Option<&Path>,
    ingested_sources: &[IngestRecord],
    ingest_enabled: bool,
) -> Result<Option<ResolvedImport>, String> {
    let stored = values.get(PATH_KEY).and_then(Value::as_str).unwrap_or("");
    if stored.is_empty() {
        return Ok(None);
    }
    let primary = resolve_stored(stored, document_directory);
    let alternate = alternate_path(values, document_directory);
    let mut attempted = Vec::new();
    for (kind, candidate) in [
        (ImportSourceKind::Path, Some(primary.clone())),
        (ImportSourceKind::Alternate, alternate.clone()),
    ] {
        let Some(candidate) = candidate else { continue };
        attempted.push(candidate.display().to_string());
        if candidate.is_file() {
            let hash = hash_file(&candidate)?;
            let warning = (kind == ImportSourceKind::Alternate).then(|| {
                format!(
                    "Source path \u{201c}{}\u{201d} was missing; used the alternate path \u{201c}{}\u{201d}.",
                    primary.display(),
                    candidate.display()
                )
            });
            return Ok(Some(ResolvedImport {
                kind,
                path: Some(candidate),
                bytes: None,
                content_hash: hash,
                warning,
            }));
        }
    }
    if let Some(record) = ingest_record_for(values, ingested_sources, ingest_enabled) {
        let bytes = decode_ingest_record(record)?;
        let actual = content_hash(&bytes);
        if actual != record.content_hash {
            return Err(format!(
                "Embedded copy of {}\u{201d} is corrupt: its content hash does not match.",
                record.file_name
            ));
        }
        return Ok(Some(ResolvedImport {
            kind: ImportSourceKind::Ingested,
            path: None,
            bytes: Some(bytes),
            content_hash: actual,
            warning: Some(format!(
                "Original not found; using the embedded copy of \u{201c}{}\u{201d}.",
                record.file_name
            )),
        }));
    }
    Err(format!(
        "Source file not found: {}{}",
        attempted.join(" or "),
        if ingest_enabled { " (no embedded copy is available)" } else { "" }
    ))
}

/// Encode one source as an ingested record. One record is produced per distinct
/// content hash; callers deduplicate by [`IngestRecord::content_hash`].
pub fn ingest_record(file_name: &str, bytes: &[u8]) -> Result<IngestRecord, String> {
    use std::io::Write as _;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(bytes).map_err(|error| error.to_string())?;
    let compressed = encoder.finish().map_err(|error| error.to_string())?;
    Ok(IngestRecord {
        schema_version: INGEST_SCHEMA_VERSION,
        file_name: file_name.to_owned(),
        byte_length: bytes.len() as u64,
        content_hash: content_hash(bytes),
        compression: INGEST_COMPRESSION.to_owned(),
        embedded_at: Some(now_utc_iso8601()),
        data: base64::engine::general_purpose::STANDARD.encode(compressed),
    })
}

/// Build a record from a file on disk. The bytes read here also seed the shared
/// fingerprint cache, so the next import/evaluation does not read them again.
pub fn ingest_record_from_file(path: &Path) -> Result<IngestRecord, String> {
    let key = cache_key(path);
    let before = FileFingerprint::read(path)?;
    let observed_generation = generation(&lock_hash_cache(), &key);
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("source");
    let record = ingest_record(name, &bytes)?;
    store_hash(&key, before, path, &record.content_hash, observed_generation);
    Ok(record)
}

/// Decode an ingested record's original bytes.
pub fn decode_ingest_record(record: &IngestRecord) -> Result<Vec<u8>, String> {
    if record.schema_version != INGEST_SCHEMA_VERSION {
        return Err(format!(
            "Embedded copy of {} uses unsupported schema version {}.",
            record.file_name, record.schema_version
        ));
    }
    if record.compression != INGEST_COMPRESSION {
        return Err(format!(
            "Embedded copy of {} uses unsupported compression {:?}.",
            record.file_name, record.compression
        ));
    }
    let compressed = base64::engine::general_purpose::STANDARD
        .decode(record.data.as_bytes())
        .map_err(|error| format!("Embedded copy of {}: {error}", record.file_name))?;
    use std::io::Read as _;
    let mut decoder = flate2::read::ZlibDecoder::new(compressed.as_slice());
    let mut bytes = Vec::with_capacity(record.byte_length as usize);
    decoder
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Embedded copy of {}: {error}", record.file_name))?;
    if bytes.len() as u64 != record.byte_length {
        return Err(format!(
            "Embedded copy of {} is truncated ({} of {} bytes).",
            record.file_name,
            bytes.len(),
            record.byte_length
        ));
    }
    Ok(bytes)
}

/// Write an embedded record's original bytes to `destination`. The decoded
/// payload is verified against the record's content hash before anything is
/// written, so a saved copy always has exactly the embedded identity. Callers
/// run this off the UI thread, like every other source read/write.
///
/// # Errors
/// The record cannot be decoded, is corrupt, or the destination cannot be
/// written.
pub fn save_ingest_copy(record: &IngestRecord, destination: &Path) -> Result<(), String> {
    let bytes = decode_ingest_record(record)?;
    let actual = content_hash(&bytes);
    if actual != record.content_hash {
        return Err(format!(
            "Embedded copy of {} is corrupt: its content hash does not match.",
            record.file_name
        ));
    }
    std::fs::write(destination, &bytes)
        .map_err(|error| format!("{}: {error}", destination.display()))
}

/// The relative form of `path` under `directory`, when it is inside it.
#[must_use]
pub fn document_relative(path: &Path, directory: &Path) -> Option<PathBuf> {
    let relative = path.strip_prefix(directory).ok()?;
    (!relative.as_os_str().is_empty()).then(|| relative.to_path_buf())
}

/// A path stored in `pathMode` form, for a new import selection.
///
/// A relative mode is only possible when the document folder exists and the
/// file is under it. Otherwise the absolute path is stored (and, for absolute
/// mode, its relative form is kept as the alternate).
#[must_use]
pub fn store_path(
    path: &Path,
    mode: &str,
    document_directory: Option<&Path>,
) -> (String, Option<String>, Option<String>) {
    let absolute = path.to_path_buf();
    let relative = document_directory.and_then(|directory| document_relative(&absolute, directory));
    match mode {
        "absolute" => (
            absolute.to_string_lossy().into_owned(),
            relative.as_ref().map(|relative| relative.to_string_lossy().into_owned()),
            None,
        ),
        _ => match relative {
            Some(relative) => (relative.to_string_lossy().into_owned(), None, None),
            None => (
                absolute.to_string_lossy().into_owned(),
                None,
                Some(format!(
                    "Stored an absolute path because \u{201c}{}\u{201d} is outside the document folder, so it cannot be made relative.",
                    absolute.display()
                )),
            ),
        },
    }
}

/// The stored form after switching `pathMode` in the inspector, or `None` when
/// there is no path to rewrite. This keeps an empty importer empty.
#[must_use]
pub fn switch_path_mode(
    values: &Value,
    mode: &str,
    document_directory: Option<&Path>,
) -> Option<(String, Option<String>, Option<String>)> {
    let absolute = primary_path(values, document_directory)?;
    Some(store_path(&absolute, mode, document_directory))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_step(directory: &Path, name: &str) -> PathBuf {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives::create_box;
        let path = directory.join(name);
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep =
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).expect("box");
        nodecraft_step::write_breps_exact(path.to_str().expect("utf-8"), &[brep]).expect("write");
        path
    }

    fn temp_dir(label: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("sn-import-io-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("temp directory");
        directory
    }

    #[test]
    fn resolving_an_unchanged_source_reuses_the_hash_cache_until_it_changes() {
        let directory = temp_dir("hash-cache");
        let path = directory.join("part.step");
        std::fs::write(&path, b"first source bytes").expect("write source");
        let values = serde_json::json!({"path": "part.step", "pathMode": "relative"});

        let first = resolve_import(&values, Some(&directory), &[], false)
            .expect("first resolve")
            .expect("configured");
        assert_eq!(file_hash_reads(&path), 1, "the first resolve hashes once");

        // Same bytes and fingerprint: the shared cache answers without a read.
        let second = resolve_import(&values, Some(&directory), &[], false)
            .expect("second resolve")
            .expect("configured");
        assert_eq!(second.content_hash, first.content_hash);
        assert_eq!(file_hash_reads(&path), 1, "an unchanged resolve must not hash again");

        // A different size/mtime is a different fingerprint and must re-hash.
        std::fs::write(&path, b"second source bytes, deliberately longer").expect("replace source");
        let third = resolve_import(&values, Some(&directory), &[], false)
            .expect("third resolve")
            .expect("configured");
        assert_ne!(third.content_hash, second.content_hash);
        assert_eq!(file_hash_reads(&path), 2, "a changed fingerprint hashes again");

        // An explicit Reload forces a read through the same cache.
        invalidate_file_hash(&path);
        let fourth = resolve_import(&values, Some(&directory), &[], false)
            .expect("fourth resolve")
            .expect("configured");
        assert_eq!(fourth.content_hash, third.content_hash);
        assert_eq!(file_hash_reads(&path), 3, "invalidate forces exactly one fresh read");
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn relative_path_resolves_against_the_document_folder() {
        let directory = temp_dir("relative");
        write_step(&directory, "part.step");
        let values = serde_json::json!({"path": "part.step", "pathMode": "relative"});
        let resolved = resolve_import(&values, Some(&directory), &[], false)
            .expect("resolve")
            .expect("configured");
        assert_eq!(resolved.kind, ImportSourceKind::Path);
        assert_eq!(resolved.path, Some(directory.join("part.step")));
        assert!(resolved.warning.is_none());
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn absolute_path_is_used_as_is() {
        let directory = temp_dir("absolute");
        let path = write_step(&directory, "part.step");
        let values =
            serde_json::json!({"path": path.display().to_string(), "pathMode": "absolute"});
        let resolved = resolve_import(&values, Some(Path::new("/nonexistent")), &[], false)
            .expect("resolve")
            .expect("configured");
        assert_eq!(resolved.path, Some(path));
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn alternate_relative_form_is_tried_before_ingest() {
        let directory = temp_dir("alternate");
        write_step(&directory, "part.step");
        let values = serde_json::json!({
            "path": "/old/project/part.step",
            "pathMode": "absolute",
            "relativePath": "part.step",
        });
        let resolved = resolve_import(&values, Some(&directory), &[], false)
            .expect("resolve")
            .expect("configured");
        assert_eq!(resolved.kind, ImportSourceKind::Alternate);
        assert_eq!(resolved.path, Some(directory.join("part.step")));
        assert!(resolved.warning.expect("alternate warning").contains("alternate path"));
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn ingest_record_round_trips_and_deduplicates_by_hash() {
        let bytes = b"nodecraft ingest payload without Part 21 framing";
        let first = ingest_record("part.step", bytes).expect("record");
        let second = ingest_record("copy.step", bytes).expect("record");
        assert_eq!(first.content_hash, content_hash(bytes));
        assert_eq!(first.content_hash, second.content_hash);
        assert_eq!(decode_ingest_record(&first).expect("decode"), bytes);
        let record: IngestRecord =
            serde_json::from_value(serde_json::to_value(&first).unwrap()).unwrap();
        assert_eq!(record, first);
    }

    #[test]
    fn a_new_record_has_an_embedded_at_timestamp_and_old_records_still_load() {
        let record = ingest_record("part.step", b"bytes").expect("record");
        let embedded_at = record.embedded_at.as_deref().expect("new records are stamped");
        assert!(
            embedded_at.ends_with('Z') && embedded_at.len() == 20,
            "UTC ISO-8601 instant, got {embedded_at:?}"
        );

        let mut legacy = serde_json::to_value(&record).expect("json");
        legacy.as_object_mut().expect("object").remove("embeddedAt");
        let loaded: IngestRecord = serde_json::from_value(legacy).expect("legacy record");
        assert_eq!(loaded.embedded_at, None);
        assert_eq!(loaded.content_hash, record.content_hash);
    }

    #[test]
    fn inspector_status_uses_the_same_path_and_record_lookup_without_hashing() {
        let directory = temp_dir("status");
        let bytes = b"embedded source bytes";
        let record = ingest_record("bracket.step", bytes).expect("record");
        let values = serde_json::json!({
            "path": "missing/bracket.step",
            "pathMode": "relative",
            "ingest": true,
            "ingestHash": record.content_hash,
        });
        assert_eq!(
            import_status(&values, Some(&directory), std::slice::from_ref(&record), true),
            Some(ImportStatus::Ingested {
                missing: directory.join("missing/bracket.step"),
                copy: EmbeddedCopyStatus {
                    file_name: "bracket.step".into(),
                    byte_length: bytes.len() as u64,
                    embedded_at: record.embedded_at.clone(),
                },
            })
        );
        assert_eq!(
            import_status(&values, Some(&directory), &[], true),
            Some(ImportStatus::Missing { path: directory.join("missing/bracket.step") })
        );
        assert_eq!(
            import_status(&serde_json::json!({"path": ""}), Some(&directory), &[], true),
            None
        );

        let present = directory.join("bracket.step");
        std::fs::write(&present, bytes).expect("write");
        let present_values = serde_json::json!({ "path": "bracket.step", "pathMode": "relative" });
        assert_eq!(
            import_status(&present_values, Some(&directory), &[], true),
            Some(ImportStatus::Available { kind: ImportSourceKind::Path, path: present })
        );
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn saving_an_embedded_copy_writes_bytes_with_the_recorded_hash() {
        let directory = temp_dir("save-copy");
        let bytes = b"exact embedded bytes";
        let record = ingest_record("bracket.step", bytes).expect("record");
        let destination = directory.join("saved.step");
        save_ingest_copy(&record, &destination).expect("save copy");
        let saved = std::fs::read(&destination).expect("read saved copy");
        assert_eq!(saved, bytes);
        assert_eq!(content_hash(&saved), record.content_hash);
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn missing_file_falls_back_to_the_embedded_copy() {
        let bytes = b"source bytes";
        let record = ingest_record("part.step", bytes).expect("record");
        let values = serde_json::json!({
            "path": "part.step",
            "pathMode": "relative",
            "ingest": true,
            "ingestHash": record.content_hash,
        });
        let resolved = resolve_import(
            &values,
            Some(Path::new("/nonexistent")),
            std::slice::from_ref(&record),
            true,
        )
        .expect("embedded fallback")
        .expect("configured");
        assert_eq!(resolved.kind, ImportSourceKind::Ingested);
        assert_eq!(resolved.bytes.as_deref(), Some(bytes.as_slice()));
        assert_eq!(
            resolved.warning.as_deref(),
            Some("Original not found; using the embedded copy of \u{201c}part.step\u{201d}."),
            "the warning names the embedded copy without a hash"
        );
    }

    #[test]
    fn an_empty_path_resolves_to_no_source_instead_of_an_error() {
        let directory = temp_dir("empty-path");
        assert!(
            resolve_import(&serde_json::json!({ "path": "" }), Some(&directory), &[], false)
                .expect("empty path is not an error")
                .is_none(),
            "an unconfigured importer has no source and no error"
        );
        assert!(
            resolve_import(&serde_json::json!({}), Some(&directory), &[], false)
                .expect("missing path is not an error")
                .is_none(),
            "a missing path key is the same unconfigured state"
        );
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn path_mode_prefers_an_explicit_field_and_derives_it_from_the_stored_path() {
        assert_eq!(
            path_mode(&serde_json::json!({"path": "/abs/x.step"})),
            "absolute",
            "an old document's absolute path is full-path mode"
        );
        assert_eq!(path_mode(&serde_json::json!({"path": "parts/x.step"})), "relative");
        assert_eq!(path_mode(&serde_json::json!({"path": ""})), "relative");
        assert_eq!(path_mode(&serde_json::json!({})), "relative");
        assert_eq!(
            path_mode(&serde_json::json!({"path": "parts/x.step", "pathMode": "absolute"})),
            "absolute",
            "an explicit mode wins over the stored path's form"
        );
        assert_eq!(
            path_mode(&serde_json::json!({"path": "/abs/x.step", "pathMode": "relative"})),
            "relative",
            "an explicit mode wins over the stored path's form"
        );
    }

    #[test]
    fn full_paths_are_full_on_every_os_and_relative_paths_stay_relative() {
        let directory = Path::new("/documents/project");
        for stored in
            ["/models/x.step", "\\models\\x.step", "C:\\models\\x.step", "C:/models/x.step"]
        {
            assert!(is_full_path(stored), "{stored:?} must be a full path");
            assert_eq!(
                path_mode(&serde_json::json!({ "path": stored })),
                "absolute",
                "{stored:?} must derive absolute mode"
            );
            assert_eq!(
                primary_path(&serde_json::json!({ "path": stored }), Some(directory)),
                Some(PathBuf::from(stored)),
                "{stored:?} must be used as-is, not joined to the document folder"
            );
        }
        for stored in ["models/x.step", "./x.step", "..\\x.step"] {
            assert!(!is_full_path(stored), "{stored:?} must stay relative");
            assert_eq!(
                path_mode(&serde_json::json!({ "path": stored })),
                "relative",
                "{stored:?} must derive relative mode"
            );
            assert_eq!(
                primary_path(&serde_json::json!({ "path": stored }), Some(directory)),
                Some(directory.join(stored)),
                "{stored:?} must resolve against the document folder"
            );
        }
    }

    #[test]
    fn switching_mode_does_not_invent_a_path_when_none_is_stored() {
        let values = serde_json::json!({"path": "", "pathMode": "relative"});
        assert!(switch_path_mode(&values, "absolute", Some(Path::new("/project"))).is_none());
    }

    #[test]
    fn store_path_rewrites_relative_and_keeps_absolute_with_alternate() {
        let directory = Path::new("/project");
        let (relative, alternate, warning) =
            store_path(Path::new("/project/model.step"), "relative", Some(directory));
        assert_eq!(relative, "model.step");
        assert!(alternate.is_none());
        assert!(warning.is_none());

        let (absolute, alternate, warning) =
            store_path(Path::new("/project/model.step"), "absolute", Some(directory));
        assert_eq!(absolute, "/project/model.step");
        assert_eq!(alternate.as_deref(), Some("model.step"));
        assert!(warning.is_none());
    }
}
