//! Reusable worker-side utilities.
//!
//! The node-graph auto-layout implementations live here so the desktop app can
//! call exactly the same algorithms as the Swift/WebView worker bridge.

pub mod layout;
