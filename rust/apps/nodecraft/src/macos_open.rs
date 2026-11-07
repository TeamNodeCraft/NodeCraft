//! Finder / LaunchServices open-document handoff.
//!
//! Winit owns the application delegate, so replacing it is not safe. A launch
//! observer adds `application:openURLs:` and `application:openFiles:` to
//! winit's delegate class just before AppKit finishes launching; Finder-opened
//! documents are queued and drained by the UI.

#![cfg(target_os = "macos")]

use std::ffi::{CStr, c_char};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use objc2::ffi::class_addMethod;
use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
use objc2::{define_class, msg_send, DefinedClass, MainThreadOnly};
use objc2_app_kit::{NSApplication, NSApplicationWillFinishLaunchingNotification};
use objc2_foundation::{
    MainThreadMarker, NSArray, NSNotification, NSNotificationCenter, NSObject, NSObjectProtocol,
    NSString, NSURL,
};

static PENDING: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
static CONTEXT: OnceLock<egui::Context> = OnceLock::new();
static INSTALLED: OnceLock<()> = OnceLock::new();
static OBSERVER: OnceLock<()> = OnceLock::new();

define_class!(
    // SAFETY: NSObject has no subclassing requirements and this class does not
    // implement Drop.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "NodeCraftLaunchObserver"]
    #[ivars = ()]
    struct LaunchObserver;

    impl LaunchObserver {}

    unsafe impl NSObjectProtocol for LaunchObserver {
        #[unsafe(method(applicationWillFinishLaunching:))]
        fn will_finish_launching(&self, _notification: &NSNotification) {
            install_delegate_methods();
        }
    }
);

impl LaunchObserver {
    fn new(mtm: MainThreadMarker) -> objc2::rc::Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(());
        unsafe { msg_send![super(this), init] }
    }
}

fn push_path(path: PathBuf) {
    if let Ok(mut pending) = PENDING.lock() {
        pending.push(path);
    }
    if let Some(ctx) = CONTEXT.get() {
        ctx.request_repaint();
    }
}

unsafe extern "C-unwind" fn open_urls(
    _this: *mut AnyObject,
    _selector: Sel,
    _application: *mut NSApplication,
    urls: *mut NSArray<NSURL>,
) {
    if urls.is_null() {
        return;
    }
    let urls = unsafe { &*urls };
    for index in 0..urls.count() {
        let url = urls.objectAtIndex(index);
        if !url.isFileURL() {
            continue;
        }
        let path = unsafe {
            CStr::from_ptr(url.fileSystemRepresentation().as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        push_path(PathBuf::from(path));
    }
}

unsafe extern "C-unwind" fn open_files(
    _this: *mut AnyObject,
    _selector: Sel,
    _application: *mut NSApplication,
    files: *mut NSArray<NSString>,
) {
    if files.is_null() {
        return;
    }
    let files = unsafe { &*files };
    for index in 0..files.count() {
        let name = files.objectAtIndex(index);
        let ptr = name.UTF8String();
        if !ptr.is_null() {
            let path = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
            push_path(PathBuf::from(path));
        }
    }
}

fn add_method(class: &AnyClass, name: &CStr, function: Imp) -> bool {
    let types = c"v@:@@";
    unsafe {
        class_addMethod(
            class as *const _ as *mut _,
            Sel::register(name),
            function,
            types.as_ptr() as *const c_char,
        )
        .as_bool()
    }
}

fn install_delegate_methods() {
    if INSTALLED.get().is_some() {
        return;
    }
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let application = NSApplication::sharedApplication(mtm);
    let Some(delegate) = application.delegate() else {
        return;
    };
    let class = unsafe {
        &*objc2::ffi::object_getClass(&*delegate as *const _ as *const AnyObject)
    };
    let urls = add_method(
        class,
        c"application:openURLs:",
        unsafe { std::mem::transmute::<
            unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut NSApplication, *mut NSArray<NSURL>),
            Imp,
        >(open_urls) },
    );
    let files = add_method(
        class,
        c"application:openFiles:",
        unsafe {
            std::mem::transmute::<
                unsafe extern "C-unwind" fn(
                    *mut AnyObject,
                    Sel,
                    *mut NSApplication,
                    *mut NSArray<NSString>,
                ),
                Imp,
            >(open_files)
        },
    );
    if urls || files {
        let _ = INSTALLED.set(());
    }
}

/// Register the launch observer before the event loop starts.
pub(crate) fn install_observer() {
    if OBSERVER.get().is_some() {
        return;
    }
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let observer = LaunchObserver::new(mtm);
    let center = NSNotificationCenter::defaultCenter();
    unsafe {
        center.addObserver_selector_name_object(
            &observer,
            objc2::sel!(applicationWillFinishLaunching:),
            Some(&NSApplicationWillFinishLaunchingNotification),
            None,
        );
    }
    // The notification center does not retain the observer.
    std::mem::forget(observer);
    let _ = OBSERVER.set(());
}

/// Set the context used to wake the UI after a document arrives.
pub(crate) fn install(ctx: egui::Context) {
    let _ = CONTEXT.set(ctx);
    install_delegate_methods();
}

/// Take documents delivered by macOS since the previous frame.
pub(crate) fn take_pending() -> Vec<PathBuf> {
    PENDING
        .lock()
        .map(|mut pending| std::mem::take(&mut *pending))
        .unwrap_or_default()
}
