#![cfg(feature = "onnx")]

mod support;

/// ONNX Runtime 1.23 (Ubuntu's `libonnxruntime1.23`) is older than ort rc.15 accepts;
/// `NLP_STACK_OLD_ORT_DYLIB` points at another pre-1.24 build.
#[test]
fn zero_shot_with_old_onnx_runtime_reports_too_old_instead_of_hanging() {
    let old = std::env::var_os("NLP_STACK_OLD_ORT_DYLIB")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23".into());
    if !old.is_file() {
        eprintln!("skipping: no pre-1.24 ONNX Runtime at {}", old.display());
        return;
    }
    let root = support::fake_zero_shot_bundle_root("old-ort");
    std::env::set_var("ORT_DYLIB_PATH", &old);

    let message = support::zero_shot_error(&root);

    assert!(
        message.contains("ONNX Runtime unavailable: version 1.23"),
        "{message}"
    );
    assert!(message.contains("< required 1."), "{message}");
    assert!(message.contains("ORT_DYLIB_PATH"), "{message}");
    std::fs::remove_dir_all(&root).ok();
}
