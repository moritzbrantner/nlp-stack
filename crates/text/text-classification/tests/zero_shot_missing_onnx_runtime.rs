#![cfg(feature = "onnx")]

mod support;

#[test]
fn zero_shot_without_onnx_runtime_reports_unavailable_instead_of_hanging() {
    let root = support::fake_zero_shot_bundle_root("missing-ort");
    std::env::set_var("ORT_DYLIB_PATH", root.join("libonnxruntime-missing.so"));

    let message = support::zero_shot_error(&root);

    assert!(message.contains("ONNX Runtime unavailable"), "{message}");
    assert!(message.contains("libonnxruntime-missing.so"), "{message}");
    assert!(message.contains("ORT_DYLIB_PATH"), "{message}");
    std::fs::remove_dir_all(&root).ok();
}
