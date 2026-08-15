#![cfg(feature = "external-tests")]

use std::path::{Path, PathBuf};

use model_runtime::ModelBundle;
use runtime_onnx::{
    f32_output_by_name_or_index, f32_output_by_preferred_name_or_index, OnnxNamedTensor,
    OnnxTensor, OnnxTensorValue,
};

fn bundle(name: &str) -> Option<ModelBundle> {
    let root = PathBuf::from(".model-runtime").join(name).join("main");
    if !root.join("manifest.json").is_file() {
        eprintln!(
            "skipping external ONNX smoke test; missing {}",
            root.display()
        );
        return None;
    }
    Some(ModelBundle::load(Path::new(&root)).expect("load model bundle"))
}

#[test]
#[ignore = "requires local RoBERTa SQuAD2 ONNX bundle in .model-runtime"]
fn roberta_onnx_bundle_and_runtime_helpers_regressions() {
    let Some(bundle) = bundle("roberta-base-squad2-onnx") else {
        return;
    };
    assert_eq!(bundle.manifest.task.as_protocol_str(), "question_answering");
    assert!(bundle.manifest.files.values().any(|file| {
        Path::new(&file.remote_path)
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("onnx")
    }));

    let outputs = vec![
        OnnxNamedTensor {
            name: "start_logits".to_string(),
            tensor: OnnxTensorValue::F32(OnnxTensor::new(vec![1, 3], vec![0.1, 0.2, 0.3]).unwrap()),
        },
        OnnxNamedTensor {
            name: "end_logits".to_string(),
            tensor: OnnxTensorValue::F32(OnnxTensor::new(vec![1, 3], vec![0.3, 0.2, 0.1]).unwrap()),
        },
    ];
    assert_eq!(
        f32_output_by_name_or_index(&outputs, "end_logits", 0)
            .unwrap()
            .values,
        vec![0.3, 0.2, 0.1]
    );
    assert_eq!(
        f32_output_by_preferred_name_or_index(&outputs, &["end_logits"], 0)
            .unwrap()
            .values,
        vec![0.3, 0.2, 0.1]
    );
    assert!(f32_output_by_name_or_index(&outputs, "missing", 1).is_ok());
}
