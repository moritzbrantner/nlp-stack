use text_core::{build_annotation_graph, TextProcessingOptions};

fn main() {
    let graph = build_annotation_graph(
        "Alice tagged #東京 from Berlin. Rust keeps café, emoji 👍, and offsets intact.",
        &TextProcessingOptions::default(),
    );

    println!("{}", serde_json::to_string_pretty(&graph).unwrap());
}
