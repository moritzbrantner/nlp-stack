use text_generation::{synthesize_from_terms, TermPrompt, TextSynthesisOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let generated = synthesize_from_terms(
        "doc-1",
        &[
            TermPrompt::new("alice", 2.0),
            TermPrompt::new("roadmap", 1.5),
            TermPrompt::new("berlin", 1.0),
        ],
        TextSynthesisOptions {
            sentence_count: 1,
            ..TextSynthesisOptions::default()
        },
    )?;

    println!("{}", generated.value.text);
    Ok(())
}
