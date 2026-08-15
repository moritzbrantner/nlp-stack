use text_linguistics::{TextNlpConfig, TextNlpPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = TextNlpPipeline::new(TextNlpConfig::rich());
    let analysis = pipeline.analyze_text("Alice presented the roadmap in Berlin.")?;

    println!("profile={:?}", analysis.profile);
    println!("tokens={}", analysis.graph.tokens.len());
    println!("entities={}", analysis.entities.len());
    println!("provenance={:?}", analysis.provenance);

    Ok(())
}
