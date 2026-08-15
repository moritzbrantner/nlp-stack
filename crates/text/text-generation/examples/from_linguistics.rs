use text_generation::{MarkovChain, MarkovInputMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _mode = MarkovInputMode::Normalized;
    let mut chain = MarkovChain::new(2)?;
    chain.train_text("scene transitions follow strong visual changes");

    for prediction in chain.predict_next(["scene", "transition"], 3)? {
        println!("{} {:.3}", prediction.token, prediction.probability);
    }

    Ok(())
}
