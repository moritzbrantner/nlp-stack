use text_generation::MarkovChain;

#[test]
fn trains_predicts_generates_and_scores_from_public_api() {
    let mut chain = MarkovChain::new(2).unwrap();
    chain.train_documents([
        "rust cargo builds crates",
        "rust cargo runs tests",
        "video frames detect scenes",
    ]);

    assert_eq!(chain.order(), 2);
    assert!(chain.contexts() >= 3);
    assert_eq!(
        chain
            .starts()
            .get(&vec!["rust".to_string(), "cargo".to_string()]),
        Some(&2)
    );

    let predictions = chain.predict_next(["rust", "cargo"], 2).unwrap();
    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0].probability, 0.5);
    assert_eq!(
        predictions
            .iter()
            .map(|prediction| prediction.token.as_str())
            .collect::<Vec<_>>(),
        vec!["builds", "runs"]
    );

    let generated = chain.generate(&["rust"], 4).unwrap();
    assert_eq!(generated.tokens, vec!["rust", "cargo", "builds", "crates"]);
    assert_eq!(generated.text, "rust cargo builds crates");

    let perplexity = chain.perplexity("rust cargo builds crates").unwrap();
    assert!(perplexity.is_finite());
}

#[test]
fn returns_empty_predictions_for_unknown_contexts_without_mutating_model() {
    let mut chain = MarkovChain::new(1).unwrap();
    chain.train_text("alpha beta gamma");
    let contexts = chain.contexts();

    let predictions = chain.predict_next(["missing"], 3).unwrap();

    assert!(predictions.is_empty());
    assert_eq!(chain.contexts(), contexts);
}
