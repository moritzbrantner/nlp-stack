use text_index::{IndexBuildOptions, IndexDocument, IndexQuery, MemoryTextIndex};

fn index() -> MemoryTextIndex {
    MemoryTextIndex::new_memory()
        .expect("memory index must construct")
        .with_options(IndexBuildOptions {
            chunk_tokens: 4,
            chunk_overlap_tokens: 1,
            ..IndexBuildOptions::default()
        })
}

#[test]
fn repeated_upsert_preserves_index_state_and_search_results() {
    let mut index = index();
    let document = IndexDocument::new(
        "doc-1",
        "alpha beta gamma delta epsilon zeta eta theta",
    );

    let first = index
        .upsert_documents(std::slice::from_ref(&document))
        .expect("initial upsert must succeed");
    let first_state = index.inspect().expect("index must be inspectable");
    let first_results = index
        .search(&IndexQuery::lexical("alpha epsilon", 10))
        .expect("search must succeed");

    let second = index
        .upsert_documents(std::slice::from_ref(&document))
        .expect("repeated upsert must succeed");
    let second_state = index.inspect().expect("index must remain inspectable");
    let second_results = index
        .search(&IndexQuery::lexical("alpha epsilon", 10))
        .expect("search must remain stable");

    assert_eq!(first.documents_replaced, 0);
    assert_eq!(second.documents_replaced, 1);
    assert_eq!(first_state, second_state);
    assert_eq!(first_results, second_results);
}

#[test]
fn replacing_a_document_removes_stale_searchable_state() {
    let mut index = index();
    index
        .upsert_documents(&[IndexDocument::new(
            "doc-1",
            "obsolete vocabulary survives only in the old revision",
        )])
        .expect("initial upsert must succeed");

    let replacement = index
        .upsert_documents(&[IndexDocument::new(
            "doc-1",
            "current terminology belongs to the replacement revision",
        )])
        .expect("replacement upsert must succeed");

    assert_eq!(replacement.documents_replaced, 1);
    assert!(index
        .search(&IndexQuery::lexical("obsolete", 10))
        .expect("old-term search must succeed")
        .is_empty());

    let current_results = index
        .search(&IndexQuery::lexical("current terminology", 10))
        .expect("replacement-term search must succeed");
    assert_eq!(current_results.len(), 1);
    assert_eq!(current_results[0].document_id, "doc-1");

    let state = index.inspect().expect("index must be inspectable");
    assert_eq!(state.document_count, 1);
}

#[test]
fn repeated_remove_is_idempotent() {
    let mut index = index();
    index
        .upsert_documents(&[IndexDocument::new("doc-1", "alpha beta gamma")])
        .expect("upsert must succeed");

    let first = index
        .remove_documents(&["doc-1".to_string()])
        .expect("first removal must succeed");
    let second = index
        .remove_documents(&["doc-1".to_string()])
        .expect("repeated removal must succeed");

    assert_eq!(first.documents_removed, 1);
    assert_eq!(second.documents_removed, 0);
    let state = index.inspect().expect("index must be inspectable");
    assert_eq!(state.document_count, 0);
    assert_eq!(state.chunk_count, 0);
    assert_eq!(state.vector_count, 0);
}

#[test]
fn insertion_order_does_not_change_deterministic_search_results() {
    let documents = [
        IndexDocument::new("doc-a", "alpha beta shared phrase"),
        IndexDocument::new("doc-b", "alpha gamma shared phrase"),
        IndexDocument::new("doc-c", "delta epsilon unrelated text"),
    ];

    let mut forward = index();
    forward
        .upsert_documents(&documents)
        .expect("forward upsert must succeed");

    let mut reverse = index();
    reverse
        .upsert_documents(&documents.iter().cloned().rev().collect::<Vec<_>>())
        .expect("reverse upsert must succeed");

    let query = IndexQuery::lexical("alpha shared phrase", 10);
    let forward_results = forward.search(&query).expect("forward search must succeed");
    let reverse_results = reverse.search(&query).expect("reverse search must succeed");

    assert_eq!(forward_results, reverse_results);
}
