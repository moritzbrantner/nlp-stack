use super::*;
use crate::syntax_backend::SyntaxSentenceRequest;

fn request<'a>(text: &'a str, sentences: &'a [SyntaxSentenceRequest]) -> SyntaxRequest<'a> {
    SyntaxRequest { text, sentences }
}

fn sentence(
    text: &str,
    index: usize,
    start: usize,
    end: usize,
    language: &str,
) -> SyntaxSentenceRequest {
    SyntaxSentenceRequest {
        sentence_index: index,
        span: TextSpan::from_byte_range(text, start, end).unwrap(),
        language: language.to_string(),
    }
}

const MODELS: &str = r#"{"repository":"stanfordnlp/stanza-es","revision":"abc","packages":{"pos":"combined_charlm","depparse":"combined_charlm"}}"#;

fn response(tokens: &str) -> String {
    format!(
        r#"{{"protocol":1,"stanza_version":"1.15.0","sentences":[{{"id":1,"language":"es","models":{MODELS},"tokens":[{tokens}]}}]}}"#
    )
}

#[test]
fn converts_code_point_offsets_to_utf8_byte_spans() {
    // "Hi. " precedes the sentence; "niño" has a two-byte "ñ".
    let text = "Hi. El niño vino del río.";
    let sentences = [sentence(text, 1, 4, text.len(), "es")];
    let body = response(concat!(
        r#"{"start_char":0,"end_char":2,"text":"El","words":[{"id":1,"text":"El","lemma":"el","upos":"DET","xpos":null,"feats":"Definite=Def","head":2,"deprel":"det"}]},"#,
        r#"{"start_char":3,"end_char":7,"text":"niño","words":[{"id":2,"text":"niño","lemma":"niño","upos":"NOUN","xpos":null,"feats":null,"head":3,"deprel":"nsubj"}]},"#,
        r#"{"start_char":8,"end_char":12,"text":"vino","words":[{"id":3,"text":"vino","lemma":"venir","upos":"VERB","xpos":null,"feats":null,"head":0,"deprel":"root"}]},"#,
        r#"{"start_char":13,"end_char":16,"text":"del","words":[{"id":4,"text":"de","lemma":"de","upos":"ADP","xpos":null,"feats":null,"head":6,"deprel":"case"},{"id":5,"text":"el","lemma":"el","upos":"DET","xpos":null,"feats":null,"head":6,"deprel":"det"}]},"#,
        r#"{"start_char":17,"end_char":20,"text":"río","words":[{"id":6,"text":"río","lemma":"río","upos":"NOUN","xpos":null,"feats":null,"head":3,"deprel":"obl:arg"}]},"#,
        r#"{"start_char":20,"end_char":21,"text":".","words":[{"id":7,"text":".","lemma":".","upos":"PUNCT","xpos":null,"feats":null,"head":3,"deprel":"punct"}]}"#,
    ));
    let result = decode_stanza_response(&request(text, &sentences), &body).unwrap();
    let sentence = &result[0];
    assert_eq!(sentence.provenance.backend_version, "1.15.0");
    assert_eq!(
        sentence.provenance.producer("es"),
        "stanza-1.15.0:es/combined_charlm"
    );
    let spans = sentence
        .tokens
        .iter()
        .map(|token| &text[token.span.byte_start..token.span.byte_end])
        .collect::<Vec<_>>();
    assert_eq!(spans, ["El", "niño", "vino", "del", "río", "."]);
    let del = &sentence.tokens[3];
    assert_eq!(del.word_count, 2);
    assert_eq!(sentence.words[4].form, "el");
    assert_eq!(sentence.words[3].head, Some(5));
    assert_eq!(sentence.words[2].head, None);
    assert_eq!(sentence.words[5].subtype.as_deref(), Some("arg"));
    assert!(sentence.words[0].feats.has("Definite", "Def"));
}

#[test]
fn refuses_offsets_that_do_not_reproduce_the_token() {
    let text = "El niño.";
    let sentences = [sentence(text, 1, 0, text.len(), "es")];
    let body = response(
        r#"{"start_char":0,"end_char":3,"text":"El","words":[{"id":1,"text":"El","lemma":null,"upos":"DET","xpos":null,"feats":null,"head":0,"deprel":"root"}]}"#,
    );
    let error = decode_stanza_response(&request(text, &sentences), &body).unwrap_err();
    assert!(
        matches!(error, SyntaxBackendError::Protocol { .. }),
        "{error}"
    );
}

#[test]
fn refuses_a_response_that_leaves_source_words_uncovered() {
    let hola = r#"{"start_char":0,"end_char":4,"text":"Hola","words":[{"id":1,"text":"Hola","lemma":"hola","upos":"INTJ","xpos":null,"feats":null,"head":0,"deprel":"root"}]}"#;
    let text = "Hola mundo";
    let sentences = [sentence(text, 1, 0, text.len(), "es")];
    let error = decode_stanza_response(&request(text, &sentences), &response(hola)).unwrap_err();
    assert!(error.to_string().contains("not covered"), "{error}");
    let error = decode_stanza_response(&request(text, &sentences), &response("")).unwrap_err();
    assert!(error.to_string().contains("not covered"), "{error}");
    let text = "Hola ";
    let sentences = [sentence(text, 1, 0, text.len(), "es")];
    assert!(decode_stanza_response(&request(text, &sentences), &response(hola)).is_ok());
}

#[test]
fn refuses_other_versions_and_mismatched_sentences() {
    let text = "El niño.";
    let sentences = [sentence(text, 1, 0, text.len(), "es")];
    let other_version = response("").replace("1.15.0", "1.14.0");
    let error = decode_stanza_response(&request(text, &sentences), &other_version).unwrap_err();
    assert!(error.to_string().contains("pins 1.15.0"), "{error}");
    let other_sentence = [sentence(text, 0, 0, text.len(), "es")];
    let error = decode_stanza_response(&request(text, &other_sentence), &response("")).unwrap_err();
    assert!(matches!(error, SyntaxBackendError::Protocol { .. }));
}

#[test]
fn maps_sidecar_errors_to_typed_errors() {
    let missing = decode_error(
        503,
        r#"{"error":{"kind":"model_missing","message":"run setup.sh","language":"de"}}"#,
    );
    assert_eq!(
        missing,
        SyntaxBackendError::ModelMissing {
            backend: "stanza".to_string(),
            language: "de".to_string(),
            message: "run setup.sh".to_string(),
        }
    );
    let unsupported = decode_error(
        422,
        r#"{"error":{"kind":"unsupported_language","message":"no","language":"fr"}}"#,
    );
    assert!(
        matches!(unsupported, SyntaxBackendError::UnsupportedLanguage { language, .. } if language == "fr")
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn unreachable_sidecar_is_a_typed_error() {
    // Bind and drop a listener so the port is very likely closed.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let sidecar = StanzaSidecar::new("127.0.0.1", port).timeout(std::time::Duration::from_secs(2));
    let text = "Hello there.";
    let sentences = [sentence(text, 0, 0, text.len(), "en")];
    let error = sidecar.analyze(&request(text, &sentences)).unwrap_err();
    assert!(
        matches!(error, SyntaxBackendError::Unavailable { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("sidecars/stanza/serve"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn speaks_http_to_a_sidecar() {
    use std::io::{Read, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let body = response(
        r#"{"start_char":0,"end_char":4,"text":"Hola","words":[{"id":1,"text":"Hola","lemma":"hola","upos":"INTJ","xpos":null,"feats":null,"head":0,"deprel":"root"}]}"#,
    );
    let served = body.clone();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = stream.read(&mut buffer).unwrap();
            request.extend_from_slice(&buffer[..read]);
        }
        let head = String::from_utf8_lossy(&request).to_string();
        let length = head
            .lines()
            .find_map(|line| line.strip_prefix("Content-Length: "))
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap();
        let start = request.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        while request.len() < start + length {
            let read = stream.read(&mut buffer).unwrap();
            request.extend_from_slice(&buffer[..read]);
        }
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{served}",
            served.len()
        )
        .unwrap();
        String::from_utf8(request[start..].to_vec()).unwrap()
    });
    let text = "Hola";
    let sentences = [sentence(text, 1, 0, 4, "es")];
    let result = StanzaSidecar::from_url(&format!("http://127.0.0.1:{port}"))
        .unwrap()
        .analyze(&request(text, &sentences))
        .unwrap();
    let sent = server.join().unwrap();
    assert_eq!(
        sent,
        r#"{"sentences":[{"id":1,"language":"es","text":"Hola"}]}"#
    );
    assert_eq!(result[0].words[0].upos, Some(PosTag::Intj));
}
