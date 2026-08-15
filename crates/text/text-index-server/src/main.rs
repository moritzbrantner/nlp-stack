use std::env;

fn main() -> std::io::Result<()> {
    let addr = env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:0".to_string());
    text_index_server::serve(&addr)
}
