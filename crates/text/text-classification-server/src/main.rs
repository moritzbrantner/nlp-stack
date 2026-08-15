use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "text-classification-server",
    version,
    about = "Thin HTTP API adapter for text-classification"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!(
        "text-classification-server listening on http://{}",
        args.addr
    );
    text_classification_server::serve(&args.addr)
}
