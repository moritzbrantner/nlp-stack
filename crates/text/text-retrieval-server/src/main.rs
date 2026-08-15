use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "text-retrieval-server",
    version,
    about = "Thin HTTP API adapter for text-retrieval"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!("text-retrieval-server listening on http://{}", args.addr);
    text_retrieval_server::serve(&args.addr)
}
