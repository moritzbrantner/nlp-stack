use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "text-transcripts-server",
    version,
    about = "Thin HTTP API adapter for text-transcripts"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!("text-transcripts-server listening on http://{}", args.addr);
    text_transcripts_server::serve(&args.addr)
}
