use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "text-core-server",
    version,
    about = "Thin HTTP API adapter for text-core"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!("text-core-server listening on http://{}", args.addr);
    text_core_server::serve(&args.addr)
}
