use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "text-analysis-server",
    version,
    about = "HTTP adapter for text-analysis"
)]
struct Cli {
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    text_analysis_server::serve(&cli.addr)
}
