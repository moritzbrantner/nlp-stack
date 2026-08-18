use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "nlp-package-registry-server",
    version,
    about = "Thin HTTP adapter for the NLP package registry"
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!(
        "nlp-package-registry-server listening on http://{}",
        args.addr
    );
    nlp_package_registry_server::serve(&args.addr)
}
