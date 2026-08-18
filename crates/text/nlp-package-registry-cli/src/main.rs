use clap::{Parser, Subcommand};
use runtime_core::cli::read_json_input;

#[derive(Debug, Parser)]
#[command(
    name = "nlp-package-registry-cli",
    version,
    about = "Thin CLI adapter for the NLP package registry"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Info {
        #[arg(long)]
        json: bool,
    },
    Schema {
        #[arg(long)]
        json: bool,
    },
    Operations {
        #[arg(long)]
        json: bool,
    },
    Run {
        #[arg(long, default_value = "registry.describe")]
        operation: String,
        #[arg(long)]
        json: Option<String>,
        #[arg(long)]
        file: Option<String>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse()
        .command
        .unwrap_or(Command::Info { json: false })
    {
        Command::Info { json } => print_payload(
            json,
            "NLP package registry",
            &nlp_package_registry_cli::package_metadata_json(),
        ),
        Command::Schema { json } => print_payload(
            json,
            "NLP package registry command schema",
            &nlp_package_registry_cli::command_schema_json(),
        ),
        Command::Operations { json } => print_payload(
            json,
            "NLP package registry operations",
            &serde_json::to_string(&nlp_package_registry_cli::package_surface().operations)?,
        ),
        Command::Run {
            operation,
            json,
            file,
        } => {
            let response =
                nlp_package_registry_cli::run_operation(&operation, read_json_input(json, file)?)
                    .map_err(std::io::Error::other)?;
            println!("{}", serde_json::to_string(&response)?);
        }
    }
    Ok(())
}

fn print_payload(json: bool, title: &str, payload: &str) {
    if !json {
        println!("{title}");
    }
    println!("{payload}");
}
