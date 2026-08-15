use clap::{Parser, Subcommand};
use runtime_core::cli::read_json_input;

#[derive(Debug, Parser)]
#[command(
    name = "text-analysis-cli",
    version,
    about = "Thin CLI adapter for text-analysis"
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
        #[arg(long, default_value = "analysis.describe")]
        operation: String,
        #[arg(long)]
        json: Option<String>,
        #[arg(long)]
        file: Option<String>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Info { json: false }) {
        Command::Info { json } => print_payload(
            json,
            "text-analysis",
            &text_analysis_cli::package_metadata_json(),
        ),
        Command::Schema { json } => print_payload(
            json,
            "text-analysis command schema",
            &text_analysis_cli::command_schema_json(),
        ),
        Command::Operations { json } => {
            let payload = serde_json::to_string(&text_analysis_cli::package_surface().operations)?;
            print_payload(json, "text-analysis operations", &payload);
        }
        Command::Run {
            operation,
            json,
            file,
        } => {
            let input = read_json_input(json, file)?;
            let response = text_analysis_cli::run_operation(&operation, input)
                .map_err(std::io::Error::other)?;
            println!("{}", serde_json::to_string(&response)?);
        }
    }
    Ok(())
}

fn print_payload(json: bool, title: &str, payload: &str) {
    if json {
        println!("{payload}");
    } else {
        println!("{title}");
        println!("{payload}");
    }
}
