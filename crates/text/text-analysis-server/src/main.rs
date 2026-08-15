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

    #[arg(long)]
    cuda: bool,

    #[arg(long, requires = "cuda")]
    cuda_device_index: Option<usize>,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    configure_candle_device(cli.cuda, cli.cuda_device_index)?;
    text_analysis_server::serve(&cli.addr)
}

fn configure_candle_device(cuda: bool, cuda_device_index: Option<usize>) -> std::io::Result<()> {
    let preference = if cuda {
        text_model_runtime::CandleDevicePreference::Cuda {
            device_index: cuda_device_index.unwrap_or(0),
        }
    } else {
        text_model_runtime::CandleDevicePreference::Cpu
    };
    text_model_runtime::set_candle_device_preference(preference);
    text_model_runtime::validate_candle_device_preference(preference)
        .map_err(|error| std::io::Error::other(error.to_string()))
}
