use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "text-linguistics-server",
    version,
    about = "Thin HTTP API adapter for text-linguistics"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,

    /// Use Candle CUDA execution.
    #[arg(long)]
    cuda: bool,

    /// CUDA device index for Candle execution.
    #[arg(long, requires = "cuda")]
    cuda_device_index: Option<usize>,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    configure_candle_device(args.cuda, args.cuda_device_index)?;
    eprintln!("text-linguistics-server listening on http://{}", args.addr);
    text_linguistics_server::serve(&args.addr)
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
