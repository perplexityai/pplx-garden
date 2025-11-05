use clap::{Parser, ValueEnum};
use is_terminal::IsTerminal;
use tracing::{Dispatch, dispatcher};
use tracing_log::AsLog;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
pub struct LoggingOpts {
    #[clap(long, env = "PPLX_LOG_FORMAT", default_value = "json")]
    pub log_format: LogFormat,

    #[clap(long, env = "PPLX_LOG_COLOR", default_value = "auto")]
    pub log_color: LogColor,

    /// Additional debug level flags in the RUST_LOG format to configure loggin on a
    /// per-target basis. If both this and RUST_LOG set a log level for a target,
    /// the RUST_LOG setting will take priority.
    pub log_directives: Option<String>,
}

pub fn init(opts: &LoggingOpts) -> Result<(), anyhow::Error> {
    let color = match opts.log_color {
        // tracing_subscriber::fmt uses stdout:
        // https://docs.rs/tracing-subscriber/latest/tracing_subscriber/fmt/index.html
        LogColor::Auto => std::io::stdout().is_terminal(),

        LogColor::Always => true,
        LogColor::Never => false,
    };

    // Get log levels from whatever directives were passed, if any,
    // then override with what's in the RUST_LOG env var
    let mut log_filter_builder = EnvFilter::builder();
    if let Some(directives) = &opts.log_directives {
        log_filter_builder =
            log_filter_builder.with_default_directive(directives.parse()?);
    }

    let log_filter = log_filter_builder.from_env_lossy();
    let builder = tracing_subscriber::fmt().with_env_filter(log_filter);

    #[cfg(test)]
    let builder = builder.with_test_writer();

    #[cfg(not(test))]
    let builder = builder.with_writer(std::io::stderr);

    let dispatch: Dispatch = match opts.log_format {
        LogFormat::Text => {
            let subscriber = builder.with_ansi(color).finish();
            subscriber.into()
        }
        LogFormat::Json => {
            let subscriber = builder.json().finish();
            subscriber.into()
        }
    };
    dispatcher::set_global_default(dispatch)?;

    tracing_log::LogTracer::builder()
        // Note that we must call this *after* setting the global default
        // subscriber, so that we get its max level hint.
        .with_max_level(tracing_core::LevelFilter::current().as_log())
        .init()?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum)]
pub enum LogColor {
    Auto,
    Always,
    Never,
}

impl std::str::FromStr for LogColor {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(LogColor::Auto),
            "always" => Ok(LogColor::Always),
            "never" => Ok(LogColor::Never),
            s => Err(format!(
                "{s} is not a valid option, expected `auto`, `always` or `never`"
            )),
        }
    }
}
