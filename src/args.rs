use clap::{Parser, Subcommand};

/// Example CLI with subcommands: list, pull, verify
#[derive(Parser, Debug)]
#[command(name = "llmup")]
#[command(about = "CLI tool to install and managed use LLM models", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// List available model
    List {
        /// Optional filter for the list
        #[arg(short, long)]
        filter: Option<String>,
    },
    /// Pull an model by name
    Pull {
        /// The name of the model to pull
        name: String,
    },
    /// Remove a model by name
    Remove {
        /// The name of the model to remove
        name: String,
    },
    /// Verify install
    Verify {
        /// Flag to verify blobs (might take a long time)
        #[arg(short, long, default_value_t = false)]
        blobs: bool,
    },
    /// Information about a model
    Info {
        /// The name of the model to get info
        name: String,
    },
    /// Run a model
    Run {
        /// The name of the model to run
        name: String,
        /// Debug information
        #[arg(long, default_value_t = false)]
        debug: bool,
        /// Use model path directly (no ollama)
        #[arg(long, default_value_t = false)]
        model_path: bool,
    },
    /// Bench model generation
    Bench {
        /// The name of the model to run
        name: String,
        #[arg(short, long)]
        max_tokens: Option<u64>,
    },
    /// Embedding generation
    Embed {
        /// The name of the model to run
        name: String,
    },
}
