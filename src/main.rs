use std::{
    path::PathBuf,
    str::FromStr,
    sync::atomic::AtomicBool,
    time::{Duration, SystemTime},
};

use anyhow::Context;
use clap::Parser;
use llmup_download::ollama::OllamaConfig;
use llmup_run::ollama as ollama_run;
use llmup_store::ollama::{Model, OllamaStore, Registry, Variant};

use llmup_llama_cpp as llama;
use reqwest::ClientBuilder;

mod args;
mod human;
mod progressbar;
mod run;

use args::Cli;
use progressbar::ProgressBar;

use crate::human::bench_duration_units;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        args::Commands::List { filter } => cmd_list(filter).await,
        args::Commands::Pull { name } => cmd_pull(name).await,
        args::Commands::Remove { name } => cmd_remove(name).await,
        args::Commands::Verify { blobs } => cmd_verify(blobs).await,
        args::Commands::Run { name, debug } => cmd_run(name, debug).await,
        args::Commands::Bench { name, max_tokens } => cmd_bench(name, max_tokens).await,
    }
}

async fn cmd_bench(name: String, max_tokens: Option<u64>) -> anyhow::Result<()> {
    let (model, variant) = parse_name(&name)?;
    let ollama_run::OllamaRun {
        model_path,
        template: _,
        params: _,
    } = ollama_run::model_prepare_run(&model, &variant)?;

    let max_tokens = max_tokens.unwrap_or(u64::MAX);

    run::llama_init_logging(false);

    let model_params = llama::ModelParams::default();
    let model = llama::Model::load(&model_path, &model_params)?;
    let vocab = model.vocab();

    let context_params = llama::ContextParams::default();
    let mut context = model.new_context(&context_params)?;

    const BENCHMARK_CONTEXT: &str = "this is a context for doing tokens benchmarks";
    let tokens = vocab.tokenize(BENCHMARK_CONTEXT.as_bytes(), true);
    context.append_tokens(&tokens);

    let sampler = llama::Sampler::new();

    let mut token_generated = 0u64;
    let start = SystemTime::now();
    let bar = indicatif::ProgressBar::new_spinner();

    bar.set_style(
        indicatif::ProgressStyle::with_template(
            "{pos:>7} tokens generated in {elapsed_precise} ({per_sec})",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    loop {
        match context.next_token(&sampler, &vocab) {
            None => break,
            Some(t) => {
                context.append_tokens(&[t]);
                token_generated += 1;
                bar.set_position(token_generated);
                if token_generated >= max_tokens {
                    break;
                }
            }
        }
    }

    let end = SystemTime::now();
    bar.finish();

    let dur = end.duration_since(start).unwrap_or(Duration::ZERO);

    let dur_per_token = dur
        .checked_div(token_generated as u32)
        .unwrap_or(Duration::ZERO);

    let tps = token_generated as f64 / dur.as_secs_f64();
    let time_token = bench_duration_units(dur_per_token);

    println!("model              : {}", name);
    println!("tokens generated   : {}", token_generated);
    println!("elapsed            : {}", bench_duration_units(dur));
    println!("tokens per seconds : {:.4}", tps);
    println!("time per token     : {}", time_token);

    Ok(())
}

async fn cmd_run(name: String, debug: bool) -> anyhow::Result<()> {
    let (model, variant) = parse_name(&name)?;
    let ollama_run::OllamaRun {
        model_path,
        template: _,
        params: _,
    } = ollama_run::model_prepare_run(&model, &variant)?;

    tracing_subscriber::fmt::init();
    run::llama_init_logging(debug);

    let model_params = llama::ModelParams::default();
    let model = llama::Model::load(&model_path, &model_params).unwrap();

    let context_params = llama::ContextParams::default();
    let mut context = model.new_context(&context_params).unwrap();

    let mut rl = rustyline::DefaultEditor::new()?;
    let readline = rl.readline(">> ");
    match readline {
        Err(e) => {
            anyhow::bail!("error {:?}", e);
        }
        Ok(line) => {
            run::llama_run(&mut context, &line)?;
            Ok(())
        }
    }
}

async fn cmd_list(filter: Option<String>) -> anyhow::Result<()> {
    let store = OllamaStore::default();
    let regs = store.list_registries()?;

    let mut model_lines = Vec::new();

    let now = SystemTime::now();
    for reg in regs {
        println!("{:?}", reg);
        let models = store.list_models(&reg)?;
        for model in models {
            let variants = store.list_model_variants(&reg, &model)?;
            for variant in variants {
                let manifest_path =
                    store.manifest_registry_model_variant_path(&reg, &model, &variant);
                let fs = tokio::fs::File::open(manifest_path).await?;
                let metadata = fs.metadata().await?;
                let modified = metadata.modified()?;

                let manifest = store.get_manifest(&reg, &model, &variant)?;
                let size = manifest.size();
                let name = format!("{}:{}", model.as_str(), variant.as_str());
                let acceptable = if let Some(filter) = &filter {
                    name.starts_with(filter)
                } else {
                    true
                };
                if acceptable {
                    let dur = now.duration_since(modified).unwrap_or(Duration::ZERO);
                    model_lines.push((name, size, dur))
                }
            }
        }
    }

    model_lines.sort_by(|(_, _, m1), (_, _, m2)| m1.cmp(m2));

    println!("{:40} {:15} {:15}", "NAME", "SIZE", "MODIFIED");
    for (model_name, size, modified) in model_lines {
        println!(
            "{:40} {:15} {:15}",
            model_name,
            human::size_units(size),
            format!("{} ago", human::duration_units(modified)),
        )
    }
    Ok(())
}

async fn cmd_pull(name: String) -> anyhow::Result<()> {
    let (model, variant) = parse_name(&name)?;

    let store = OllamaStore::default();
    let config = OllamaConfig::default();
    let client = ClientBuilder::new()
        .user_agent("llmup/0.1")
        //.redirect(Policy::none())
        .build()
        .unwrap();

    let registry = Registry::from_str(&config.host()).unwrap();
    llmup_download::ollama::download_model::<ProgressBar>(
        &client, &config, &store, &registry, &model, &variant,
    )
    .await;

    Ok(())
}

async fn cmd_remove(name: String) -> anyhow::Result<()> {
    let (model, variant) = parse_name(&name)?;

    let store = OllamaStore::default();
    let config = OllamaConfig::default();
    let registry = Registry::from_str(&config.host()).unwrap();

    store.remove_manifest(&registry, &model, &variant)?;

    Ok(())
}

async fn cmd_verify(blobs: bool) -> anyhow::Result<()> {
    let store = OllamaStore::default();
    let regs = store.list_registries()?;

    for reg in regs {
        let models = store.list_models(&reg)?;
        for model in models {
            let variants = store.list_model_variants(&reg, &model)?;
            for variant in variants {
                let manifest = store.get_manifest(&reg, &model, &variant)?;
                let digests = manifest.all_digests();

                let mut failed = Vec::new();

                for blob in digests.iter() {
                    if !store.blob_exists(blob) {
                        failed.push(format!("missing {}", blob));
                        continue;
                    }
                    if blobs {
                        let verified = store.blob_self_verify(blob)?;
                        if !verified {
                            failed.push(format!("invalid blob {}", blob))
                        }
                    }
                }

                if failed.is_empty() {
                    println!("{}:{}: OK", model.as_str(), variant.as_str())
                } else {
                    println!("{}:{}: FAILED", model.as_str(), variant.as_str());
                    for f in failed {
                        println!(" * {}", f)
                    }
                }
            }
        }
    }
    Ok(())
}

fn parse_name(name: &str) -> anyhow::Result<(Model, Variant)> {
    let Some((model_name, variant_name)) = name.split_once(':') else {
        anyhow::bail!("'{}' should have <model>:<tag> format", name);
    };

    let model = Model::from_str(model_name).map_err(|_| anyhow::anyhow!("invalid model name"))?;
    let variant =
        Variant::from_str(variant_name).map_err(|_| anyhow::anyhow!("invalid variant name"))?;
    Ok((model, variant))
}
