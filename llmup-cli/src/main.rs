use std::{
    path::PathBuf,
    str::FromStr,
    time::{Duration, SystemTime},
};

use anyhow::Context;
use clap::Parser;
use llmup_download::ollama::OllamaConfig;
use llmup_run::{ModelDescr, ModelParameters, ollama as ollama_run};
use llmup_store::ollama::OllamaStore;

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
        args::Commands::Set { name, key, value } => cmd_set(name, key, value).await,
        args::Commands::Verify { blobs } => cmd_verify(blobs).await,
        args::Commands::Run {
            name,
            debug,
            model_path,
            no_prompt,
            system,
            input,
            output,
        } => cmd_run(name, debug, model_path, no_prompt, system, input, output).await,
        args::Commands::Info { name } => cmd_info(name).await,
        args::Commands::Bench { name, max_tokens } => cmd_bench(name, max_tokens).await,
        args::Commands::Embed { name } => cmd_embed(name).await,
    }
}

async fn cmd_set(name: String, key: String, value: String) -> anyhow::Result<()> {
    let model_descr = parse_ollama_descr(&name)?;
    let store = OllamaStore::default();
    let mut manifest = store.get_manifest(&model_descr)?;

    match key.as_str() {
        "model" => {
            let file_path = PathBuf::from(&value);
            if file_path.exists() {
                let src_file = std::fs::File::open(file_path)?;
                let blob = store.add_blob_from_file(src_file)?;
                println!("added blob {}", blob);
                if let Some(layer) =
                    manifest.find_media_type_mut(llmup_store::ollama::MEDIA_TYPE_IMAGE_MODEL)
                {
                    layer.digest = blob;
                } else {
                    anyhow::bail!("manifest doesn't have a model")
                }

                store.add_manifest(
                    &model_descr.registry,
                    &model_descr.model,
                    &model_descr.variant,
                    &manifest,
                )?;
            } else {
                anyhow::bail!("value should be a model file")
            }
        }
        _ => {
            anyhow::bail!("unknown key to set {}", key)
        }
    }

    Ok(())
}

async fn cmd_info(name: String) -> anyhow::Result<()> {
    let model_descr = parse_model_descr(&name)?;
    let model = llmup_run::Model::load(&model_descr)?;

    if let Some(chat_template) = model.model.chat_template() {
        println!("chat-template:\n{}", chat_template)
    }

    Ok(())
}

async fn cmd_embed(name: String) -> anyhow::Result<()> {
    let model_descr = parse_model_descr(&name)?;
    let model = llmup_run::Model::load(&model_descr)?;

    run::llama_init_logging(false);

    let mut context = model.new_context_embeddings().1;

    let pooling_type = context.pooling_type();

    println!("pooling type: {:?}", pooling_type);

    let tokens = model.vocab.tokenize(b"test", true);

    let e = context.embeddings(&tokens)?;

    println!("embedding {:?}", e);
    Ok(())
}

async fn cmd_bench(name: String, max_tokens: Option<u64>) -> anyhow::Result<()> {
    let model_descr = parse_model_descr(&name)?;

    let max_tokens = max_tokens.unwrap_or(u64::MAX);

    run::llama_init_logging(false);

    let model = llmup_run::Model::load(&model_descr)?;

    let mut context = model.new_context().1;
    let vocab = model.vocab;

    const BENCHMARK_CONTEXT: &str = "this is a context for doing tokens benchmarks";
    let tokens = vocab.tokenize(BENCHMARK_CONTEXT.as_bytes(), true);
    context.append_tokens(&tokens)?;

    let mut sampler = run::llama_sampler();

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
        match context.next_token(&mut sampler, &vocab) {
            None => break,
            Some(t) => {
                context.append_tokens(&[t])?;
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

async fn cmd_run(
    name: String,
    debug: bool,
    model_path: bool,
    no_prompt: bool,
    system: Option<String>,
    input: Option<String>,
    output: Option<String>,
) -> anyhow::Result<()> {
    const DEFAULT_SYSTEM_PROMPT: &str = "you are a chatbot answering question";

    let model_descr = if model_path {
        ModelDescr::Path(PathBuf::from(name))
    } else {
        ModelDescr::Ollama(llmup_run::ollama::ModelDescr::from_str(&name).unwrap())
    };

    run::llama_init_logging(debug);
    tracing_subscriber::fmt::init();

    let model = llmup_run::Model::load(&model_descr)?;

    let input_data = if let Some(input_file) = input {
        std::fs::read_to_string(&input_file)
            .with_context(|| format!("reading input file {}", input_file))?
    } else {
        String::new()
    };

    let prompt = if no_prompt {
        input_data
    } else {
        let mut rl = rustyline::DefaultEditor::new()?;
        match rl.readline(">> ") {
            Err(e) => {
                anyhow::bail!("error {:?}", e);
            }
            Ok(line) => format!("{}\n{}", input_data, line),
        }
    };

    let system = system.unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

    let parameters = ModelParameters { system, prompt };
    let template = model.model_template_render(&parameters);

    let mut context = model.new_context();
    run::llama_run(&mut context, &template, &output)?;
    Ok(())
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

                let descr = ollama_run::ModelDescr {
                    registry: reg.clone(),
                    model: model.clone(),
                    variant: variant.clone(),
                };
                let manifest = store.get_manifest(&descr)?;
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
    let model_descr = parse_ollama_descr(&name)?;

    let store = OllamaStore::default();
    let config = OllamaConfig::default();
    let client = ClientBuilder::new()
        .user_agent("llmup/0.1")
        //.redirect(Policy::none())
        .build()
        .unwrap();

    let download_results = llmup_download::ollama::download_model::<ProgressBar>(
        &client,
        &config,
        &store,
        &model_descr.registry,
        &model_descr.model,
        &model_descr.variant,
    )
    .await?;

    for (download_name, download_result) in download_results {
        let r = match download_result {
            llmup_download::ollama::DownloadResult::Skipped(blob) => {
                format!("{} already downloaded", blob)
            }
            llmup_download::ollama::DownloadResult::Success(blob) => format!("{} downloaded", blob),
        };
        println!("{}: {}", download_name, r)
    }

    Ok(())
}

async fn cmd_remove(name: String) -> anyhow::Result<()> {
    let llmup_run::ModelDescr::Ollama(model_descr) = parse_model_descr(&name)? else {
        anyhow::bail!("ollama invalid name")
    };
    let store = OllamaStore::default();
    store.remove_manifest(
        &model_descr.registry,
        &model_descr.model,
        &model_descr.variant,
    )?;

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
                let model_descr = ollama_run::ModelDescr {
                    registry: reg.clone(),
                    model: model.clone(),
                    variant: variant.clone(),
                };
                let manifest = store.get_manifest(&model_descr)?;
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

fn parse_ollama_descr(name: &str) -> anyhow::Result<ollama_run::ModelDescr> {
    ollama_run::ModelDescr::from_str(name).map_err(|_| {
        anyhow::anyhow!("Invalid Ollama model description: expecting <registry>/<model>:<variant> or <model>:<variant>")
    })
}

fn parse_model_descr(name: &str) -> anyhow::Result<llmup_run::ModelDescr> {
    ollama_run::ModelDescr::from_str(name).map_err(|_| {
        anyhow::anyhow!("Invalid Ollama model description: expecting <registry>/<model>:<variant> or <model>:<variant>")
    }).map(llmup_run::ModelDescr::Ollama)
}
