use std::{
    path::PathBuf,
    str::FromStr,
    sync::atomic::AtomicBool,
    time::{Duration, SystemTime},
};

use anyhow::Context;
use clap::Parser;
use llmup_download::ollama::OllamaConfig;
use llmup_store::ollama::{Model, OllamaStore, Registry, Variant};

use llmup_llama_cpp as llama;
use reqwest::ClientBuilder;

mod args;
mod human;
mod progressbar;

use args::Cli;
use progressbar::ProgressBar;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        args::Commands::List { filter } => cmd_list(filter).await,
        args::Commands::Pull { name } => cmd_pull(name).await,
        args::Commands::Remove { name } => cmd_remove(name).await,
        args::Commands::Verify { blobs } => cmd_verify(blobs).await,
        args::Commands::Run { name } => cmd_run(name).await,
        args::Commands::Bench { name } => cmd_bench(name).await,
    }
}

async fn cmd_bench(name: String) -> anyhow::Result<()> {
    let OllamaRun {
        model_path,
        template: _,
    } = ollama_model_prepare_run(&name)?;

    llama_init_logging();

    let model_params = llama::ModelParams::default();
    let model = llama::Model::load(&model_path, &model_params).unwrap();
    let vocab = model.vocab();

    let context_params = llama::ContextParams::default();
    let mut context = model.new_context(&context_params).unwrap();

    Ok(())
}

async fn cmd_run(name: String) -> anyhow::Result<()> {
    let OllamaRun {
        model_path,
        template: _,
    } = ollama_model_prepare_run(&name)?;
    tracing_subscriber::fmt::init();

    llama_init_logging();

    let model_params = llama::ModelParams::default();
    let model = llama::Model::load(&model_path, &model_params).unwrap();
    let vocab = model.vocab();

    let context_params = llama::ContextParams::default();
    let mut context = model.new_context(&context_params).unwrap();

    let mut rl = rustyline::DefaultEditor::new()?;
    let readline = rl.readline(">> ");
    match readline {
        Err(e) => {
            anyhow::bail!("error {:?}", e);
        }
        Ok(line) => {
            let mut tokens = vocab.tokenize(line.as_bytes(), true);
            context.append_tokens(&mut tokens);

            let sampler = llama::Sampler::new();

            let quit_requested = std::sync::Arc::new(AtomicBool::new(false));
            let quit_requested_2 = quit_requested.clone();
            ctrlc::set_handler(move || {
                quit_requested_2.store(true, std::sync::atomic::Ordering::Relaxed);
            })
            .expect("Error setting Ctrl-C handler");

            pub struct Output {
                utf8_errors: usize,
            }

            impl Output {
                pub fn new() -> Self {
                    Self { utf8_errors: 0 }
                }

                pub fn append(&mut self, bytes: &[u8]) {
                    match std::str::from_utf8(bytes) {
                        Ok(valid) => {
                            print!("{}", valid);
                            use std::io::Write;
                            std::io::stdout().flush().unwrap();
                        }
                        Err(_) => {
                            self.utf8_errors += 1;
                        }
                    }
                }
            }

            let mut output = Output::new();
            let mut tokens = Vec::new();
            while !quit_requested.load(std::sync::atomic::Ordering::Relaxed) {
                let n = context.next_token(&sampler, &vocab);
                match n {
                    None => break,
                    Some(t) => {
                        tokens.push(t);
                        context.append_tokens(&[t]);
                        let bytes = vocab.as_bytes(t);
                        output.append(&bytes);
                    }
                }
            }

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

pub struct OllamaRun {
    model_path: PathBuf,
    template: gtmpl::Template,
}

fn ollama_model_prepare_run(name: &str) -> anyhow::Result<OllamaRun> {
    let (model, variant) = parse_name(&name)?;

    let store = llmup_store::ollama::OllamaStore::default();
    let registry = Registry::from_str(&OllamaConfig::default().host()).unwrap();

    let manifest = store.get_manifest(&registry, &model, &variant)?;

    let Some(model_layer) = manifest.find_media_type(llmup_store::ollama::MEDIA_TYPE_IMAGE_MODEL)
    else {
        anyhow::bail!("no model found in {}", name);
    };

    let Some(template_layer) =
        manifest.find_media_type(llmup_store::ollama::MEDIA_TYPE_IMAGE_TEMPLATE)
    else {
        anyhow::bail!("no template found in {}", name);
    };
    let template_data = store.blob_read_string(&template_layer.digest)?;

    println!("== template layer");
    println!("{}", template_data);

    let mut template = gtmpl::Template::default();
    template
        .parse(&template_data)
        .with_context(|| "parsing ollama template string")?;

    let path = store.blob_path(&model_layer.digest);
    Ok(OllamaRun {
        model_path: path,
        template,
    })
}

fn llama_init_logging() {
    llama::llama_logging(Box::new(|level, key, t| {
        if ![llama::LogKey::ModelLoader].iter().any(|k| *k == key) {
            return;
        }
        println!("{:5?} | {:?} | {}", level, key, t)
    }));
}
