use std::{path::PathBuf, str::FromStr, sync::atomic::AtomicBool};

use anyhow::Context;
use llmup_llama_cpp as llama;

use llmup_download::ollama::OllamaConfig;
use llmup_store::ollama;

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

pub fn llama_init_logging() {
    llama::llama_logging(Box::new(|level, key, t| {
        /*
        if ![llama::LogKey::ModelLoader].iter().any(|k| *k == key) {
            return;
        }
        */
        println!("{:5?} | {:?} | {}", level, key, t)
    }));
}

pub fn llama_run(context: &mut llama::Context, line: &str) -> anyhow::Result<()> {
    let model = context.model();
    let vocab = model.vocab();

    let mut tokens = vocab.tokenize(line.as_bytes(), true);
    context.append_tokens(&mut tokens);

    let sampler = llama::Sampler::new();

    let quit_requested = std::sync::Arc::new(AtomicBool::new(false));
    let quit_requested_inner = quit_requested.clone();
    ctrlc::set_handler(move || {
        quit_requested_inner.store(true, std::sync::atomic::Ordering::Relaxed);
    })
    .expect("Error setting Ctrl-C handler");

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

pub struct OllamaRun {
    pub model_path: PathBuf,
    pub template: gtmpl::Template,
}

pub fn ollama_model_prepare_run(
    model: &ollama::Model,
    variant: &ollama::Variant,
) -> anyhow::Result<OllamaRun> {
    let store = llmup_store::ollama::OllamaStore::default();
    let registry = ollama::Registry::from_str(&OllamaConfig::default().host()).unwrap();

    let manifest = store.get_manifest(&registry, &model, &variant)?;

    let Some(model_layer) = manifest.find_media_type(llmup_store::ollama::MEDIA_TYPE_IMAGE_MODEL)
    else {
        anyhow::bail!("no model found for {:?}:{:?}", model, variant);
    };

    let Some(template_layer) =
        manifest.find_media_type(llmup_store::ollama::MEDIA_TYPE_IMAGE_TEMPLATE)
    else {
        anyhow::bail!("no template found for {:?}:{:?}", model, variant);
    };
    let template_data = store.blob_read_string(&template_layer.digest)?;

    println!("== template layer");
    println!("{}", template_data);

    fn gtmpl_fn_slice(values: &[gtmpl::Value]) -> Result<gtmpl::Value, gtmpl::FuncError> {
        println!("slice: {:?}", values);
        todo!()
    }

    fn gtmpl_fn_current_date(values: &[gtmpl::Value]) -> Result<gtmpl::Value, gtmpl::FuncError> {
        println!("currentDate: {:?}", values);
        todo!()
    }

    let mut template = gtmpl::Template::default();
    template.add_func("slice", gtmpl_fn_slice);
    template.add_func("currentDate", gtmpl_fn_current_date);
    template
        .parse(&template_data)
        .with_context(|| "parsing ollama template string")?;

    let path = store.blob_path(&model_layer.digest);
    Ok(OllamaRun {
        model_path: path,
        template,
    })
}
