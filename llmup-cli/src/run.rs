use std::io::Write;
use std::{path::Path, sync::atomic::AtomicBool};

use skelm_llama_cpp as llama;

pub struct Output {
    handle: Option<std::fs::File>,
    utf8_errors: usize,
}

impl Output {
    pub fn new() -> Self {
        Self {
            utf8_errors: 0,
            handle: None,
        }
    }

    pub fn new_file<P: AsRef<Path>>(p: P) -> std::io::Result<Self> {
        let file = std::fs::File::create(p)?;
        Ok(Self {
            utf8_errors: 0,
            handle: Some(file),
        })
    }

    pub fn append(&mut self, bytes: &[u8]) {
        if let Some(file) = &mut self.handle {
            file.write_all(bytes).unwrap();
        } else {
            match std::str::from_utf8(bytes) {
                Ok(valid) => {
                    print!("{}", valid);
                    std::io::stdout().flush().unwrap();
                }
                Err(_) => {
                    self.utf8_errors += 1;
                }
            }
        }
    }
}

pub fn llama_init_logging(debug: bool) {
    llama::llama_logging(Box::new(move |level, key, t| {
        if level != llama::LogLevel::Error
            && (!debug && ![llama::LogKey::ModelLoader].contains(&key))
        {
            return;
        }
        eprintln!(
            "{:<5} | {:<22} | {}",
            format!("{}", level),
            format!("{:?}", key),
            t
        )
    }));
}

pub fn llama_sampler() -> impl llama::Sampler {
    let mut sampler = llama::SamplerChain::new();
    sampler.add(Box::new(llama::SamplerMinP::new(0.05, 1)));
    sampler.add(Box::new(llama::SamplerTemperature::new(0.8)));
    sampler.add(Box::new(llama::SamplerDistance::new(0xFFFF_FFFF)));
    //sampler.add(Box::new(llama::SamplerGreedy));

    sampler
}

pub fn llama_run(
    context: &mut llmup_run::Context,
    line: &str,
    output: &Option<String>,
) -> anyhow::Result<()> {
    let model = context.model().clone();
    let vocab = model.vocab;

    context.append_bytes(line.as_bytes());

    let context = &mut context.1;

    let mut sampler = llama_sampler();

    let quit_requested = std::sync::Arc::new(AtomicBool::new(false));
    let quit_requested_inner = quit_requested.clone();
    ctrlc::set_handler(move || {
        quit_requested_inner.store(true, std::sync::atomic::Ordering::Relaxed);
    })
    .expect("Error setting Ctrl-C handler");

    let mut output = output
        .as_ref()
        .map(|o| Output::new_file(o))
        .unwrap_or(Ok(Output::new()))?;
    let mut tokens = Vec::new();
    while !quit_requested.load(std::sync::atomic::Ordering::Relaxed) {
        let n = context.next_token(&mut sampler, &vocab);
        match n {
            None => break,
            Some(t) => {
                tokens.push(t);
                context.append_tokens(&[t])?;
                let attr = vocab.token_attr(t);
                if attr.is_control() {
                    continue;
                }
                let bytes = vocab.as_bytes(t);
                output.append(&bytes);
            }
        }
    }

    Ok(())
}
