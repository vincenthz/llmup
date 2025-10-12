use std::sync::atomic::AtomicBool;

use llmup_llama_cpp as llama;

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

pub fn llama_init_logging(debug: bool) {
    llama::llama_logging(Box::new(move |level, key, t| {
        if level != llama::LogLevel::Error
            && (!debug && ![llama::LogKey::ModelLoader].iter().any(|k| *k == key))
        {
            return;
        }
        println!(
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

pub fn llama_run(context: &mut llama::Context, line: &str) -> anyhow::Result<()> {
    let model = context.model();
    let vocab = model.vocab();

    let mut tokens = vocab.tokenize(line.as_bytes(), true);
    context.append_tokens(&mut tokens)?;

    let mut sampler = llama_sampler();

    let quit_requested = std::sync::Arc::new(AtomicBool::new(false));
    let quit_requested_inner = quit_requested.clone();
    ctrlc::set_handler(move || {
        quit_requested_inner.store(true, std::sync::atomic::Ordering::Relaxed);
    })
    .expect("Error setting Ctrl-C handler");

    let mut output = Output::new();
    let mut tokens = Vec::new();
    while !quit_requested.load(std::sync::atomic::Ordering::Relaxed) {
        let n = context.next_token(&mut sampler, &vocab);
        match n {
            None => break,
            Some(t) => {
                tokens.push(t);
                context.append_tokens(&[t])?;
                let bytes = vocab.as_bytes(t);
                output.append(&bytes);
            }
        }
    }
    Ok(())
}
