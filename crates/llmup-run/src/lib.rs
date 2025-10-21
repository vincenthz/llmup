pub mod ollama;
mod template;

use std::hash::Hash;
use std::path::PathBuf;
use std::sync::Mutex;
use std::{collections::HashMap, sync::Arc};

use thiserror::Error;

use llmup_llama_cpp as llama;

pub use template::chat_template;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ModelDescr {
    Ollama(ollama::ModelDescr),
    Path(PathBuf),
}

#[derive(Clone)]
pub struct Model {
    pub model: llama::Model,
    pub vocab: llama::Vocab,
    pub config: Arc<ModelConfig>,
}

#[derive(Clone)]
pub enum ModelConfig {
    Ollama(ollama::ModelConfig),
    Implicit,
}

#[derive(Debug, Error)]
pub enum ModelLoadError {
    #[error("Llama model failed to load model")]
    LlamaModelFailedLoading(#[from] llama::ModelLoadError),
    #[error("Ollama config get error {0}")]
    OllamaConfigGetError(#[from] ollama::ModelConfigGetError),
}

impl Model {
    pub fn load(descr: &ModelDescr) -> Result<Self, ModelLoadError> {
        let (config, model_path) = match descr {
            ModelDescr::Ollama(model_descr) => {
                let config = ollama::model_config_get(model_descr)?;
                let model_path = config.model_path.clone();
                (ModelConfig::Ollama(config), model_path)
            }
            ModelDescr::Path(path_buf) => (ModelConfig::Implicit, path_buf.clone()),
        };
        let params = llama::ModelParams::default();
        llama::Model::load(model_path, &params)
            .map_err(ModelLoadError::LlamaModelFailedLoading)
            .map(|m| Model {
                vocab: m.vocab(),
                model: m,
                config: Arc::new(config),
            })
    }

    pub fn new_context(&self) -> Context {
        let params = llama::ContextParams::default();
        Context(self.clone(), self.model.new_context(&params).unwrap())
    }

    pub fn new_context_embeddings(&self) -> Context {
        if self.model.has_encoder() && self.model.has_decoder() {
            panic!("cannot generate embeddings in models with encoder-decoder")
        }
        let params = llama::ContextParams {
            embeddings: true,
            ..llama::ContextParams::default()
        };
        Context(self.clone(), self.model.new_context(&params).unwrap())
    }

    pub fn model_template_render(&self, parameters: &ModelParameters) -> String {
        match self.config.as_ref() {
            ModelConfig::Ollama(_model_config) => {
                implicit_model_template(self, &parameters.system, &parameters.prompt)
            }
            ModelConfig::Implicit => {
                implicit_model_template(self, &parameters.system, &parameters.prompt)
            }
        }
    }
}

fn implicit_model_template(model: &Model, system: &str, prompt: &str) -> String {
    if let Some(template) = model.model.chat_template() {
        //println!("template:\n{}", template);
        match chat_template(&template, &system, &prompt) {
            Err(e) => {
                eprintln!("rendering chat template failed: {}", e);
                eprintln!("chat template:");
                for (i, l) in template.lines().enumerate() {
                    eprintln!("{:03} {}", i + 1, l)
                }
                prompt.to_string()
            }
            Ok(render) => {
                //println!("rendered:\n{}", render);
                render
            }
        }
    } else {
        prompt.to_string()
    }
}

pub struct ModelParameters {
    pub system: String,
    pub prompt: String,
}

pub struct Context(pub Model, pub llama::Context);

impl Context {
    pub fn model(&self) -> &Model {
        &self.0
    }

    pub fn append_bytes(&mut self, bytes: &[u8]) {
        let mut tokens = self.0.vocab.tokenize(bytes, true);
        self.1.append_tokens(&mut tokens).unwrap();
    }
}

pub struct Models<K> {
    models: Arc<Mutex<HashMap<K, Model>>>,
}

impl<K> Clone for Models<K> {
    fn clone(&self) -> Self {
        Self {
            models: self.models.clone(),
        }
    }
}

impl<K: Hash + Eq> Models<K> {
    pub fn new() -> Self {
        Self {
            models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn set(&self, key: K, model: Model) {
        let mut models = self.models.lock().unwrap();
        models.insert(key, model.clone());
    }

    pub fn get(&self, key: K) -> Option<Model> {
        let models = self.models.lock().unwrap();
        models.get(&key).map(|m| m.clone())
    }
}
