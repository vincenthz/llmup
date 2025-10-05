use anyhow::Context;
use chrono::Local;
use llmup_download::ollama::OllamaConfig;
use llmup_store::ollama;
use std::{path::PathBuf, str::FromStr};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    System,
    Assistant,
}

impl FromStr for Role {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "user" => Ok(Self::User),
            "system" => Ok(Self::System),
            "assistant" => Ok(Self::Assistant),
            _ => Err(()),
        }
    }
}

pub struct Message(String);
pub struct Tool();

pub struct OllamaRunParams {
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub prompt: String,
    pub suffix: String,
    pub think: bool,
    pub think_level: String,
    pub is_think_set: bool,
}

pub struct OllamaRun {
    pub model_path: PathBuf,
    pub template: gtmpl::Template,
    pub params: serde_json::Value,
}

pub fn model_prepare_run(
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

    let Some(params_layer) = manifest.find_media_type(llmup_store::ollama::MEDIA_TYPE_IMAGE_PARAMS)
    else {
        anyhow::bail!("no params found for {:?}:{:?}", model, variant);
    };
    let params_data = store.blob_read_string(&params_layer.digest)?;
    let params_json = serde_json::Value::from_str(&params_data)?;

    println!("== template layer");
    println!("{}", template_data);

    let mut template = gtmpl::Template::default();
    template.add_func("slice", gtmpl_fn_slice);
    template.add_func("currentDate", gtmpl_fn_current_date);
    match template.parse(&template_data) {
        Err(e) => {
            println!("error parsing template {}", e);
        }
        Ok(()) => (),
    }

    let path = store.blob_path(&model_layer.digest);
    Ok(OllamaRun {
        model_path: path,
        template,
        params: params_json,
    })
}

fn gtmpl_fn_slice(args: &[gtmpl::Value]) -> Result<gtmpl::Value, gtmpl::FuncError> {
    if args.is_empty() {
        return Err(gtmpl::FuncError::ExactlyXArgs("slice".to_string(), 1));
    }
    match &args[0] {
        gtmpl::Value::String(s) => {
            let mut indices = Vec::new();
            for arg in &args[1..] {
                if let gtmpl::Value::Number(n) = arg {
                    if let Some(i) = n.as_i64() {
                        indices.push(i as usize);
                    } else {
                        return Err(gtmpl::FuncError::Generic(
                            "slice bounds out of range".to_string(),
                        ));
                    }
                } else {
                    return Err(gtmpl::FuncError::Generic(
                        "slice bounds must be numbers".to_string(),
                    ));
                }
            }
            let result = match indices.len() {
                0 => s.clone(),
                1 => s[indices[0]..].to_string(),
                2 => s[indices[0]..indices[1]].to_string(),
                3 => {
                    if indices[0] <= indices[1] && indices[1] <= s.len() {
                        s[indices[0]..indices[1]].to_string()
                    } else {
                        return Err(gtmpl::FuncError::Generic(
                            "slice bounds out of range".to_string(),
                        ));
                    }
                }
                _ => return Err(gtmpl::FuncError::ExactlyXArgs("a".to_string(), 4)),
            };
            Ok(gtmpl::Value::String(result))
        }
        gtmpl::Value::Array(arr) => {
            let mut indices = Vec::new();
            for arg in &args[1..] {
                if let gtmpl::Value::Number(n) = arg {
                    if let Some(i) = n.as_i64() {
                        indices.push(i as usize);
                    } else {
                        return Err(gtmpl::FuncError::Generic(
                            "slice bounds out of range".to_string(),
                        ));
                    }
                } else {
                    return Err(gtmpl::FuncError::Generic(
                        "slice bounds must be numbers".to_string(),
                    ));
                }
            }
            let result = match indices.len() {
                0 => arr.clone(),
                1 => arr[indices[0]..].to_vec(),
                2 => arr[indices[0]..indices[1]].to_vec(),
                3 => {
                    if indices[0] <= indices[1] && indices[1] <= arr.len() {
                        arr[indices[0]..indices[1]].to_vec()
                    } else {
                        return Err(gtmpl::FuncError::Generic(
                            "slice bounds out of range".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(gtmpl::FuncError::ExactlyXArgs(
                        "Doesn't enough arg".to_string(),
                        4,
                    ));
                }
            };
            Ok(gtmpl::Value::Array(result))
        }
        _ => Err(gtmpl::FuncError::Generic("slice".to_string())),
    }
}

fn gtmpl_fn_current_date(args: &[gtmpl::Value]) -> Result<gtmpl::Value, gtmpl::FuncError> {
    if !args.is_empty() {
        return Err(gtmpl::FuncError::ExactlyXArgs(
            "current_date".to_string(),
            0,
        ));
    }

    let date = Local::now().date_naive();

    Ok(gtmpl::Value::String(format!("{}", date)))
}
