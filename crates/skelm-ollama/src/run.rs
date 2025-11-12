use std::{path::PathBuf, str::FromStr};

use crate::storage::*;
use thiserror::*;

#[derive(Clone)]
pub struct ModelConfig {
    pub model_path: PathBuf,
    pub template: Option<String>,
    pub params: serde_json::Value,
}

#[derive(Error, Debug)]
pub enum ModelConfigGetError {
    #[error("manifest error {0} for {1}")]
    ManifestError(std::io::Error, ModelDescr),
    #[error("model image not found for {0}")]
    ModelImageNotFound(ModelDescr),
    #[error("error reading template file {0} for {1}")]
    ReadingTemplateError(std::io::Error, ModelDescr),
    #[error("manifest has no parameters layer for {0}")]
    ParametersMissing(ModelDescr),
    #[error("error reading parameters file {0} for {1}")]
    ReadingParameterError(std::io::Error, ModelDescr),
    #[error("error reading parameters JSON {0} for {1}")]
    ParameterFileNotJson(serde_json::error::Error, ModelDescr),
}

pub fn model_config_get(model_descr: &ModelDescr) -> Result<ModelConfig, ModelConfigGetError> {
    let store = OllamaStore::default();
    let manifest = store
        .get_manifest(&model_descr)
        .map_err(|e| ModelConfigGetError::ManifestError(e, model_descr.clone()))?;

    let Some(model_layer) = manifest.find_media_type(MEDIA_TYPE_IMAGE_MODEL) else {
        return Err(ModelConfigGetError::ModelImageNotFound(model_descr.clone()));
    };

    let template_data = if let Some(template_layer) =
        manifest.find_media_type(MEDIA_TYPE_IMAGE_TEMPLATE)
    {
        Some(
            store
                .blob_read_string(&template_layer.digest)
                .map_err(|e| ModelConfigGetError::ReadingTemplateError(e, model_descr.clone()))?,
        )
    } else {
        None
    };

    let Some(params_layer) = manifest.find_media_type(MEDIA_TYPE_IMAGE_PARAMS) else {
        return Err(ModelConfigGetError::ParametersMissing(model_descr.clone()));
    };
    let params_data = store
        .blob_read_string(&params_layer.digest)
        .map_err(|e| ModelConfigGetError::ReadingParameterError(e, model_descr.clone()))?;
    let params_json = serde_json::Value::from_str(&params_data)
        .map_err(|e| ModelConfigGetError::ParameterFileNotJson(e, model_descr.clone()))?;

    /*
    let template = if let Some(template_data) = template_data {
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
        };
        Some(template)
    } else {
        None
    };
    */

    let path = store.blob_path(&model_layer.digest);
    Ok(ModelConfig {
        model_path: path,
        template: template_data,
        params: params_json,
    })
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn gtmpl_fn_current_date(args: &[gtmpl::Value]) -> Result<gtmpl::Value, gtmpl::FuncError> {
    if !args.is_empty() {
        return Err(gtmpl::FuncError::ExactlyXArgs(
            "current_date".to_string(),
            0,
        ));
    }

    let date = chrono::Local::now().date_naive();

    Ok(gtmpl::Value::String(format!("{}", date)))
}
