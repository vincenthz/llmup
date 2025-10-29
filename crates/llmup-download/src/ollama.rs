use crate::{ProgressDisplay, http::HttpError};

use super::http;
use llmup_store::ollama;
use reqwest::StatusCode;
use thiserror::Error;
use url::Url;

#[derive(Clone)]
pub struct OllamaConfig {
    pub base_url: Url,
    pub version: String,
}

const BASE_URL: &str = "https://registry.ollama.ai/";
const VERSION: &str = "v2";

impl Default for OllamaConfig {
    fn default() -> Self {
        let base_url = Url::parse(BASE_URL).unwrap();
        Self {
            base_url,
            version: String::from(VERSION),
        }
    }
}

impl OllamaConfig {
    pub fn host(&self) -> String {
        format!(
            "{}",
            self.base_url
                .host()
                .expect("valid host for ollama config registry")
        )
    }
}

pub fn blob_url(config: &OllamaConfig, blob: &ollama::Blob) -> Url {
    config
        .base_url
        .join(&format!(
            "{}/library/registry/blobs/{}",
            &config.version,
            &blob.as_path_name()
        ))
        .unwrap()
}

pub fn manifest_url(
    config: &OllamaConfig,
    model: &ollama::Model,
    variant: &ollama::Variant,
) -> Url {
    config
        .base_url
        .join(&format!(
            "{}/library/{}/manifests/{}",
            &config.version,
            model.as_str(),
            variant.as_str()
        ))
        .unwrap()
}

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("HTTP Error downloading {0}")]
    HttpError(#[from] HttpError),
    #[error("Fail to download manifest http-code={0}")]
    ManifestError(StatusCode),
    #[error("Fail to add manifest {0:?}")]
    ManifestAddingFailed(std::io::Error),
    #[error("Fail to commit blob {0} : {1}")]
    BlobCommitFailed(ollama::Blob, std::io::Error),
    #[error("Downloaded blob doesn't match expected {0} but got {1}")]
    InvalidBlobDownloaded(ollama::Blob, ollama::Blob),
}

pub async fn download_model<PB: ProgressDisplay>(
    client: &reqwest::Client,
    config: &OllamaConfig,
    store: &ollama::OllamaStore,
    registry: &ollama::Registry,
    model: &ollama::Model,
    variant: &ollama::Variant,
) -> Result<Vec<(String, DownloadResult)>, DownloadError> {
    let manifest_url = manifest_url(config, model, variant);

    let request = client.get(manifest_url).header(
        reqwest::header::CONTENT_TYPE,
        "application/vnd.docker.distribution.manifest.v2+json",
    );

    let response = request.send().await.unwrap();

    if response.status() == reqwest::StatusCode::OK {
        let bytes = response.bytes().await.unwrap();
        let manifest = ollama::Manifest::from_json_bytes(&bytes).unwrap();
        //println!("{:#?}", manifest);

        download_model_with_manifest::<PB>(
            client, config, store, &manifest, registry, model, variant,
        )
        .await
    } else {
        Err(DownloadError::ManifestError(response.status()))
        //println!("failed to download manifest : {}", response.status())
    }
}

pub enum DownloadResult {
    Skipped(ollama::Blob),
    Success(ollama::Blob),
}

async fn download_model_with_manifest<PB: ProgressDisplay>(
    client: &reqwest::Client,
    config: &OllamaConfig,
    store: &ollama::OllamaStore,
    manifest: &ollama::Manifest,
    registry: &ollama::Registry,
    model: &ollama::Model,
    variant: &ollama::Variant,
) -> Result<Vec<(String, DownloadResult)>, DownloadError> {
    let mut results = Vec::new();
    let manifest_result =
        download_model_blob::<PB>(client, config, store, &manifest.config.digest).await?;

    results.push(("manifest".to_string(), manifest_result));
    for layer in &manifest.layers {
        let r = download_model_blob::<PB>(client, config, store, &layer.digest).await?;
        results.push((layer.media_type.clone(), r))
    }

    store
        .add_manifest(registry, model, variant, manifest)
        .map_err(|e| DownloadError::ManifestAddingFailed(e))?;
    Ok(results)
}

async fn download_model_blob<PB: ProgressDisplay>(
    client: &reqwest::Client,
    config: &OllamaConfig,
    store: &ollama::OllamaStore,
    blob: &ollama::Blob,
) -> Result<DownloadResult, DownloadError> {
    if store.blob_exists(blob) {
        return Ok(DownloadResult::Skipped(blob.clone()));
    }

    let blob_url = blob_url(config, blob);
    let blob_tmp_path = store.blob_path_tmp(blob);

    let mut blob_context = ollama::BlobContext::new_from_blob_type(blob);

    http::download::<_, PB>(client, &blob_url, &blob_tmp_path, &mut blob_context).await?;

    let got_blob = blob_context.finalize();
    if &got_blob != blob {
        std::fs::remove_file(blob_tmp_path).unwrap();
        return Err(DownloadError::InvalidBlobDownloaded(blob.clone(), got_blob));
    }

    std::fs::rename(blob_tmp_path, store.blob_path(blob))
        .map_err(|e| DownloadError::BlobCommitFailed(blob.clone(), e))?;
    Ok(DownloadResult::Success(blob.clone()))
}
