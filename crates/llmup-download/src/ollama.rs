use crate::ProgressDisplay;

use super::http;
use llmup_store::ollama;
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

pub async fn download_model<PB: ProgressDisplay>(
    client: &reqwest::Client,
    config: &OllamaConfig,
    store: &ollama::OllamaStore,
    registry: &ollama::Registry,
    model: &ollama::Model,
    variant: &ollama::Variant,
) -> () {
    let manifest_url = manifest_url(config, model, variant);

    let request = client.get(manifest_url).header(
        reqwest::header::CONTENT_TYPE,
        "application/vnd.docker.distribution.manifest.v2+json",
    );

    let response = request.send().await.unwrap();

    if response.status() == reqwest::StatusCode::OK {
        let bytes = response.bytes().await.unwrap();
        let manifest = ollama::Manifest::from_json_bytes(&bytes).unwrap();
        println!("{:#?}", manifest);

        download_model_with_manifest::<PB>(
            client, config, store, &manifest, registry, model, variant,
        )
        .await
    } else {
        println!("failed to download manifest : {}", response.status())
    }
}

async fn download_model_with_manifest<PB: ProgressDisplay>(
    client: &reqwest::Client,
    config: &OllamaConfig,
    store: &ollama::OllamaStore,
    manifest: &ollama::Manifest,
    registry: &ollama::Registry,
    model: &ollama::Model,
    variant: &ollama::Variant,
) {
    download_model_blob::<PB>(client, config, store, &manifest.config.digest).await;
    for layer in &manifest.layers {
        download_model_blob::<PB>(client, config, store, &layer.digest).await
    }

    store
        .add_manifest(registry, model, variant, manifest)
        .unwrap();
}

async fn download_model_blob<PB: ProgressDisplay>(
    client: &reqwest::Client,
    config: &OllamaConfig,
    store: &ollama::OllamaStore,
    blob: &ollama::Blob,
) {
    if store.blob_exists(blob) {
        return;
    }

    let blob_url = blob_url(config, blob);
    let blob_tmp_path = store.blob_path_tmp(blob);

    let mut blob_context = ollama::BlobContext::new_from_blob_type(blob);

    http::download::<_, PB>(client, &blob_url, &blob_tmp_path, &mut blob_context)
        .await
        .unwrap();

    let got_blob = blob_context.finalize();
    if &got_blob != blob {
        std::fs::remove_file(blob_tmp_path).unwrap();
        panic!("blob {} invalid got {}", blob, got_blob);
    }

    std::fs::rename(blob_tmp_path, store.blob_path(blob)).unwrap();
}
