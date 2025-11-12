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

    pub fn blob_url(&self, blob: &super::Blob) -> Url {
        self.base_url
            .join(&format!(
                "{}/library/registry/blobs/{}",
                &self.version,
                &blob.as_path_name()
            ))
            .unwrap()
    }

    pub fn manifest_url(&self, model: &super::Model, variant: &super::Variant) -> Url {
        self.base_url
            .join(&format!(
                "{}/library/{}/manifests/{}",
                &self.version,
                model.as_str(),
                variant.as_str()
            ))
            .unwrap()
    }
}
