use std::{
    fs::DirEntry,
    io::{Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

use serde::{Deserialize, Serialize};

pub struct OllamaStore {
    path: PathBuf,
}

impl Default for OllamaStore {
    fn default() -> Self {
        let home = std::env::home_dir().unwrap();
        let ollama_default_path = home.join(".ollama");
        Self::new(ollama_default_path)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model(String);

impl Model {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl FromStr for Model {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(s.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Variant(String);

impl Variant {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl FromStr for Variant {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(s.to_string()))
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Blob {
    Sha256([u8; 32]),
}

impl std::fmt::Debug for Blob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Blob::Sha256(h) => write!(f, "sha256:{}", hex::encode(h)),
        }
    }
}

impl std::fmt::Display for Blob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Blob::Sha256(h) => write!(f, "sha256:{}", hex::encode(h)),
        }
    }
}

impl serde::Serialize for Blob {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for Blob {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<Blob>().map_err(serde::de::Error::custom)
    }
}

impl Blob {
    pub fn as_path_name(&self) -> String {
        match self {
            Blob::Sha256(h) => format!("sha256-{}", hex::encode(h)),
        }
    }

    pub fn from_path_name(s: &str) -> Result<Self, String> {
        Self::from_str_sep(s, '-')
    }

    fn from_str_sep(s: &str, separator: char) -> Result<Self, String> {
        let Some((prefix, hex)) = s.split_once(separator) else {
            return Err(format!(
                "invalid format for blob string, no separator found"
            ));
        };
        let Ok(hex_val) = hex::decode(hex) else {
            return Err(format!("invalid blob name sha256 not hexadecimal"));
        };

        match prefix {
            "sha256" => {
                let Ok(hash) = <&[u8; 32]>::try_from(&hex_val[..]) else {
                    return Err(format!("invalid blob name sha256 not 32 bytes"));
                };
                Ok(Blob::Sha256(*hash))
            }
            _ => Err(format!("invalid blob unknown hash method {}", prefix)),
        }
    }
}

impl FromStr for Blob {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Blob::from_str_sep(s, ':')
    }
}

pub enum BlobContext {
    Sha256(cryptoxide::hashing::sha2::Context256),
}

impl BlobContext {
    pub fn new_from_blob_type(blob: &Blob) -> Self {
        match blob {
            Blob::Sha256(_) => Self::new_sha256(),
        }
    }

    pub fn new_sha256() -> Self {
        Self::Sha256(cryptoxide::hashing::sha2::Context256::new())
    }

    pub fn update(&mut self, data: &[u8]) {
        match self {
            BlobContext::Sha256(context256) => context256.update_mut(data),
        }
    }

    pub fn finalize(self) -> Blob {
        match self {
            BlobContext::Sha256(context256) => Blob::Sha256(context256.finalize()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Registry(String);

impl FromStr for Registry {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(s.to_string()))
    }
}

impl OllamaStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let path: PathBuf = path.as_ref().into();
        if !path.exists() {
            std::fs::create_dir_all(&path).unwrap();
            std::fs::create_dir_all(&path.join("models")).unwrap();
        }
        Self { path }
    }

    fn model_path(&self) -> PathBuf {
        self.path.join("models")
    }

    fn blobs_path(&self) -> PathBuf {
        self.model_path().join("blobs")
    }

    fn manifests_path(&self) -> PathBuf {
        self.model_path().join("manifests")
    }

    fn manifest_registry_path(&self, registry: &Registry) -> PathBuf {
        self.model_path().join("manifests").join(&registry.0)
    }

    pub fn manifest_registry_model_variant_path(
        &self,
        registry: &Registry,
        model: &Model,
        variant: &Variant,
    ) -> PathBuf {
        let reg_path = self.manifest_registry_path(registry);
        reg_path.join("library").join(&model.0).join(&variant.0)
    }

    pub fn list_blobs(&self) -> std::io::Result<Vec<Blob>> {
        let mut blobs = Vec::new();
        let dir = std::fs::read_dir(self.blobs_path())?;
        for d in dir {
            let Ok(name) = entry_read_dir_string(d, false) else {
                continue;
            };
            let Ok(blob) = Blob::from_str(&name) else {
                continue;
            };
            blobs.push(blob)
        }
        Ok(blobs)
    }

    pub fn blob_path(&self, blob: &Blob) -> PathBuf {
        self.blobs_path().join(&blob.as_path_name())
    }

    pub fn blob_path_tmp(&self, blob: &Blob) -> PathBuf {
        self.blobs_path()
            .join(&blob.as_path_name())
            .with_extension("tmp")
    }

    pub fn blob_read(&self, blob: &Blob) -> std::io::Result<Vec<u8>> {
        let path = self.blob_path(blob);
        let mut file = std::fs::File::open(path)?;
        let mut out = Vec::new();
        file.read_to_end(&mut out)?;
        Ok(out)
    }

    pub fn blob_read_string(&self, blob: &Blob) -> std::io::Result<String> {
        let path = self.blob_path(blob);
        let mut file = std::fs::File::open(path)?;
        let mut out = Vec::new();
        file.read_to_end(&mut out)?;
        String::from_utf8(out).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("utf8 error reading blob {}", e),
            )
        })
    }

    pub fn list_registries(&self) -> std::io::Result<Vec<Registry>> {
        let mut regs = Vec::new();
        let dir = std::fs::read_dir(self.manifests_path())?;
        for d in dir {
            let Ok(name) = entry_read_dir_string(d, true) else {
                continue;
            };
            if name == "." || name == ".." {
                continue;
            }
            regs.push(Registry(name))
        }
        Ok(regs)
    }

    pub fn list_models(&self, registry: &Registry) -> std::io::Result<Vec<Model>> {
        let mut models = Vec::new();
        let reg_path = self.manifest_registry_path(registry);
        let dir = std::fs::read_dir(reg_path.join("library"))?;
        for d in dir {
            let Ok(name) = entry_read_dir_string(d, true) else {
                continue;
            };
            if name == "." || name == ".." {
                continue;
            }
            models.push(Model(name))
        }
        Ok(models)
    }

    pub fn list_model_variants(
        &self,
        registry: &Registry,
        model: &Model,
    ) -> std::io::Result<Vec<Variant>> {
        let mut variants = Vec::new();
        let reg_path = self.manifest_registry_path(registry);
        let dir = std::fs::read_dir(reg_path.join("library").join(&model.0))?;
        for d in dir {
            let Ok(name) = entry_read_dir_string(d, false) else {
                continue;
            };
            variants.push(Variant(name))
        }
        Ok(variants)
    }

    pub fn get_manifest(
        &self,
        registry: &Registry,
        model: &Model,
        variant: &Variant,
    ) -> std::io::Result<Manifest> {
        let path = self.manifest_registry_model_variant_path(registry, model, variant);
        let content = std::fs::read_to_string(path)?;

        serde_json::from_reader(std::io::Cursor::new(content)).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("json manifest invalid: {}", e),
            )
        })
    }

    pub fn remove_manifest(
        &self,
        registry: &Registry,
        model: &Model,
        variant: &Variant,
    ) -> std::io::Result<()> {
        let path = self.manifest_registry_model_variant_path(registry, model, variant);

        if !path.exists() {
            return Ok(());
        }
        std::fs::remove_file(path)?;
        Ok(())
    }

    pub fn add_manifest(
        &self,
        registry: &Registry,
        model: &Model,
        variant: &Variant,
        manifest: &Manifest,
    ) -> std::io::Result<()> {
        let path = self.manifest_registry_model_variant_path(registry, model, variant);
        let manifest_data = serde_json::to_string(manifest).unwrap();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(path)?;
        file.write_all(&manifest_data.as_bytes())?;
        Ok(())
    }

    pub fn blob_exists(&self, blob: &Blob) -> bool {
        let path = self.blob_path(blob);
        path.exists()
    }

    pub fn blob_self_verify(&self, blob: &Blob) -> std::io::Result<bool> {
        let path = self.blob_path(blob);
        let mut fs = std::fs::File::open(path)?;
        let mut buf = [0; 16384];
        let mut ctx = BlobContext::new_from_blob_type(blob);
        loop {
            let n = fs.read(&mut buf)?;
            if n == 0 {
                break;
            };
            ctx.update(&buf[0..n]);
        }
        let blob_got = ctx.finalize();
        Ok(&blob_got == blob)
    }

    pub fn write_blob_data(&self, blob: &Blob, data: &[u8]) -> std::io::Result<()> {
        let path = self.blob_path(blob);

        if path.exists() {
            return Ok(());
        }

        let mut file = std::fs::File::create_new(path)?;
        file.write_all(data)?;
        Ok(())
    }

    pub fn verify_blob(&self, blob: &Blob) -> std::io::Result<bool> {
        let path = self.blob_path(blob);
        let mut file = std::fs::File::open(path)?;
        let mut buf = vec![0; 16834];
        let mut context = BlobContext::new_from_blob_type(blob);
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            context.update(&buf[0..n]);
        }
        let found_blob = context.finalize();
        Ok(&found_blob == blob)
    }
}

fn entry_read_dir_string(
    de: std::io::Result<DirEntry>,
    expected_dir: bool,
) -> std::io::Result<String> {
    let d = de?;
    let metadata = d.metadata()?;
    match expected_dir {
        true => {
            if !metadata.is_dir() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("entry not a directory"),
                ));
            }
        }
        false => {
            if !metadata.is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("entry not a file"),
                ));
            }
        }
    }
    let Ok(name) = d.file_name().into_string() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("entry not a valid utf8 string"),
        ));
    };
    Ok(name)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub schema_version: u64,
    pub media_type: String,
    pub config: ManifestConfig,
    pub layers: Vec<ManifestLayer>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestConfig {
    pub media_type: String,
    pub digest: Blob,
    pub size: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestLayer {
    pub media_type: String,
    pub digest: Blob,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
}

impl Manifest {
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_reader(std::io::Cursor::new(s))
    }

    pub fn from_json_bytes(s: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_reader(std::io::Cursor::new(s))
    }

    pub fn size(&self) -> u64 {
        self.layers
            .iter()
            .fold(self.config.size, |acc, l| acc + l.size)
    }

    pub fn all_digests(&self) -> Vec<Blob> {
        let mut digests = Vec::new();
        digests.push(self.config.digest.clone());
        for d in self.layers.iter().map(|l| l.digest.clone()) {
            digests.push(d)
        }
        digests
    }

    pub fn find_media_type(&self, ty: &str) -> Option<&ManifestLayer> {
        self.layers.iter().find(|l| l.media_type == ty)
    }
}

pub const MEDIA_TYPE_IMAGE_MODEL: &str = "application/vnd.ollama.image.model";
pub const MEDIA_TYPE_IMAGE_LICENSE: &str = "application/vnd.ollama.image.license";
pub const MEDIA_TYPE_IMAGE_TEMPLATE: &str = "application/vnd.ollama.image.template";
pub const MEDIA_TYPE_IMAGE_PARAMS: &str = "application/vnd.ollama.image.params";
