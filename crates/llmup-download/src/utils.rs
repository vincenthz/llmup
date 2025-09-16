use llmup_store::ollama;
use tokio::io::AsyncReadExt;

pub trait ProgressDisplay {
    fn progress_start(size: Option<u64>) -> Self;
    fn progress_update(&self, position: u64);
    fn progress_finalize(self);
}

#[derive(Clone)]
pub struct NoProgress;

impl ProgressDisplay for NoProgress {
    fn progress_start(_size: Option<u64>) -> Self {
        Self
    }

    fn progress_update(&self, _position: u64) {}

    fn progress_finalize(self) {}
}

pub trait DataUpdatable {
    fn ctx_new() -> Self;
    fn ctx_update(&mut self, data: &[u8]);
    fn ctx_update_read_file(
        &mut self,
        file: &mut tokio::fs::File,
    ) -> impl Future<Output = std::io::Result<()>> {
        async {
            let mut buf = vec![0; 16384];
            loop {
                match file.read_buf(&mut buf).await {
                    Ok(n) => {
                        if n == 0 {
                            break;
                        }
                        self.ctx_update(&buf[0..n]);
                    }
                    Err(e) => {
                        panic!("issue : {}", e);
                    }
                }
            }
            Ok(())
        }
    }
}

pub struct DataUpdatableNoop;

impl Default for DataUpdatableNoop {
    fn default() -> Self {
        Self
    }
}

impl DataUpdatable for DataUpdatableNoop {
    fn ctx_new() -> Self {
        Self
    }
    fn ctx_update(&mut self, _data: &[u8]) {}
    async fn ctx_update_read_file(&mut self, _file: &mut tokio::fs::File) -> std::io::Result<()> {
        Ok(())
    }
}

impl DataUpdatable for ollama::BlobContext {
    fn ctx_new() -> Self {
        ollama::BlobContext::new_sha256()
    }

    fn ctx_update(&mut self, data: &[u8]) {
        self.update(data)
    }
}
