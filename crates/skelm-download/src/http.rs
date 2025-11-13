use std::{io::SeekFrom, path::Path};

use futures_util::StreamExt;
use reqwest::Client;
use thiserror::Error;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use url::Url;

use super::utils::{DataUpdatable, ProgressDisplay};

#[derive(Debug, Error)]
pub enum HttpError {
    #[error("I/O Error {0}")]
    IO(#[from] std::io::Error),
    #[error("HTTP Error {0}")]
    HTTP(#[from] reqwest::Error),
}

/// Interface to download from resumable HTTP with an incremental hash and a potential progress report
pub async fn download<H: DataUpdatable, PB: ProgressDisplay>(
    client: &Client,
    url: &Url,
    destination: &Path,
    hash_ctx: &mut H,
) -> Result<(), HttpError> {
    let mut downloaded: u64 = 0;
    let mut file = if destination.exists() {
        let mut f = tokio::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .open(destination)
            .await?;
        downloaded = f.metadata().await?.len();

        hash_ctx.ctx_update_read_file(&mut f).await?;
        // in case the update_read_file is not reading anything, seek to the end
        f.seek(SeekFrom::Start(downloaded)).await?;
        f
    } else {
        tokio::fs::File::create(destination).await?
    };

    let mut request = client.get(url.clone());
    if downloaded > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={}-", downloaded));
    }

    let response = request.send().await?;

    let total_size = response.content_length().map(|len| len + downloaded);

    let pb = PB::progress_start(total_size);
    pb.progress_update(downloaded);

    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        hash_ctx.ctx_update(&chunk);
        downloaded += chunk.len() as u64;
        pb.progress_update(downloaded);
    }

    pb.progress_finalize();

    Ok(())
}
