use std::path::PathBuf;

use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};

use crate::catalog::{self, ModelEntry};

/// Download a model by name from the catalog.
pub async fn download_model(name: &str) -> anyhow::Result<()> {
    let entry = catalog::find(name).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown model '{}'. Use 'onnx-http download --list' to see available models.",
            name
        )
    })?;

    let dest_dir = PathBuf::from("models").join(entry.name);

    let model_path = dest_dir.join("model.onnx");
    let tokenizer_path = dest_dir.join("tokenizer.json");

    if model_path.exists() && tokenizer_path.exists() {
        eprintln!("✅ Model '{}' already exists at {}", entry.name, dest_dir.display());
        return Ok(());
    }

    std::fs::create_dir_all(&dest_dir)?;

    eprintln!(
        "📥 Downloading '{}' ({} dims, {}K tokens) — {}",
        entry.name, entry.dimensions, entry.max_tokens / 1000, entry.description
    );
    eprintln!("   Source: huggingface.co/{}", entry.repo);
    eprintln!();

    download_hf_file(entry, entry.onnx_path, &model_path).await?;
    download_hf_file(entry, entry.tokenizer_path, &tokenizer_path).await?;

    // Download extra files (e.g., model.onnx_data for large models)
    for (remote_path, local_name) in entry.extra_files {
        let dest = dest_dir.join(local_name);
        download_hf_file(entry, remote_path, &dest).await?;
    }

    // Write model_config.json with per-model settings
    let config = format!("{{\"max_tokens\": {}}}\n", entry.max_tokens);
    std::fs::write(dest_dir.join("model_config.json"), config)?;

    eprintln!();
    eprintln!("✅ Model '{}' installed to {}", entry.name, dest_dir.display());
    eprintln!("   Use with: {{\"model\": \"{}\", \"input\": \"...\"}}", entry.name);

    Ok(())
}

async fn download_hf_file(
    entry: &ModelEntry,
    file_path: &str,
    dest: &PathBuf,
) -> anyhow::Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        entry.repo, file_path
    );

    let file_name = dest.file_name().unwrap_or_default().to_string_lossy();
    eprintln!("   Downloading {file_name}...");

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to download {url}: {e}"))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Download failed: HTTP {} for {}",
            response.status(),
            url
        );
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("   [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );

    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = response.bytes_stream();

    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| anyhow::anyhow!("Download error: {e}"))?;
        file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }

    pb.finish();
    Ok(())
}

/// Print the list of available models.
pub fn print_catalog() {
    eprintln!("Available models:");
    eprintln!();
    eprintln!(
        "  {:<30} {:>6} {:>8}  {}",
        "NAME", "DIMS", "TOKENS", "DESCRIPTION"
    );
    eprintln!("  {}", "─".repeat(86));

    let models_dir = PathBuf::from("models");
    for entry in catalog::CATALOG {
        let installed = models_dir.join(entry.name).join("model.onnx").exists();
        let marker = if installed { " ✓" } else { "" };
        eprintln!(
            "  {:<30} {:>6} {:>8}  {}{}",
            entry.name, entry.dimensions, entry.max_tokens, entry.description, marker
        );
    }

    eprintln!();
    eprintln!("Download with: onnx-http download <name>");
}

/// Print installed models found in the models/ directory.
pub fn print_installed() {
    let models_dir = PathBuf::from("models");

    if !models_dir.is_dir() {
        eprintln!("No models directory found. Download a model with: onnx-http download <name>");
        return;
    }

    let mut entries: Vec<_> = std::fs::read_dir(&models_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            p.is_dir() && p.join("model.onnx").exists() && p.join("tokenizer.json").exists()
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    if entries.is_empty() {
        eprintln!("No models installed. Download one with: onnx-http download <name>");
        return;
    }

    eprintln!("Installed models:");
    eprintln!();
    eprintln!(
        "  {:<30} {:>6} {:>8}  {}",
        "NAME", "DIMS", "TOKENS", "SOURCE"
    );
    eprintln!("  {}", "─".repeat(76));

    for entry in &entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let catalog_entry = catalog::find(&name);

        let dims = catalog_entry
            .map(|e| format!("{}", e.dimensions))
            .unwrap_or_else(|| "?".to_string());

        let tokens = catalog_entry
            .map(|e| format!("{}", e.max_tokens))
            .unwrap_or_else(|| "?".to_string());

        let source = catalog_entry
            .map(|e| format!("huggingface.co/{}", e.repo))
            .unwrap_or_else(|| "custom".to_string());

        eprintln!("  {:<30} {:>6} {:>8}  {}", name, dims, tokens, source);
    }

    eprintln!();
    eprintln!("{} model(s) installed", entries.len());
}
