/// Known models that can be downloaded with `onnx-http download <name>`.
pub struct ModelEntry {
    /// Model name (used as directory name under models/)
    pub name: &'static str,
    /// HuggingFace repository ID
    pub repo: &'static str,
    /// Path to the ONNX model file within the repo
    pub onnx_path: &'static str,
    /// Path to the tokenizer.json within the repo
    pub tokenizer_path: &'static str,
    /// Additional files to download (e.g., model.onnx_data for large models)
    pub extra_files: &'static [(&'static str, &'static str)],
    /// Embedding dimensions
    pub dimensions: u32,
    /// Maximum token sequence length
    pub max_tokens: usize,
    /// Short description
    pub description: &'static str,
}

pub const CATALOG: &[ModelEntry] = &[
    ModelEntry {
        name: "all-MiniLM-L6-v2",
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        extra_files: &[],
        dimensions: 384,
        max_tokens: 512,
        description: "Fast, lightweight general-purpose embeddings",
    },
    ModelEntry {
        name: "all-MiniLM-L12-v2",
        repo: "sentence-transformers/all-MiniLM-L12-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        extra_files: &[],
        dimensions: 384,
        max_tokens: 512,
        description: "Better quality than L6, still fast",
    },
    ModelEntry {
        name: "all-mpnet-base-v2",
        repo: "sentence-transformers/all-mpnet-base-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        extra_files: &[],
        dimensions: 768,
        max_tokens: 512,
        description: "Best all-around sentence-transformers model",
    },
    ModelEntry {
        name: "bge-base-en-v1.5",
        repo: "BAAI/bge-base-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        extra_files: &[],
        dimensions: 768,
        max_tokens: 512,
        description: "Top retrieval/search quality, great for documents",
    },
    ModelEntry {
        name: "bge-large-en-v1.5",
        repo: "BAAI/bge-large-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        extra_files: &[],
        dimensions: 1024,
        max_tokens: 512,
        description: "Highest quality BGE model, 1024 dimensions",
    },
    ModelEntry {
        name: "bge-m3",
        repo: "BAAI/bge-m3",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "onnx/tokenizer.json",
        extra_files: &[("onnx/model.onnx_data", "model.onnx_data")],
        dimensions: 1024,
        max_tokens: 8192,
        description: "8K context, multilingual, dense+sparse retrieval (2.3GB)",
    },
    ModelEntry {
        name: "jina-embeddings-v2-small-en",
        repo: "jinaai/jina-embeddings-v2-small-en",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        extra_files: &[],
        dimensions: 512,
        max_tokens: 8192,
        description: "8K context, lightweight English embeddings (130MB)",
    },
    ModelEntry {
        name: "embeddinggemma-300m",
        repo: "onnx-community/embeddinggemma-300m-ONNX",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        extra_files: &[("onnx/model.onnx_data", "model.onnx_data")],
        dimensions: 768,
        max_tokens: 2048,
        description: "Google Gemma, 2K context, multilingual (1.2GB)",
    },
    ModelEntry {
        name: "multi-qa-mpnet-base-cos-v1",
        repo: "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        extra_files: &[],
        dimensions: 768,
        max_tokens: 512,
        description: "Trained for semantic search and QA",
    },
];

pub fn find(name: &str) -> Option<&'static ModelEntry> {
    CATALOG.iter().find(|m| m.name.eq_ignore_ascii_case(name))
}
