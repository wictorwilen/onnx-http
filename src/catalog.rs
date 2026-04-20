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
    /// Embedding dimensions
    pub dimensions: u32,
    /// Short description
    pub description: &'static str,
}

pub const CATALOG: &[ModelEntry] = &[
    ModelEntry {
        name: "all-MiniLM-L6-v2",
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dimensions: 384,
        description: "Fast, lightweight general-purpose embeddings",
    },
    ModelEntry {
        name: "all-MiniLM-L12-v2",
        repo: "sentence-transformers/all-MiniLM-L12-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dimensions: 384,
        description: "Better quality than L6, still fast",
    },
    ModelEntry {
        name: "all-mpnet-base-v2",
        repo: "sentence-transformers/all-mpnet-base-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dimensions: 768,
        description: "Best all-around sentence-transformers model",
    },
    ModelEntry {
        name: "bge-base-en-v1.5",
        repo: "BAAI/bge-base-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dimensions: 768,
        description: "Top retrieval/search quality, great for documents",
    },
    ModelEntry {
        name: "bge-large-en-v1.5",
        repo: "BAAI/bge-large-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dimensions: 1024,
        description: "Highest quality BGE model, 1024 dimensions",
    },
    ModelEntry {
        name: "multi-qa-mpnet-base-cos-v1",
        repo: "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        dimensions: 768,
        description: "Trained for semantic search and QA",
    },
];

pub fn find(name: &str) -> Option<&'static ModelEntry> {
    CATALOG.iter().find(|m| m.name.eq_ignore_ascii_case(name))
}
