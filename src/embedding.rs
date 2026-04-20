use ndarray::{Array2, Array3, Axis};
use ort::value::Tensor;
use tracing::debug;

use crate::model::OnnxModel;

/// Generate embeddings for a single text input.
#[allow(dead_code)]
pub fn embed(model: &OnnxModel, text: &str) -> anyhow::Result<Vec<f32>> {
    let results = embed_batch(model, &[text.to_string()])?;
    results
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No embedding produced"))
}

/// Generate embeddings for a batch of text inputs.
pub fn embed_batch(model: &OnnxModel, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
    let batch_size = texts.len();
    debug!("Embedding batch of {} text(s)", batch_size);

    // Tokenize all inputs
    let encodings = model
        .tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

    // Find max sequence length for padding
    let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
    if max_len == 0 {
        return Err(anyhow::anyhow!("All inputs produced empty token sequences"));
    }

    // Build padded input_ids, attention_mask, and token_type_ids as flat vecs
    let mut input_ids_vec = vec![0i64; batch_size * max_len];
    let mut attention_mask_vec = vec![0i64; batch_size * max_len];
    let mut token_type_ids_vec = vec![0i64; batch_size * max_len];

    for (i, encoding) in encodings.iter().enumerate() {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        for j in 0..ids.len() {
            input_ids_vec[i * max_len + j] = ids[j] as i64;
            attention_mask_vec[i * max_len + j] = mask[j] as i64;
            token_type_ids_vec[i * max_len + j] = type_ids[j] as i64;
        }
    }

    // Create tensors using (shape, data) tuple form
    let input_ids_tensor =
        Tensor::from_array((vec![batch_size, max_len], input_ids_vec.clone().into_boxed_slice()))
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {e}"))?;
    let attention_mask_tensor =
        Tensor::from_array((vec![batch_size, max_len], attention_mask_vec.clone().into_boxed_slice()))
            .map_err(|e| anyhow::anyhow!("Failed to create attention_mask tensor: {e}"))?;

    // Run ONNX inference and extract output while session lock is held
    let (shape, hidden_data) = {
        let mut session = model
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock session: {e}"))?;

        let outputs = if model.has_token_type_ids {
            let token_type_ids_tensor =
                Tensor::from_array((vec![batch_size, max_len], token_type_ids_vec.into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("Failed to create token_type_ids tensor: {e}"))?;
            session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
        }
        .map_err(|e| anyhow::anyhow!("ONNX inference failed: {e}"))?;

        let (out_shape, out_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output tensor: {e}"))?;
        let shape: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        let data = out_data.to_vec();
        (shape, data)
    };
    debug!("Model output shape: {:?}", shape);

    if shape.len() != 3 {
        return Err(anyhow::anyhow!(
            "Expected 3D output [batch, seq, hidden], got shape {:?}",
            shape
        ));
    }

    let hidden = Array3::<f32>::from_shape_vec(
        (shape[0], shape[1], shape[2]),
        hidden_data,
    )?;

    // Mean pooling with attention mask
    let attention_mask_2d =
        Array2::<f32>::from_shape_vec((batch_size, max_len), attention_mask_vec.iter().map(|&v| v as f32).collect())?;
    let mask_expanded = attention_mask_2d.insert_axis(Axis(2));

    let masked = &hidden * &mask_expanded;
    let sum = masked.sum_axis(Axis(1)); // [batch, hidden]
    let count = mask_expanded.sum_axis(Axis(1)); // [batch, 1]
    let count = count.mapv(|v| if v == 0.0 { 1.0 } else { v });
    let embeddings = &sum / &count; // [batch, hidden]

    let result: Vec<Vec<f32>> = embeddings
        .outer_iter()
        .map(|row: ndarray::ArrayView1<f32>| row.to_vec())
        .collect();

    Ok(result)
}
