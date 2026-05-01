use ndarray::{Array2, Array3, Axis};
use ort::value::Tensor;
use std::time::Instant;
use tokenizers::Encoding;
use tracing::{debug, info};

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
/// Sorts inputs by token length and splits into sub-batches to minimize padding waste.
pub fn embed_batch(model: &OnnxModel, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
    let total_start = Instant::now();
    let batch_size = texts.len();

    // Tokenize all inputs upfront
    let tokenize_start = Instant::now();
    let encodings = model
        .tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let tokenize_ms = tokenize_start.elapsed().as_secs_f64() * 1000.0;

    let total_tokens: usize = encodings.iter().map(|e| e.get_ids().len()).sum();
    let overall_max_seq: usize = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
    if overall_max_seq == 0 {
        return Err(anyhow::anyhow!("All inputs produced empty token sequences"));
    }

    // Sort indices by token length to group similar-length texts
    let mut sorted_indices: Vec<usize> = (0..batch_size).collect();
    sorted_indices.sort_by_key(|&i| encodings[i].get_ids().len());

    // Split into sub-batches: start a new group when longest > 2x shortest
    let sub_batches = build_sub_batches(&sorted_indices, &encodings);
    let num_sub_batches = sub_batches.len();

    // Process each sub-batch and collect (original_index, embedding) pairs
    let mut all_results: Vec<(usize, Vec<f32>)> = Vec::with_capacity(batch_size);
    let mut total_inference_ms = 0.0f64;

    for (sb_idx, sb_indices) in sub_batches.iter().enumerate() {
        let sb_encodings: Vec<&Encoding> = sb_indices.iter().map(|&i| &encodings[i]).collect();
        let sb_max_len = sb_encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        debug!(
            "Sub-batch {}/{}: {} text(s), max_seq={}",
            sb_idx + 1, num_sub_batches, sb_indices.len(), sb_max_len
        );

        let inference_start = Instant::now();
        let embeddings = run_inference(model, &sb_encodings, sb_max_len)?;
        total_inference_ms += inference_start.elapsed().as_secs_f64() * 1000.0;

        for (j, emb) in embeddings.into_iter().enumerate() {
            all_results.push((sb_indices[j], emb));
        }
    }

    // Reassemble in original order
    all_results.sort_by_key(|&(idx, _)| idx);
    let result: Vec<Vec<f32>> = all_results.into_iter().map(|(_, emb)| emb).collect();

    let total_chars: usize = texts.iter().map(|t| t.len()).sum();
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Embedded {} text(s) ({} chars, {} tokens, max_seq={}) in {} sub-batch(es): tokenize={:.1}ms, inference={:.1}ms, total={:.1}ms",
        batch_size, total_chars, total_tokens, overall_max_seq, num_sub_batches,
        tokenize_ms, total_inference_ms, total_ms
    );

    Ok(result)
}

/// Group sorted indices into sub-batches where max_len <= 2 * min_len.
fn build_sub_batches(sorted_indices: &[usize], encodings: &[Encoding]) -> Vec<Vec<usize>> {
    if sorted_indices.is_empty() {
        return vec![];
    }

    let mut sub_batches: Vec<Vec<usize>> = Vec::new();
    let mut current_batch: Vec<usize> = vec![sorted_indices[0]];
    let mut current_min_len = encodings[sorted_indices[0]].get_ids().len();

    for &idx in &sorted_indices[1..] {
        let len = encodings[idx].get_ids().len();
        // Start a new sub-batch if this text would cause >2x padding ratio
        if len > current_min_len * 2 && !current_batch.is_empty() {
            sub_batches.push(std::mem::take(&mut current_batch));
            current_min_len = len;
        }
        current_batch.push(idx);
    }

    if !current_batch.is_empty() {
        sub_batches.push(current_batch);
    }

    sub_batches
}

/// Run ONNX inference on a sub-batch of pre-tokenized encodings.
fn run_inference(
    model: &OnnxModel,
    encodings: &[&Encoding],
    max_len: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let batch_size = encodings.len();

    // Build padded input tensors
    let mut input_ids_vec = vec![0i64; batch_size * max_len];
    let mut attention_mask_vec = vec![0i64; batch_size * max_len];
    let mut token_type_ids_vec = vec![0i64; batch_size * max_len];

    for (i, encoding) in encodings.iter().enumerate() {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        for j in 0..ids.len().min(max_len) {
            input_ids_vec[i * max_len + j] = ids[j] as i64;
            attention_mask_vec[i * max_len + j] = mask[j] as i64;
            token_type_ids_vec[i * max_len + j] = type_ids[j] as i64;
        }
    }

    let input_ids_tensor =
        Tensor::from_array((vec![batch_size, max_len], input_ids_vec.clone().into_boxed_slice()))
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {e}"))?;
    let attention_mask_tensor =
        Tensor::from_array((vec![batch_size, max_len], attention_mask_vec.clone().into_boxed_slice()))
            .map_err(|e| anyhow::anyhow!("Failed to create attention_mask tensor: {e}"))?;

    // Run ONNX inference using a session from the pool
    let (shape, hidden_data) = {
        let mut session = model.pool.acquire();

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
        (shape, out_data.to_vec())
    };

    if shape.len() != 3 {
        return Err(anyhow::anyhow!(
            "Expected 3D output [batch, seq, hidden], got shape {:?}",
            shape
        ));
    }

    let hidden = Array3::<f32>::from_shape_vec((shape[0], shape[1], shape[2]), hidden_data)?;

    // Mean pooling with attention mask
    let attention_mask_2d =
        Array2::<f32>::from_shape_vec((batch_size, max_len), attention_mask_vec.iter().map(|&v| v as f32).collect())?;
    let mask_expanded = attention_mask_2d.insert_axis(Axis(2));

    let masked = &hidden * &mask_expanded;
    let sum = masked.sum_axis(Axis(1));
    let count = mask_expanded.sum_axis(Axis(1));
    let count = count.mapv(|v| if v == 0.0 { 1.0 } else { v });
    let embeddings = &sum / &count;

    Ok(embeddings
        .outer_iter()
        .map(|row| row.to_vec())
        .collect())
}
