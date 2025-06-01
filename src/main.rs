use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::eig::Eig;
use rayon::prelude::*;

const IMAGE_DIM: usize = 28;
const NUM_PIXELS: usize = IMAGE_DIM * IMAGE_DIM;

fn read_labels(file_path: &Path) -> Result<Vec<u8>, String> {
    let data = fs::read(file_path).map_err(|e| e.to_string())?;
    if &data[0..4] == b"\x00\x00\x08\x01" {
        Ok(data[8..].to_vec())
    } else {
        Err("Invalid MNIST label file".to_string())
    }
}

fn read_images(file_path: &Path, num_images: usize) -> Result<Vec<Vec<f32>>, String> {
    let data = fs::read(file_path).map_err(|e| e.to_string())?;
    if &data[0..4] != b"\x00\x00\x08\x03" {
        return Err("Invalid MNIST image file".to_string());
    }

    let images: Vec<Vec<f32>> = (0..num_images).into_par_iter().map(|i| {
        let start = 16 + i * NUM_PIXELS;
        let end = start + NUM_PIXELS;
        let image_data = &data[start..end];
        image_data.iter().map(|&p| p as f32 / 255.0).collect()
    }).collect();
    Ok(images)
}

fn compute_pca(images: &[Vec<f32>], k: usize) -> Result<(Array2<f32>, Array1<f32>, Array2<f32>), String> {
    let n = images.len();
    let flat_data: Vec<f32> = images.iter().flatten().cloned().collect();
    let data = Array2::from_shape_vec((n, NUM_PIXELS), flat_data).map_err(|_| "Invalid image shape")?;

    let mean = data.mean_axis(Axis(0)).ok_or("Failed to compute mean")?;
    let centered = &data - &mean;

    let cov = centered.t().dot(&centered) / n as f32;
    let (_eigenvalues, eigenvectors) = cov.eig().map_err(|e| e.to_string())?;

    // Sort eigenvectors by eigenvalue norm
    let mut eig_pairs: Vec<_> = eigenvectors.axis_iter(Axis(1)).collect();
    eig_pairs.truncate(k);

    let projection = Array2::from_shape_fn((NUM_PIXELS, k), |(i, j)| eig_pairs[j][i].re);
    Ok((projection.clone(), mean, centered.dot(&projection)))
}

fn project(image: &[f32], mean: &Array1<f32>, projection: &Array2<f32>) -> Vec<f32> {
    let centered = Array1::from(image.to_vec()) - mean;
    projection.t().dot(&centered).to_vec()
}

fn knn_distance_metric(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>()
}

fn get_knn(k: usize, image_to_compare: &[f32], labels: &[u8], projected_images: &[Vec<f32>]) -> Vec<u8> {
    let mut distances: Vec<(usize, f32)> = projected_images
        .iter()
        .enumerate()
        .map(|(i, img)| (i, knn_distance_metric(img, image_to_compare)))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.iter().take(k).map(|(i, _)| labels[*i]).collect()
}

fn most_common_label(labels: &[u8]) -> u8 {
    let mut counts = HashMap::new();
    for &label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|&(_, count)| count).map(|(label, _)| label).unwrap_or(0)
}

fn main() -> Result<(), String> {
    let start = std::time::Instant::now();
    let training_time = std::time::Instant::now();
    let k = 5; // k-NN
    let num_train = 60000;
    let num_test = 10000;
    let pca_components = 30;


    let train_labels = read_labels(Path::new("./MNIST/train-labels.idx1-ubyte"))?;
    let train_images = read_images(Path::new("./MNIST/train-images.idx3-ubyte"), num_train)?;
    let (projection, mean, train_projected) = compute_pca(&train_images, pca_components)?;
    let train_projected_vecs = train_projected.outer_iter().map(|row| row.to_vec()).collect::<Vec<_>>();
    let train_speed =  training_time.elapsed().as_secs_f32();
    let test_time = std::time::Instant::now();
    let test_labels = read_labels(Path::new("./MNIST/t10k-labels.idx1-ubyte"))?;
    let test_images = read_images(Path::new("./MNIST/t10k-images.idx3-ubyte"), num_test)?;

    let correct = Arc::new(AtomicUsize::new(0));
    test_images.par_iter().enumerate().for_each(|(i, image)| {
        let projected = project(image, &mean, &projection);
        let neighbors = get_knn(k, &projected, &train_labels, &train_projected_vecs);
        let predicted = most_common_label(&neighbors);

        if predicted == test_labels[i] {
            correct.fetch_add(1, Ordering::Relaxed);
        }
    });
    println!("PCA computed in {:.2} seconds", train_speed);
    println!("test completed in {:.2} seconds", test_time.elapsed().as_secs_f32());
    println!("correct: {:?}", correct);
    println!("Total time: {:.2} seconds", start.elapsed().as_secs_f32());
    Ok(())
}
