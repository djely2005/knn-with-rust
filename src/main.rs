use std::fs;
use std::path::Path;
use std::collections::HashMap;
use image::{GrayImage, Luma};

fn save_image(image_data: &[f32; 784], file_path: &Path) {
    let mut img = GrayImage::new(28, 28); // MNIST images are 28x28 pixels
    for (i, pixel) in image_data.iter().enumerate() {
        let x = (i % 28) as u32;
        let y = (i / 28) as u32;
        let pixel_value = (pixel * 255.0) as u8; // Convert normalized value back to 0-255
        img.put_pixel(x, y, Luma([pixel_value]));
    }
    img.save(file_path).expect("Failed to save image");
}
fn main() {
    let change = 6;
    let start = std::time::Instant::now();
    let label_test_path = Path::new("./MNIST/t10k-labels.idx1-ubyte");
    let image_test_path = Path::new("./MNIST/t10k-images.idx3-ubyte");

    let image_to_compare = read_images(image_test_path).expect("Failed to read images")[change];
    let label = read_labels(label_test_path).expect("Failed to read labels")[change];
    println!("Label to compare: {}", label);

    let k = 5; // Number of nearest neighbors
    println!("Finding {} nearest neighbors...", k);
    let guessed = get_knn(k, &image_to_compare);
    println!("Finished finding nearest neighbors.");
    let most_common = most_common_label(&guessed);
    println!("Most common: {}", most_common);
    let elapsed = start.elapsed();
    println!("Elapsed time: {:.2?}", elapsed);
    let test_output_path = Path::new("./image/output.png");
    save_image(&image_to_compare, test_output_path);

}

fn read_labels(file_path: &Path) -> Result<Vec<u8>, String> {
    let data = fs::read(file_path).map_err(|e| e.to_string())?;
    if &data[0..4] == b"\x00\x00\x08\x01" {
        Ok(data[8..].to_vec())
    } else {
        Err("Invalid MNIST label file".to_string())
    }
}

fn read_images(file_path: &Path) -> Result<Vec<[f32; 784]>, String> {
    let data = fs::read(file_path).map_err(|e| e.to_string())?;
    if &data[0..4] == b"\x00\x00\x08\x03" {
        let number_of_images = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let mut images = Vec::with_capacity(number_of_images);
        for i in 0..number_of_images {
            let start = 16 + i * 784;
            let end = start + 784;
            let image_data = &data[start..end];
            let normalized_image: [f32; 784] = image_data.iter().map(|&p| normalize_pixel_value(p)).collect::<Vec<_>>().try_into().unwrap();
            images.push(normalized_image);
        }
        Ok(images)
    } else {
        Err("Invalid MNIST image file".to_string())
    }
}

fn normalize_pixel_value(pixel_value: u8) -> f32 {
    pixel_value as f32 / 255.0
}

fn knn_distance_metric(image1: &[f32; 784], image2: &[f32; 784]) -> f32 {
    image1.iter().zip(image2).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt()
}

fn get_knn(k: usize, image_to_compare: &[f32; 784]) -> Vec<u8> {
    let label_path = Path::new("./MNIST/train-labels.idx1-ubyte");
    let image_path = Path::new("./MNIST/train-images.idx3-ubyte");

    let labels = read_labels(label_path).expect("Failed to read labels");
    let images = read_images(image_path).expect("Failed to read images");

    let mut distances: Vec<(usize, f32)> = images.iter()
        .enumerate()
        .map(|(i, image)| (i, knn_distance_metric(image, image_to_compare)))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let guessed = distances.iter().take(k);
    for (i, (index, _)) in guessed.clone().enumerate() {
        save_image(&images[*index], format!("./image/output_lookalike{}.png", i).as_ref());
    }
    guessed.map(|&(index, _)| labels[index]).collect()
}

fn most_common_label(labels: &[u8]) -> u8 {
    let mut label_count = HashMap::new();
    for &label in labels {
        *label_count.entry(label).or_insert(0) += 1;
    }
    label_count.into_iter().max_by_key(|&(_, count)| count).map(|(label, _)| label).unwrap_or(0)
}