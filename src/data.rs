use ndarray::prelude::*;
use ndarray::{stack, Array, Array1, Array2, Axis}; // for matrices

use ndarray_rand::rand_distr::StandardNormal; // for randomness
use ndarray_rand::RandomExt; // for randomness
use rand::rngs::SmallRng;
use rand::SeedableRng; // for from_seed // for randomness

use std::f32::consts::PI; // for math functions

/**
 * point data with labels
 */
#[derive(Default)]
pub struct Data {
    pub points: Array2<f32>, // points position
    pub labels: Array1<u32>, // points labels
}

impl Data {
    // num_sample: num of data for each label class
    // radius: radius of the circle of data points position
    // span: each data arm ratate span
    pub fn init(
        &mut self,
        num_classes: u32,
        num_samples: u32,
        radius: f32,
        span: f32,
        rand_max: f32,
    ) {
        // For array creating convenience
        let num_classes = num_classes as usize;
        let num_samples = num_samples as usize;

        let num_data = num_classes * num_samples;
        self.points = Array::zeros((num_data, 2));
        self.labels = Array::zeros(num_data);
        for i in 0..num_classes {
            let rho = Array::linspace(0f32, radius, num_samples);
            let begin = i as f32 * (2f32 * PI / num_classes as f32);

            let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let mut rng = SmallRng::from_seed(seed);
            let theta = Array::linspace(begin, begin - span, num_samples)
                        // will be changed later to use span to generate randomness to avoid points flickering
                        + Array::<f32, _>::random_using(num_samples, StandardNormal, &mut rng) * rand_max;

            let xs = (theta.mapv(f32::sin) * &rho)
                .into_shape((num_samples, 1))
                .unwrap();
            let ys = (theta.mapv(f32::cos) * &rho)
                .into_shape((num_samples, 1))
                .unwrap();
            let mut class_points = self
                .points
                .slice_mut(s![i * num_samples..(i + 1) * num_samples, ..]);
            class_points.assign(&stack![Axis(1), xs, ys]);
            let mut class_labels = self
                .labels
                .slice_mut(s![i * num_samples..(i + 1) * num_samples]);
            class_labels.fill(i as u32);
            // or:
            //class_labels.assign(&(Array::ones(num_samples) * i));
        }
    }
}
