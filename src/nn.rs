use ndarray::{Array, Array1, Array2, Axis, Zip}; // for matrices

use ndarray_rand::rand_distr::StandardNormal; // for randomness
use ndarray_rand::RandomExt; // for randomness
use rand::rngs::SmallRng;
use rand::SeedableRng; // for from_seed // for randomness

/**
 * single layer neural network
 */
#[derive(Default)]
pub struct Network {
    pub w1: Array2<f32>,
    pub b1: Array2<f32>,
    pub w2: Array2<f32>,
    pub b2: Array2<f32>,
}

impl Network {
    pub fn init(&mut self, input_size: u32, fc_size: u32, output_size: u32, rand_max: f32) {
        let input_size = input_size as usize;
        let fc_size = fc_size as usize;
        let output_size = output_size as usize;
        // according to rand::rngs/mod.rs line 121
        let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut rng = SmallRng::from_seed(seed);
        *self = Network {
            w1: Array::random_using((input_size, fc_size), StandardNormal, &mut rng) * rand_max,
            w2: Array::random_using((fc_size, output_size), StandardNormal, &mut rng) * rand_max,
            b1: Array::random_using((1, fc_size), StandardNormal, &mut rng) * rand_max,
            b2: Array::random_using((1, output_size), StandardNormal, &mut rng) * rand_max,
            /* random version but commented because strange behaviour of random in wasm leads to panic
            w1: Array::ones((input_size, fc_size)) * rand_max,
            w2: Array::ones((fc_size, output_size)) * rand_max,
            b1: Array::ones((1, fc_size)) * rand_max,
            b2: Array::ones((1, output_size)) * rand_max,
            */
        }
    }

    pub fn descent(
        &mut self,
        dw1: &Array2<f32>,
        db1: &Array2<f32>,
        dw2: &Array2<f32>,
        db2: &Array2<f32>,
        descent_rate: f32,
    ) {
        let rate = descent_rate;
        self.w1 -= &(rate * dw1);
        self.b1 -= &(rate * db1);
        self.w2 -= &(rate * dw2);
        self.b2 -= &(rate * db2);
    }

    pub fn forward_propagation(&self, points: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let act1 = &points.dot(&self.w1) + &self.b1;
        let fc_layer = act1.mapv(|x| x.max(0f32)); // relu process
        let act2 = &fc_layer.dot(&self.w2) + &self.b2;
        let scores = act2;
        let exp_scores = scores.mapv(f32::exp);
        let softmax = &exp_scores / &exp_scores.sum_axis(Axis(1)).insert_axis(Axis(1));
        // println!("{:#?}", softmax);
        (fc_layer, softmax)
    }

    pub fn loss(
        &self,
        softmax: &Array2<f32>,
        labels: &Array1<u32>,
        regular_rate: f32,
    ) -> (f32, f32) {
        let num_data = softmax.nrows();
        let mut probs_correct: Array1<f32> = Array::zeros(num_data);
        Zip::from(&mut probs_correct)
            .and(softmax.genrows())
            .and(labels)
            .apply(|prob_correct, prob, &label| {
                *prob_correct = prob[label as usize];
            });
        let infos = probs_correct.mapv(|x| -f32::ln(x));
        //println!("{:#?}", &probs_correct);
        let data_loss = infos.mean().unwrap();
        let regular_loss =
            0.5f32 * regular_rate * ((&self.w1 * &self.w1).sum() + (&self.w2 * &self.w2).sum());
        //println!("data loss: {} regular loss: {}", data_loss, regular_loss);
        (data_loss, regular_loss)
    }

    pub fn back_propagation(
        &self,
        points: &Array2<f32>,
        fc_layer: &Array2<f32>,
        softmax: &Array2<f32>,
        labels: &Array1<u32>,
        regular_rate: f32,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let num_data = softmax.nrows();
        let mut dscores = softmax.clone();
        for (i, mut dscore) in dscores.axis_iter_mut(Axis(0)).enumerate() {
            dscore[[labels[[i]] as usize]] -= 1f32;
        }
        dscores /= num_data as f32;
        let dact2 = dscores;
        let dfc_layer = dact2.dot(&self.w2.t());
        let mut dact1 = dfc_layer.clone();
        Zip::from(&mut dact1).and(fc_layer).apply(|act1, &fc| {
            if fc == 0f32 {
                *act1 = 0f32;
            }
        });

        let dw2 = fc_layer.t().dot(&dact2) + regular_rate * &self.w2;
        let db2 = dact2.sum_axis(Axis(0)).insert_axis(Axis(0));
        let dw1 = points.t().dot(&dact1) + regular_rate * &self.w1;
        let db1 = dact1.sum_axis(Axis(0)).insert_axis(Axis(0));
        (dw1, db1, dw2, db2)
    }
}
