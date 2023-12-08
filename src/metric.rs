use crate::datasets::vec_dataset::VecDataset;

#[derive(Debug)]
pub enum Metric {
    MeanSquaredError,
}

impl Metric {
    pub fn forward(&self, x: &VecDataset<f32>, y: &VecDataset<f32>) -> f32 {
        match self {
            Metric::MeanSquaredError => {
                let mut sum = 0.0;
                for (x, y) in x.as_ref().iter().zip(y.as_ref().iter()) {
                    sum += (x - y).powi(2);
                }
                sum / x.len() as f32
            }
        }
    }
}
