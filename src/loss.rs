use crate::datasets::vec_dataset::VecDataset;

#[derive(Debug)]
pub enum Loss {
    MeanSquaredError,
    MeanEuclideanError,
}

impl Loss {
    pub fn forward(&self, input: &VecDataset<f32>, target: &VecDataset<f32>) -> f32 {
        input
            .as_ref()
            .iter()
            .zip(target.as_ref().iter())
            .map(|x| match self {
                Loss::MeanSquaredError => (x.0 - x.1).powi(2),
                Loss::MeanEuclideanError => (x.0 - x.1).abs(),
            })
            .sum::<f32>()
            / input.len() as f32
    }

    pub fn backward(&self, input: &VecDataset<f32>, target: &VecDataset<f32>) -> VecDataset<f32> {
        VecDataset::from_vec(
            input
                .as_ref()
                .iter()
                .zip(target.as_ref().iter())
                .map(|x| match self {
                    Loss::MeanSquaredError => x.0 - x.1,
                    Loss::MeanEuclideanError => x.0 - x.1,
                })
                .collect(),
            input.len(),
            input.dim(),
        )
    }
}
