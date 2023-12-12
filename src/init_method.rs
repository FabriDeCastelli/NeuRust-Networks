use crate::datasets::vec_dataset::VecDataset;

#[derive(Debug, Clone)]
pub enum InitMethod {
    Random,
    FanIn,
    Zeros,
}

impl InitMethod {
    pub fn init_weights(&self, n: usize, m: usize) -> VecDataset<f32> {
        match self {
            InitMethod::Random => VecDataset::from_vec(
                (0..n * m)
                    .map(|_| 2.0 * rand::random::<f32>() - 1.0)
                    .collect(),
                n,
                m,
            ),

            InitMethod::FanIn => VecDataset::from_vec(
                (0..n * m)
                    .map(|_| (2.0 / (n as f32)).sqrt() * rand::random::<f32>() - 1.0 / (n as f32))
                    .collect(),
                n,
                m,
            ),
            InitMethod::Zeros => VecDataset::from_vec(vec![0.0; n * m], n, m),
        }
    }

    pub fn init_bias(&self, n: usize) -> VecDataset<f32> {
        match self {
            InitMethod::Random => {
                VecDataset::from_vec((0..n).map(|_| rand::random::<f32>()).collect(), n, 1)
            }
            InitMethod::FanIn => VecDataset::from_vec(
                (0..n)
                    .map(|_| (2.0 / (n as f32)).sqrt() * rand::random::<f32>() - 1.0 / (n as f32))
                    .collect(),
                n,
                1,
            ),
            InitMethod::Zeros => VecDataset::from_vec(vec![0.0; n], n, 1),
        }
    }
}
