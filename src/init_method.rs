use crate::datasets::vec_dataset::VecDataset;

pub enum InitMethod {
    Random,
    FanIn,
}

impl InitMethod {
    pub fn init_weights(&self, n: usize, m: usize) -> VecDataset<f32> {
        match self {
            InitMethod::Random => VecDataset::from_vec(
                (0..n * m).map(|_| rand::random::<f32>() - 0.5).collect(),
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
        }
    }

    pub fn init_bias(&self, n: usize) -> VecDataset<f32> {
        match self {
            InitMethod::Random => {
                VecDataset::from_vec((0..n).map(|_| rand::random::<f32>() - 0.5).collect(), n, 1)
            }
            InitMethod::FanIn => VecDataset::from_vec(
                (0..n)
                    .map(|_| (2.0 / (n as f32)).sqrt() * rand::random::<f32>() - 1.0 / (n as f32))
                    .collect(),
                n,
                1,
            ),
        }
    }
}
