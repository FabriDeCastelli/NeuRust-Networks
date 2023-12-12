use std::convert::identity;
use crate::datasets::vec_dataset::VecDataset;

#[derive(Debug, Clone)]
pub enum Activation {
    Identity,
    ReLU,
    Sigmoid,
    Tanh,
}

impl Activation {
    pub fn forward(&self, input: &VecDataset<f32>) -> VecDataset<f32> {
        VecDataset::from_vec(
            input
                .as_ref()
                .iter()
                .map(|&x| match self {
                    Activation::Identity => identity(x),
                    Activation::ReLU => relu(x),
                    Activation::Sigmoid => sigmoid(x),
                    Activation::Tanh => tanh(x),
                })
                .collect(),
            input.len(),
            input.dim(),
        )
    }

    pub fn backward(&self, input: &VecDataset<f32>) -> VecDataset<f32> {
        VecDataset::from_vec(
            input
                .as_ref()
                .iter()
                .map(|&x| match self {
                    Activation::Identity => 1.0,
                    Activation::ReLU => relu_prime(x),
                    Activation::Sigmoid => sigmoid_prime(x),
                    Activation::Tanh => tanh_prime(x),
                })
                .collect(),
            input.len(),
            input.dim(),
        )
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[inline]
fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[inline]
fn tanh_prime(x: f32) -> f32 {
    1.0 - x.tanh().powi(2)
}

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn relu_prime(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}
