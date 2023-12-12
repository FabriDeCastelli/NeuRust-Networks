use crate::activation::Activation;
use crate::datasets::vec_dataset::VecDataset;

pub trait Layer {
    fn forward(&mut self, input: &VecDataset<f32>) -> VecDataset<f32>;
    fn backward(&mut self, input: &VecDataset<f32>) -> VecDataset<f32>;
    fn update_params(&mut self, delta: &VecDataset<f32>, lr: f32);
    fn get_weights(&self) -> &VecDataset<f32>;
    fn get_bias(&self) -> &VecDataset<f32>;
    fn get_activation(&self) -> &Activation;
    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;
    fn print_params(&self);
}
