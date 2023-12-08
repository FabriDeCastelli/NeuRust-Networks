use crate::activation::Activation;
use crate::datasets::vec_dataset::VecDataset;
use crate::init_method::InitMethod;
use crate::models::layers::layer::Layer;

pub struct Dense {
    input_size: usize,
    output_size: usize,
    activation: Activation,
    weights: VecDataset<f32>,
    bias: VecDataset<f32>,
    input: VecDataset<f32>,
    // net
    output: VecDataset<f32>,
}

impl Layer for Dense {
    fn forward(&mut self, input: &VecDataset<f32>) -> VecDataset<f32> {
        // println!("input dim: {:?}, neuron dim {:?}", input.dim(), self.input_size);
        assert_eq!(
            input.dim(),
            self.input_size,
            "input size must match layer input size"
        );

        self.input = input.clone();
        self.output = input.dot(&self.weights).plus(&self.bias.transpose());
        // println!("input dimensions: {}*{:?}", self.input.len(), self.input.dim());

        // println!("net dimensions: {}*{:?}", self.output.len(), self.output.dim());
        self.activation.forward(&self.output)
    }

    fn backward(&mut self, delta: &VecDataset<f32>) -> VecDataset<f32> {
        assert_eq!(
            delta.dim(),
            self.output_size,
            "delta size must match layer output size"
        );

        /*
        println!("delta dimensions: {}*{:?}", delta.len(), delta.dim());
        println!("weights dimensions: {}*{:?}", self.weights.len(), self.weights.dim());
        println!("output dimensions: {}*{:?}", self.output.len(), self.output.dim());

         */

        let delta = self
            .activation
            .backward(&self.output)
            .element_wise_mul(delta);
        delta.dot(&self.weights.transpose())
    }

    fn update_params(&mut self, delta: &VecDataset<f32>, learning_rate: f32) {
        // println!("delta dimensions: {}*{:?}", delta.len(), delta.dim());
        // println!("bias dimensions: {}*{:?}", self.bias.len(), self.bias.dim());
        let delta_weights = self.input.transpose().dot(delta);
        let delta_bias = delta.sum_axis(0).transpose();

        self.weights = self.weights.minus(&delta_weights.times(learning_rate));
        self.bias = self.bias.minus(&delta_bias.times(learning_rate));
    }

    fn get_weights(&self) -> &VecDataset<f32> {
        &self.weights
    }

    fn get_bias(&self) -> &VecDataset<f32> {
        &self.bias
    }

    fn get_activation(&self) -> &Activation {
        &self.activation
    }

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn get_output_size(&self) -> usize {
        self.output_size
    }
}

impl Dense {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        init: InitMethod,
    ) -> Dense {
        Dense {
            input_size,
            output_size,
            activation,
            weights: init.init_weights(input_size, output_size),
            bias: init.init_bias(output_size),
            input: VecDataset::with_dim(input_size),
            output: VecDataset::with_dim(output_size),
        }
    }
}
