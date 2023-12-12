use crate::activation::Activation;
use crate::datasets::vec_dataset::VecDataset;
use crate::init_method::InitMethod;
use crate::models::layers::layer::Layer;

#[derive(Debug, Clone)]
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
        self.output = self.activation.forward(&input.dot(&self.weights).plus(&self.bias.transpose()));
        self.output.clone()
    }

    fn backward(&mut self, delta: &VecDataset<f32>) -> VecDataset<f32> {
        assert_eq!(
            delta.dim(),
            self.output_size,
            "delta size must match layer output size"
        );

        self.activation
            .backward(&self.output)
            .element_wise_mul(delta)
            .dot(&self.weights.transpose())
    }

    fn update_params(&mut self, delta: &VecDataset<f32>, lr: f32) {
        let grad_weights = self.input.transpose().dot(delta);
        let grad_bias = delta.sum_axis(0).transpose();


        self.weights = self.weights.minus(&grad_weights.times(lr));
        self.bias = self.bias.minus(&grad_bias.times(lr));
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

    fn print_params(&self) {
        println!("Weights:  {:?}", self.get_weights().data());
        println!("Bias: {:?}", self.get_bias().data());
    }
}

impl Dense {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        weights_initializer: InitMethod,
        bias_initializer: InitMethod,
    ) -> Dense {
        Dense {
            input_size,
            output_size,
            activation,
            weights: weights_initializer.init_weights(input_size, output_size),
            bias: bias_initializer.init_bias(output_size),
            input: VecDataset::with_dim(input_size),
            output: VecDataset::with_dim(output_size),
        }
    }
}

pub struct DenseBuilder {
    input_size: usize,
    output_size: usize,
    activation: Activation,
    weights_initializer: InitMethod,
    bias_initializer: InitMethod,
}

impl DenseBuilder {
    pub fn new() -> DenseBuilder {
        DenseBuilder {
            input_size: 1,
            output_size: 1,
            activation: Activation::Identity,
            weights_initializer: InitMethod::Random,
            bias_initializer: InitMethod::Zeros,
        }
    }

    pub fn input_size(mut self, input_size: usize) -> DenseBuilder {
        self.input_size = input_size;
        self
    }

    pub fn output_size(mut self, output_size: usize) -> DenseBuilder {
        self.output_size = output_size;
        self
    }

    pub fn activation(mut self, activation: Activation) -> DenseBuilder {
        self.activation = activation;
        self
    }

    pub fn weights_initializer(mut self, weight_initializer: InitMethod) -> DenseBuilder {
        self.weights_initializer = weight_initializer;
        self
    }

    pub fn bias_initializer(mut self, bias_initializer: InitMethod) -> DenseBuilder {
        self.bias_initializer = bias_initializer;
        self
    }

    pub fn build(self) -> Dense {
        Dense::new(
            self.input_size,
            self.output_size,
            self.activation,
            self.weights_initializer,
            self.bias_initializer,
        )
    }
}
