use crate::dataset::Dataset;
use crate::datasets::vec_dataset::VecDataset;
use crate::loss::Loss;
use crate::metric::Metric;
use crate::models::layers::layer::Layer;
use crate::optimizer::Optimizer;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Optimizer,
    loss: Loss,
    metrics: Vec<Metric>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential {
            layers: Vec::new(),
            optimizer: Optimizer::SGD,
            loss: Loss::MeanSquaredError,
            metrics: vec![],
        }
    }

    fn set_optimizer(&mut self, optimizer: Optimizer) {
        self.optimizer = optimizer;
    }

    fn set_loss(&mut self, loss: Loss) {
        self.loss = loss;
    }

    fn set_metrics(&mut self, metrics: Vec<Metric>) {
        self.metrics = metrics;
    }

    pub fn get_layers(&self) -> &Vec<Box<dyn Layer>> {
        &self.layers
    }

    pub fn add(&mut self, layer: impl Layer + 'static) {
        self.layers.push(Box::new(layer));
    }

    pub fn compile(&mut self, optimizer: Optimizer, loss: Loss, metrics: Vec<Metric>) {
        self.set_loss(loss);
        self.set_optimizer(optimizer);
        self.set_metrics(metrics);
    }

    pub fn fit(
        &mut self,
        x: VecDataset<f32>,
        y: VecDataset<f32>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        verbose: bool,
    ) {
        assert_eq!(
            x.len(),
            y.len(),
            "training data and labels must have the same number of rows"
        );

        let input = &x;

        for epoch in 0..epochs {
            for (data, labels) in input.iter_batch(batch_size).zip(y.iter_batch(20)) {
                let vec_data = VecDataset::from_vec(data.to_vec(), batch_size, 1);
                let vec_labels = VecDataset::from_vec(labels.to_vec(), batch_size, 1);
                self.train_one_step(vec_data, vec_labels, learning_rate);
            }

            if verbose && epoch % 100 == 0 {
                println!(
                    "Epoch: {}  -  Loss : {}",
                    epoch,
                    self.evaluate(x.clone(), y.clone())
                );
            }
        }
    }

    fn train_one_step(&mut self, x: VecDataset<f32>, y: VecDataset<f32>, learning_rate: f32) {
        let mut input = x;

        for layer in self.layers.iter_mut() {
            input = layer.forward(&input);
        }

        let mut delta = self.loss.backward(&input, &y);

        for layer in self.layers.iter_mut().rev() {
            layer.update_params(&delta, learning_rate);
            delta = layer.backward(&delta);
        }
    }

    pub fn predict(&self, _: Vec<f32>) -> Vec<f32> {
        todo!()
    }

    pub fn evaluate(&mut self, x_test: VecDataset<f32>, y_test: VecDataset<f32>) -> f32 {
        let mut input = x_test;
        for layer in self.layers.iter_mut() {
            input = layer.forward(&input);
        }

        self.loss.forward(&input, &y_test)
    }

    pub fn summary(&self) {
        println!("Model Summary");
        println!("-------------");
        println!("Optimizer: {:?}", self.optimizer);
        println!("Loss: {:?}", self.loss);
        println!("Metrics: {:?}", self.metrics);
        println!("-------------Layers-------------");
        // Print weights and biases for each layer
        for (i, layer) in self.layers.iter().enumerate() {
            println!("Layer {}", i + 1);
            println!("Input Neurons: {:?}", layer.get_input_size());
            println!("Output Neurons: {:?}", layer.get_output_size());
            println!(
                "Weights Dimension: {:?} * {:?}",
                layer.get_weights().len(),
                layer.get_weights().dim()
            );
            println!(
                "Biases Dimension: {:?} * {:?}",
                layer.get_bias().len(),
                layer.get_bias().dim()
            );
            println!("Activation: {:?}", layer.get_activation());
            println!("  ");
        }
    }

    pub fn print_params(&self) {
        for layer in self.layers.iter() {
            layer.print_params()
        }
    }
}
