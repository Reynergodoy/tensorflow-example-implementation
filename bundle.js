const model = tf.sequential(); // initialize the neural network sequential model

function parameters (inputShape, kernelSize, filters, strides, activation, kernelInitializer) {
     return { // adds a convolution
        inputShape,
        kernelSize,
        filters,
        strides,
        activation,
        kernelInitializer
     }
}

function addConvLayer (model, params) {
    if (params.inputShape === 0) {
        delete params.inputShape;
    }
    model.add(tf.layers.conv2d(params)); // adds a convolution layer
}

function addPoolingLayer (model, poolSize, strides) {
    model.add(tf.layers.maxPooling2d({
        poolSize,
        strides
    }));
}

function flatten (model) {
    model.add(tf.layers.flatten());
}

function addDenseLayer (model, units, kernelInitializer, activation) { // adds a fully connected layer / dense layer
    model.add(tf.layers.dense({
        units,
        kernelInitializer,
        activation
    }));
}

function compileModel (model, optimizer, loss, metrics) {
   model.compile({
      optimizer,
      loss,
      metrics
  });
}

addConvLayer(model, parameters([28, 28, 1], 5, 8, 1, 'relu', 'VarianceScaling'));

addPoolingLayer(model, [2,2], [2,2]);

addConvLayer(model, parameters(0, 5, 16, 1, 'relu', 'VarianceScaling'))

addPoolingLayer(model, [2,2], [2,2]);

flatten(model);

/* Adds a fully connected layer with softmax activation function and a variance
scaling initialization */
addDenseLayer(model, 36, 'VarianceScaling', 'softmax');

/* Defining an optimizer */
const optimizer = tf.train.sgd(0.15); // utilizes a stochastic gradient descent optimizer, also known as SGD optimizer

/* Compile the model */
compileModel(model, optimizer, 'categoricalCrossentropy', ['accuracy']);

     
