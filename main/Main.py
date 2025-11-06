from data import dataset_manager
from optimizers import Adam
from network import NeuralNetwork
from src.neural_network.layers.layer import Sigmoid, WeightInitializers
from main.losses import LossFunction
from sklearn.preprocessing import OneHotEncoder

# ------------- GLOBAL VARIABLES ------------
dataset_paths = {
    "mnist": "../data/datasets/mnist.npz",
    "iris": "../data/datasets/iris.npz",
}
# ------------- GLOBAL VARIABLES ------------


# ------------- NEURAL NETWORK HYPERPARAMETERS ------------

epochs = 50
batch_size = 16
learning_rate = 0.01

# ------------- NEURAL NETWORK HYPERPARAMETERS ------------


#dataset_manager.download_iris_dataset()
#dataset_manager.download_mnist_dataset()
X_train, y_train, X_val, y_val, X_test, y_test = dataset_manager.split_and_preprocess_dataset(dataset_paths["iris"], random_seed=1)

# Transpose X
X_train = X_train.T  # shape: (4, 105)
X_val = X_val.T
X_test = X_test.T

# One-hot encode y
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1, 1)).T  # shape: (3, 105)
y_val = encoder.transform(y_val.reshape(-1, 1)).T
y_test = encoder.transform(y_test.reshape(-1, 1)).T


nn = NeuralNetwork(
    input_size=4,
    layers_num_neurons=[8, 6, 3],
    layers_activation_functions=[Sigmoid(), Sigmoid(), Sigmoid()],
    layers_initializers=[WeightInitializers.he, WeightInitializers.glorot, WeightInitializers.glorot],
    optimizer=Adam(learning_rate=learning_rate),
    loss_function=LossFunction.mse
)

print(nn.__repr__())

print("Training...")
nn.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
print("[OK] Training finished")
nn.plot_losses()


