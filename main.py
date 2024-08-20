import time
import warnings
import os
import numpy as np
import torch
import torch.nn as nn
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from typing import Optional, Tuple
from torch import Tensor
from torch.nn.utils import weight_norm as wn
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch import fit_gpytorch_mll

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex
import argparse
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


def parse(file):
    with open(f'tcnn_data/{file}') as f:
        x = []
        for i in range(12):
            x.append(f.readline().split())

    x = np.array(x)
    x = x.astype('float32')
    return x

target = []
for file in os.listdir('tcnn_data'):
  if file[0] == 'f' and file[-1] == 't':
    target.append(parse(file))
  # target.append(extract_features(f'tcnn_images/{file}'))
target = np.array(target)

from random import shuffle
image_1d = target.reshape(-1,192)

# prepare input and target arrays

def prepare_data(seq_len):
  X = []
  y = []

  seq_len = 16
  for i in range(target.shape[0]-seq_len):
      X.append(image_1d[i:i+seq_len,:])
      y.append(target[i+1:i+seq_len+1,:,:])

  X = np.array(X)
  # X = X.reshape(X.shape[0], seq_len, target.shape[1], target.shape[2], 1)
  y = np.array(y)
  y = y.reshape(X.shape[0], seq_len,target.shape[1], target.shape[2])

  indices = [i for i in range(len(X))]
  shuffle(indices)
  X = X[indices]
  y = y[indices]

  train_size = int(X.shape[0]*0.7)
  val_size = int(X.shape[0]*0.9)
  trainX, trainY = X[:train_size], y[:train_size]
  valX, valY = X[train_size:val_size], y[train_size:val_size]
  testX, testY = X[val_size:], y[val_size:]
  return trainX, trainY, valX, valY, testX, testY

trainX, trainY, valX, valY, testX, testY = prepare_data(16)

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True, padding=0):
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias)

        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(CausalConv1d, self).forward(x)

class ResBlock(nn.Module):
    def __init__(self, num_layers=3, dilations=[1,2,4], dropout_prob=0.25, filters=64, padding='causal', input=False):
        super(ResBlock, self).__init__()

        self.conv = nn.Conv1d
        # self.conv = CausalConv1d
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.filters = filters
        self.padding = 'same' if padding == 'causal' else padding
        self.num_layers = num_layers
        self.dilations = dilations

        # Assert whether the number of layers and the len of dilations match or not
        assert num_layers == len(dilations), "Dilations undefined for some layers"

        self.layers = nn.ModuleList()
        self.start = 192 if input else self.filters
        self.make_layers()


        self.recon_conv = self.conv(self.start, filters, kernel_size=1)

    def make_layers(self):
      self.layers.extend([
                wn(self.conv(self.start, self.filters, kernel_size=2, dilation=self.dilations[0],
                          padding=self.padding)),
                self.relu,
                self.dropout
            ])
      for i in range(1,self.num_layers):
        self.layers.extend([
                wn(self.conv(self.filters, self.filters, kernel_size=2, dilation=self.dilations[i],
                          padding=self.padding)),
                self.relu,
                self.dropout
            ])

    def forward(self, x):
        assert len(self.layers) > 0, "Call make_layers function before forward pass"

        input_tensor = x
        for i, layer in enumerate(self.layers):
          # print("hello", i, len(self.layers), layer)
          if isinstance(layer, nn.Dropout):
            x = layer(x)
            continue
          x = layer(x)

        input1 = self.recon_conv(input_tensor)
        input1 = self.relu(x + input1)
        x = input1

        return x


# class ResBlock(nn.Module):
#     def __init__(self, num_layers=3, dilations=[1,2,4], dropout_prob=0.25, filters=64):
#         super(ResBlock, self).__init__()

#         self.conv = CausalConv1d
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_prob)
#         self.filters = filters
#         self.num_layers = num_layers
#         self.dilations = dilations

#         assert num_layers == len(dilations), "Dilations undefined for some layers"

#         self.layers = nn.ModuleList()
#         self.make_layers()

#         self.recon_conv = self.conv(filters, filters, kernel_size=1)

#     def make_layers(self):
#         for i in range(self.num_layers):
#             self.layers.extend([
#                 self.conv(self.filters, self.filters, kernel_size=2, dilation=self.dilations[i]),
#                 self.relu,
#                 self.dropout
#             ])

#     def forward(self, x):
#         assert len(self.layers) > 0, "Call make_layers function before forward pass"

#         input_tensor = x
#         for layer in self.layers:
#             if isinstance(layer, nn.Dropout):
#                 x = layer(x)
#                 continue
#             x = layer(x)

#         input1 = self.recon_conv(input_tensor)
#         input1 = self.relu(x + input1)
#         x = input1

#         return x

res = ResBlock(input=True)
res.layers, res(torch.tensor(trainX[:2].reshape(-1, 192, 16))).shape

class TCNN(nn.Module):
    def __init__(self, num_blocks=3, num_layers=3, dilations=None, dropout_prob=0.25, filters=64, padding='causal'):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.num_layers = int(num_layers)
        self.dilations = [2**i for i in range(int(num_layers))] if dilations is None else dilations
        self.dropout_prob = dropout_prob
        self.filters = int(filters)**2
        self.padding = padding

        self.blocks = nn.ModuleList()
        self.make_model()

        self.ffn = nn.Linear(self.filters, 192)
        self.softmax = nn.Softmax(dim=-1)

    def make_model(self):
        block = ResBlock(self.num_layers, self.dilations, self.dropout_prob, self.filters, self.padding, input=True)
        self.blocks.append(block)
        for i in range(1, self.num_blocks):
            block = ResBlock(self.num_layers, self.dilations, self.dropout_prob, self.filters, self.padding)
            self.blocks.append(block)

    def forward(self, x):
        batch_size = x.shape[0]

        for block in self.blocks:
            # print('kkkdk')
            x = block(x)

        x = x.reshape(batch_size, -1, self.filters)
        x = self.ffn(x)
        x = self.softmax(x)

        seq_len = x.shape[1]
        output = x.view(batch_size, seq_len, 12, 16)

        return output

sample = TCNN(num_blocks=5, filters=6, num_layers=4, dilations=[1,2,4,8])
output = sample(torch.tensor(trainX.reshape(-1, 192, 16)))
output.shape

### setup our problem for TCNN
### minimise the number of params and the error for the prediction

def train_and_evaluate(num_blocks, filters, num_layers, learning_rate, decay_factor, step_size):
  # num_blocks, filters, num_layers, step_size =
  model = TCNN(num_blocks=num_blocks, filters=filters, num_layers=num_layers)
  # Assuming 'model' is your PyTorch model
  model.train()

  trainX, trainY, valX, valY, testX, testY = prepare_data(16)

  # Convert data to PyTorch tensors
  trainX = torch.tensor(trainX, dtype=torch.float32)
  trainY = torch.tensor(trainY, dtype=torch.float32)
  valX = torch.tensor(valX, dtype=torch.float32)
  valY = torch.tensor(valY, dtype=torch.float32)

  # Create DataLoaders
  train_dataset = TensorDataset(trainX, trainY)
  val_dataset = TensorDataset(valX, valY)
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

  # Define loss function and optimizer
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=7e-4)


  def step_decay_schedule(optimizer, initial_lr=7e-4, decay_factor=0.9, step_size=76):
      def schedule(epoch):
          return decay_factor ** np.floor(epoch / step_size)

      return LambdaLR(optimizer, lr_lambda=schedule)

  scheduler = step_decay_schedule(optimizer, initial_lr=learning_rate, decay_factor=decay_factor, step_size=step_size)

  # Training loop
  num_epochs = 800
  for epoch in range(num_epochs):
      model.train()
      train_loss = 0.0
      for batch_x, batch_y in train_loader:
          optimizer.zero_grad()
          batch_x = batch_x.reshape(-1, 192, 16)
          outputs = model(batch_x)
          loss = criterion(outputs, batch_y)
          loss.backward()
          optimizer.step()
          train_loss += loss.item()

      # Validation
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for batch_x, batch_y in val_loader:
              batch_x = batch_x.reshape(-1, 192, 16)
              outputs = model(batch_x)
              loss = criterion(outputs, batch_y)
              val_loss += loss.item()

      # Print metrics
      # print(f"Epoch {epoch+1}/{num_epochs}")
      # print(f"Train loss: {train_loss/len(train_loader)}")
      # print(f"Val loss: {val_loss/len(val_loader)}")

      # Step the scheduler
      scheduler.step()

  # After training, set the model to evaluation mode
  model.eval()

  model_size = sum(p.numel() for p in model.parameters())

  return torch.tensor([val_loss, model_size])


class TCNNProblem(MultiObjectiveTestProblem):
    r"""Two objective problem for image classification.
    Objectives:
    1. Number of parameters (to be minimized)
    2. Model accuracy (to be maximized)

    Hyperparameters:
    1. Learning rate (log scale)
    2. Number of filters
    3. Number of layers
    """
    dim = 6 # the number of inputs needed to compute the objectives
    num_objectives = 2
    _bounds = [(2, 5), (3, 6), (2, 5), (5e-4,  5e-3), (0.75, 0.9), (40, 60)] # num_blocks, filters, num_layers, learning_rate, decay_factor, step_size
    _ref_point = [1e7, 0.5]  # 10M parameters, 50% accuracy

    def __init__(
        self,
        noise_std: Optional[Tensor] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @staticmethod
    def _compute_objectives(X) -> Tensor:
        num_params = []
        errors = []

        for num_blocks, filters, num_layers, learning_rate, decay_factor, step_size in X:
          # Dummy calculations - replace these with your actual model training and evaluation
          error, params = train_and_evaluate(num_blocks.item(), filters.item(), num_layers.item(), learning_rate.item(), decay_factor.item(), step_size.item())
          errors.append(error)
          num_params.append(params)

        errors = torch.tensor(errors)
        num_params = torch.tensor(num_params)

        return torch.stack([-num_params, -errors], dim=-1)

    def evaluate_true(self, X):
        # print("hello")
        return self._compute_objectives(X)

# num_blocks, filters, num_layers, learning_rate, decay_factor, step_size
problem = TCNNProblem()
X = torch.tensor([[3, 4, 3, 0.001, 0.8, 50]])
# problem(X)

from botorch.utils.sampling import draw_sobol_samples


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_initial_data(n=6):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1).to(device)
    print(train_x)
    train_obj_true = problem(train_x).to(device)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * 0*0
    return train_x, train_obj, train_obj_true

train_x, train_obj, train_obj_true = generate_initial_data(n=2 * (problem.dim + 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_x = torch.load('drive/MyDrive/multi_objective/windpmf_train_x.pt', map_location=device)
train_obj = torch.load('drive/MyDrive/multi_objective/windpmf_train_obj.pt', map_location=device).to(torch.float32)
train_obj_true = torch.load('drive/MyDrive/multi_objective/windpmf_train_obj_true.pt', map_location=device).to(torch.float32)

train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = (train_x.to(torch.float32), train_obj.to(torch.float32), train_obj_true.to(torch.float32))



SMOKE_TEST = os.environ.get("SMOKE_TEST")
MC_SAMPLES = 128 if not SMOKE_TEST else 16

tkwargs = {
    "dtype": torch.float32,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

BATCH_SIZE = 2
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
verbose = True

standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1

def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model.to(torch.float32),
        ref_point=problem.ref_point.to(torch.float32).tolist(),  # use known reference point
        X_baseline=normalize(train_x.to(device), problem.bounds.to(device).to(torch.float32)),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds.to(device)).to(device)
    new_obj_true = problem(new_x).to(device)
    new_obj = new_obj_true + torch.randn_like(new_obj_true).to(device) * 0
    return new_x, new_obj, new_obj_true

def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    train_x = normalize(train_x.to(device), problem.bounds.to(device)).to(device).to(torch.float32)
    models = []
    train_obj = train_obj.to(device)
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_yvar = torch.full_like(train_y,0  ** 2)
        models.append(
            FixedNoiseGP(
                train_x, train_y, train_yvar.to(device), outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll.to(device), model.to(device)

# mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi.to(device), train_obj_qnehvi.to(device))

# compute hypervolume
bd = DominatedPartitioning(ref_point=problem.ref_point.to(device), Y=train_obj_true.to(device)).to(device)
volume = bd.compute_hypervolume().item()

hvs_qnehvi = []
hvs_qnehvi.append(volume)
print(hvs_qnehvi, type(hvs_qnehvi))

# run N_BATCH rounds of BayesOpt after the initial random batch
N_BATCH = 3
for iteration in range(1, N_BATCH + 1):

    t0 = time.monotonic()

    # fit the models
    fit_gpytorch_mll(mll_qnehvi)

    # define the qEI and qNEI acquisition modules using a QMC sampler
    qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])).to(torch.float32)

    # optimize acquisition functions and get new observations
    (
        new_x_qnehvi,
        new_obj_qnehvi,
        new_obj_true_qnehvi,
    ) = optimize_qnehvi_and_get_observation(
        model_qnehvi, train_x_qnehvi.to(torch.float32), train_obj_qnehvi.to(torch.float32), qnehvi_sampler.to(torch.float32)
    )
    # update training points
    train_x_qnehvi = torch.cat([train_x_qnehvi.to(device), new_x_qnehvi])
    train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
    train_obj_true_qnehvi = torch.cat([train_obj_true_qnehvi, new_obj_true_qnehvi])

    bd = DominatedPartitioning(ref_point=problem.ref_point.to(device), Y=train_obj_true_qnehvi.to(device)).to(device)
    volume = bd.compute_hypervolume().item()
    hvs_qnehvi.append(volume)

    # reinitialize the models so they are ready for fitting on next iteration
    # Note: we find improved performance from not warm starting the model hyperparameters
    # using the hyperparameters from the previous iteration
    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_true_qnehvi)

    t1 = time.monotonic()

    if verbose:
        print(
            f"\nBatch {iteration:>2}: Hypervolume (NEHVI) = "
            f"(8{hvs_qnehvi[-1]:>4.2f}), "
            f"time = {t1-t0:>4.2f}.",
            end="",
        )
    else:
        print(".", end="")

def plot():
    fig, ax = plt.subplots(figsize=(10, 7))
    algo = "qNEHVI"
    cm = plt.get_cmap("viridis")

    batch_number = torch.cat(
        [
            torch.zeros(2 * (problem.dim + 1)),
            torch.arange(1, 5 + 1 + 3 + 3).repeat(BATCH_SIZE, 1).t().reshape(-1),
        ]
    ).numpy()

    sc = ax.scatter(
        train_obj_true_qnehvi[:, 0].cpu().numpy(),
        train_obj_true_qnehvi[:, 1].cpu().numpy(),
        c=batch_number,
        alpha=0.8,
    )
    ax.set_title(algo)
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")

    norm = plt.Normalize(batch_number.min(), batch_number.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.set_title("Iteration")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-objective optimization for image classification")

    parser.add_argument("--load_data", type=int, default=1, help="1 if we need to load the initial data")
    # parser.add_argument("--batch_size", type=int, default=4, help="Batch size for optimization")
    # parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    # parser.add_argument("--output_file", type=str, default="results.txt", help="File to save optimization results")
    args = parser.parse_args()

    # Use the parsed arguments
    num_iterations = args.num_iterations
    batch_size = args.batch_size
    verbose = args.verbose
    output_file = args.output_file

    # Run your main optimization loop here, using the parsed arguments
    # For example:
    # run_optimization(num_iterations=num_iterations, batch_size=batch_size, verbose=verbose, output_file=output_file)