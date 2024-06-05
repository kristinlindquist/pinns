import torch

from dynnn.layers.simulator import SimulatorModel
from dynnn.layers.pinn import PINN
from dynnn.simulation.mve_ensemble import MveEnsembleMechanics
from dynnn.train.dnn_train import dnn_train
from dynnn.types import ModelArgs


def simulator_train(args: dict, data: dict, plot_loss_callback: Callable | None = None):
    """
    Training loop for the RL model
    """
    sim_model = SimulatorModel(args.rl_state_dim, args.rl_param_dim)
    optimizer = torch.optim.Adam(sim_model.parameters(), lr=args.learn_rate)
    y0, masses = get_initial_conditions(args.n_bodies, args.n_dims)

    possible_params = {
        "dataset_args": {
            "num_samples": 1,
            "test_split": 0.8,
        },
        "trajectory_args": {
            "y0": y0,
            "masses": masses,
            "time_scale": 3,
            "odeint_rtol": 1e-10,
            "odeint_atol": 1e-6,
            "odeint_solver": "tsit5",
        },
        "model_args": {
            "domain": (0, 10),
            "t_span": (0, 100),
            "system_type": "hamiltonian",
        },
    }

    for episode in range(num_episodes):
        state = {}

        for step in range(max_steps):
            # Choose parameters using the RL model
            params = sim_model(state, possible_params)

            # Generate dataset using the simulator
            mechanics = MveEnsembleMechanics(ModelArgs(*params["model_args"]))
            data = mechanics.get_dataset(
                params["dataset_args"],
                params["trajectory_args"],
            )

            # Train the DNN on the generated dataset
            pinn = PINN()
            pinn, pinn_stats = dnn_train(args, data, plot_loss_callback)

            # Evaluate the performance of the DNN
            reward = evaluate_dnn(pinn)

            # Update the RL model based on the reward
            loss = -reward  # Negative reward as the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the state based on the chosen parameters
            state = {}  # Update the state representation
