from torch import optim

from dynnn.layers.param_rl import RLModel
from dynnn.layers.pinn import DynNN
from dynnn.train.dnn_train import dnn_train


def rl_train(args: dict, data: dict, plot_loss_callback: Callable | None = None):
    """
    Training loop for the RL model
    """
    rl_model = RLModel(args.rl_state_dim, args.rl_param_dim)
    optimizer = optim.Adam(rl_model.parameters(), lr=args.learn_rate)

    for episode in range(num_episodes):
        state = ...  # Initialize the state

        for step in range(max_steps):
            # Choose parameters using the RL model
            params = rl_model(state)

            # Generate dataset using the simulator
            dataset = simulator(params)

            # Train the DNN on the generated dataset
            pinn = DynNN()
            pinn, pinn_stats = dnn_train(args, daata, plot_loss_callback)

            # Evaluate the performance of the DNN
            reward = evaluate_dnn(pinn)

            # Update the RL model based on the reward
            loss = -reward  # Negative reward as the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the state based on the chosen parameters
            state = ...  # Update the state representation
