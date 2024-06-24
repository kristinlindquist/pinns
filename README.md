# PINNs
Various experiments with PINNs for molecular dynamics simulation.
More generally, I'm exploring how to give models intuition based on formal systems with a sort of FEP/active learning cycle. 

### Rough Idea
Attempt to solve a hairy "outer problem" (e.g. protein-protein-interaction) by teaching a model to choose and learn from a large number of physics simulations. In a vaguely FEP sense, have the model attempt to gather more data in areas corresponding to its greatest surprisal. "Gather more data" meaning choosing + parameterizing simulations along with its objective. It will explore parameter space driven by reward based on learning objectives and effort (compute time).
<img width="1346" alt="Screenshot 2024-06-11 at 3 14 53 PM" src="https://github.com/kristinlindquist/pinns/assets/9382486/e62f7ebd-8faa-4ea3-8a83-ba93b0df2f4d">

### Notebooks
[MVE Ensemble with RL simulator param search](src/dynnn/simulation/mve_ensemble/run.ipynb) (WIP!)

![mve_ensemble](https://github.com/kristinlindquist/pinns/assets/9382486/8a06b6df-d560-47a1-9234-5b5d1361d115)


### Resources
- [Steve Brunton's lectures on physics-informed neural networks](https://www.youtube.com/watch?v=JoFW2uSd3Uo&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa)
- [Erik Lindahl's lectures on Molecular Biophysics](https://www.youtube.com/@eriklindahl/playlists)
- [Port-Hamiltonian systems](https://www.math.rug.nl/~arjan/DownloadVarious/PHbook.pdf)
- [Karl Friston (FEP) on Sean Carroll's podcast](https://www.preposterousuniverse.com/podcast/2020/03/09/87-karl-friston-on-brains-predictions-and-free-energy/)


### Related Work
* Jarek Liesen, Chris Lu, Andrei Lupu, Jakob N. Foerster, Henning Sprekeler, Robert T. Lange: “Discovering Minimal Reinforcement Learning Environments”, 2024; [http://arxiv.org/abs/2406.12589 arXiv:2406.12589].

### Acknowledgements
I started by using [Hamiltonian-NN](https://github.com/greydanus/hamiltonian-nn/tree/master) and [Lagrangian-NNs](https://github.com/MilesCranmer/lagrangian_nns) as templates (and evolved from there).
