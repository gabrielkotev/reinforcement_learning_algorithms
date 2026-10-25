# Reinforcement Learning Algorithms

Implementations of reinforcement learning algorithms that go with the tutorials on my blog, [TATIKA – AI Blog](https://tatika.pythonanywhere.com/).
Videos of the trained agents are on my [YouTube channel](https://www.youtube.com/@gabrielkotev).

## Contents

| Folder | Algorithm | Environment | Tutorial |
|---|---|---|---|
| [ars/](ars/) | Augmented Random Search (multithreaded) | Gym | [ARS](https://tatika.pythonanywhere.com/post_group/8) |
| [dqn/](dqn/) | Deep Q-Network | CartPole | [DQN](https://tatika.pythonanywhere.com/post_group/10) |
| [evolution/](evolution/) | Genetic algorithm (neuroevolution) | Gym continuous control | [RL with genetic algorithms](https://tatika.pythonanywhere.com/post_group/11) |
| [reinforce/](reinforce/) | REINFORCE with baseline | Acrobot | [REINFORCE to Actor-Critic](https://tatika.pythonanywhere.com/post_group/12) |
| [actor_critic/](actor_critic/) | Actor-Critic | Acrobot | [REINFORCE to Actor-Critic](https://tatika.pythonanywhere.com/post_group/12) |
| [ars_hexapod_coppeliasim/](ars_hexapod_coppeliasim/) | ARS for hexapod walking (4- and 6-legged) | CoppeliaSim | – |

## Requirements

- Python 3, NumPy, Matplotlib
- PyTorch (DQN, genetic algorithm, REINFORCE, Actor-Critic)
- `gym` (ARS, DQN, genetic algorithm) or `gymnasium` (REINFORCE, Actor-Critic)
- Hexapod: [CoppeliaSim](https://www.coppeliarobotics.com/), `coppeliasim-zmqremoteapi-client`, `ray`, `mss`, `ffmpeg`

## Usage

Each algorithm is a single script. Run it directly, for example:

```bash
python dqn/cart_pole_dqn.py
```

For the hexapod, set `<coppeliasim_path>` in `hexapod_walk_ars_4l.py` / `hexapod_walk_ars_6l.py` to your CoppeliaSim install and run the script. It starts the simulator instances in parallel and loads the scenes from `ars_hexapod_coppeliasim/scenes/`. Recording video needs `ffmpeg`.

## License

See [LICENSE](LICENSE).
