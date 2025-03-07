# Hopfield Network Project

## Overview
This project implements a Continuous Attractor Neural Network using PyTorch. The Hopfield Network is a form of recurrent artificial neural network that serves as a content-addressable ("associative") memory system with binary threshold nodes. This implementation allows for the simulation of the network's dynamics, energy calculations, and visualization of its behavior over time.

## Project Structure
```
continuous_attractor_project/
├── src/
│   ├── __init__.py
│   ├── network.py           # Defines the HopfieldNetwork class.
│   ├── update_strategies.py # Contains abstract and concrete update strategies.
│   ├── simulator.py         # Contains simulation logic.
│   └── visualization.py     # Contains plotting functions.
├── tests/
│   ├── __init__.py
│   └── test_network.py      # Basic tests for network functionality.
├── main.py                  # Entry point.
└── requirements.txt         # Python package dependencies.
```

## Installation
To set up the project, clone the repository and install the required dependencies. You can do this using pip:

```bash
pip install -r requirements.txt
```

## Usage
To run the Continuous Attractor Network simulation, execute the `main.py` script. You can modify the hyperparameters directly in the `main.py` file.

```bash
python main.py
```

## Hyperparameters
The following hyperparameters can be adjusted in `main.py`:

- `NUM_NEURONS`: Number of neurons in the network.
- `NOISE`: Noise (temperature) parameter.
- `SYN_FAIL`: Synaptic failure parameter.
- `SPON_REL`: Spontaneous release parameter.
- `FIELD_WIDTH`: Receptive field width fraction.
- `FRACTION_ACTIVE`: Fraction of neurons that are active.
- `CONSTRICT`: Constriction term scaling factor.
- `I_DIR`: Directional stimulus reference index.
- `I_STR`: Stimulus strength.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.