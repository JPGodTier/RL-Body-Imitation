# RL-Body-Imitation
Reinforcement learning Environment for Body Imitation using Poppy torso

## Getting Started

### Dependencies

Refer to `requirements.txt` for a full list of dependencies.

### Installing

#### For Users:

* To install our project: 

```
git clone https://github.com/JPGodTier/RL-Body-Imitation
cd RL-Body-Imitation
pip install .
```

#### For Developers/Contributors:

If you're planning to contribute or test the latest changes, you should first set up a virtual environment and then install the package in "editable" mode. This allows any changes you make to the source files to immediately affect the installed package without requiring a reinstall.
test
* Clone the repository:

```
git clone https://github.com/JPGodTier/RL-Body-Imitation
cd RL-Body-Imitation
```

* Set up a virtual environment:

```
python3 -m venv rlbi_env
source rlbi_env/bin/activate  # On Windows, use: rlbi_env\Scripts\activate
```

* Install the required dependencies:

```
pip install -r requirements.txt
```

* Install the project in editable mode:

```
pip install -e . 
```

### Executing program

Launch the different model Runners:  
```
python3 bin/Runner.py
```

## Authors

* Paul Aristidou
* Olivier Lapabe-Goastat
* Anatole Reffet

## Version History

* **1.0.0** - Initial release