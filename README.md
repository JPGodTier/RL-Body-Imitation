# RL-Body-Imitation
Reinforcement learning Environment for Body Imitation using Poppy torso

## Project report

### Goal

The goal of this project is to have a Poppy Torso robot imitating the movements made on the provided video.

### Code structure and main files

To achieve this, we have trained a PPO model using the stablebaselines3 library.
You can see in the code of the repository the different scripts. Below is a short description of the main ones:
* [Runner.py](./bin/Runner.py) is the script that is executed to learn and test our model.
* [PPO_inference.py](./bin/PPO_inference.py) is the script that is executed to infer our model and plot the positions (x,y) of both hands in the video versus the ground truths.
* [PoppyTorsoEnv.py](./src/Poppy/PoppyTorsoEnv.py) is the gym environment that we created for the project, including the step and the calculate_reward functions (among others).
* [PoppyChannel.py](./src/CoppeliaComs/PoppyChannel.py) manages the interface with coppeliaSim
* The [Tests](./Tests) folder would include the Tensorboard logs and the saved model

### Results

After launching the training of the RL PPO model, we see the robot moving while it is learning. (see [Training_recording.mov](./Training_recording.mov))
Unfortunately, the API with copeliaSim seems very slow, and it is doing approximately 1 step per second during training. We have tried to adapt directly the copeliaSim API to increase the speed of each step simulation, but it unfortunately did not work.
To train properly the model, we would need several 100k of steps, but we are here very much limited.

Nevertheless, we have launched the model training during 24 hours. Please see below the results of the training:
<div align="center">
    <img src="https://github.com/JPGodTier/RL-Body-Imitation/blob/main/Results/Training_ep_rew_mean.png" width="50%" height="auto">
    <p>Evolution of the mean reward per episode during training</p>
</div>

We can see the reward properly increasing along the training, which show that the robot learns increasingly better to imitate the video.

We also had prepared a script to plot the movements of the robot imitating the video, but we were not able to run it (as we were not able to fully train our PPO model, as the copeliaSim API is too slow).

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
