# Install 
```bash
pip install numpy    # Needed for UCRL to install properly in next step
pip install -e .
```

# Generate Learning Curves
The data for each of the three objects in Figure 1 of the paper is stored in a `.npz` file labeled by the object name within the data folder. To rollout algorithms on a given object, simply run the `tools/run_algorithm.py` script with the corresponding data file:
```bash
python tools/run_algorithm.py data/vase.npz
```
This script will save a plot with the reward curves for each policy in the root directory like the ones in Figure 1. The policies and settings (i.e., running with or without toppling or changing the number of rollouts or rollout steps) can be changed in the `cfg/run_algorithm.yaml` file. Note that changing the `plot_ci` argument to 1 will cause plot generation to be much slower, but will include error bands on the learning curves.

# Viewing Stable Poses
Note that this requires OpenGL. Stable poses of a mesh and their respective probabilities can be viewed by running the `tools/view_stable_poses.py`:
```bash
python tools/view_stable_poses.py data/vase.npz
```

# Observations and Grasps
Observations and grasps for each of the objects can be found in each of the object data files as well. While we do not provide a script to inspect them, the reviewers are welcome to open the data file and view them.

# References
The authors appreciate both Ronan Fruit and Jeff Mahler for making their implementations of UCRL and GQCNN available. Included in this code is the UCRL implementation that can be found at https://github.com/RonanFR/UCRL.