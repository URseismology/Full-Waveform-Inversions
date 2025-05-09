Note: This file defines the component of the Full Waveform Inversion in Timedomain with Jax


## Project Description 
This project demonstrate the Power of JAX in performing full waveform inversion (FWI) in the time domain. The project primary aim is to reconstruct a medical image (spatial strcuture of tissues involving a breast in the presence of  malignant tumor). The project uses channel data recorded (sound pulse transmitted and received) across 256 elements in a Ring Array structure surrounding the specimen under observation. It then uses the FWI Time Domain technique to reconstruct the spatial structure of the tissue. 

This project was developed as a part of the Inverse Problem in Imaging Cource (ECE485) taught by Dr. Rehman Ali at the University of Rochester.


## Requirements 
The following libraries will be required to run the codes:
    > jax, jaxopt, matplotlib, PIL, interpax, numpy, scipy, math, h5py, time, glob, tqdm


## Folders/Files 
The following details out the information about the folder/files present in this document.

**Root Folder: TimeDomainFWI_JAX**
Note: it is important to cd to this directory and source python from here before running all subsequent codes discussed below.
|--

**Folder 1: data (data source)** <br/> 
data<br/> 
    |-- breast_ct.jpg (sample of image the original breast cancer)<br/> 
    |-- recordedData.mat (the recorded channel data across sensors that needs to be inverted)<br/> 


**Folder 2: utils (core computation functions/libraries)** <br/> 
utils <br/> 
    |-- coreFWI.py (the core functions needed to run the Time Domain FWI)<br/> 
    |-- gaussPulse.py (function to generate a source pulse for waveform simulation)<br/> 
    |-- getSoundSpeed.py (empiracally construct the original velocity map of sound across the breast cancer tissue this data can be used to synthetically produce channel data that is in-practice recorded during a medical imaging)<br/> 
    |-- sample_circle.py (generates the ring array strucutre and records the position of each sensor element across 2D space)<br/> 
    |-- visuals.py (the function to generates gifs images for all simulations)<br/> 


**Folder 3: results (resulting images of simulations. mainly used ot build gifs)** <br/> 
results<br/> 
    |-- 4wd_*** (contains the images of forward waveform simulations, recorded channel data simulations)<br/> 
    |-- lbfgs_iterations (contains the images of the intermediate steps during the optimization step of lbfgs solver)<br/> 
    |-- sensor_locations (contains images of how transmitters and receivers change over time)<br/> 
Note: Folder 3 (lbfgs_iterations) is particularly important as it is used to monitor the intermediate steps of the solver. Rather than visualizing the results directly in the terminal/notebook, images are generated/saved for easier analysis and tracking.<br/> 


**Folder 4: models (to save initial tree strctures of solver)** <br/> 
models<br/> 
    |-- empty<br/> 
Note: This folder for now does not have anything meanigful. But it is created to particularly save init_state results of the solver particularly for long running processes. Since the init_state steps take the longest time to build the tree structure of the jit therefore saving it can reduce time wastage for re-running the solvers.<br/> 


**Folder 5: examples (contains some example notebook files)** <br/> 
examples<br/> 
    |-- visualizeSoundSpeed.ipynb (notebook using the functions of getSoundSpeed.py to construct velocity map of sound while passing through breast tissues)<br/> 
    |-- visualizeTimeDomainSimulation.ipynb (notebook that performes the time stepping algorithm to simulate waveform propagation sourced by a gaussian pulse and simulate recordings of the propagation at the receiver locations)<br/> 


### Code Entry Point 
**Main Files that Run the Optmization (codes that run the inversion)**<br/> 
    |-- timedomainFWI_LBFGS.ipynb (give run all from a notebook file to generate inversion results using lbfgs solver. This code does not look into each step of the solver rather uses solver.run and waits for it to finish)<br/> 
    |-- timedomainFWI_LBFGS_stepiter.ipynb (give run all from a notebook file to generate inversion results using lbfgs solver. This file monitors each step while lbfgs uses solver.update to update the gradients)<br/> 
    |-- timedomainFWI_LCG.ipynb (give run all from a notebook file to generate inversion results using LinearizedCG  solver. This code does not look into each step of the solver rather uses solver.run and waits for it to finish)<br/> 


### Contact 
Please reach out to Sayan Swar (sswar@ur.rochester.edu) for any questions/suggestions regarding this approch.

Thank You<br/> 
**Sayan Kr. Swar**<br/> 
**Earth Imaging: Signals and Algorithms Lab**<br/> 
Geoscience Dept, ECE Department<br/> 
University of Rochester<br/> 
