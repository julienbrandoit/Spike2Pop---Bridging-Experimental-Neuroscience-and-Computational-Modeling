# Releases


## Version 1.1.1
- **Release Date**: 2025 August 5
- **Description**: 
1. ***Bug fix***. The seeding of the random number generator was not working correctly. The seed was common accross the processes, leading to the same results for each process. This has been fixed.
2. ***New feature***. In the simulation panel, we can now choose wether to save the full traces or only the spikes. This is done by checking the corresponding checkbox in the simulation panel. The columns of the CSV file are `'simulation_V'` (as before) and `'spiking_times'` (new); such that this is consistent with the input format.
- **Download**: [Google Drive](https://drive.google.com/file/d/1wVXigLndAHXa_hFnm6614b3-cq4UdEe-/view?usp=sharing)

- ***Known issue***: 
1. We are aware of a bug when infering with the DA module *without GPU*. The PyTorch message reports a bad loading of the LoRA. This will be fixed in the next release.

## Version 1.1.0
- **Release Date**: 2025 May 11
- **Description**: Added simulation panel for simulating generated populations and visualizing the results.
- **Download**: *Not available before the version 1.1.1 that patches a key feature.*

## Version 1.0.0
- **Release Date**: 2025 May 7
- **Description**: Initial release of Spike2Pop, a tool for generating spike time populations from recorded sequences.
- **Download**: *Not available before the version 1.1.1 that patches a key feature.*