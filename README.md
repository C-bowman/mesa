# MESA

MESA is a python package for optimizing simulation results. There
are two simulations available: SOLPS and VSim.  For SOLPS, MESA matches 
experiment and SOLPS autonomously.  It uses Gaussian-process optimization
 provided by [inference-tools](https://github.com/C-bowman/inference-tools)
to maximise agreement between experimental data and SOLPS-ITER
predictions without the need for hand-tuning of SOLPS parameters.
For VSim, simulation setup parameters
can be optimized to minimize/maximize some figure of merit.