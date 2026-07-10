Lorenz Dögnitz, Théo Konc, Linus Mattauch
# Green Transitions under Changing Preferences: Optimal Intertemporal Policy Response

## Overview
- The whole simulation is contained in optimization.py. Parameters and model functions are defined first. Then, the backward induction alogrithm is applied by calling functions optimization() and tracking(). The loop for the sensitivity analysis can be found at the very end of the file.
- Running file optimization_d=0.py recomputes the optimization with a value of 0 of the preference transition speed parameter d.
- The file welfare_differences_d=0.py computes the welfare loss of wrongly assuming that preferences are fixed (not endogenous). The formulas are explained in Appendix A.4.
- File global_maximum_check.py performs a sanity check of the solver results as explained in Appendix B.1.
- File plots.py creates plots from the data created and stored while running optimization.py.
- File plots_d-sensitivity.py creates plots for further analysis of the influence of parameter d. 

## Data Information

- All data is computed by running optimization.py and optimization_d=0.py. Make sure that there is an empty folder "data" in your working directory so that data can be stored.

## Instructions to Replicators

- You can recreate our results by calling optimization.py and optimization_d=0.py first, and then welfare_differences_d=0.py, plots.py, and plots_d-sensitivity.py.
- The python files only require numpy, scipy, and matplotlib.
- For plots, MiKTeX (https://miktex.org) must be installed on your system.

## List of Tables and and Figures

- Table 1: Parameter values set in optimization.py
- Table 2: Welfare loss computed in welfare_differences_d=0.py
- Figure 1: Results for the basecase parameter set, data is created by running optimization.py and plots are created by plots.py
- Figure 2: Results of optimization.py and optimization_d=0.py, plotted in plots_d-sensitivity.py
- Figure 3 (Appendix): Created using global_maximum_check.py (for basecase parameters, period 0)
- Figures 4-6, 8-14 (Appendix): Created by running plots.py
- Figure 7 (Appendix): Created by running plots_d-sensitivity.py


