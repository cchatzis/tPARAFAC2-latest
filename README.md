Installation steps
------------------

This has been tested on Windows 11, Ubuntu 22.04 and macOS 14 Sonoma using python 3.10.12.

For Linux/Mac:
`$  ./setup.sh`

For Windows:
`$ setup.bat`

This will create a virtual environment, install the required packages, and launch the Jupyter Notebook.

The implementation is based on matcouply (https://matcouply.readthedocs.io/en/latest/), and we adjust tensorly-viz to work for PARAFAC2 factors (https://tensorly.org/viz/stable/). The code that peforms the AO-ADMM updates can be found in `myenv/lib/python3.10/site-packages/matcouply/decomposition.py` while the code for the custom penalty class for the temporal smoothness of tPARAFAC2 can be found in `my_factor_tools.py`. `out_dit.gif` contains the animation of the evolving metabolite factors for the Metabolomics application of the paper.
