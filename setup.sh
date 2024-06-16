#!/bin/bash

# Delete virtual environment folder if it exists
if [ -d "myenv" ]; then
    rm -rf myenv
fi

# Create and activate virtual environment
python3.10 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir tensorly-viz-0.1.7.tar.gz
pip install --no-cache-dir matcouply-0.1.6.tar.gz

pip install jupyter ipykernel
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

# Launch Jupyter Notebook
jupyter-notebook tPARAFAC2.ipynb
