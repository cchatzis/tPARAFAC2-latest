@echo off

:: Create the virtual environment
:: Delete virtual environment folder if it exists
rmdir /S /Q myenv

:: Create and activate virtual environment
py -3.10 -m venv myenv
call myenv\Scripts\activate.bat

:: Install dependencies using requirements.txt
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir tensorly-viz-0.1.7.tar.gz
pip install --no-cache-dir matcouply-0.1.6.tar.gz

pip install jupyter ipykernel
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

:: Launch Jupyter Notebook
jupyter-notebook tPARAFAC2.ipynb
