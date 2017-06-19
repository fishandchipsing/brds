# brds
A not-so-intelligent intelligent thing

# Install

Build a virtual environment so we do not step on toes

~~~
virtualenv venv
source venv/bin/activate
~~~

When you are finished be sure to deactivate the virualenv

~~~
deactivate
~~~

Install requirements

~~~
python -m pip install -r requirements.txt
~~~

Install pyfluidsynth
https://github.com/nwhitehead/pyfluidsynth

Download the MIR-QBSH-corpus to the Dataset folder

Add a recon/ folder or else you will get a directory not found error

Download the soundfont to the folder
https://packages.debian.org/sid/fluid-soundfont-gm

~~~
wget http://mirlab.org/dataSet/public/MIR-QBSH-corpus.rar
~~~

Unarchive and then the notebook will point to it.

# Notebook

~~~
jupyter notebook
~~~

Open the query-by-humming notebook, and run interactively from there.

# Run

~~~
python decomp.py
~~~
