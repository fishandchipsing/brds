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

Install fluidsynth
~~~
brew install fluidsynth
~~~

Install pyfluidsynth
https://github.com/nwhitehead/pyfluidsynth

Download the MIR-QBSH-corpus to the Dataset folder

~~~
wget http://mirlab.org/dataSet/public/MIR-QBSH-corpus.rar
~~~

Add a recon/ folder or else you will get a directory not found error

Install timidity, needed for auto conversion to wav

~~~
brew install timidity
~~~

Download the soundfont to the folder
https://packages.debian.org/sid/fluid-soundfont-gm

Create the folder Dataset/soundfonts and place fluid_r3_gm2.sf2 in it or any other Soundfont you like

Create the folder Dataset/recordings to old the saved files.


# Run standalone code

Record for five seconds humming and then translate to midi and play back (it will)

~~~
python birds.py
~~~

Use existing .wav humming recording from the MIR-QBSH-corpus

~~~
python birds.py --fname 'Dataset/MIR-QBSH-corpus/waveFile/year2003/person00001/00014.wav'
~~~



# Notebook Version

~~~
jupyter notebook
~~~

Open the query-by-humming notebook, and run interactively from there.

# Run

~~~
python decomp.py
~~~
