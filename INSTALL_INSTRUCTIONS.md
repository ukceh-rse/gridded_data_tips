## ANACONDA INSTALLATION INSTRUCTIONS

To get the notebook to run you will need to install jupyter and several python modules that are
not available on the standard linux set up at UKCEH.  
The easiest way to do this is to install anaconda in your home directory, and using
the .txt file I have provided in this directory to tell anaconda which modules to install,
replicate my setup.
This can be done in such a way as to not interfere with your current setup, and only be
enabled when needed to run the notebook.

1. Download anaconda from https://www.anaconda.com/distribution/
You want the python 3.7, 64bit (x86) installer for linux.
As of 17/09/2020 this can be obtained by running: wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
But note that the link to the most up to date version will likely change, so don't rely on this.

2. Install anaconda by running: bash the-script-you-just-downloaded.sh
By default it will install in your home directory, requiring no admin/root permissions.
At the end of the installation it will ask you whether you want to run conda-init.
Say no to this.

3. Add the following line to the .bashrc file in your linux home (/users/sgsys/username) directory.
(To see if you have one run 'ls -a'. If you don't have one, just create a new text file called .bashrc in your home directory)
alias loadconda='export PATH="/users/sgsys/username/anaconda3/bin:$PATH"'
changing 'username' to your ukceh linux username. Run 'whoami' if you're not sure what this is. 

4. Start a new bash shell by running bash, then load anaconda by running loadconda (the command we just created above).

5. Setup a 'ceh' environment that contains all the modules you need to run this notebook (and many more)
using the conda_env_file_ceh.txt.
In your home directory run 'bash' if you are not already using the bash shell (by default ceh uses the csh shell)
Then conda create --name ceh --file conda_env_file_ceh.txt (after copying the txt file to your home dir).

6. Run 'source activate ceh' to load this environment, and you should be good to go.

Now, instead of sourcing the python modules from the default setup, or your own, it will source them
from ~/anaconda3/envs/ceh/.
To get back to your own python setup, simply open a new terminal/shell.

Anytime you want to run the notebook, or use my python setup, run:
bash
loadconda
source activate ceh

To check which python you are using, or which anaconda environment you are using,
run 'which python'.
It will be /usr/bin/python (or similar) for the standard system one,
/users/sgsys/username/anaconda3... for an anaconda one

## `uv` installation

https://earthly.dev/blog/python-uv

https://alberto-agudo.github.io/posts/02-uv-for-package-management-in-python/index.html

```
uv venv gridded
. ./gridded/bin/activate
uv sync
```

## Issues installing cartopy

Cartopy requires python >=3.10
https://github.com/SciTools/cartopy/blob/main/pyproject.toml

`uv lock` by default generates "resolution markers" for python versions down to 3.7

You can `uv python pin` and add a version file https://github.com/astral-sh/uv/issues/6780 

This doesn't affect the behaviour of `uv add cartopy` / `uv sync` though


TBC






