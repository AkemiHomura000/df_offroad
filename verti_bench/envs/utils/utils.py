# PyChrono imports
import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.sensor as sens

import math
import numpy as np
import os
from PIL import Image

def SetChronoDataDirectories():
    """
    Set data directory

    This is useful so data directory paths don't need to be changed everytime
    you pull from or push to github. To make this useful, make sure you perform
    step 2, as defined for your operating system.

    For Linux or Mac users:
      Replace bashrc with the shell your using. Could be .zshrc.
      1. echo 'export CHRONO_DATA_DIR=<chrono's data directory>' >> ~/.bashrc
          Ex. echo 'export CHRONO_DATA_DIR=/home/user/chrono/data/' >> ~/.zshrc
      2. source ~/.zshrc

    For Windows users:
      Link as reference: https://helpdeskgeek.com/how-to/create-custom-environment-variables-in-windows/
      1. Open the System Properties dialog, click on Advanced and then Environment Variables
      2. Under User variables, click New... and create a variable as described below
          Variable name: CHRONO_DATA_DIR
          Variable value: <chrono's data directory>
              Ex. Variable value: C:\ Users\ user\ chrono\ data\
    """
    from pathlib import Path
    import os

    CONDA_PREFIX = os.environ.get('CONDA_PREFIX')
    CHRONO_DATA_DIR = os.environ.get('CHRONO_DATA_DIR')
    if CONDA_PREFIX and not CHRONO_DATA_DIR:
        CHRONO_DATA_DIR = os.path.join(
            CONDA_PREFIX, 'share', 'chrono', 'data', '')
    if not CHRONO_DATA_DIR:
        CHRONO_DATA_DIR = os.path.join(Path(os.path.dirname(
            os.path.realpath(__file__))).parents[1], 'chrono', 'data', '')
    elif not CHRONO_DATA_DIR:
        raise Exception(
            'Cannot find the chrono data directory. Please verify that CHRONO_DATA_DIR is set correctly.')

    chrono.SetChronoDataPath(CHRONO_DATA_DIR)
    veh.SetDataPath(os.path.join(CHRONO_DATA_DIR, 'vehicle', ''))

