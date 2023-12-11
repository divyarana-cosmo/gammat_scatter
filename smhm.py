import numpy as np
import matplotlib.pyplot as plt
import galsim
from halopy import halo
from stellarpy import stellar
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import FlatLambdaCDM

import argparse
import yaml


def smhm(logMh, logM0, logM1, gamma1, gamma2):



