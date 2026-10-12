import numpy as np
#import regions
import astropy.units as u
from astropy.coordinates import SkyCoord
#from astropy.io import fits
import matplotlib.pyplot as plt
import pathlib
# for benching

from time import time
# to clean up after ourselves
import os

from killms_v2.catalog import xcatalog

import pytest



def test_xcatalog_initialisation():
    '''Check initialisation with default behaviour.'''
    ### initialise xJones object from function calls
    # first test  - explicitly call all arguments
    test_xcat = xcatalog(name='test',
                         reference_freq=144e6*u.Hz,
                         source_names=np.array(["source1",
                                                "source2",
                                                "source3"]),
                         source_types=np.array(["PoInT",
                                                "POINT",
                                                "point"]),
                         source_ras=np.array([0,.1,.2])<<u.rad,
                         source_decs=np.array([0,.1,.2])<<u.rad,
                         source_fluxes_I=np.array([1,2,3])<<u.Jy,
                         source_fluxes_Q=np.array([0,0,0])<<u.Jy,
                         source_fluxes_U=np.array([0,0,0])<<u.Jy,
                         source_fluxes_V=np.array([0,0,0])<<u.Jy,
                         source_spectra=np.array([[-0.8,0.2],
                                                  [-0.8,-0.1],
                                                  [0.1,-0.4]]),
                         source_gauss_sigRAs=np.zeros(3)<<u.rad,
                         source_gauss_sigDecs=np.zeros(3)<<u.rad,
                         source_gauss_RotAngs=np.zeros(3)<<u.rad,
                         comments=np.array(["Initialisation ex nihilo"])
                         )
    print(test_xcat)

# second test - use default values for all optional params
test_xcat = xcatalog(name='test',
                        reference_freq=144e6*u.Hz,
                        source_names=np.array(["source1",
                                            "source2",
                                            "source3"]),
                        source_types=np.array(["PoInT",
                                            "POINT",
                                            "point"]),
                        source_ras=np.array([0,.1,.2])<<u.rad,
                        source_decs=np.array([0,.1,.2])<<u.rad,
                        source_fluxes_I=np.array([1,2,3])<<u.Jy,
                        comments=np.array(["Initialisation ex nihilo"])
                        )

def test_bbs_skymodel_init_serialize():
    test_xcat = xcatalog(name='test',
                        reference_freq=144e6*u.Hz,
                        source_names=np.array(["source1",
                                            "source2",
                                            "source3"]),
                        source_types=np.array(["PoInT",
                                            "POINT",
                                            "point"]),
                        source_ras=np.array([0,.1,.2])<<u.rad,
                        source_decs=np.array([0,.1,.2])<<u.rad,
                        source_fluxes_I=np.array([1,2,3])<<u.Jy,
                        source_spectra=np.array([[-0.8,0.2],
                                                 [-0.8,-0.01],
                                                 [-0.4,0.0]]),
                        comments=np.array(["Initialisation ex nihilo"])
                        )
    test_xcat.to_bbs_skymodel(filename="test.skymodel")

# test json serialisation
def test_json_serialize():
    test_xcat = xcatalog(name='test_serialize',
                        reference_freq=144e6*u.Hz,
                        source_names=np.array(["source1",
                                            "source2",
                                            "source3"]),
                        source_types=np.array(["PoInT",
                                            "POINT",
                                            "point"]),
                        source_ras=np.array([0,.1,.2])<<u.rad,
                        source_decs=np.array([0,.1,.2])<<u.rad,
                        source_fluxes_I=np.array([1,2,3])<<u.Jy,
                        comments=np.array(["Test serialisation"])
                        )
    test_xcat.to_json("serialise_test.json",include_data=True)
    # clean up
    os.system("rm serialise_test.json")

def test_json_init_serialize():
    test_xcat = xcatalog(name='test_serialize',
                        reference_freq=144e6*u.Hz,
                        source_names=np.array(["source1",
                                            "source2",
                                            "source3"]),
                        source_types=np.array(["PoInT",
                                            "POINT",
                                            "point"]),
                        source_ras=np.array([0,.1,.2])<<u.rad,
                        source_decs=np.array([0,.1,.2])<<u.rad,
                        source_fluxes_I=np.array([1,2,3])<<u.Jy,
                        comments=np.array(["Test serialisation"])
                        )
    test_xcat.to_json("serialise_test.json",include_data=True)
    test_xcat1 = xcatalog.from_json('serialise_test.json',include_data=True)
    assert test_xcat == test_xcat1
    # clean up
    os.system("rm serialise_test.json")

def test_skymodel_init_serialize():
    test_xcat = xcatalog(name='test_serialize',
                        reference_freq=144e6*u.Hz,
                        source_names=np.array(["source1",
                                            "source2",
                                            "source3"]),
                        source_types=np.array(["PoInT",
                                            "POINT",
                                            "point"]),
                        source_ras=np.array([0,.1,.2])<<u.rad,
                        source_decs=np.array([0,.1,.2])<<u.rad,
                        source_fluxes_I=np.array([1,2,3])<<u.Jy,
                        source_spectra=np.array([[-0.8,0.2],
                                                [-0.8,-0.01],
                                                [-0.4,0.0]]),
                        comments=np.array(["Test serialisation"])
                        )
    test_xcat.to_bbs_skymodel(filename="test.skymodel",hmsdms=False)
    test_xcat1=xcatalog.from_bbs_skymodel("test.skymodel")
    # Update test_xcat comments, name, and type to match from_bbs_skymodel metadata
    test_xcat.comments = np.array(['xcatalog initialised from BBS-style skymodel test.skymodel'])
    test_xcat.cat.attrs['comments'] = np.array(['xcatalog initialised from BBS-style skymodel test.skymodel'])
    test_xcat.name = "test.skymodel"
    test_xcat.cat.attrs['name'] = "test.skymodel"
    test_xcat.type = "BBS gaul"
    # validate data homogeneity
    assert test_xcat==test_xcat1
    # clean up
    os.system("rm test.skymodel")

def test_read_hmsdms_skymodel():
    test_xcat = xcatalog(name='test_serialize',
                        reference_freq=144e6*u.Hz,
                        source_names=np.array(["source1",
                                            "source2",
                                            "source3"]),
                        source_types=np.array(["PoInT",
                                            "POINT",
                                            "point"]),
                        source_ras=np.array([0,.21,.12])<<u.rad,
                        source_decs=np.array([0,.21,.2])<<u.rad,
                        source_fluxes_I=np.array([1,2,3])<<u.Jy,
                        source_spectra=np.array([[-0.8,0.2],
                                                [-0.8,-0.01],
                                                [-0.4,0.0]]),
                        comments=np.array(["Test serialisation"])
                        )
    test_xcat.to_bbs_skymodel(filename="test.skymodel",hmsdms=True)
    test_xcat1=xcatalog.from_bbs_skymodel("test.skymodel")
    # ok, the to and from seem to work.
    # precision errors in the conversion to and from skycoords corrupts the exact values.
    # TODO: fix this.
    os.system("rm test.skymodel")

def test_generate_skymodel():
    test_xcat = xcatalog.generate_skymodel(centrecoords=SkyCoord(np.pi*u.rad,0*u.rad,frame='fk5'),
                                           npoints=500)
    test_xcat.show()

def test_generate_skymodel_dr3():
    target_coords = SkyCoord(np.pi*u.rad,0*u.rad,frame='fk5')
    test_xcat = xcatalog.generate_skymodel_dr3(pointing_centre=target_coords)
    test_xcat.to_bbs_skymodel("test.skymodel")
    test_xcat.show()
