from __future__ import annotations
from typing import Any, cast
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

def validate_axis(input_var,
                    desired_physical_type:str,
                    output_unit:u.Unit|u.IrreducibleUnit) -> u.Quantity:
    '''
    Function to take in sloppy inputs for xjones coordinate axes
    and reformat them to build nice xarray coordinates.
    
    :param input_var: coord value or array 
    :param desired_physical_type: physical type for this axis, such as time, frequency, Stokes...
    :type desired_physical_type: str
    :param output_unit: Fundamental unit to return the coord in, such as s, Hz...
    :type output_unit: u.Unit | u.IrreducibleUnit
    :return: Outputs a reformatted astropy Unit array using the provided values and requested units.
    :rtype: Quantity
    '''
    # since u.Quantity doesn't say if it's scalar or array, check that first.
    if isinstance(input_var,u.Quantity):
        output_var = cast(u.Quantity, input_var) # inform pylance
        # check if it is scalar or vector
        if output_var.isscalar:
            output_var = cast(u.Quantity,np.array([output_var]))
        # check if it has correct unit physical type
        if isinstance(output_var, u.Quantity) \
            and output_var.unit is not None \
                and output_var.unit.physical_type != desired_physical_type:
            raise ValueError(f"Unit mismatch: assigning data of type ({output_var.unit.physical_type}) to column requiring type ({desired_physical_type})")
        # if it does, cast to default unit (e.g. seconds rather than minutes)
        output_var = input_var.to(output_unit)
    else:
        output_var=np.asarray(input_var) # check this works. pylance complains less at least.
        # if it is not a quantity, make it into an array if it is not already
        if np.isscalar(input_var):
            output_var=np.array([input_var])
        elif not isinstance(input_var, np.ndarray):
            output_var=np.asarray(input_var)
        # not a quantity to begin with; just add requested units
        output_var = cast(u.Quantity, output_var * output_unit)
    return output_var

class MissingDicoSolsKey(Exception):
    """
    Exception raised due to missing coordinates when serialising xJones as DicoSols.
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def serialize_complex(arr:np.ndarray[Any]):
    '''
    Splits complex numpy array to allow JSON serialisation.
    Returns a list, split into real and imag if values are complex.
    
    :param arr: Numpy-like array of complex values to split
    :type arr: np.ndarray[np.complexfloating]
    '''
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return {
            "real": np.real(arr).tolist(),
            "imag": np.imag(arr).tolist(),
            "_complex_": True
        }
    return arr.tolist()


def deserialize_complex(data: Any) -> np.ndarray:
    """Reverse serialize_complex function."""
    if isinstance(data, dict) and data.get("_complex_", False):
        return np.array(data["real"]) + 1j * np.array(data["imag"])
    elif isinstance(data, list):
        return np.array(data)
    else:
        return np.array([data])

def get_quantity_from_data(data_dict:dict,
                           var_name:str,
                           default_unit:u.Unit|u.IrreducibleUnit, 
                           default_value:Any=None):
    """Helper to extract a quantity from data dict, or create default."""
    if var_name in data_dict:
        var_data = data_dict[var_name]
        values = deserialize_complex(var_data["values"])
        return values * default_unit
    return default_value

def get_array_from_data(data_dict:dict,
                        var_name:str,
                        default_value:Any=None):
    """Helper to extract a plain array from data or create default."""
    if var_name in data_dict:
        var_data = data_dict[var_name]
        return deserialize_complex(var_data["values"])
    return default_value


def get_lmn(phasecentre:SkyCoord,
            sourcecoords:SkyCoord):
    '''
    Function that builds l,m,n values for set of coordinates given a phasecentre
    '''
    refra  = np.asarray(phasecentre.ra.rad) # type: ignore[union-attr]
    refdec = np.asarray(phasecentre.dec.rad) # type: ignore[union-attr]
    ras    = np.asarray(sourcecoords.ra.rad) # type: ignore[union-attr]
    decs   = np.asarray(sourcecoords.dec.rad) # type: ignore[union-attr]
    l = np.cos(decs) * np.sin(ras - refra)
    m = np.sin(decs) * np.cos(refdec) - np.cos(decs) * np.sin(refdec) * np.cos(ras - refra)
    n = np.sqrt(1 - l**2 - m**2)
    return l,m,n

def calc_plot_nx_ny(n_plots:int,
                    ratio:float=1.2) -> np.ndarray:
    '''
    Function that splits n_plots into X by Y subplots. 
    Returns appropriate nx, ny.
    ratio states rough desired ratio of nx to ny. 
    Returns [nx, ny] as a list of integers.

    :param n_plots: Number of subplots required
    :param ratio: Desired ratio of nx to ny (default 1.5)
    :return: [nx, ny] - grid dimensions
    '''
    # first, check easy case
    if n_plots <= 1:
        return np.array([1, 1])
    # then, make first reasonable guess
    ny = max(1, int(np.sqrt(n_plots / ratio)))
    nx = max(1, -(-n_plots//ny))

    return np.array([nx,ny])


def casa_stokes_enum():
    return  [
            'Undefined',
            'I',
            'Q',
            'U',
            'V',
            'RR',
            'RL',
            'LR',
            'LL',
            'XX',
            'XY',
            'YX',
            'YY',
            'RX',
            'RY',
            'LX',
            'LY',
            'XR',
            'XL',
            'YR',
            'YL',
            'PP',
            'PQ',
            'QP',
            'QQ',
            'RCircular',
            'LCircular',
            'Linear',
            'Ptotal',
            'Plinear',
            'PFtotal',
            'PFlinear',
            'Pangle'
            ]
