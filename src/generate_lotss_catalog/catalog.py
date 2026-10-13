# Defines Catalog object properties.
from __future__ import annotations
from typing import Any, Optional
import astropy.units as u
import xarray as xr
from attrs import define
import numpy as np
from pathlib import Path
# for serialisation
import json
# for nice __repr__
import pint_xarray 
# for regex string manipulation
import re
# to handle HMSDMS values
from astropy.coordinates import SkyCoord

# import from local libs
#from killms_v2.utils import validate_axis, serialize_complex, deserialize_complex, get_quantity_from_data, get_array_from_data
import matplotlib.pyplot as plt
from .utils import generate_n_values
import numpy as np
from importlib.resources import files


class xcatalog():
    '''
    Class that contains an xarray catalog, along with metadata as explicit class attributes.
    '''

    def __init__(self,
                 name:str,
                 reference_freq:u.Quantity,                         # scalar quantity in Hz
                 source_names:np.typing.NDArray[np.str_],     
                 source_types:np.typing.NDArray[np.str_],           # make this into literal POINT and GAUSSIAN
                 source_ras:u.Quantity,                             # array of radians
                 source_decs:u.Quantity,                            # array of radians
                 source_fluxes_I:u.Quantity,                        # array of I in Jy
                 source_fluxes_Q:Optional[u.Quantity]=None,         # array of Q in Jy
                 source_fluxes_U:Optional[u.Quantity]=None,         # array of Q in Jy
                 source_fluxes_V:Optional[u.Quantity]=None,         # array of Q in Jy
                 source_spectra:Optional[np.typing.NDArray[np.float64]]=None,      # shape (n_sources, n_spi_terms)
                 source_gauss_sigRAs:Optional[u.Quantity]=None,     # gaussian semimajor axis sigma in rads
                 source_gauss_sigDecs:Optional[u.Quantity]=None,    # gaussian semiminor axis sigma in rads
                 source_gauss_RotAngs:Optional[u.Quantity]=None,    # gaussian rotation angle in rads (NOT astronomical unit! Actual angle of rotation of the gaussian!)
                 comments:Optional[np.typing.NDArray[np.str_]]=np.array([""])
                 ) -> None:
        # build class metadata
        self.name = name
        '''Name of this catalog'''
        self.type = "Unknown"
        '''If applicable, type of this catalog. E.g. BBS_gaul-like, CARMENES-like, ...'''
        self.comments=comments
        '''Optional comments'''

        # base column length on name; use this to check conformity.
        source_names = np.asarray(source_names)
        cat_length = len(source_names)
        # check that reference_freq is a frequency
        if  isinstance(reference_freq, u.Quantity)==False:
            raise ValueError(f"Reference frequency must be a Quantity with frequency-like units")
        elif reference_freq.unit is not None and \
             reference_freq.unit.physical_type!="frequency":
                raise ValueError(f"Reference frequency must have frequency-like units")
        # check that source_types contains only point and gaussian types. Remove casing
        source_types = np.array([s.lower().rstrip() for s in source_types])
        for source_type in set(source_types):
            if source_type not in {'point', 'gaussian'}:
                raise ValueError(f"Source types can only be (case-insensitive) \'point\' and \'gaussian\'")
        ### Type
        source_types = self.check_column_length(source_types,cat_length)
        ### Position
        source_ras = self.check_quantity(source_ras,desired_physical_type='angle',desired_length=cat_length,output_unit=u.rad)
        source_decs = self.check_quantity(source_decs,desired_physical_type='angle',desired_length=cat_length,output_unit=u.rad)
        ### Flux
        source_fluxes_I = self.check_quantity(source_fluxes_I,desired_physical_type='spectral flux density',desired_length=cat_length,output_unit=u.Jy)
        if source_fluxes_Q is not None:
            source_fluxes_Q = self.check_quantity(source_fluxes_Q,desired_physical_type='spectral flux density',desired_length=cat_length,output_unit=u.Jy)
        else:
            source_fluxes_Q = np.zeros(cat_length) * u.Jy
        if source_fluxes_U is not None:
            source_fluxes_U = self.check_quantity(source_fluxes_U,desired_physical_type='spectral flux density',desired_length=cat_length,output_unit=u.Jy)
        else:
            source_fluxes_U = np.zeros(cat_length) * u.Jy
        if source_fluxes_V is not None:
            source_fluxes_V = self.check_quantity(source_fluxes_V,desired_physical_type='spectral flux density',desired_length=cat_length,output_unit=u.Jy)
        else:
            source_fluxes_V = np.zeros(cat_length) * u.Jy
        ### frequency
        if source_spectra is not None:
            # check array length conformance
            if source_spectra.shape[0] != cat_length:
                raise ValueError(f"Only {source_spectra.shape[0]} sets of spectral parameters were provided for {cat_length} sources")
            # check array shape conformance
            if len(source_spectra.shape)==1:
                # only spectral index provided, no curvature etc. Reshape for convenience.
                source_spectra=source_spectra.reshape(cat_length,1)
            if len(source_spectra.shape) > 2:
                raise ValueError(f"Spectral parameters must be provided as a numpy array of shape (n_sources, n_params)=({cat_length},...), not {source_spectra.shape}")
        else:
            # assume default synchrotron spectrum
            source_spectra = -0.8 * np.ones((cat_length,1))
        ### gaussian
        if source_gauss_sigRAs is not None:
            # check array conformance
            source_gauss_sigRAs=self.check_quantity(source_gauss_sigRAs,desired_physical_type='angle',desired_length=cat_length,output_unit=u.rad)
            # input OK, set values
        else:
            # Set all gaussian sources to point sources unless specified.
            source_gauss_sigRAs = np.zeros(cat_length)*u.rad
        if source_gauss_sigDecs is not None:
            # check array conformance
            source_gauss_sigDecs=self.check_quantity(source_gauss_sigDecs,desired_physical_type='angle',desired_length=cat_length,output_unit=u.rad)
            # input OK, set values
        else:
            # Set all gaussian sources to point sources unless specified.
            source_gauss_sigDecs = np.zeros(cat_length)*u.rad
        if source_gauss_RotAngs is not None:
            # check array conformance
            source_gauss_RotAngs=self.check_quantity(source_gauss_RotAngs,desired_physical_type='angle',desired_length=cat_length,output_unit=u.rad)
            # input OK, set values
        else:
            # Set all gaussian sources to point sources unless specified.
            source_gauss_RotAngs = np.zeros(cat_length)*u.rad

        ### with all data loaded, build unit and value maps.
        unit_map = {
            "RA": str(source_ras.unit),
            "DEC": str(source_decs.unit),
            "Flux": str(source_fluxes_I.unit),
            # "Flux_Q": str(source_fluxes_Q.unit),  # type: ignore[union-attr]
            # "Flux_U": str(source_fluxes_U.unit),  # type: ignore[union-attr]
            # "Flux_V": str(source_fluxes_V.unit),  # type: ignore[union-attr]
            "GaussSigma_RA": str(source_gauss_sigRAs.unit),  # type: ignore[union-attr]
            "GaussSigma_DEC": str(source_gauss_sigDecs.unit),  # type: ignore[union-attr]
            "GaussRotAngle": str(source_gauss_RotAngs.unit)  # type: ignore[union-attr]
        }


        ### Follow pandas-to-xarray conversion convention.
        stokes=["I","Q","U","V"]
        coords={"Name":source_names,
                "Stokes":stokes}
        data={
            "Type":("Name",source_types),
            "RA":(["Name"],source_ras.value),
            "DEC":(["Name"],source_decs.value),
            "Flux":(["Name", "Stokes"],np.array([source_fluxes_I.value, 
                                     source_fluxes_Q.value, # type: ignore[union-attr]
                                     source_fluxes_U.value, # type: ignore[union-attr]
                                     source_fluxes_V.value]).transpose() # type: ignore[union-attr]
                                     )# type: ignore[union-attr]
            # "Flux_Q":(["Name"],source_fluxes_Q.value),  # type: ignore[union-attr]
            # "Flux_U":(["Name"],source_fluxes_U.value),  # type: ignore[union-attr]
            # "Flux_V":(["Name"],source_fluxes_V.value)  # type: ignore[union-attr]
            }
        # add spectral properties to data dict
        data["SpectralIndex"] = (["Name"],source_spectra[:,0])
        # below - we enforced a 2D array, but pylance doesn't see it. we type-ignore accordingly.
        for n_spi_param in range(1,source_spectra.shape[1]): # type: ignore[index]
            data[f"SpectralIndex_{n_spi_param}"] = (["Name"],source_spectra[:,n_spi_param])


        # add gaussian properties to data dict
        data["GaussSigma_RA"] = (["Name"],source_gauss_sigRAs.value)  # type: ignore[union-attr]
        data["GaussSigma_DEC"] = (["Name"],source_gauss_sigDecs.value)  # type: ignore[union-attr]
        data["GaussRotAngle"] = (["Name"],source_gauss_RotAngs.value)  # type: ignore[union-attr]

            # "SpectralIndex":None,
            #}
        # build xarray catalog
        cat = xr.Dataset(
                        data_vars=data,
                        coords=coords,
                        attrs=dict(reference_freq=reference_freq.value,
                                   reference_freq_unit=str(reference_freq.unit),
                                   name=name,
                                   comments=comments),
                    )
        # add unit labels
        for var_name, unit_str in unit_map.items():
            cat[var_name].attrs["unit"] = unit_str
        # enable nice repr
        self.cat = cat.pint.quantify()
        '''xarray containing the catalog.'''

    def __repr__(self) -> str:
        '''
        Return the underlying xarray catalog representation.
        Modify it to show units explicitly.
        
        '''
        #return self.cat.pint.quantify().__repr__()
        lines = []
        lines.append("<xarray.Dataset>")
        dims = ", ".join(f"{k}: {v}" for k, v in self.cat.sizes.items())
        lines.append(f"Dimensions:    ({dims})")

        # Coordinates
        if self.cat.coords:
            lines.append("Coordinates:")
            coord_items = []
            for coord in self.cat.coords:
                var = self.cat.coords[coord]
                unit = var.attrs.get('unit', '')
                coord_items.append((coord, unit, str(var.dims), str(var.dtype), str(var.shape)))

            if coord_items:
                max_name = max(len(item[0]) for item in coord_items)
                max_unit = max(len(f"[{item[1]}]") for item in coord_items) if any(item[1] for item in coord_items) else 0
                max_dims = max(len(item[2]) for item in coord_items)
                max_dtype = max(len(item[3]) for item in coord_items)
                for name, unit, dims, dtype, shape in coord_items:
                    unit_str = f"[{unit}]" if unit else ""
                    lines.append(f"  * {name:<{max_name}}  {unit_str:>{max_unit}}  {dims:>{max_dims}}  {dtype:<{max_dtype}}  {shape}")

        # Data variables
        lines.append("Data variables:")
        var_items = []
        for var_name in self.cat.data_vars:
            var = self.cat[var_name]
            unit = var.attrs.get('unit', '')
            values = var.values
            val_str = " ".join(map(str, values[:3])) + (" ..." if len(values) > 3 else "")
            var_items.append((var_name, unit, str(var.dims), str(var.dtype), var.nbytes, val_str))

        if var_items:
            max_name = max(len(item[0]) for item in var_items)
            max_unit = max(len(f"[{item[1]}]") for item in var_items) if any(item[1] for item in var_items) else 0
            max_dims = max(len(item[2]) for item in var_items)
            max_dtype = max(len(item[3]) for item in var_items)
            max_bytes = max(len(f"{item[4]}B") for item in var_items)
            for name, unit, dims, dtype, nbytes, val_str in var_items:
                unit_str = f"[{unit}]" if unit else ""
                lines.append(f"    {name:<{max_name}}  {unit_str:>{max_unit}}  {dims:>{max_dims}}  {dtype:<{max_dtype}}  {nbytes:>{max_bytes}}B  {val_str}")

        # Attributes
        if self.cat.attrs:
            lines.append("Attributes:")
            for k, v in self.cat.attrs.items():
                lines.append(f"    {k}: {v}")

        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        '''
        Compare two xcatalog objects for equality.
        Returns True if all attributes and data are equal.
        '''
        if not isinstance(other, xcatalog):
            return False
        # Compare class attributes
        if self.name != other.name or self.type != other.type:
            return False
        if not np.array_equal(np.asarray(self.comments), np.asarray(other.comments)):
            return False
        # Compare the underlying xarray Dataset using .equals() method
        return self.cat.equals(other.cat)


    def add_coords(self,
                   coordname:str,
                   coordvals:u.Quantity,
                   physical_type:str,
                   unit:u.Unit|u.IrreducibleUnit) -> None:
        '''
        Add coordinate axis specified by coordname with coordvals values. 
        :param self: xcatalog object
        :param coordname: Name of new coordinate axis
        :type coordname: str
        :param coordvals: Values of new coordinates in axis
        :type coordvals: u.Quantity
        '''
        self.catalog = self.catalog.assign_coords({coordname:self.check_quantity(coordvals,
                                                                                 physical_type,
                                                                                 desired_length=len(coordvals),
                                                                                 output_unit=unit)})

    def del_coords(self, coordname:str) -> None:
        '''
        Remove coordinate axis specified by coordname.
        
        :param self: xJones object
        :param coordname: name of coordinate axis to remove
        :type coordname: str
        '''
        self.catalog=self.catalog.drop_dims(coordname)

    def add_attr(self,
                 attrname:str,
                 attrvals:Any,
                 ) -> None:
        '''Add attribute to catalog as specified by attrname, attrvals'''
        self.catalog = self.catalog.assign_attrs({attrname:attrvals})

    def del_attr(self,
                 attrname:str) -> None:
        '''Remove attribute attrname from catalog'''
        attrs = self.catalog.attrs
        del attrs[attrname]
        self.catalog = self.catalog.assign_attrs(attrs)
    
    def check_column_length(self,
                            col_in:np.typing.NDArray[Any],
                            length) -> np.typing.NDArray[Any]:
        '''Function to validate that xcatalog.__init__ input ndarrays are of the correct length'''
        if len(col_in)==length:
            return col_in
        else:
            raise ValueError(f"Length mismatch: assigning data column of length ({len(col_in)}) to catalog of length ({length})")

    def check_quantity(self,
                       quant_in:u.Quantity,
                       desired_physical_type:str,
                       desired_length:int,
                       output_unit:u.Unit|u.IrreducibleUnit) -> u.Quantity:
        '''
        Function to validate that xcatalog.__init__ input quantities are arrays of the correct type and length
           
        :param quant_in: input quantity array of length desired_length
        :param desired_physical_type: physical type for this axis, such as time, frequency, Stokes...
        :type desired_physical_type: str
        :param output_unit: Fundamental unit to return the coord in, such as s, Hz...
        :type output_unit: u.Unit | u.IrreducibleUnit
        :return: Outputs a reformatted astropy Unit array using the provided values and requested units.
        :rtype: Quantity
        '''
        if np.isscalar(quant_in):
            raise ValueError(f"Length mismatch: assigning scalar quantity to catalog of length ({desired_length})")
        elif len(quant_in)!=desired_length:
            raise ValueError(f"Length mismatch: assigning data column of length ({len(quant_in)}) to catalog of length ({desired_length})")
        # with dimensionality checked, validate unit types
        if quant_in.unit is not None and quant_in.unit.physical_type != desired_physical_type:
            raise ValueError(f"Unit mismatch: assigning data of type ({quant_in.unit.physical_type}) to column requiring type ({desired_physical_type})")
        # finally, set to default unit (e.g. seconds rather than minutes)
        quant_in = quant_in.to(output_unit)
        return quant_in
    
    def show(self):
        '''
        Function to plot the catalog in space
        
        :param self: xcatalog object
        '''
        sizes = 10*np.log10(self.cat["Flux"].sel(Stokes="I").values)
        plt.scatter(self.cat["RA"].values,self.cat["DEC"].values,s=sizes)
        plt.xlabel("RA [rad]")
        plt.ylabel("DEC [rad]")
        plt.show()
    

    @classmethod
    def generate_skymodel(cls,
                          centrecoords:SkyCoord,
                          fov:u.Quantity=5*u.deg,
                          npoints:int=5,
                          ngauss:int=0,
                          max_flux_point:float=10,
                          min_flux_point:float=0,
                          flux_spectrum_point:float=0,
                          max_flux_gauss:float=1,
                          flux_spectrum_gauss:float=0,
                          ref_freq=144*u.MHz
                          ) -> 'xcatalog':
        '''
        Generate a random skymodel with gaussians and point
        sources located within the requested FoV around the
        stated centrecoords. 
        
        :param cls: Description
        :param centrecoords: Description
        :type centrecoords: SkyCoord
        :param fov: Description
        :param npoints: Description
        :type npoints: int
        :param ngauss: Description
        :type ngauss: int
        :param flux_dist_spectrum: Description
        :type flux_dist_spectrum: float
        :param ref_freq: Description
        :return: Description
        :rtype: xcatalog
        '''
        # get total number of sources
        nsources = ngauss + npoints
        # make list of names
        snames = np.array([f"sources_{i}" for i in range(0,nsources)])
        # make list of types
        stypes=np.array(["gauss"]*ngauss + ["point"]*npoints)
        # build lists
        sras=[]
        sdecs=[]
        fluxes=[]
        for i in range(nsources):
            # make coords
            scoord = centrecoords.directional_offset_by(np.random.uniform(0,2*np.pi)*u.rad, np.random.uniform(0,fov.value)*u.deg)
            sras.append(scoord.ra.to(u.rad).value) # type: ignore ### pylance annoying
            sdecs.append(scoord.dec.to(u.rad).value) # type: ignore ### pylance annoying
            # make fluxes
            fluxes.append(np.random.uniform(min_flux_point,max_flux_point))
        # make source ras, decs
        source_ras = sras * u.rad
        source_decs = sdecs *u.rad
        # make source fluxes
        fluxes = fluxes * u.Jy
        # make polfracs
        ...
        # spectra
        ### TODO
        # comments
        comments = np.array([f"Synthetic skymodel of {ngauss} gaussians and {npoints} points lying within {float(fov.to(u.deg).value)} degrees of {centrecoords.to_string('hmsdms')}"])
        # add not implemented warnings
        if ngauss!=0:
            raise NotImplementedError(f"Gaussian simulation not implemented yet!")
        if flux_spectrum_point != 0:
            raise NotImplementedError(f"Source density per flux bin not implemented yet!")


        synth_cat = xcatalog(name="simul",
                             reference_freq=ref_freq,                         # scalar quantity in Hz
                             source_names=snames,     
                             source_types=stypes,           # make this into literal POINT and GAUSSIAN
                             source_ras=source_ras,                             # array of radians
                             source_decs=source_decs,                            # array of radians
                             source_fluxes_I=fluxes,                        # array of I in Jy
                             source_fluxes_Q=None,         # array of Q in Jy
                             source_fluxes_U=None,         # array of Q in Jy
                             source_fluxes_V=None,         # array of Q in Jy
                             source_spectra=None,      # shape (n_sources, n_spi_terms)
                             source_gauss_sigRAs=None,     # gaussian semimajor axis sigma in rads
                             source_gauss_sigDecs=None,    # gaussian semiminor axis sigma in rads
                             source_gauss_RotAngs=None,    # gaussian rotation angle in rads (NOT astronomical unit! Actual angle of rotation of the gaussian!)
                             comments=comments
                             )

        return synth_cat
    
    @classmethod
    def generate_skymodel_dr3(cls,
                          pointing_centre:SkyCoord,
                          fov:u.Quantity=5*u.deg,
                          type:str="LoTSS-DR3",
                          n_sources:int=None,
                          flux_min=0,
                          flux_max=np.inf,
                          gauss_sigma_min=0,
                          gauss_sigma_max=np.inf):
        
        if type not in ["LoTSS-DR3"]:
            raise NotImplementedError(f"Catalog simulations following the distributions of {type} are not implemented yet.")
        else:
            # this is the pickle for LOTSS_DR3
            pickle_filename="data/lotss_dr3.distribs.pickle.npy"
            data_file = (files("generate_lotss_catalog")/ "data"/ "lotss_dr3.distribs.pickle.npy")
            # generate sources based on LoTSS-DR3 source density OR requested number
            if n_sources is None:
                # estimate number of points to draw from density
                n_sources_in_lotss=16943656
                area_lotss_deg2= (0.4649947210785588 * u.steradian).to(u.deg**2).value # numerical value is the sky coverage of lotss-dr3 in steradians.
                lotss_source_density_deg2=n_sources_in_lotss/area_lotss_deg2
                requested_fov_area_deg2=(0.5*fov.to(u.deg).value)**2
                n_sources_in_fov = int(lotss_source_density_deg2 *  requested_fov_area_deg2)
            else:
                n_sources_in_fov=n_sources
            # check that pickle is present
            picklepath = Path(data_file)
            if picklepath.exists()==False:
                raise FileNotFoundError(f"LoTSS-DR3 statistics file {pickle_filename} not found.")
            else: 
                ppath=str(picklepath.absolute())
            # make list of names
            snames = np.array([f"ID_{i}" for i in range(0,n_sources_in_fov)])
            # make list of types
            stypes=np.array(["gaussian"]*n_sources_in_fov)
            # generate simulated RAs
            ra_min = max(0,pointing_centre.ra.to(u.deg).value-0.5*fov.to(u.deg).value)
            ra_max = min(360,pointing_centre.ra.to(u.deg).value+0.5*fov.to(u.deg).value)
            source_ras = generate_n_values(ppath,n_sources_in_fov,keylist=["RA"], val_min=ra_min, val_max=ra_max) * u.deg
            # generate simulated Decs
            dec_min = max(0,pointing_centre.dec.to(u.deg).value-0.5*fov.to(u.deg).value)
            dec_max = min(90,pointing_centre.dec.to(u.deg).value+0.5*fov.to(u.deg).value)
            source_decs = generate_n_values(ppath,n_sources_in_fov,keylist=["DEC"], val_min=dec_min, val_max=dec_max) * u.deg
            # generate simulated fluxes
            fluxes = 10**generate_n_values(ppath,n_sources_in_fov,keylist=["log_Total_flux"], val_min=flux_min, val_max=flux_max) * u.mJy
            # generated simulated gaussian params
            gauss_sigRAs = generate_n_values(ppath,n_sources_in_fov,keylist=["Maj"], val_min=gauss_sigma_min, val_max=gauss_sigma_max) * u.arcsec 
            gauss_sigDecs = generate_n_values(ppath,n_sources_in_fov,keylist=["Min"], val_min=gauss_sigma_min, val_max=gauss_sigma_max) * u.arcsec
            gausss_PAs = generate_n_values(ppath,n_sources_in_fov,keylist=["PA"]) * u.deg
            # generate dummy xcat
            dr3_cat = xcatalog(name="LoTSS_DR3_simul",
                               reference_freq=144*u.MHz,                         # scalar quantity in Hz
                               source_names=snames,     
                               source_types=stypes,           # make this into literal POINT and GAUSSIAN
                               source_ras=source_ras,                             # array of radians
                               source_decs=source_decs,                            # array of radians
                               source_fluxes_I=fluxes,                        # array of I in Jy
                               source_fluxes_Q=None,         # array of Q in Jy
                               source_fluxes_U=None,         # array of Q in Jy
                               source_fluxes_V=None,         # array of Q in Jy
                               source_spectra=None,      # shape (n_sources, n_spi_terms)
                               source_gauss_sigRAs=gauss_sigRAs,     # gaussian semimajor axis sigma in rads
                               source_gauss_sigDecs=gauss_sigDecs,    # gaussian semiminor axis sigma in rads
                               source_gauss_RotAngs=gausss_PAs,    # gaussian rotation angle in rads (NOT astronomical unit! Actual angle of rotation of the gaussian!)
                               comments=["Simulated LoTSS-DR3 distribution"]
                               )
            return dr3_cat


    @classmethod
    def from_bbs_skymodel(cls,
                          filename:str) -> 'xcatalog':
        ### check for the file
        filepath = Path(filename)
        if filepath.exists()==False:
            raise FileExistsError(f"File {filename} not found.")
        with open(filepath, 'r') as f:
            # first line must always be header
            header = f.readline()
            ### build initial key values for the dictionary
            keystr = re.search(r'\((.*?)\)', header)
            if keystr is None:
                raise ValueError("Skymodel headers must be of the form: #format=(Name, Type, ....) with parentheses.")
            keys = keystr.group(1).replace(" ","").lower().split(",")
            # fix key case
            for ik in range(len(keys)):
                # first, hard-coded uppercases
                if keys[ik]=="i": keys[ik]='I'
                elif keys[ik]=="q": keys[ik]='Q'
                elif keys[ik]=="u": keys[ik]='U'
                elif keys[ik]=="v": keys[ik]='V'
                elif keys[ik]=="ra": keys[ik]='RA'
                elif keys[ik]=="dec": keys[ik]='Dec'
                else:
                    # then, capitalise words
                    if 'name' in keys[ik]: keys[ik]=keys[ik].replace('name','Name')
                    if 'type' in keys[ik]: keys[ik]=keys[ik].replace('type','Type')
                    if 'patch' in keys[ik]: keys[ik]=keys[ik].replace('patch','Patch')
                    if 'reference' in keys[ik]: keys[ik]=keys[ik].replace('reference','Reference')
                    if 'reference' in keys[ik]: keys[ik]=keys[ik].replace('frequency','Frequency')
                    if 'reference' in keys[ik]: keys[ik]=keys[ik].replace('spectral','Spectral')
                    if 'reference' in keys[ik]: keys[ik]=keys[ik].replace('index','Index')
                    if 'reference' in keys[ik]: keys[ik]=keys[ik].replace('major','Major')
                    if 'reference' in keys[ik]: keys[ik]=keys[ik].replace('minor','Minor')
                    if 'reference' in keys[ik]: keys[ik]=keys[ik].replace('axis','Axis')
                    if 'reference' in keys[ik]: keys[ik]=keys[ik].replace('orientation','Orientation')
            ### check that core skymodel values are present
            required_keys = {'Name', 'RA', 'Dec', 'I'}
            if not required_keys.issubset(keys):
                missing = required_keys - set(keys)
                if len(missing)>0:
                    raise ValueError(f"Missing required keys in skymodel: {missing}")                
            ### clean up keys, and read reference frequency if SpectralIndex
            for key in keys:
                if "referencefrequency" in key.lower():
                    if "=" not in key:
                        raise ValueError(f"Reference frequency being sought in {key}, but no value is provided. Expected format is \'ReferenceFrequency=value\' in Hz")
                    else:
                        reference_freq = float(key.split("=")[1].strip("\'"))*u.Hz
                        keys[keys.index(key)] = "ReferenceFrequency"
                if "spectralindex" in key.lower():
                    keys[keys.index(key)] = "SpectralIndex"            
            skymodel_dict = {key: [] for key in keys}
            ### find spectral index key index
            try:
                idx = keys.index("SpectralIndex") # find spi index to add new ones if needed
            except KeyError:
                skymodel_dict["SpectralIndex"]=[]
                idx = keys.index("SpectralIndex")
            n_spi=1 # assume only 1 spectral index value provided, to start
            n_lines=0
            ### populate sky model dictionary line by line
            for line in f.readlines():
                # header has already been skipped, so now clean up the line and check if it is empty, is a Patch, or is what we want
                # clean up the line
                line = line.replace(" ","").replace("\n","")
                this_line_contents=line.split(",")
                if line=="":
                    continue # this line is empty
                elif "[" not in line: # this line is a patch. 
                    ... # TODO: learn to assign patches.
                elif len(this_line_contents) >= len(keys): # this line is part of skymodel
                    n_lines=n_lines+1
                    # assign all variables before spectral indices
                    vals=line.split("[")[0].split(",")
                    for i in range(0,idx):
                        skymodel_dict[keys[i]].append(vals[i])
                    # assign all variables after spectral indices
                    vals=line.split("],")[1].split(",")
                    for i in range(idx+1,len(keys)):
                        skymodel_dict[keys[i]].append(vals[i-idx-1])
                    # assign spectral index values
                    if "[" not in line or "]" not in line: # check correct encoding
                        raise ValueError("Skymodel spectral indices must be contained in brackets in the form: [a,b,...]")
                    spivals = line.split("[")[1].split("],")[0].split(",")#np.fromstring(line.split("[")[1].split("],")[0], sep=',', dtype=float)
                    skymodel_dict["SpectralIndex"].append(spivals[0])
                    # if there are more spivals than usual, create new empty column for to fill later
                    if len(spivals)>n_spi:
                        for nsp in range(n_spi,len(spivals)):
                            skymodel_dict[f"SpectralIndex_{nsp}"]=[0]*(n_lines-1)
                        n_spi = len(spivals)
                    # if there are fewer spivals than current columns, buffer with zeroes
                    elif len(spivals)<n_spi:
                        for nsp in range(len(spivals),n_spi):
                            spivals.append("0")
                    # write to all known additional spectral index columns
                    for i in range(1,n_spi):
                        skymodel_dict[f"SpectralIndex_{i}"].append(spivals[i])
                else:
                    # this line is funkadelic, raise valuerror baby
                    raise ValueError(f"Don't know what to do with this line of {filename} file: {','.join(this_line_contents)}")
                        
        # build default values for keys that were missing
        expected_keys={"Q", "U", "V", "SpectralIndex", 'GaussSigma_RA', 'GaussSigma_DEC', 'GaussRotAngle'}
        for key in expected_keys:
            if key not in skymodel_dict.keys():
                skymodel_dict[key]=[0]*n_lines

        # check if RA, Dec are in HHMMSS or already floats
        if ":" in skymodel_dict["RA"][0]: # they are in HHMMSS
            if len(skymodel_dict["Dec"][0].split("."))==1:
                raise ValueError(f"You seem to have RA values {skymodel_dict["RA"][0]} in HMS, but Dec values {skymodel_dict["Dec"][0]} not in DMS. Please choose HMSDMS or rad rad units.")
            else:
                decs = [r.replace('.', ':',2) for r in skymodel_dict["Dec"]]
                coords = SkyCoord(ra=skymodel_dict["RA"], dec=decs, unit=(u.hourangle, u.deg), frame='fk5')
                skymodel_dict["RA"] = coords.ra.to(u.rad).value # type: ignore ## pylance incomplete here
                skymodel_dict["Dec"] = coords.dec.to(u.rad).value # type: ignore ## pylance incomplete here
        else:
            skymodel_dict["RA"] = [float(x) for x in skymodel_dict["RA"]]
            skymodel_dict["Dec"] = [float(x) for x in skymodel_dict["Dec"]]

        # Create the xcatalog instance
        obj = cls(
            name=filename,
            reference_freq=reference_freq, # type: ignore[possibly-unbound]
            source_names=np.array(skymodel_dict["Name"],dtype=np.str_),
            source_types=np.array(skymodel_dict["Type"],dtype=np.str_),
            source_ras=skymodel_dict["RA"]*u.rad,
            source_decs=skymodel_dict["Dec"]*u.rad,
            source_fluxes_I=np.array(skymodel_dict["I"],dtype=float)<<u.Jy,
            source_fluxes_Q=np.array(skymodel_dict["Q"],dtype=float)<<u.Jy,
            source_fluxes_U=np.array(skymodel_dict["U"],dtype=float)<<u.Jy,
            source_fluxes_V=np.array(skymodel_dict["V"],dtype=float)<<u.Jy,
            source_spectra=np.array(skymodel_dict["SpectralIndex"],dtype=float),
            source_gauss_sigRAs=np.array(skymodel_dict["GaussSigma_RA"],dtype=float)<<u.rad,
            source_gauss_sigDecs=np.array(skymodel_dict["GaussSigma_DEC"],dtype=float)<<u.rad,
            source_gauss_RotAngs=np.array(skymodel_dict["GaussRotAngle"],dtype=float)<<u.rad,
            comments=np.array([f"xcatalog initialised from BBS-style skymodel {filename}"])
        )

        # add other data info
        for n_spi_param in range(1,n_spi): # type: ignore[index]
            obj.cat[f"SpectralIndex_{n_spi_param}"] = (["Name"],np.array(skymodel_dict[f"SpectralIndex_{n_spi_param}"],dtype=float))
            obj.cat[f"SpectralIndex_{n_spi_param}"].attrs["unit"] = ""

        # Set class attrs
        obj.type = "BBS gaul"
        obj.name = filename
        obj.comments=[f"xcatalog initialised from BBS-style skymodel {filename}"]

        # Add any additional coordinates that aren't part of the core structure
        for coord_name, coord_data in coordinates.items():
            if coord_name in ["Name", "RA", "DEC"]:
                continue  # Already handled

            if isinstance(coord_data, dict):
                coord_values = deserialize_complex(coord_data["values"])
                if "unit" in coord_data:
                    unit_str = coord_data["unit"]
                    try:
                        unit = u.Unit(unit_str)
                        coord_quantity = coord_values * unit
                        obj.cat = obj.cat.assign_coords({coord_name: coord_quantity})
                    except Exception:
                        obj.cat = obj.cat.assign_coords({coord_name: coord_values})
                else:
                    obj.cat = obj.cat.assign_coords({coord_name: coord_values})
            else:
                obj.cat = obj.cat.assign_coords({coord_name: deserialize_complex(coord_data)})

        # Set attributes
        for attr_name, attr_value in attrs.items():
            if attr_name == "reference_freq":
                continue  # Already handled in __init__
            obj.cat = obj.cat.assign_attrs({attr_name: attr_value})
        return obj


    def to_bbs_skymodel(self,
                        filename:str,
                        hmsdms:bool=True,
                        hmsdms_precision:int=2,
                        overwrite=True):
        # check for the file
        filepath = Path(filename)
        if filepath.exists() and  not overwrite:
            raise FileExistsError(f"File {filename} already exists. Set overwrite=True to overwrite.")
        # load keys once
        cat_keys = self.cat.keys()
        # check for HMSDMS
        if hmsdms:
            coords = SkyCoord(ra=self.cat["RA"],dec=self.cat["DEC"],unit=u.rad,frame="fk5")
        # build header
        head="#format=(Name, "
        for key in cat_keys:
            # for Flux key, write out IQUV
            if "Flux"==key:
                head+="I, Q, U, V, "
            # curvature etc is written into SpectralIndex, not SpectralIndex_{i}
            elif "SpectralIndex" not in str(key):
                head+=str(key)+", "
            elif "_" not in str(key):
                head+=f"ReferenceFrequency={self.cat.attrs['reference_freq']}, "+str(key)+"='[]', "
        head = head[:-2]+")"
        output_lines=[head]
        # add customary empty line
        output_lines.append("")
        # do patches - TODO
#        if self.cat["direction"] ... :
            # write out patch information
#            ...
        # build spi column
        self.cat["SpectralIndex"]
        # do catalog lines
        lines=self.cat["Name"].values.astype(str)
        for key in self.cat.keys():
            if key=="RA":
                if hmsdms:
                    lines = [f"{a}, {b}" for a,b in zip(lines, coords.ra.to_string(u.hour,sep=":",precision=hmsdms_precision))] # type: ignore[possibly-unbound]
                else:
                    lines = [f"{a}, {b}" for a,b in zip(lines, self.cat[key].values.astype(str))]
            elif key=="DEC": 
                if hmsdms:
                    lines = [f"{a}, {b}" for a,b in zip(lines, coords.dec.to_string(u.degree,sep=".",precision=hmsdms_precision))] # type: ignore[possibly-unbound]
                else:
                    lines = [f"{a}, {b}" for a,b in zip(lines, self.cat[key].values.astype(str))]
            elif "Spectral" not in str(key):
                if key=="Flux":
                    lines = [f"{a}, {b}" for a,b in zip(lines, [", ".join(row.astype(str)) for row in self.cat[key].values])]
                else:
                    lines = [f"{a}, {b}" for a,b in zip(lines, self.cat[key].values.astype(str))]
            elif "_" not in str(key):
                # build spi column value, triggered when "_" is absent so it is only done once
                spi_str = self.cat["SpectralIndex"].values.astype(str)
                # check for further spectral index parameters
                for sp_key in self.cat.keys():
                    if "SpectralIndex" in str(sp_key):
                        if "_" in str(sp_key):
                            spi_str = [f"{a}, {b}" for a,b in zip(spi_str, self.cat[sp_key].values.astype(str))]
                # add reference frequency line
                ###lines = [f"{a}, {self.cat.attrs['reference_freq']}" for a in lines] ### new way, maybe? unconfirmed...
                lines = [f"{a}, " for a in lines] ### old way - no repetition of ref freq information 
                # add spi
                lines = [f"{a}, [{b}]" for a, b in zip(lines,spi_str)]
        

        # start writing to file
        with open(filepath, 'w') as f:
            f.write("\n".join(output_lines) + "\n")
            f.write("\n".join(lines) + "\n")

        

    def to_json(self,
                filename:str, 
                include_data:bool=False,
                overwrite:bool=True,
                indent:int=2) -> None:
        '''
         Serialise xcatalog object to JSON file.
        
        :param self: xcatalog object
        :param filename: Output filename
        :type filename: str
        :param include_data: include data values in JSON file. Default False.
        :type include_data: bool
        :param overwrite: overwrite existing file if present. Default True.
        :type overwrite: bool
        :param indent: JSON indentation level. Default 2.
        :type indent: int
        '''
        filepath = Path(filename)
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filename} already exists. Set overwrite=True to overwrite.")
        # build xcat dict
        cat_dict = self.cat.to_dict(data=include_data)

        # Serialize coordinates
        cat_dict["coordinates"]={}
        for coord_name, coord in self.cat.coords.items():
            if hasattr(coord, 'values'):
                if hasattr(coord, 'unit'):
                    cat_dict["coordinates"][coord_name] = {
                                "values": serialize_complex(coord.values),
                                "unit": str(coord.unit)
                    }
                else:
                    cat_dict["coordinates"][coord_name] = serialize_complex(coord.values)
            else:
                cat_dict["coordinates"][coord_name] = str(coord)

        # Serialize data variables if requested
        if include_data==True:
            cat_dict["data"]={}
            for var_name in self.cat.data_vars:
                var = self.cat[var_name]
                cat_dict["data"][var_name] = {
                            "values": serialize_complex(var.values),
                            "dims": list(var.dims)
                }
        with open(filepath, 'w') as f:
            json.dump(cat_dict, f, indent=indent)

    @classmethod
    def from_json(cls,
                  filename: str,
                  include_data:bool=True) -> 'xcatalog':
        """
        Initialize an xcatalog object from a JSON file such as created by xcatalog.to_json.

        :param cls: The xcatalog class
        :param filename: Path to the JSON file
        :type filename: str
        :param include_data: Whether to expect and load data variables from JSON. Default True.
        :type include_data: bool
        :return: An xcatalog object reconstructed from the JSON file
        :rtype: xcatalog
        """
        # check file
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} does not exist.")
        # load file
        with open(filepath, 'r') as f:
            cat_dict = json.load(f)

        # Extract metadata, giving defaults if absent
        attrs = cat_dict.get("attrs", {})
        name = attrs.get("name", "Unnamed")
        cat_type = attrs.get("type", "Unknown")
        comments = np.array(attrs.get("comments", [""]))

        # Extract attributes
        attrs = cat_dict.get("attrs", {})
        reference_freq = attrs.get("reference_freq", None)
        if reference_freq is not None:
            try:
                reference_freq = float(reference_freq) * u.Hz
            except (ValueError, TypeError):
                reference_freq = 1.4e9 * u.Hz  # Default to 1.4 GHz

        # Extract coordinates
        coordinates = cat_dict.get("coordinates", {})
        source_names = None
        if "Name" in coordinates:
            name_data = coordinates["Name"]
            source_names = deserialize_complex(name_data)
            if isinstance(source_names, np.ndarray) and source_names.dtype == np.object_:
                source_names = source_names.astype(np.str_)
            elif isinstance(source_names, list):
                source_names = np.array(source_names, dtype=np.str_)
        else:
            raise ValueError("JSON is missing required 'Name' coordinate")
        cat_length = len(source_names)
        # read data vars
        data = cat_dict.get("data", {})


        # Extract source_types from data. default to point unless specified
        source_types = get_array_from_data(data,"Type", default_value=np.array(["point"] * cat_length))
        if isinstance(source_types, np.ndarray) and source_types.dtype == np.object_:
            source_types = source_types.astype(np.str_)
        # Fluxes
        if "Flux" in data:
            flux_data = data["Flux"]
            flux_values = np.array(flux_data["values"])
            # Get Stokes coordinate order
            stokes_coord = coordinates.get("Stokes", ["I", "Q", "U", "V"])
            # Find indices for each Stokes component
            stokes_indices = {stokes: idx for idx, stokes in enumerate(stokes_coord)}
            # Extract each component
            source_fluxes_I = flux_values[:, stokes_indices.get("I", 0)] * u.Jy
            source_fluxes_Q = flux_values[:, stokes_indices.get("Q", 1)] * u.Jy
            source_fluxes_U = flux_values[:, stokes_indices.get("U", 2)] * u.Jy
            source_fluxes_V = flux_values[:, stokes_indices.get("V", 3)] * u.Jy
        else:
            # if missing, default to zero
            source_fluxes_I = np.zeros(cat_length) * u.Jy
            source_fluxes_Q = np.zeros(cat_length) * u.Jy
            source_fluxes_U = np.zeros(cat_length) * u.Jy
            source_fluxes_V = np.zeros(cat_length) * u.Jy


        # Spectral indices, default to fully flat spectrum
        spectral_data = {}
        for i in range(10):  # Support up to 10 spectral parameters
            var_name = "SpectralIndex" if i == 0 else f"SpectralIndex_{i}"
            if var_name in data:
                spectral_data[i] = get_array_from_data(data,var_name)
        if spectral_data:
            # Build source_spectra array
            n_params = len(spectral_data)
            source_spectra = np.zeros((cat_length, n_params))
            for i, values in spectral_data.items():
                source_spectra[:, i] = values
        else:
            # default to basic synchrotron spectrum if nothing else provided
            source_spectra = -0.8 * np.ones((cat_length, 1))

        # Gaussian parameters. default to zeros if not provided.
        source_gauss_sigRAs = get_quantity_from_data(data,"GaussSigma_RA", u.rad, np.zeros(cat_length) * u.rad)
        source_gauss_sigDecs = get_quantity_from_data(data,"GaussSigma_DEC", u.rad, np.zeros(cat_length) * u.rad)
        source_gauss_RotAngs = get_quantity_from_data(data,"GaussRotAngle", u.rad, np.zeros(cat_length) * u.rad)

        # Coordinates for RA/Dec
        source_ras = get_quantity_from_data(data,"RA", u.rad)
        source_decs = get_quantity_from_data(data,"DEC", u.rad)

        # If RA/Dec not in data, check coordinates
        if source_ras is None and "RA" in coordinates:
            ra_data = coordinates["RA"]
            if isinstance(ra_data, dict):
                ra_values = deserialize_complex(ra_data["values"])
                unit_str = ra_data.get("unit", "rad")
                source_ras = ra_values * u.Unit(unit_str)
            else:
                source_ras = deserialize_complex(ra_data) * u.rad

        if source_decs is None and "DEC" in coordinates:
            dec_data = coordinates["DEC"]
            if isinstance(dec_data, dict):
                dec_values = deserialize_complex(dec_data["values"])
                unit_str = dec_data.get("unit", "rad")
                source_decs = dec_values * u.Unit(unit_str)
            else:
                source_decs = deserialize_complex(dec_data) * u.rad

        if reference_freq is None:
            raise ValueError(f"{filename} is missing required 'reference_freq' attribute")

        # Create the xcatalog instance
        obj = cls(
            name=name,
            reference_freq=reference_freq,
            source_names=source_names,
            source_types=source_types,
            source_ras=source_ras,
            source_decs=source_decs,
            source_fluxes_I=source_fluxes_I,
            source_fluxes_Q=source_fluxes_Q,
            source_fluxes_U=source_fluxes_U,
            source_fluxes_V=source_fluxes_V,
            source_spectra=source_spectra,
            source_gauss_sigRAs=source_gauss_sigRAs,
            source_gauss_sigDecs=source_gauss_sigDecs,
            source_gauss_RotAngs=source_gauss_RotAngs,
            comments=comments
        )

        # Set class attrs
        obj.type = cat_type
        obj.name = name
        obj.comments=comments

        # Add any additional coordinates that aren't part of the core structure
        for coord_name, coord_data in coordinates.items():
            if coord_name in ["Name", "RA", "DEC"]:
                continue  # Already handled

            if isinstance(coord_data, dict):
                coord_values = deserialize_complex(coord_data["values"])
                if "unit" in coord_data:
                    unit_str = coord_data["unit"]
                    try:
                        unit = u.Unit(unit_str)
                        coord_quantity = coord_values * unit
                        obj.cat = obj.cat.assign_coords({coord_name: coord_quantity})
                    except Exception:
                        obj.cat = obj.cat.assign_coords({coord_name: coord_values})
                else:
                    obj.cat = obj.cat.assign_coords({coord_name: coord_values})
            else:
                obj.cat = obj.cat.assign_coords({coord_name: deserialize_complex(coord_data)})

        # Set attributes
        for attr_name, attr_value in attrs.items():
            if attr_name == "reference_freq":
                continue  # Already handled in __init__
            obj.cat = obj.cat.assign_attrs({attr_name: attr_value})
        return obj

    def to_fits(self) -> None:
        '''
        Serialise catalog to fits model image
        '''
        raise NotImplementedError(f"TODO")
