
import numpy as np
import pandas as pd
from astropy.io import fits
import pylab


def generate_hists(cat_filename:str,
         writefilename:str,
         nrows=5000,
         nbins=100):
    print(f"Regenerating {cat_filename} parameter distributions")
    # initialise dict
    distribs={}
    # load catalog
    with fits.open(cat_filename, memmap=True) as hdul:
        data = hdul[1].data # type:ignore
        for key in data.columns.names:
            if "name" in key.lower() \
                or "id" in key.lower() \
                or "code" in key.lower() \
                or "isl" in key.lower():
                continue
            else:
                print(f"Generating distribution for {key}")
                dist = np.zeros(nbins)
                # find max for the key
                this_max = -np.inf
                this_min = np.inf
                # for flux values, do log-scale
                if "_flux" in key:
                    for i in range(0, len(data), nrows):
                        dchunk = np.log10(data[key][i:i + nrows])
                        this_max = max(np.max(dchunk),this_max)
                        this_min = min(np.min(dchunk),this_min)
                    # build the distribution
                    bin_edges = np.linspace(this_min, this_max, nbins + 1)
                    dist = np.zeros(nbins)
                    for i in range(0, len(data), nrows):
                        dchunk = np.log10(data[key][i:i + nrows])
                        indices = np.searchsorted(bin_edges, dchunk, side='right') - 1
                        indices = indices[(indices >= 0) & (indices < nbins)]  # Filter out-of-bounds
                        np.add.at(dist, indices, 1)
                    distribs["log_"+key] = dist/np.sum(dist)
                    distribs["log_"+key+"_bin_edges"] = bin_edges
                else:
                    for i in range(0, len(data), nrows):
                        dchunk = data[key][i:i + nrows]
                        this_max = max(np.max(dchunk),this_max)
                        this_min = min(np.min(dchunk),this_min)
                    # build the distribution
                    bin_edges = np.linspace(this_min, this_max, nbins + 1)
                    dist = np.zeros(nbins)
                    for i in range(0, len(data), nrows):
                        dchunk = data[key][i:i + nrows]
                        indices = np.searchsorted(bin_edges, dchunk, side='right') - 1
                        indices = indices[(indices >= 0) & (indices < nbins)]  # Filter out-of-bounds
                        np.add.at(dist, indices, 1)
                    distribs[key] = dist/np.sum(dist)
                    distribs[key+"_bin_edges"] = bin_edges
    # pickle file
    np.save(writefilename,distribs) # type:ignore
    print(f"Wrote catalog distribution histograms to {writefilename}")

def plot_distribs(pickle_filename):
    # open pickle
    distribs = np.load(pickle_filename+".npy",allow_pickle=True).item()
    for key in distribs.keys():
        if "bin_edges" not in key:
            xvals = 0.5*(distribs[key+"_bin_edges"][:-1] + distribs[key+"_bin_edges"][1:])
            if "_flux" in key.lower():
                vals = np.cumsum(distribs[key])#np.cumsum(np.log(distribs[key]*1e6))
                pylab.scatter(xvals,vals)
                pylab.title(key)
                pylab.show()

            else:
                vals = np.cumsum(distribs[key])
                # pylab.scatter(xvals,vals)
                # pylab.title(key)
                # pylab.show()
                

def build_cdf(pickle_filename:str,
              overwrite:bool=False) -> None:
    '''
    Function to generate cumulative distribution
    functions for all keys in the previously-generated
    histogram dictionary stored in pickle_filename
    '''
    # check for file
    ...
    # open pickle
    distribs = np.load(pickle_filename+".npy",allow_pickle=True).item()
    initkeys = distribs.copy().keys() # fuck you python
    for key in initkeys:
        if "bin_edges" not in key: # skip bin_edge parts of dict
            # check that CDF not already generated
            if key+"_cdf" in distribs.keys() and overwrite==False:
                continue
            else:
                bin_centres = 0.5*(distribs[key+"_bin_edges"][:-1] + distribs[key+"_bin_edges"][1:])
                cdf = np.cumsum((distribs[key]))
                cdf = cdf / cdf[-1]
                distribs[key+"_cdf"] = cdf
    # repickle
    np.save(pickle_filename,distribs) # type:ignore

def generate_n_values(pickle_filename:str,
                      n_vals:int,
                      keylist:list=None,
                      val_min:float=-np.inf,
                      val_max:float=np.inf,
                      plot=False) -> np.ndarray:
    '''
    Function which returns n random values drawn from the distribution associated
    to the provided key, and bound within val_min and val_max.
    '''
    # open pickle
    distribs = np.load(pickle_filename,allow_pickle=True).item()
    if keylist is None:
        keylist = distribs.keys()
    for key in keylist:
        if key not in distribs.keys():
            raise ValueError(f"Requested key {key} not in the dictionary. Correct keys include: {[key for key in distribs.keys()]}")
        if "bin_edges" not in key and "cdf" not in key: # skip bin_edge, cdf parts of dict
            # build bin centres
            bin_centres = 0.5*(distribs[key+"_bin_edges"][:-1] + distribs[key+"_bin_edges"][1:])
            # apply edges
            mask = (bin_centres > val_min) * (bin_centres < val_max)
            bin_centres = bin_centres[mask]
            # build masked CDF
            cdf = np.cumsum((distribs[key][mask]))
            # renormalise
            cdf = cdf / cdf[-1]           
            n_values = np.random.rand(n_vals)
            value_bins = np.searchsorted(cdf, n_values)
            random_from_cdf = bin_centres[value_bins]
            if plot:
                import pylab as plt
#                bin_centres = bin_centres = 0.5*(10**distribs[key+"_bin_edges"][:-1] + 10**distribs[key+"_bin_edges"][1:])
                print(f"Plotting {key}")
                plt.suptitle(key)
                plt.subplot(121)
                plt.hist(distribs[key+"_bin_edges"][:-1], distribs[key+"_bin_edges"][1:], weights=distribs[key],density=True)
                xmin = distribs[key+"_bin_edges"][0]
                xmax = distribs[key+"_bin_edges"][-1]
                plt.xlim((xmin,xmax))
#                plt.hist(distribs[key+"_bin_edges"][:-1][::10], distribs[key+"_bin_edges"][1:][::10], weights=distribs[key][::10],density=True)
                # make histogram for plotting
                plt.title("Real distrib")
                plt.subplot(122)
                plt.hist(random_from_cdf, 100,density=True)
                plt.title("Simu distrib")
                plt.xlim((xmin,xmax))
                plt.savefig(key+"_simu.png")
                plt.clf()
                print(f"Saved distrib as {key}_simu.png")

    return random_from_cdf



if __name__ == "__main__":
    cat_filename = "LoTSS_DR3_v1.0.gaul.fits"
    pickle_filename = "lotss_dr3.distribs.pickle"
    #generate_hists(cat_filename, pickle_filename,nbins=int(1e4))
    #plot_distribs(pickle_filename)
    #build_cdf(pickle_filename,overwrite=True)
    vals=generate_n_values(pickle_filename,250,keylist=["DEC"], val_min=25, val_max=75)
    print(vals)