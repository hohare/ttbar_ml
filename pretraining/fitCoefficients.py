import numpy as np
import awkward as ak
import time

wcs = [
    "cSM",
    "ctGIm",
    "ctGRe",
    "cQj38",
    "cQj18",
    "cQu8",
    "cQd8",
    "ctj8",
    "ctu8",
    "ctd8",
    "cQj31",
    "cQj11",
    "cQu1",
    "cQd1",
    "ctj1",
    "ctu1",
    "ctd1",
]

def strnum2float(num: str):
    return float(num.replace("m", "-").replace("p", "."))
def float2strnum(num: float):
    """Rewrite decimal as str notation"""
    mystr = ""
    if num<0: 
        mystr += "m"
        num = abs(num)
    mystr += str(int(num)%10)
    mystr += "p"
    mystr += str(int(num%1 *1e4)).strip("0")
    return mystr
#def find_wc_index():
    

def vec(S):
    """Why do this?"""
    # Acquiring upper triangle of array
    j, i = np.tril_indices(S.shape[-1])
    # Replacing diagonal with 1, all others 2
    scale = np.where(i == j, 1.0, np.sqrt(2))
    return S[..., i, j] * scale

def mat(s, n=None):
    """Inverse of vec() operation"""
    if n is None:
        l = s.shape[-1]
        n = int(0.5 * (np.sqrt(1 + 8 * l) - 1))
    S = np.empty(s.shape[:-1] + (n, n))
    j, i = np.tril_indices(n)
    #scaled = np.where(i == j, 1.0, 0.5) * s
    scaled = np.where(i == j, 1.0, 1.0) * s
    S[..., i, j] = scaled
    S[..., j, i] = scaled
    return S

def get_points(fields):
    """Convert reweighting strings to array"""
    prefix = 'EFTrwgt'
    reweightpoints=[]
    for wname in fields:
        if wname == "sm_point":
            continue
            vector = np.zeros(len(wcs))
            vector[wcs.index("cSM")] = 1.0
            reweightpoints.append(vector)
            print('adding cSM')
            continue
        elif wname.startswith(prefix):
            i = wname.find("_")
            num = wname[:i].replace(prefix,"")
            rwpoint = wname[i+1:]
            
            vector = np.zeros(len(wcs))
            vector[wcs.index("cSM")] = 1.0
    
            rwpoint = rwpoint.split("_")
            for name, value in zip(rwpoint[::2], rwpoint[1::2]):
                fval = strnum2float(value)
                vector[wcs.index(name)] = fval
            reweightpoints.append(vector)
        #nick's version does not account for anything other than 
        #sm_point and something starting with EFTrwgt
    reweightpoints_array = np.array(reweightpoints)
    return reweightpoints_array

def calculate(fields, weights):
    fields = [field for field in fields if field.startswith('EFTrwgt')]# or field.startswith('sm')]
    # Acquiring the WC values per reweighting
    reweightpoints_array = get_points(fields)
    # Acquiring a weighted lower triangle built from products of wc pairs
    reweightpoints_outer = vec(reweightpoints_array[:, :, None] * reweightpoints_array[:, None, :])
    # Filling an array with 
    weightsvector = np.empty((len(weights), len(reweightpoints_array)))
    for i, field in enumerate(fields):
        weightsvector[:, i] = weights[field]
    # Solving the system
    x, residuals, rank, s = np.linalg.lstsq(reweightpoints_outer, weightsvector.T, rcond=None)
    
    return x, residuals