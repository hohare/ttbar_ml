import numpy as np
import matplotlib.pyplot as plt

def strnum2float(num: str):
    return float(num.replace("m", "-").replace("p", "."))

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

def get_points(fields):
    # Convert reweighting names
    prefix = 'EFTrwgt'
    reweightpoints=[]
    reweightnumber=[]
    wclist = {"cSM"}

    for wname in fields:
        if wname == "reference_point":
            continue
            vector = np.ones(len(wcs))*999
            reweightpoints.append(vector)
        if wname == "sm_point":
            vector = np.zeros(len(wcs))
            vector[wcs.index("cSM")] = 1.0
            reweightpoints.append(vector)
        elif wname.startswith(prefix):
            i = wname.find("_")
            num = wname[:i].replace("EFTrwgt","")
            rwpoint = wname[i+1:]
            
            vector = np.zeros(len(wcs))
            vector[wcs.index("cSM")] = 1.0
    
            rwpoint = rwpoint.split("_")
            for name, value in zip(rwpoint[::2], rwpoint[1::2]):
                fval = strnum2float(value)
                wclist.update([name])
                vector[wcs.index(name)] = fval
            reweightpoints.append(vector)
            reweightnumber.append(num)
    reweightpoints_array = np.array(reweightpoints)
    reweightnumber_array = np.array(reweightnumber)
    return reweightpoints_array, reweightnumber_array#, wclist

def parse_rwgt_card(cardpath):
    with open(cardpath, "r") as file:
        wholefile = file.read()
    
    lines = wholefile.split('\n')
    lines = [l.replace('launch --rwgt_name=','') for l in lines if l.startswith('launch')]
    pts, nums = get_points(lines)

    return pts, nums#, wclist

def acquire_SM_rwgt_index(rwgtcard):
    pts, nrwgt = parse_rwgt_card(rwgtcard)
    SMweight_idx = np.where((np.sum(pts[:,1:],axis=1) == 0.))[0]
    return SMweight_idx[0]

def plot_rwgt_card(points):
    fig, ax = plt.subplots(figsize=(6,4))
    umm = plt.scatter(points[:,7],points[:,0]);
    yticks = np.arange(-1, 2.1, 1)
    xticks = np.arange(-11, 11.1, 1)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks);
    ax.grid(which='both',ls='--', alpha=0.3);
    
if __name__ == '__main__':
    parse_rwgt_card()
    