import awkward as ak
from coffea.analysis_tools import PackedSelection

def isClean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

def genObjectSelection(events):
    ######## Initialize objets ########
    leps  = events.GenDressedLepton
    jets  = events.GenJet
    el    = leps[abs(leps.pdgId) == 11]
    mu    = leps[abs(leps.pdgId) == 13]
    nu    = leps[(abs(leps.pdgId) == 12) | (abs(leps.pdgId) == 14)]

    ######## Lep selection ########
    el    = el[(el.pt>20) & (abs(el.eta)<2.5)]
    mu    = mu[(mu.pt>20) & (abs(mu.eta)<2.5)]
    leps  = ak.concatenate([el,mu],axis=1)

    ######## Jet selection ########
    jets  = jets[(jets.pt>30) & (abs(jets.eta)<2.5)]
    jets  = jets[isClean(jets, leps, drmin=0.4) & isClean(jets, nu, drmin=0.4)]

    return leps, jets

def genEventSelection(leps, jets,):
    nleps = ak.num(leps)
    njets = ak.num(jets)

    exactly_one_lep = ak.fill_none(nleps==1, False)
    at_least_four_jets = ak.fill_none(njets>=4, False)
    
    selections = PackedSelection()
    selections.add('1l', exactly_one_lep)
    selections.add('4j', at_least_four_jets)
    event_selection_mask = selections.all('1l', '4j')
    
    return event_selection_mask