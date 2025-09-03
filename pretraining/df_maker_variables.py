import awkward as ak
import numpy as np
import pandas as pd

from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

from pretraining.accumulators import DataframeAccumulator
from pretraining.analysis_tools import genObjectSelection

NanoAODSchema.warn_missing_crossrefs = False
np.seterr(divide='ignore', invalid='ignore', over='ignore')

class SemiLepProcessor(processor.ProcessorABC):
    def __init__(self, SMweight_idx=-1, runVariables=False):
        if SMweight_idx == -1: print('FUCK')
        self.SMweight_idx = SMweight_idx
        self.runVariables = runVariables
        
        self._accumulator = processor.dict_accumulator({
            "dataframe": DataframeAccumulator(pd.DataFrame()),
            "metadata": processor.defaultdict_accumulator(float),
        })

    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns
        
    def process(self, events):     
        fname = events.metadata['filename']
        if "smeft" in fname or "ctq8" in fname: isEFT=True
        else: isEFT=False
        output = self._accumulator.identity()
        output['metadata']['nInputEvents'] += len(events)
        output['metadata']["sumGenWeights"] += ak.sum(events.genWeight)
        if self.SMweight_idx>0:
            output['metadata']["sumSMreweights"] += ak.sum(events.LHEReweightingWeight[:,self.SMweight_idx])
        
        df = pd.DataFrame()

        ######## Object Selection  ########
        # tops
        genpart = events.GenPart
        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])
        gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)
        # leps and jets
        leps, jets = genObjectSelection(events)

        ######## Event selections ########
        selections = PackedSelection()
        
        exactly_two_tops = ak.fill_none(ak.num(gen_top)==2, False)
        exactly_one_lep  = ak.num(leps)==1
        minimum_four_jets = ak.num(jets)>=4
        minimum_met = events.GenMET.pt > 20.
        
        selections.add('2t', exactly_two_tops)
        selections.add('1Lep', exactly_one_lep)
        selections.add('4pJets', minimum_four_jets)
        selections.add('minMET20', minimum_met)
        event_selection_mask = selections.all('2t', '1Lep', '4pJets', 'minMET20')

        output['metadata']['nSelectedEvents'] += ak.sum(event_selection_mask)
        
        ######## Apply selections ########
        tops = gen_top[event_selection_mask]
        tops_idx = ak.argsort(tops.pt, ascending=False)
        tops = tops[tops_idx]
        leps  = leps[event_selection_mask]
        jets  = jets[event_selection_mask]
        met   = events.GenMET[event_selection_mask]

        ######## Create Variables ########
        mtt = (tops[:,0] + tops[:,1]).mass
        tops_pt = tops.sum().pt
        avg_top_pt = np.divide(tops_pt, 2.0)

        # get EFT reweighted to SM
        weight_originalXWGTUP = ak.to_numpy(events.LHEWeight.originalXWGTUP[event_selection_mask])
        if isEFT:
            weights = ak.to_numpy(events.LHEReweightingWeight[:,self.SMweight_idx][event_selection_mask])
        else: # set all weights to one      
            #weights = np.ones_like(events['event'])[event_selection_mask]
            weights = ak.to_numpy(events.genWeight[event_selection_mask])
        
        ######## Fill pandas dataframe ########s
        df['weights']   = weights
        #df['weight_originalXWGTUP'] = weight_originalXWGTUP
        df['top1pt']   = tops.pt[:,0]
        df['top1eta']  = tops.eta[:,0]
        df['top1phi']  = tops.phi[:,0]
        df['top1mass'] = tops.mass[:,0]
        df['top2pt']   = tops.pt[:,1]
        df['top2eta']  = tops.eta[:,1]
        df['top2phi']  = tops.phi[:,1]
        df['top2mass'] = tops.mass[:,1]

        df['mtt']       = mtt
        df['tops_pt']   = tops_pt
        df['avg_top_pt']= avg_top_pt
        
        df['lep_pt']   = leps.pt[:,0]
        df['lep_eta']  = leps.eta[:,0]
        df['lep_phi']  = leps.phi[:,0]
        df['lep_mass'] = leps.mass[:,0]
        df['met_pt']  = met.pt
        df['met_phi'] = met.phi
        df['jet1_pt']   = jets.pt[:,0]
        df['jet1_eta']  = jets.eta[:,0]
        df['jet1_phi']  = jets.phi[:,0]
        df['jet1_mass'] = jets.mass[:,0]
        df['jet2_pt']   = jets.pt[:,1]
        df['jet2_eta']  = jets.eta[:,1]
        df['jet2_phi']  = jets.phi[:,1]
        df['jet2_mass'] = jets.mass[:,1]
        df['jet3_pt']   = jets.pt[:,2]
        df['jet3_eta']  = jets.eta[:,2]
        df['jet3_phi']  = jets.phi[:,2]
        df['jet3_mass'] = jets.mass[:,2]
        df['jet4_pt']   = jets.pt[:,3]
        df['jet4_eta']  = jets.eta[:,3]
        df['jet4_phi']  = jets.phi[:,3]
        df['jet4_mass'] = jets.mass[:,3]

        output['dataframe'] = DataframeAccumulator(df)
        return output
    
    def postprocess(self, accumulator):
        return accumulator
