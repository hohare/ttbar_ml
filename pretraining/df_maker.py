import awkward as ak
import numpy as np
import pandas as pd

#from coffea import processor
from coffea.processor import ProcessorABC, dict_accumulator, defaultdict_accumulator
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

from .DataframeAccumulator import DataframeAccumulator
#from pretraining.selection import is_clean

NanoAODSchema.warn_missing_crossrefs = False
np.seterr(divide='ignore', invalid='ignore', over='ignore')

class SemiLepProcessor(ProcessorABC):
    def __init__(self, SMweight_idx=-1):
        self.SMweight_idx = SMweight_idx
        
        self._accumulator = dict_accumulator({
            "dataframe": DataframeAccumulator(pd.DataFrame()),
            "metadata": defaultdict_accumulator(float),
        })

    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns
        
    def process(self, events):
        fname = events.metadata['filename']
        
        if "smeft" in fname or "honor" in fname: isEFT=True
        else: isEFT=False
        output = self._accumulator.identity()
        output['metadata']['nInputEvents'] += ak.num(events, axis=0)
        output['metadata']["sumGenWeights"] += np.float64(ak.sum(events.genWeight))
        if self.SMweight_idx>0:
            output['metadata']["sumSMreweights"] += np.float64(ak.sum(events.LHEReweightingWeight[:,self.SMweight_idx]))

        df = pd.DataFrame()

        ######## Initialize Objects  ########
        genpart = events.GenPart
        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])
        #ele  = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        #mu   = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        #nu_ele = genpart[is_final_mask & (abs(genpart.pdgId) == 12)]
        #nu_mu = genpart[is_final_mask & (abs(genpart.pdgId) == 14)]
        #nu = ak.concatenate([nu_ele,nu_mu],axis=1)

        ######## Top selection ########
        gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)

        ######## Event selections ########
        ntops = ak.num(gen_top)
        
        selections = PackedSelection()
        exactly_two_tops = ak.fill_none(ntops==2, False)
        
        selections.add('2t', exactly_two_tops)
        event_selection_mask = selections.all('2t')

        output['metadata']['nSelectedEvents'] += ak.sum(event_selection_mask)
        
        ######## Apply selections ########
        #leps  = leps[event_selection_mask]
        #jets  = jets[event_selection_mask]
        tops = gen_top[event_selection_mask]
        tops_idx = ak.argsort(tops.pt, ascending=False)
        tops = tops[tops_idx]


        ######## Create Variables ########
        mtt = (tops[:,0] + tops[:,1]).mass
        tops_pt = tops.sum().pt
        avg_top_pt = np.divide(tops_pt, 2.0)
        
        ######## Fill pandas dataframe ########
        if isEFT:
            df['SMweights'] = ak.to_numpy(events.LHEReweightingWeight[:,self.SMweight_idx][event_selection_mask])
        else:
            df['genweights'] = ak.to_numpy(events.genWeight[event_selection_mask])
            
        df['weight_originalXWGTUP'] = ak.to_numpy(events.LHEWeight.originalXWGTUP[event_selection_mask])

        df['top1pt']=tops.pt[:,0]
        df['top1eta']=tops.eta[:,0]
        df['top1phi']=tops.phi[:,0]
        df['top1mass']=tops.mass[:,0]

        df['top2pt']=tops.pt[:,1]
        df['top2eta']=tops.eta[:,1]
        df['top2phi']=tops.phi[:,1]
        df['top2mass']=tops.mass[:,1]

        df['mtt'] = mtt

        output['dataframe'] = DataframeAccumulator(df)
        return output
    
    def postprocess(self, accumulator):
        return accumulator
