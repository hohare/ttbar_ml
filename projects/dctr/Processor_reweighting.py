import awkward as ak
import numpy as np
import pandas as pd
import yaml
import json
import hist
from coffea.processor import ProcessorABC, dict_accumulator, defaultdict_accumulator
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

import pretraining.analysis_tools as anaT
import projects.dctr.EventReweighting as rwgt

NanoAODSchema.warn_missing_crossrefs = False
np.seterr(divide='ignore', invalid='ignore', over='ignore')

with open('ttbar_utilities/inputs/luminosity_in_fb.json') as f:
    lumis = json.load(f)
with open('Inputs/histogram_settings.yml') as f:
    histo_settings = yaml.safe_load(f)

class ReweightingProc(ProcessorABC):
    def __init__(self, hlist, metadata, runRwgt, config):
        self.hlist = hlist
        self.metadata = metadata
        self.runRwgt = runRwgt
        self.config = config

        output = {}
        for variable in self.hlist:
            if variable in histo_settings.keys():
                output[variable] = hist.Hist(
                    hist.axis.Regular(histo_settings[variable]['nbins'], 
                                      histo_settings[variable]['min'], 
                                      histo_settings[variable]['max'], 
                                      name=variable, label=variable),
                    hist.axis.StrCategory(["powheg","eft"], name='dataset', label='dataset'),
                    hist.axis.IntCategory([0,1], name='postrwgt', label='postrwgt'),
                    storage='Weight'
                )
            else:
                print(f'Histogram for variable {variable} not defined.')
                self.hlist.remove(variable)
        output["r"] = hist.Hist(
            hist.axis.Regular(50, 0., 10., name="r", label="r"),
            hist.axis.StrCategory(["powheg","eft"], name='dataset', label='dataset'),
        )
        output["wgts"] = hist.Hist(
            hist.axis.Regular(100, 0., 50., name='wgts', label='wgts'),
            hist.axis.StrCategory(["powheg","eft"], name='dataset', label='dataset'),
        )
        output["metadata"] = defaultdict_accumulator(float)
        self._accumulator = output

    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns
        
    def process(self, events):
        ######## Parameters ########
        fname = events.metadata['filename']
        if "smeft" in fname or "honor" in fname: 
            isEFT=True
            labels = "eft"
            dataset = 'modCentral'
        else: 
            isEFT=False
            labels = "powheg"
            dataset = 'powheg'
        
        output = self._accumulator
        year = self.metadata[dataset]['year']
        
        ######## Fill counters ########
        output['metadata']['nInputEvents'] += ak.num(events, axis=0)
        
        ######## Top Selection ########
        tops, exactly_two_tops = anaT.topObjectSelection(events.GenPart)
        output['metadata']['twoTops'] += ak.sum(exactly_two_tops)
        selections = PackedSelection()
        selections.add('2t', exactly_two_tops)

        top_event_mask = selections.all('2t')
        tops_idx = ak.argsort(tops.pt, ascending=False)
        tops = tops[tops_idx]

        top_info = {
            "top1pt": tops[:,0].pt[top_event_mask],
            "top1eta": tops[:,0].eta[top_event_mask],
            "top1phi": tops[:,0].phi[top_event_mask],
            "top1mass": tops[:,0].mass[top_event_mask],
            "top2pt": tops[:,1].pt[top_event_mask],
            "top2eta": tops[:,1].eta[top_event_mask],
            "top2phi": tops[:,1].phi[top_event_mask],
            "top2mass": tops[:,1].mass[top_event_mask]
        }

        ######## Reweight Calculation ########
        if self.runRwgt: # Calculate reweight to powheg NLO
            factor = rwgt.calculate_reweight(top_info, self.config, isEFT)
        else: # Don't reweight to powheg NLO
            factor = 1.
        # Calculate weights
        sow = self.metadata[dataset]['nSumOfWeights']
        norm = self.metadata[dataset]['xsec']/sow * lumis[year]*1000.0
        if hasattr(events.LHEWeight, 'sm_point'):
            weights = events.LHEWeight.sm_point
        else:
            weights = events.genWeight
        evt_weights_orig = norm * weights[top_event_mask]
        evt_weights_rwgt = evt_weights_orig * factor
        
        output['wgts'].fill(wgts=evt_weights_orig, dataset=labels)
        
        ######## Lep, Jet Selection ########
        #leps, jets = anaT.genObjectSelection(events)
        
        ######## Event selections ########

        #exactly_one_lep  = ak.num(leps)==1
        #minimum_four_jets = ak.num(jets)>=4
        #minimum_met = events.GenMET.pt > 20.

        #selections.add('1l', exactly_one_lep)
        #selections.add('4pJets', minimum_four_jets)
        #selections.add('minMET20', minimum_met)
        #event_selection_mask = selections.all('2t', '1Lep', '4pJets', 'minMET20')
        
        #output['metadata']['nSelectedEvents'] += ak.sum(event_selection_mask)
    
        ######## Apply event selections ########
    
        ######## Create Variables ########
        #top_info['mtt'] = (tops[:,0] + tops[:,1]).mass
        #tops_pt = tops.sum().pt
        #avg_top_pt = np.divide(tops_pt, 2.0)
        
        ######## Histogram filling ########
        for variable in self.hlist:
            #for now only handling top info
            if "top" in variable:
                # Pre-reweight
                fill_info = {
                    variable: top_info[variable],
                    "dataset": labels,
                    "postrwgt": 0,
                    "weight": evt_weights_orig
                }
                output[variable].fill(**fill_info)
                if self.runRwgt:
                    # Post-reweight
                    fill_info = {
                        variable: top_info[variable],
                        "dataset": labels,
                        "postrwgt": 1,
                        "weight": evt_weights_rwgt
                    }
                    output[variable].fill(**fill_info)

                    #output['s'].fill(s=)

        return output

    def postprocess(self, accumulator):
        return accumulator