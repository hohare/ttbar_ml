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
    def __init__(self, hlist, metadata, runRwgt, config, runVars=False):
        self.hlist = hlist
        self.metadata = metadata
        self.runRwgt = runRwgt
        self.config = config
        self.runVars = runVars

        output = {}
        # PRE SELECTION
        for variable in config['features']:
            if variable in histo_settings.keys():
                output[variable] = hist.Hist(
                    hist.axis.Regular(histo_settings[variable]['nbins'], 
                                      histo_settings[variable]['min'], 
                                      histo_settings[variable]['max'], 
                                      name=variable, label=variable),
                    hist.axis.StrCategory(["powheg","eft"], name='dataset', label='dataset'),
                    hist.axis.IntCategory([0,1], name='postrwgt', label='postrwgt'),
                    hist.axis.Regular(50, 0., 6., name="factor", label="factor"),
                    storage='Weight'
                )
            else:
                print(f'Histogram for variable {variable} not defined.')
                self.hlist.remove(variable)
        # POST SELECTION
        for variable in self.hlist:
            if variable in histo_settings.keys():
                output[variable+"_sel"] = hist.Hist(
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
                
        output["factor"] = hist.Hist(
            hist.axis.Regular(40, 0., 6., name="factor", label="factor"),
            hist.axis.StrCategory(["powheg","eft"], name='dataset', label='dataset'),
        )
        output["wgts"] = hist.Hist(
            hist.axis.Regular(100, 0., 50., name='wgts', label='wgts'),
            hist.axis.StrCategory(["powheg","eft"], name='dataset', label='dataset'),
        )
        output["metadata"] = {
            'powheg': defaultdict_accumulator(float),
            'modCentral': defaultdict_accumulator(float)
        }
            
        self._accumulator = output

    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns
        
    def process(self, events):
        ######## Parameters ########
        fname = events.metadata['filename']
        #print(fname)
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
        output['metadata'][dataset]['nInputEvents'] += ak.num(events, axis=0)
        
        ######## Top Selection ########
        tops, exactly_two_tops = anaT.topObjectSelection(events.GenPart)
        output['metadata'][dataset]['twoTops'] += ak.sum(exactly_two_tops)
        selections = PackedSelection()
        selections.add('2t', exactly_two_tops)
        upper_mask = (tops[:,0].mass <= 193.) & (tops[:,1].mass <= 193.)
        lower_mask = (tops[:,0].mass >= 152.) & (tops[:,1].mass >= 152.)
        selections.add('m_t', upper_mask & lower_mask)

        top_event_mask = selections.all('2t', 'm_t')
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

        output['metadata'][dataset]['nGenEvents'] += ak.sum(top_event_mask)
        output['metadata'][dataset]['sowPostMask'] += ak.sum(events.genWeight[top_event_mask])

        ######## Reweight Calculation ########
            # Calculate weights
        sow = self.metadata[dataset]['sumOfWeights']   
        norm = self.metadata[dataset]['xsec'] * lumis[year]*1000.0 / sow 
        #print(norm)
        if hasattr(events.LHEWeight, 'sm_point'):
            weights = events.LHEWeight.sm_point
        else:
            weights = events.genWeight
        #wgt_mask = (weights[top_event_mask]>=0.)
        # Calculate reweight to powheg NLO
        if self.runRwgt: 
            factor = rwgt.calculate_reweight(top_info, self.config, isEFT)
        else: # Don't reweight to powheg NLO
            factor = np.ones_like(weights)
        
        evt_weights_orig = norm * weights
        evt_weights_rwgt = evt_weights_orig[top_event_mask] * factor
        
        output['wgts'].fill(wgts=evt_weights_orig[top_event_mask], dataset=labels)
        output['factor'].fill(factor=factor, dataset=labels)

        #output['metadata'][dataset]['sumWgtMask'] += ak.sum(wgt_mask)
        
        ######## Lep, Jet Selection ########
        if self.runVars:
            leps, jets = anaT.genObjectSelection(events[top_event_mask])
            
            ######## Event selections ########
            exactly_one_lep  = ak.num(leps)==1
            minimum_four_jets = ak.num(jets)>=4
            minimum_met = events.GenMET.pt[top_event_mask] >= 20.

            selections = PackedSelection()
            selections.add('1L', exactly_one_lep)
            selections.add('4pJets', minimum_four_jets)
            selections.add('minMET20', minimum_met)
            event_selection_mask = selections.all('1L', '4pJets', 'minMET20')
            
            output['metadata'][dataset]['nSelectedEvents'] += ak.sum(event_selection_mask)
        
            ######## Apply event selections ########
            leps = leps[event_selection_mask]
            jets = jets[event_selection_mask]
            met  = events.GenMET[event_selection_mask]
            
            var_info = {
                "lep_pt":   leps[:,0].pt,
                "lep_eta":  leps[:,0].eta,
                "lep_phi":  leps[:,0].phi,
                "lep_mass": leps[:,0].mass,
                "met_pt":  met.pt,
                "met_phi": met.phi,
                "jet0_pt":   jets[:,0].pt,
                "jet0_eta":  jets[:,0].eta,
                "jet0_phi":  jets[:,0].phi,
                "jet0_mass": jets[:,0].mass,
                "jet1_pt":   jets[:,1].pt,
                "jet1_eta":  jets[:,1].eta,
                "jet1_phi":  jets[:,1].phi,
                "jet1_mass": jets[:,1].mass,
            }
    
        ######## Create Variables ########
        #top_info['mtt'] = (tops[:,0] + tops[:,1]).mass
        #tops_pt = tops.sum().pt
        #avg_top_pt = np.divide(tops_pt, 2.0)
        
        ######## Histogram filling ########
        fill_info = {
            "dataset": labels,
            "postrwgt": 0,
            "weight": evt_weights_orig[top_event_mask],
            'factor': factor
        }
        fill_info_rwgt = {
            "dataset": labels,
            "postrwgt": 1,
            "weight": evt_weights_rwgt,
            'factor': factor
        }
        for variable in self.config['features']:
            #for now only handling top info
            if "top" in variable:
                # Pre-reweight
                fill_info.update({variable: top_info[variable]})
                output[variable].fill(**fill_info)
                if self.runRwgt:# Post-reweight
                    fill_info_rwgt.update({variable: top_info[variable]})
                    output[variable].fill(**fill_info_rwgt)
            del fill_info[variable]
            del fill_info_rwgt[variable]

        if self.runVars:
            fill_info = {
                "dataset": labels,
                "postrwgt": 0,
                "weight": evt_weights_orig[top_event_mask][event_selection_mask],
            }
            fill_info_rwgt = {
                "dataset": labels,
                "postrwgt": 1,
                "weight": evt_weights_rwgt[event_selection_mask],
            }
            for variable in self.hlist:
                var_name = variable+"_sel"
                fill_info.update({variable: var_info[variable]})
                output[var_name].fill(**fill_info)
                if self.runRwgt:
                    fill_info_rwgt.update({variable: var_info[variable]})
                    output[var_name].fill(**fill_info_rwgt)
                del fill_info[variable]
                del fill_info_rwgt[variable]
                
        return output

    def postprocess(self, accumulator):
        return accumulator