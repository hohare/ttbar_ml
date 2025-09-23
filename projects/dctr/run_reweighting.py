import argparse
import json
import time
import os

from coffea import processor, util
from coffea.nanoevents import NanoAODSchema

import sys
import ttbar_utilities.filing.filesets as uFiles
from utils.parse_rwgt_card import acquire_SM_rwgt_index

LST_OF_KNOWN_EXECUTORS = ["futures", "work_queue", "debug"]
LST_DATASET_FORMATS = ["sbi", "dctr"]

def main():
    parser = argparse.ArgumentParser(description='You can customize your run')