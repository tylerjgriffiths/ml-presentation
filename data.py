#!/usr/bin/env python

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import csv

## A Data class to make it easier to handle and use the datasets.
class Data:

    def __init__(self):
        self.bradley_data_file = None
        self.bradley_data = None
        self.fp_data = None
        pass
    
    def read_bradley_from_csv(self, bradley_data_file):
        with open(bradley_data_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            self.bradley_data = [[Chem.MolFromSmiles(str(row[1])),float(row[2])] for row in reader]
        self.bradley_data_file = bradley_data_file
        
    def get_bradley_fps(self, fpfunc=None):
        if not fpfunc:
            fpfunc=lambda m: AllChem.GetMorganFingerprintAsBitVect(m,2)
        errors=0
        fp_data = []
        for row in self.bradley_data:
            try:
                fp_data.append([fpfunc(row[0]),row[1]])
            except:
                errors += 1
        if errors:
            print("Encountered {} errors.".format(errors))
        self.fp_data = fp_data
        return self.fp_data
    
