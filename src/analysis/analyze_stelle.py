#!/usr/bin/env python
import os
import sys
from pathlib import Path
import pandas as pd

proj_name = 'stelle'
home = Path(__file__).resolve().home()
root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent  #home / (dir_prj+proj_name)
sys.path.insert(0, str(root_dir / 'src'))

weighted = False

from stelle.analysis_pipeline import AnalysisPipeline

if __name__ == '__main__':
    input_dir = root_dir / 'data/01_raw/stelle'
    output_dir = root_dir / ('data/02_processed/stelle')
    output_dir.mkdir(parents=True, exist_ok=True)
    for sim in input_dir.glob('mar*'):
        outdir = output_dir / sim
        indir = input_dir / sim
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Now analyzing {sim}")
        files = list(sim.glob('mar_*.dat.lammpstrj'))
        if len(files) > 0:
            pipeline = AnalysisPipeline(sim, weighted)
            # avoid re-running previous analyses if results present
            out = output_dir / str(pipeline)
            out_files = list(out.glob('*'))
            files_pqt = list(out.glob('*.pqt'))
            if len(out_files)>1:
                print(f'---Analysis of {sim} already performed, skipping.')
            else:
                pipeline.save_static_properties(output_dir)
        else:
            print(f'---Missing data in {sim}, skipping.')
