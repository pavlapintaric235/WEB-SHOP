"""Separate judge entrypoint; never opens the existing 8508/8509 database."""
import os
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parent
os.environ['WEBSHOP_JUDGE_MODE']='1'
os.environ['WEBSHOP_DATA_DIR']=str(ROOT/'data/webshop_judge_demo')
runpy.run_path(str(ROOT/'app_8509.py'),run_name='__main__')
