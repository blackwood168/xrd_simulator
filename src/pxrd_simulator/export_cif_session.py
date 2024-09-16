# coding: utf-8
import numpy as np
import pxrd_simulator as ps
omport pickle
import pickle
ps.CctbxStr.from_cif("../../tests/si.cif")
ps.CctbxStr.from_cif("../../tests/quartz.cif")
aa = ps.CctbxStr.from_cif("../../tests/quartz.cif")
pickle.dumps(aa, "test.pcl")
get_ipython().run_line_magic('pinfo', 'pickle.dumps')
aa.structure
aa.structure.as_str
aa.structure.as_str()
aa.structure.as_cif_simple()
bb = aa.structure.as_cif_simple()\
bb = aa.structure.as_cif_simple()
bb
aa.structure.
import io
from contextlib import redirect_stdout
f = io.StringIO()
with redirect_stdout(f): aa.structure.as_cif_simple()
f.getvalue
f.getvalue()
aa.structure.as_py_code()
aa.structure.as_cif_block()
f.getvalue()
