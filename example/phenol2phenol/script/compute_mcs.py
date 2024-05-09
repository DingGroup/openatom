import openmm.app as app
import pickle
from atom.functions import (
    compute_mcs_VF2,
)

liga_prmtop = app.AmberPrmtopFile("./structure/output/BNZ.prmtop")
liga_top = liga_prmtop.topology

ligb_prmtop = app.AmberPrmtopFile("./structure/output/IPH.prmtop")
ligb_top = ligb_prmtop.topology

mcs = compute_mcs_VF2(liga_top, ligb_top, timeout=30)

with open('./output/mcs.pkl', 'wb') as file_handle:
    pickle.dump(mcs, file_handle)
    