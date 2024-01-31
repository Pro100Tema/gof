# ============================================================================= 
from   __future__        import print_function
# ============================================================================= 
import ostap.fitting.roofit 
import ostap.fitting.models    as     Models 
from   ostap.core.core         import Ostap
import ostap.io.zipshelve      as     DBASE
import ostap.logger.table      as     T 
from   ostap.utils.timing      import timing
from   ostap.utils.utils       import wait
from   ostap.plotting.canvas   import use_canvas
from   ostap.fitting.variables import SETVAR 
from   ostap.utils.utils       import vrange 
from   builtins                import range
import ROOT, time
from good_fits import good_fits
# =============================================================================
# logging 
# =============================================================================
from ostap.logger.logger import getLogger
if '__main__' == __name__  or '__builtin__' == __name__ : 
    logger = getLogger ( 'tests_good_fits' )
else : 
    logger = getLogger ( __name__ )
# =============================================================================

def test_point_to_point():
    f = ROOT.TFile('dataset.root', 'read')
    dataset = f['dataset']
    f.close()

    f = ROOT.TFile('dataset_mc.root', 'read')
    dataset_mc = f['dataset_mc']
    f.close()

    T_value, p_value = good_fits(dataset, dataset_mc, ['x', 'y'], 'dism')

    print("Observed Dissimilarity Statistic:", T_value)
    print("Permutation Test P-value:", p_value)

def test_kNN():
    f = ROOT.TFile('dataset.root', 'read')
    dataset = f['dataset']
    f.close()

    distance = good_fits(dataset, method = 'kNN')
    print("Distances for dataset:", distance)

if '__main__' == __name__ :

    with timing ('Test point to point' , logger ) :
        test_point_to_point()

    with timing ('Test distance to nearest neighbor' , logger ) :
        test_kNN()
