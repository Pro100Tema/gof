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
import tracemalloc
from ROOT import TFile
# =============================================================================
# logging 
# =============================================================================
from ostap.logger.logger import getLogger
if '__main__' == __name__  or '__builtin__' == __name__ : 
    logger = getLogger ( 'tests_good_fits' )
else : 
    logger = getLogger ( __name__ )
# =============================================================================

def perform_test(data_file, mc_file, var_lst, method='PPD'):
    # Load dataset
    f = TFile(data_file, 'read')
    dataset = f['dataset']
    f.close()

    # Load MC dataset
    f = TFile(mc_file, 'read')
    dataset_mc = f['dataset_mc']
    f.close()

    # Run good_fits function
    T_value, p_value = good_fits(dataset, dataset_mc, var_lst, method)
    return T_value, p_value

def measure_memory_usage(test_func, *args):
    tracemalloc.start()

    # Run the test function
    result = test_func(*args)

    # Get current and peak memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6} MB")
    print(f"Peak memory usage: {peak / 10**6} MB")

    tracemalloc.stop()

    return result

# Test functions
def test_p2p_cdist():
    result = measure_memory_usage(perform_test, 'dataset.root', 'dataset_mc.root', ['x', 'y'], 'PPD')
    print("Permutation Test P-value:", result[1])

def test_p2p_rd_10k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc.root', ['x', 'y'], 'PPD')
    print("Permutation Test P-value:", result[1])

def test_p2p_mc_100k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc_100k.root', ['x', 'y'], 'PPD')
    print("Permutation Test P-value:", result[1])

def test_kNN():
    f = ROOT.TFile('dataset.root', 'read')
    dataset = f['dataset']
    f.close()

    distance = good_fits(dataset, method = 'kNN')
    print("Distances for dataset:", distance)

def test_LD():
    f = ROOT.TFile('dataset.root', 'read')
    dataset = f['dataset']
    f.close()

    f = TFile('dataset_mc.root', 'read')
    dataset_mc = f['dataset_mc']
    f.close()

    U_value, p_value = good_fits(dataset, dataset_mc, ['x', 'y'], method = 'LD')
    print("Permutation Test P-value:", p_value)

def test_KB():
    f = ROOT.TFile('dataset.root', 'read')
    dataset = f['dataset']
    f.close()

    f = TFile('dataset_mc.root', 'read')
    dataset_mc = f['dataset_mc']
    f.close()

    U_value, p_value = good_fits(dataset, dataset_mc, ['x', 'y'], method = 'KB')
    print("Permutation Test P-value:", p_value)

def test_MS():
    f = ROOT.TFile('dataset.root', 'read')
    dataset = f['dataset']
    f.close()

    f = TFile('dataset_mc.root', 'read')
    dataset_mc = f['dataset_mc']
    f.close()

    U_value, p_value = good_fits(dataset, dataset_mc, ['x', 'y'], method = 'MS')
    print("Permutation Test P-value:", p_value)

if '__main__' == __name__ :
    with timing ('Test p2p cdist' , logger ) :
        test_p2p_cdist()
    
    with timing ('Test p2p rd 10k' , logger ) :
        test_p2p_rd_10k()

    with timing ('Test p2p mc 100k' , logger ) :
        test_p2p_mc_100k()
    
    with timing ('Test kNN' , logger ) :
        test_kNN()
    
    with timing ('Test LD' , logger ) :
        test_LD()

    with timing ('Test KB' , logger ) :
        test_KB()
    
    with timing ('Test MS' , logger ) :
        test_MS()
