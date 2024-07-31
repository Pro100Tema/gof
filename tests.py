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

def perform_test(data_file, mc_file=None, var_lst=None, method='PPD'):
    # Load dataset
    f = TFile(data_file, 'read')
    dataset = f['dataset']
    f.close()

    if method != 'kNN' and mc_file is not None:
        # Load MC dataset
        f = TFile(mc_file, 'read')
        dataset_mc = f['dataset_mc']
        f.close()
    else:
        dataset_mc = None

    # Run good_fits function
    if method == 'kNN':
        result = good_fits(dataset, method=method)
        return result, None  # kNN method returns only one value
    else:
        if var_lst is None:
            raise ValueError("var_lst must be provided for methods other than kNN")
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
def test_ppd_cdist():
    result = measure_memory_usage(perform_test, 'dataset.root', 'dataset_mc.root', ['x', 'y'], 'PPD')
    print("Permutation Test P-value:", result[1])

def test_ppd_rd_10k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc.root', ['x', 'y'], 'PPD')
    print("Permutation Test P-value:", result[1])

def test_ppd_mc_100k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc_100k.root', ['x', 'y'], 'PPD')
    print("Permutation Test P-value:", result[1])

def test_kNN():
    result = measure_memory_usage(perform_test, 'dataset.root', None, None, 'kNN')
    print("kNN Test Result:", result[0])

def test_LD():
    result = measure_memory_usage(perform_test, 'dataset.root', 'dataset_mc.root', ['x', 'y'], 'LD')
    print("Permutation Test P-value:", result[1])

def test_LD_rd_10k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc.root', ['x', 'y'], 'LD')
    print("Permutation Test P-value:", result[1])

def test_LD_mc_100k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc_100k.root', ['x', 'y'], 'LD')
    print("Permutation Test P-value:", result[1])

def test_KB():
    result = measure_memory_usage(perform_test, 'dataset.root', 'dataset_mc.root', ['x', 'y'], 'KB')
    print("Permutation Test P-value:", result[1])

def test_KB_rd_10k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc.root', ['x', 'y'], 'KB')
    print("Permutation Test P-value:", result[1])

def test_KB_mc_100k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc_100k.root', ['x', 'y'], 'KB')
    print("Permutation Test P-value:", result[1])

def test_MS():
    result = measure_memory_usage(perform_test, 'dataset.root', 'dataset_mc.root', ['x', 'y'], 'MS')
    print("Permutation Test P-value:", result[1])

def test_MS_rd_10k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc.root', ['x', 'y'], 'MS')
    print("Permutation Test P-value:", result[1])

def test_MS_mc_100k():
    result = measure_memory_usage(perform_test, 'dataset_rd_10k.root', 'dataset_mc_100k.root', ['x', 'y'], 'MS')
    print("Permutation Test P-value:", result[1])

if __name__ == '__main__':
    with timing('Test PPD cdist', logger):
        test_ppd_cdist()
    
    with timing('Test PPD rd 10k', logger):
        test_ppd_rd_10k()

    with timing('Test PPD mc 100k', logger):
        test_ppd_mc_100k()
    
    with timing('Test kNN', logger):
        test_kNN()
    
    with timing('Test LD', logger):
        test_LD()

    with timing('Test LD rd 10k', logger):
        test_LD_rd_10k()

    with timing('Test LD mc 100k', logger):
        test_LD_mc_100k()

    with timing('Test KB', logger):
        test_KB()

    with timing('Test KB rd 10k', logger):
        test_KB_rd_10k()

    with timing('Test KB mc 100k', logger):
        test_KB_mc_100k()
    
    with timing('Test MS', logger):
        test_MS()

    with timing('Test MS rd 10k', logger):
        test_MS_rd_10k()

    with timing('Test MS mc 100k', logger):
        test_MS_mc_100k()
    with timing ('Test MS' , logger ) :
        test_MS()
