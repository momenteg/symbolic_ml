'''
SYMBOLIC MODELING TRAINING                                                                                   
'''

  
# Author   : Filippo Arcadu
# Date     : 10/11/2018
 
 
from __future__ import division , print_function
import argparse
import time
from syml_class import SYML




# =============================================================================
# Get input arguments 
# =============================================================================

def _example():
    print( '\n\nTrain model given a CSV table and a YAML config file:\n' )
    print( '"""\npython syml_train.py -i data.csv -c syml_config.yml -o ./\n"""\n\n' )


def _get_args():
    parser = argparse.ArgumentParser( description     = 'Training Symbolic Machine Learning Model',
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter    ,
                                      add_help        = False )
    
    parser.add_argument('-i', '--file_in', dest='file_in',
                        help='Specify input CSV file containing features and output variable')

    parser.add_argument('-c', '--config', dest='file_cfg',
                        help='Specify destination where to save outputs')  
    
    parser.add_argument('-o', '--path_out', dest='path_out', default='./' ,
                        help='Specify destination where to save outputs')  
    
    parser.add_argument('-l', '--label', dest='label',
                        help='Specify additional label for outputs' )

    parser.add_argument('-p', '--print_vars', dest='print_vars', action='store_true',
                        help='Print names and unique values of features and outcome variables' )

    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print example command line' )

    args = parser.parse_args()

    if args.help:
        parser.print_help()
        _example()

    if args.file_in is None:
        parser.print_help()
        sys.exit('\nERROR: Input CSV file not specified!\n')

    if os.path.isfile( args.file_in ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input CSV file does not exist!\n')

    if args.file_cfg is None:
        parser.print_help()
        sys.exit('\nERROR: Input YAML file not specified!\n')

    if os.path.isfile( args.file_cfg ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input YAML file does not exist!\n')

    return args




# =============================================================================
# MAIN 
# =============================================================================

def main():
    # Initial print
    time1 = time.time()
    
    print( '#########################################' )
    print( '#                                       #' )
    print( '#        TRAIN SYMBOLIC ML MODEL        #' )
    print( '#                                       #' )
    print( '#########################################' )


    # Get input arguments
    args = _get_args()


    # Initialize SYML class
    syml = SYML( args.file_in  ,
                 args.file_cfg , 
                 label = args.label )


    # Check prints
    print( '\nINPUTS' )
    print( 'CSV data file: ', syml._file_in )
    print( 'YAML config file: ', syml._file_cfg )
    print( 'Additional label: ', syml._add_label )
    
    print( '\nMODEL' )
    print( 'Type of task: ', syml._task_type )
    print( 'Algorithm: ', syml._alg )
    print( 'Metric: ', syml._metric )
    print( 'Output variable: ', syml._col_out )
    print( 'Selected features: ', syml._select )
    print( 'Excluded features: ', syml._exclude )

    print( '\nVALIDATION' )
    print( 'Hold-out: ', syml._hold_out )
    print( 'Constrained feature for splitting: ', syml._col_constr )
    print( 'K-fold cross-validation: ', syml._kfold_cv )
    
    if syml._kfold_cv:
        print( 'No. folds for cross-validation: ', syml._kfold_cv )

    print( '\nLabel for output files: ', syml._label_out )
    print( '\nNumber of parameter grid points: ', len( syml._param_combs ) ) 


    # Check feature and outcome variables
    if args.print_vars:
        print( '\n\nChecking feature and outcome variables ....' )
        syml._check_vars()


    # Training
    print( '\n\nStarting training ....' )
    syml._train()
    print( '.... done!' )


    # Write outputs
    print( '\n\nSaving outputs ....' )
    syml._save()
    print( '.... done!' )


    # Elapsed time
    time2 = time.time()
    print( '\n\nTime elapsed: ', time2-time1 )
    print( '\n\n' )




# =============================================================================
# CALL to MAIN 
# =============================================================================

if __name__ == '__main__':
    main()
