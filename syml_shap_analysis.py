'''
Analysis of shap values
'''

from __future__ import print_function , division
import argparse
import sys , os
import json
import numpy as np




# =============================================================================
# MATPLOTLIB WITH DISABLED X-SERVER 
# =============================================================================

#import matplotlib as mpl
#mpl.use( 'Agg' )
import matplotlib.pyplot as plt

import shap
shap.initjs()


# =============================================================================
# MY VARIABLE FORMAT 
# =============================================================================

myint    = np.int
myfloat  = np.float32
myfloat2 = np.float64




# =============================================================================
# Get input arguments 
# =============================================================================

def _example():
    print( '\n\nBar plot of feature importance by aggregating one-hot encoded variables:\n' )
    print( '"""\npython syml_shap_analysis.py -i1 syml_shap_class_2018-11-15-21-28-24_16188673153453647142.npy -i2 syml_history_class_2018-11-15-21-28-24_16188673153453647142.json\n"""\n\n' )


def _get_args():
    parser = argparse.ArgumentParser( description     = 'Create analysis plots for SHAP values',
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter    ,
                                      add_help        = False )
    
    parser.add_argument('-i1', '--file_npy', dest='file_npy',
                        help='Specify input NPY file containing shap values')

    parser.add_argument('-i2', '--file_json', dest='file_json',
                        help='Specify input JSON file containing the history')
     
    parser.add_argument('-i3', '--file_csv', dest='file_csv',
                        help='Specify input CSV file containing the input dataset')
     
    parser.add_argument('-i4', '--col_out', dest='col_out',
                        help='Specify outcome variable')
    
    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print example command line' )

    args = parser.parse_args()

    if args.help:
        parser.print_help()
        _example()
        sys.exit()
    
    if args.file_npy is None:
        parser.print_help()
        sys.exit('\nERROR: Input SYML NPY history file not specified!\n')

    if args.file_json is None:
        parser.print_help()
        sys.exit('\nERROR: Input SYML JSON history file not specified!\n')

    if os.path.isfile( args.file_json ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input SYML JSON history file does not exist!\n')
    
    if ( args.file_csv is not None and args.col_out is None ) or \
        ( args.file_csv is not None and args.col_out is None ): 
        parser.print_help()
        sys.exit('\nERROR: Both an input CSV and the outcome column have to be specified!\n')
 
    if args.file_csv is not None:
        if os.path.isfile( args.file_csv ) is False:
            parser.print_help()
            sys.exit('\nERROR: Input CSV file does not exist!\n')

    return args




# =============================================================================
# Get data from JSON
# =============================================================================

def _get_data_from_npy( file_npy ):
    arr = np.load( file_npy )
    return arr[:,:,:arr.shape[2]-1]



# =============================================================================
# Get data from JSON
# =============================================================================

def _get_data_from_json( file_json ):
    df_json = json.loads( open( file_json ).read() )
    feats   = df_json[ 'feature_cols' ]
    return feats




# =============================================================================
# Get data from JSON
# =============================================================================

def _get_data_from_csv( file_csv , col_out ):
    X = pd.read_csv( file_csv , sep=SEP )
    X = X.drop( col_out )
    return X



# =============================================================================
# Plot separate summary plots
# =============================================================================

def _plot( feats ):
    import catboost
    import pandas as pd
    model = catboost.CatBoostClassifier()
    model.load_model( 'tmp/syml_model_class_2018-11-15-22-50-06_7185573423039249820_fold00.bin' )
    X     = pd.read_csv( '../../asthma_project/ml/ml_actionable_data/lavolta_01_baseline_asl_asl2_asl3.csv' , sep=',' )
    print( X.keys() )
    X     = X.drop( [ 'NUMEX' , 'NUMEX-BINARY' , 'ANONID' ] , axis=1 ) 
    df_json = json.loads( open( 'tmp/syml_history_class_2018-11-15-22-50-06_7185573423039249820.json' ).read() )
    feats_cat = df_json[ 'feature_cols_cat' ]
    print( feats_cat )
    
    inds = []
    for i in range( len( feats_cat ) ):
        ind = feats.index( feats_cat[i] )
        inds.append( ind )
    print( inds )
    print( X.keys() )
    xpool     = catboost.Pool( X , cat_features=inds )

    explainer = shap.TreeExplainer(model)
    shaps = explainer.shap_values( xpool )
    
    fig = shap.summary_plot( shaps , X , show=False )
    plt.savefig( 'scratch.png' )
    #plt.show()        


# =============================================================================
# Plot separate summary plots
# =============================================================================

def _plot_separate_summary_plots( feats, shaps  , file_in  , X=None ):
    for i in range( shaps.shape[0] ):
        # Plot separate summary plots shap-impact
        if i < 10:
            str_num = 'fold0' + str( i )
        else:
            str_num = 'fold' + str( i )

        print( shaps[0].shape )
        fig = shap.summary_plot( shaps[0] , feats )
        save_plot = file_in[:len( file_in )] + '_shap-plot-impact-point_' + str_num + '.pdf'
        #plt.tight_layout()
        #plt.savefig( save_plot , transparent=True , dpi=300 )
        plt.show()        

        fig = shap.summary_plot( shaps[0] , feats , plot_type='bar' )
        save_plot = file_in[:len( file_in )] + '_shap-plot-impact-mean_' + str_num + '.pdf'
        plt.tight_layout()
        plt.savefig( save_plot , transparent=True , dpi=300 )

        
        # Plot dependency with respect to a variable
        if X is not None:
            for j in range( len( feats ) ):
                fig = shap.dependence_plot( feats[j] , shaps[0] , X )
                save_plot = file_in[:len( file_in )] + '_shap-plot-depend-' + feats[i] + '_' + str_num + '.pdf'
                plt.tight_layout()
                plt.savefig( save_plot , transparent=True , dpi=300 )
           

        
# =============================================================================
# Plot composite summary plots
# =============================================================================

def _plot_composite_summary_plot( feats, shaps , file_in  , X=None ):
    # Average shap values
    shaps_mean = np.mean( shaps , axis=0 )

        
    # Plot separate summary plots shap-impact
    fig = shap.summary_plot( shaps_mean , feats )
    save_plot = file_in[:len( file_in )] + '_shap-plot-impact-point_composite.pdf'
    plt.savefig( save_plot , transparent=True , dpi=300 )
        
    fig = shap.summary_plot( shaps_mean , feats , plot_type='bar' )
    save_plot = file_in[:len( file_in )] + '_shap-plot-impact-mean_composite.pdf'
    plt.tight_layout()
    plt.savefig( save_plot , transparent=True , dpi=300 )

        
    # Plot dependency with respect to a variable
    if X is not None:
        for j in range( len( feats ) ):
            fig = shap.dependence_plot( feats[j] , shaps_mean , X )
            save_plot = file_in[:len( file_in )] + '_shap-plot-depend-' + feats[i] + '_composite.pdf'
            plt.tight_layout()
            plt.savefig( save_plot , transparent=True , dpi=300 )

    

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Get input arguments
    args = _get_args()
    
    print( '\nInput NPY file with shap values: ', args.file_npy )
    print( 'Input JSON history file: ', args.file_json )
    print( 'Input CSV data file: ', args.file_csv )
    print( 'Output column: ', args.col_out )


    # Get all necessary data
    shaps = _get_data_from_npy( args.file_npy )
    feats = _get_data_from_json( args.file_json )

    if args.file_csv:
        X = _get_data_from_csv( args.file_csv , args.col_out )
    else:
        X = None
    
    print( '\nShaps shape: ', shaps.shape )
    print( '\nNumber of features: ', len( feats ) )
    print( 'Feature names:\n', feats )


    # Plot ROC curves
    print( '\nPlot SHAP separate summary plots ....' )
    _plot( feats )
    #_plot_separate_summary_plots( feats , shaps , args.file_npy , X=X )

    print( '\nPlot SHAP composite summary plot ....' )
    #_plot_composite_summary_plot( feats , shaps , args.file_npy , X=X )

    print( '\n\n' )




# =============================================================================
# CALL TO MAIN
# =============================================================================

if __name__ == '__main__':
    main()

