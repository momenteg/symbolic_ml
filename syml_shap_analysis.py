'''
Analysis of shap values
'''

from __future__ import print_function , division
import argparse
import sys , os
import json
import numpy as np
import pandas as pd

from catboost import Pool




# =============================================================================
# MATPLOTLIB WITH DISABLED X-SERVER 
# =============================================================================

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt

import shap




# =============================================================================
# MY VARIABLE FORMAT 
# =============================================================================

myint    = np.int
myfloat  = np.float32
myfloat2 = np.float64




# =============================================================================
# MATPLOTLIB WITH DISABLED X-SERVER 
# =============================================================================

SEP = ','




# =============================================================================
# Get input arguments 
# =============================================================================

def _example():
    print( '\n\nBar plot of feature importance by aggregating one-hot encoded variables:\n' )
    print( '"""\npython syml_shap_analysis.py -i1 syml_shap_class_2018-11-16-09-22-46_8305569712766266153.npy -i2 lavolta_01_baseline_asl_asl2_asl3.csv -i3 ANONID,NUMEX,NUMEX-BINARY\n"""\n\n' )


def _get_args():
    parser = argparse.ArgumentParser( description     = 'Create analysis plots for SHAP values',
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter    ,
                                      add_help        = False )
    
    parser.add_argument('-i1', '--file_npy', dest='file_npy',
                        help='Specify input NPY file containing shap values')

    parser.add_argument('-i2', '--file_csv', dest='file_csv',
                        help='Specify input CSV file containing the input dataset')
     
    parser.add_argument('-i3', '--col_excl', dest='col_excl',
                        help='Specify columns to exclude separated by a ","')
    
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
    
    if os.path.isfile( args.file_npy ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input SYML NPY file does not exist!\n')
 
    if args.file_csv is None:
        parser.print_help()
        sys.exit('\nERROR: Input CSV data not specified!\n')

    if os.path.isfile( args.file_csv ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input CSV data file does not exist!\n')
    
    return args




# =============================================================================
# Get data from NPY
# =============================================================================

def _get_data_from_npy( file_npy ):
    # Load NPY file
    shaps   = np.load( file_npy )
    shaps   = shaps[:,:,:shaps.shape[2]-1] 
    

    # Get shape
    if len( shaps.shape ) == 3:
        n_shaps = shaps.shape[0]
    elif len( shaps.shape ) == 2:
        n_shaps = 1
    else:
        sys.exit( '\nERROR ( _get_data_from_npy ): issue with shape of SHAP array ---> ' + ','.join( shaps.shape ) + '!\n\n' ) 

    return shaps , n_shaps




# =============================================================================
# Get data from CSV
# =============================================================================

def _get_data_from_csv( file_csv , col_excl ):
    # Read CSV
    X = pd.read_csv( file_csv , sep=SEP )
    
    
    # Split column names
    col_excl = col_excl.split( SEP )

    aux = []
    for i in range( len( col_excl ) ):
        if col_excl[i] in X.keys():
            aux.append( col_excl[i] )
        else:
            print( '------ Warning column ', col_excl[i],' to be excluded is not present in data frame' )

    if len( aux ):
        X = X.drop( aux , axis=1 )
    
    return X




# =============================================================================
# Get indices of categorical variables
# =============================================================================

def _get_indices_categ( X ):
    inds_cat = np.argwhere( X.dtypes == object ).reshape( -1 ).tolist()

    if len( inds_cat ) == 0:
        inds_cat = None

    return inds_cat




# =============================================================================
# Plot shap analysis
# =============================================================================

def _plot_shap_analysis( shaps , n_shaps , X , file_in , inds_cat=None , ext='.png' ):
    # Catboost data pool
    if inds_cat is None: 
        xpool   = Pool( X )
        X_nocat = X
    else:
        print( X.keys()[ inds_cat ] )
        xpool   = Pool( X , cat_features=inds_cat )
        X_nocat = X.drop( X.keys()[ inds_cat ] , axis=1 )
  
 
    # Output stem and extension
    path_out = os.path.join( os.path.dirname( file_in ) , 'shap_plots' )
    basename = os.path.basename( file_in )
    basename = basename[:len( basename )-4]

    if os.path.isdir( path_out ) is False:
        os.mkdir( path_out )

    stem_out = os.path.join( path_out , basename )


    # For loop on SHAP arrays 
    for i in range( n_shaps ):
        print( '\n----> Plotting for SHAP array n.', i )

        # Case A ---> n_shaps == 1
        if n_shaps == 1:
            shaps_aux = shaps.copy()
            str_num   = ''
            
        # Case B ---> n_shaps > 1
        else:
            # String to label the fold of CV
            shaps_aux = shaps[i].copy()

            if i < 10:
                str_num = 'fold0' + str( i )
            else:
                str_num = 'fold' + str( i )


        # Point-wise impact on modeling
        fig       = shap.summary_plot( shaps_aux , X , show=False )
        save_plot = stem_out + '_shap-plot-impact-point_' + str_num + ext
        plt.tight_layout()
        plt.savefig( save_plot , transparent=True , dpi=300 )
        plt.close()
        print( '\n------------> .... created ', save_plot )
              

        # Mean impact on modeling
        fig = shap.summary_plot( shaps_aux , X , plot_type='bar' , show=False )
        save_plot = stem_out + '_shap-plot-impact-mean_' + str_num + ext
        plt.tight_layout()
        plt.savefig( save_plot , transparent=True , dpi=300 )
        plt.close()
        print( '\n------------> .... created ', save_plot )
        

        # Plot dependency with respect to a variable
        '''
        inds_sel    = np.setdiff1d( np.arange( shaps_aux.shape[1] ) , inds_cat )
        shaps_nocat = shaps_aux[:,inds_sel]

        for j in range( len( X_nocat.keys() ) ):
            print( X_nocat.keys()[j] )
            print( shaps_nocat.shape )
            print( X_nocat.shape )
            print( np.unique( X_nocat[ X_nocat.keys()[j] ] ) )
            
            fig = shap.dependence_plot(  X_nocat.keys()[j] , shaps_nocat , X_nocat , show=False )
            save_plot = stem_out + '_shap-plot-depend-' + X_nocat.keys()[j] + '_' + str_num + ext
            plt.tight_layout()
            plt.savefig( save_plot , transparent=True , dpi=300 )
            plt.close()
            print( '\n--------------------> .... created ', save_plot )
        '''

       
    # Plot Composite shap plots
    if n_shaps > 1:
        print( '\n\n----> Plotting composite SHAP plots' )
        shaps_comp = np.mean( shaps , axis=0 )
        _plot_shap_analysis( shaps_comp , 1 , X , file_in , inds_cat=inds_cat , ext='.png' )
 


        
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
    print( 'Input CSV data file: ', args.file_csv )
    print( 'Columns to exclude: ', args.col_excl )


    # Get all necessary data
    shaps , n_shaps = _get_data_from_npy( args.file_npy )
    X               = _get_data_from_csv( args.file_csv , args.col_excl )

    print( '\nShaps shape: ', shaps.shape )
    print( '\nNumber of features: ', len( X.keys() ) )
    print( 'Feature names:\n', X.keys() )


    # Get categorical indeces
    inds_cat = _get_indices_categ( X )
    
    if inds_cat is not None:
        print( '\nIndices corresponding to categorical features: ', inds_cat )
        print( 'Categorical features: ', X.keys()[ inds_cat ] )


    # Checkpoint
    if n_shaps == 1:
        shape_aux = shaps.shape
    else:
        shape_aux = shaps.shape[1:]        

    if shape_aux != X.shape:
        sys.exit( '\nERROR ( main ): shape of SHAP values (' + str( shape_aux[0] ) + ',' + \
                str( shape_aux[1] ) + ') does not coincide ' + \
               ' with shape of data frame (' + str( X.shape[0] ) + ',' + str( X.shape[1] ) + ')!\n\n' )


    # Plot SHAP analysis
    print( '\n\nPlot SHAP analysis plots ....' )
    _plot_shap_analysis( shaps , n_shaps , X , args.file_npy , inds_cat=inds_cat )

    print( '\n\n' )




# =============================================================================
# CALL TO MAIN
# =============================================================================

if __name__ == '__main__':
    main()

