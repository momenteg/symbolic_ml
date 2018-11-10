'''
    SYMBOLIC MODELING CLASS                                                                                     
'''
 
# Author   : Filippo Arcadu
# Date     : 10/11/2018


from __future__ import division , print_function
import os , sys
import pandas as pd
import numpy as np
import yaml , json




################################################
#
# MY VARIABLE FORMAT
#
################################################

myint    = np.int
myfloat  = np.float32
myfloat2 = np.float64




################################################
#
# CSV SEPARATOR
#
################################################

SEP = ';'




################################################
#
# ML ALGORITHM LIST
#
################################################

ALGS = [ 'rf' , 'xgboost' ]




################################################
#
# CLASS SYMBOLIC ML
#
################################################

class SYML:
    
    ############################################
    #
    #  Init
    #
    ############################################

    def init( self , file_in , file_cfg ):
        # Assign to class
        self._file_in  = file_in
        self._file_cfg = file_cfg

        
        # Read input CSV file
        self._read_input_table() 


        # Read config file
        self._read_config_file()


        # Get outcome column
        self._get_outcome()


        # Get feature columns
        self._get_features()


        # Slice data frame
        self._slice_data_frame()


        # 1-hot encoding of non-numeric variable
        self._1hot_encoding()



    ############################################
    #
    #  Read input table
    #
    ############################################

    def _read_input_table( self ):
        # Read table
        self._df = pd.read_csv( self._file_in , sep=SEP )


        # Check size
        if self._df.shape[1] <= 1:
            sys.exit( 'ERROR ( SYML -- _read_input_table ): size of input table is (' + \
                      ','.join( self._df.shape ) + ')!\n\n' )


    
    ############################################
    #
    #  Read config file
    #
    ############################################

    def _read_config_file( self ):
        # Read YAML config file
        with open( self._file_cfg , 'r' ) as stream:
            cfg = yaml.load( stream )


        # Get model parameters
        self._alg     = cfg[ 'model' ][ 'algorithm' ]
        self._metric  = cfg[ 'model' ][ 'metric' ]
        self._select  = cfg[ 'model' ][ 'select' ]
        self._exclude = cfg[ 'model' ][ 'exclude' ]


        # Get validation parameters
        self._alg     = cfg[ 'validation' ][ 'testing' ]
        self._metric  = cfg[ 'validation' ][ 'col_constr' ]
        self._exclude = cfg[ 'validation' ][ 'kfold_cv' ]
        self._n_folds = cfg[ 'validation' ][ 'n_folds' ]


        # Get random forest parameters
        if self._alg == 'rf':
            chapter = 'rf_params'

            if chapter in cfg.keys():
                # No. estimators
                if 'n_estimators' in cfg[ chapter ].keys():
                    self._n_estimators = self._parse_arg( cfg[ chapter ][ 'n_estimators' ] ,
                                                      arg_type = 'int'                     ,
                                                      var      = chapter + ':n_estimators' )
                else:
                    self._n_estimators = 10

                # Criterion
                if 'criterion' in cfg[ chapter ].keys():
                    self._criterion = self._parse_arg( cfg[ chapter ][ 'criterion' ] ,
                                                       arg_type = 'string'               ,
                                                       var      = chapter + ':criterion' )
                else:
                    self._criterion = 'gini'

                # Max depth of tree
                if 'max_depth' in cfg[ chapter ].keys():
                    self._max_depth = self._parse_arg( cfg[ chapter ][ 'max_depth' ] ,
                                                       arg_type = 'int'                  ,
                                                       var      = chapter + ':max_depth' )
                else:
                    self._max_depth = None

                # Min no. samples to split node
                if 'min_samples_split' in cfg[ chapter ].keys():
                    self._min_samples_split = self._parse_arg( cfg[ chapter ][ 'min_samples_split' ] ,
                                                               arg_type = 'int'                          ,
                                                               var      = chapter + ':min_samples_split' )
                else:
                    self._min_samples_split = 2

                # Min no. samples per leaf
                if 'min_samples_leaf' in cfg[ chapter ].keys():
                    self._min_samples_leaf = self._parse_arg( cfg[ chapter ][ 'min_samples_leaf' ] ,
                                                              arg_type = 'int'                         ,
                                                              var      = chapter + ':min_samples_leaf' )
                else:
                    self._min_samples_leaf = 1 

                # Use bootstrap
                if 'bootstrap' in cfg[ chapter ].keys():
                    self._bootstrap = self._parse_arg( cfg[ chapter ][ 'bootstrap' ] ,
                                                       arg_type = 'bool'                 ,
                                                       var      = chapter + ':bootstrap' )
                else:
                    self._bootstrap = True 

                # Use out-of-bag-score
                if 'oob_score' in cfg[ chapter ].keys():
                    self._oob_score = self._parse_arg( cfg[ chapter ][ 'oob_score' ] ,
                                                       arg_type = 'bool'                 ,
                                                       var      = chapter + ':oob_score' )
                else:
                    self._oob_score = False 

                # Class balance
                if 'class_weights' in cfg[ chapter ].keys():
                    self._class_weight = self._parse_arg( cfg[ chapter ][ 'class_weights' ] ,
                                                          arg_type = 'string'                   ,
                                                          var      = chapter + ':class_weigths' )
                else:
                    self._class_weight = 1

            else:
                sys.exit( '\nERROR ( SYML -- _read_config_file ): < rf_params > not found in ' + \
                          self._file_cfg + '!\n\n' ) 

        
        # Get random forest parameters
        elif self._alg == 'xgboost':
            chapter = 'xg_params'

            if chapter in cfg.keys():
                # Booster type
                if 'booster' in cfg[ chapter ].keys():
                    self._booster = self._parse_arg( cfg[ chapter ][ 'booster' ] ,
                                                     arg_type = 'string'         ,
                                                     var      = chapter + ':booster' )
                else:
                    self._booster = 'gbtree'

                # Feature dimension
                if 'num_feature' in cfg[ chapter ].keys():
                    self._num_feature = self._parse_arg( cfg[ chapter ][ 'num_feature' ] ,
                                                         arg_type = 'string'               ,
                                                         var      = chapter + ':num_feature' )
                else:
                    self._num_feature = None

                # Step size shrinkage
                if 'eta' in cfg[ chapter ].keys():
                    self._eta = self._parse_arg( cfg[ chapter ][ 'eta' ] ,
                                                 arg_type = 'float'      ,
                                                 var      = chapter + ':eta' )
                else:
                    self._eta = 0.3

                # Minimum loss reduction 
                if 'gamma' in cfg[ chapter ].keys():
                    self._gamma = self._parse_arg( cfg[ chapter ][ 'gamma' ] ,
                                                   arg_type = 'float'        ,
                                                   var      = chapter + ':gamma' )
                else:
                    self._gamma = 0 

                # L2-regularization
                if 'lambda' in cfg[ chapter ].keys():
                    self._lambda = self._parse_arg( cfg[ chapter ][ 'lambda' ] ,
                                                    arg_type = 'float'         ,
                                                    var      = chapter + ':lambda' )
                else:
                    self._lambda = 1 

                # Max depth
                if 'max_depth' in cfg[ chapter ].keys():
                    self._max_depth = self._parse_arg( cfg[ chapter ][ 'max_depth' ] ,
                                                       arg_type = 'int'              ,
                                                       var      = chapter + ':max_depth' )
                else:
                    self._max_depth = 6

                # L1-regularization
                if 'alpha' in cfg[ chapter ].keys():
                    self._alpha = self._parse_arg( cfg[ chapter ][ 'alpha' ] ,
                                                   arg_type = 'float'        ,
                                                   var      = chapter + ':alpha' )
                else:
                    self._alpha = 0.0 

                # Class balance
                if 'tree_method' in cfg[ chapter ].keys():
                    self._tree_method = self._parse_arg( cfg[ chapter ][ 'tree_method' ] ,
                                                         arg_type = 'string'                   ,
                                                         var      = chapter + ':tree_method' )
                else:
                    self._class_weight = 1

            else:
                sys.exit( '\nERROR ( SYML -- _read_config_file ): < xg_params > not found in ' + \
                          self._file_cfg + '!\n\n' ) 
    


    ############################################
    #
    #  Parse argument from config file
    #
    ############################################
    
    def _parse_arg( self , arg , arg_type='string' , var='variable' ):
        try:
            # Case "," is present
            if ',' in arg:
                entries   = arg.split( ',' )
                n_entries = len( entries )
        
            # Case ":" is present
            elif ':' in arg:
                entries   = arg.split( ':' )
                n_entries = len( entries )

            # Case single entry
            else:
                entries   = arg
                n_entries = 1

            # Convert to either integer or float
            if arg_type != 'string' and arg_type != 'bool':
                if n_entries > 1:
                    for i in range( n_entries ):
                        if arg_type == 'int':
                            entries[i] = myint( entries[i] )
                        elif arg_type == 'float':
                            entries[i] = myfloat( entries[i] )

            return entries

        except:
            sys.exit( '\nERROR ( SYML -- _parse_arg ): issue with ' + var + '!\n\n' )



    ############################################
    #
    #  Get outcome variable
    #
    ############################################

    def _get_outcome( self ):
        if self._col_out in self._df.keys():
            self._y = self._df[ self._col_out ].values

        else:
            sys.exit( '\nERROR ( SYML -- _get_outcome ): outcome column ' + \
                      self._col_out + ' is not in input data frame!\n'    + \
                      'Available Keys: (' + ','.join( self._df.keys() ) + ')\n\n' )    

       
     
    ############################################
    #
    #  Get feature variables
    #
    ############################################

    def _get_features( self ):
        # Initialize list of feature column names
        self._feats = []


        # Do selection of columns
        if self._select is not None:
            if '*' in self._select:
                select  = self._select.strip( '*' )
               
                for i in range( len( self._df.keys() ) ):
                    if select in self._df.keys()[i]:
                        self._feats.append( self._df.keys()[i] )
                
                if len( feats ) == 0:
                    sys.exit( '\nERROR ( SYML -- _get_features ): no feature columns ' + \
                              'were selected with ' + self._select + '!\n'    + \
                              'Available Keys: (' + ','.join( self._df.keys() ) + ')\n\n' )    

            else:
                for i in range( len( self._select ) ):
                    if self._select[i] in self._df.keys():
                        self._feats.append( self._select[i] )
                    else:
                        sys.exit( '\nERROR ( SYML -- _get_features ): feature columns ' + \
                                  self._select[i] + ' is not in input data frame!\n'    + \
                                  'Available Keys: (' + ','.join( self._df.keys() ) + ')\n\n' )    

        else:
            self._feats = self._df.keys()[:]
            

        # Remove outcome variable if present
        if self._col_out in self._feats:
            self._feats.remove( self._col_out )


        # Exclude columns
        if self._exclude is not None:
            if '*' in self._exclude:
                exclude = self._exclude.strip( '*' )
                
                for i in range( len( self._feats ) ):
                    if exclude in self._feats[i]:
                        self._feats.remove( self._feats[i] )
                
                if len( self._feats ) == 0:
                    sys.exit( '\nERROR ( SYML -- _get_features ): no feature columns ' + \
                              'were selected with ' + self._exclude + '!\n'    + \
                              'Available Keys: (' + ','.join( self._df.keys() ) + ')\n\n' )    

            else:
                for i in range( len( self._exclude ) ):
                    if self._exclude[i] in self._df.keys():
                        self._feats.remove( self._exclude[i] )
                    else:
                        sys.exit( '\nERROR ( SYML -- _get_features ): feature columns ' + \
                                  self._exclude[i] + ' is not in input data frame!\n'    + \
                                  'Available Keys: (' + ','.join( self._df.keys() ) + ')\n\n' )    
            
        

    ############################################
    #
    #  Check feature amd outcome variables
    #
    ############################################

    def _check_vars( self ):
        # Outcome variable
        print( '\n\nOUTCOME' )
        print( '\nVariable: ', self._col_out )
        print( 'Unique values:\n', np.unique( self._df[ self._col_out ].values ) )
        print( 'Type: ', self._df[ self._col_out ].values.dtype )


        # Feature variables
        print( '\n\nFEATURES' )
        for i in range( len( self._feats ) ):
            print( '\nVariable: ', self._feats[i] )
            print( 'Unique values:\n', np.unique( self._df[ self._feats[i] ].values ) )
            print( 'Type: ', self._df[ self._feats[i] ].values.dtype )

       

    ############################################
    #
    #  Convert non-numeric to 1-hot encoded
    #
    ############################################

    def _slice_data_frame( self ):
        self._df_enc = self._df[ self._feats + [ self._col_out ] ]



    ############################################
    #
    #  Convert non-numeric to 1-hot encoded
    #
    ############################################

    def _1hot_encoding( self ):
        # Do 1-hot encoding
        keys = self._df_enc.keys()
        keys.remove( self._col_out )

        keys_str = []
        for i in range( len( keys ) ):
            if self._is_numeric(  keys[i] ) is False:
                keys_str.append( keys[i] )    

        if len( keys_str ):
            self._dl_enc = pd.get_dummies( self._dl_enc , columns=keys_str )


        # Do label encoding
        if self._is_numeric( self._col_out ) is False:
            from sklearn import preprocessing
                
            label_enc = preprocessing.LabelEncoding()           
            arr       = self._dl_enc[ self._col_out ].values
            
            label_enc.fit( arr )
            
            arr                           = label_enc.transform( arr )
            self._dl_enc[ self._col_out ] = arr
             
  

    def _is_numeric( self , key ):
        if self._df[ key ].dtype == myint or \
            self._df[ key ].dtype == myfloat or \
            self._df[ key ].dtype == myfloat2:
            return True

        else:
            return False



    ############################################
    #
    #  Training
    #
    ############################################

    def _train():
        # Split dataset
        self._split_data()


        # 
