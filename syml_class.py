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
import glob
import numbers
import itertools

from sklearn.metrics import accuracy_score , precision_score , recall_score , r2_score , \
                            f1_score , cohen_kappa_score , roc_curve , auc 
from sklearn.externals import joblib            




# =============================================================================
# MY VARIABLE FORMAT 
# =============================================================================

myint    = np.int
myfloat  = np.float32
myfloat2 = np.float64




# =============================================================================
# CSV SEPARATOR 
# =============================================================================
SEP = ','




# =============================================================================
# ML ALGORITHM LIST 
# =============================================================================

ALGS = [ 'rf-class' , 'rf-regr' , 'xgboost-class' , 'xgboost-regr' ]




# =============================================================================
# TYPES OF CROSS-VALIDATION
# =============================================================================

CV_TYPES = [ 'k-fold' , 'resampling' ]




# =============================================================================
# ALGORITHM PARAMETER LIST 
# =============================================================================

RF_PARAMS = [ 'n_estimators' , 'criterion' , 'max_depth' , 'min_samples_split' ,
              'min_samples_leaf' , 'bootstrap' , 'oob_score' , 'class_weight' ]

XG_PARAMS = [ 'booster' , 'num_feature' , 'eta' , 'gamma' , 'max_depth' , 
              'lambda' , 'alpha' , 'tree_method' ] 




# =============================================================================
# CLASS SYMBOLIC ML 
# =============================================================================

class SYML:
    
    # ===================================
    # Init 
    # ===================================
    
    def __init__( self , file_in , file_cfg , path_out='./' , label=None , verbose=False ):
        # Assign to class
        self._file_in   = file_in
        self._file_cfg  = file_cfg
        self._add_label = label
        self._path_out  = path_out
        self._verbose   = verbose

        
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


        # Create label for output files
        self._create_output_label()


        # Create all combinations for parameter exploration
        self._create_param_grid()



    # ===========================================
    # Read input table
    # ===========================================
    
    def _read_input_table( self ):
        # Read table
        self._df = pd.read_csv( self._file_in , sep=SEP )


        # Check size
        if self._df.shape[1] <= 1:
            sys.exit( 'ERROR ( SYML -- _read_input_table ): size of input table is (' + \
                      ','.join( self._df.shape ) + ')!\n\n' )



    # ===================================
    # Read config file 
    # ===================================
    
    def _read_config_file( self ):
        # Read YAML config file
        with open( self._file_cfg , 'r' ) as stream:
            cfg = yaml.load( stream )


        # Get model parameters
        self._alg     = cfg[ 'model' ][ 'algorithm' ]
        self._metric  = cfg[ 'model' ][ 'metric' ]
        self._col_out = cfg[ 'model' ][ 'col_out' ]
        self._select  = cfg[ 'model' ][ 'select' ]
        self._exclude = cfg[ 'model' ][ 'exclude' ]

        if self._select == 'None':
            self._select = None

        if self._exclude == 'None':
            self._exclude = None


        # Get validation parameters
        self._test_perc  = cfg[ 'validation' ][ 'testing' ]
        self._col_constr = cfg[ 'validation' ][ 'col_constr' ]
        self._cv_type    = cfg[ 'validation' ][ 'cv_type' ]
        self._n_splits   = cfg[ 'validation' ][ 'n_folds' ]

        if self._col_constr == 'None':
            self._col_constr = None
        
        if self._cv_type == 'None':
            self._cv_type = None


        # Get type of task
        if '-class' in self._alg:
            self._task_type = 'classification'
        elif '-regr' in self._alg:
            self._task_type = 'regression'


        # Get random forest parameters
        self._params = []

        if 'rf-' in self._alg:
            chapter = 'rf_params'

            if chapter in cfg.keys():
                for i in range( len( RF_PARAMS ) ):
                    if RF_PARAMS[i] in cfg[ chapter ].keys():
                        self._params.append( self._parse_arg( cfg[ chapter ][ RF_PARAMS[i] ] ,
                                                              var = chapter + ':' + RF_PARAMS[i] ) )
                    else:
                        sys.exit( 'ERROR ( SYML -- _read_config_file ): param ' + RF_PARAMS[i] + \
                                  ' is not contained inside ' + self._file_cfg + '!\n\n' )

            else:
                sys.exit( '\nERROR ( SYML -- _read_config_file ): < rf_params > not found in ' + \
                          self._file_cfg + '!\n\n' ) 

        
        # Get xgboost parameters
        elif 'xgboost-' in self._alg:
            chapter = 'xg_params'

            if chapter in cfg.keys():
                for i in range( len( XG_PARAMS ) ):
                    if XG_PARAMS[i] in cfg[ chapter ].keys():
                        self._params.append( self._parse_arg( cfg[ chapter ][ XG_PARAMS[i] ] ,
                                                              var = chapter + ':' + XG_PARAMS[i] ) )
                    else:
                        sys.exit( 'ERROR ( SYML -- _read_config_file ): param ' + XG_PARAMS[i] + \
                                  ' is not contained inside ' + self._file_cfg + '!\n\n' )

            else:
                sys.exit( '\nERROR ( SYML -- _read_config_file ): < xg_params > not found in ' + \
                          self._file_cfg + '!\n\n' )


        # Print params
        if self._verbose:
            print( 'Params:\n', self._params )
   

    
    # ===================================
    # Parse argument from config file 
    # ===================================
 
    def _parse_arg( self , arg , var='variable' ):
        try:
            # Case single entry
            if self._is_single_variable( arg ): 
                return [ arg ]


            # Case multiple entries
            else:
                # Case "," is present
                if ',' in arg:
                    entries   = arg.split( ',' )
                    n_entries = len( entries )
                    arg_type  = self._get_type( entries[0] )
        
                # Case ":" is present
                elif ':' in arg:
                    entries   = arg.split( ':' )
                    n_entries = len( entries )
                    arg_type  = self._get_type( entries[0] )

                # Convert to either integer or float
                if arg_type != str and arg_type != bool:
                    if n_entries > 1:
                        for i in range( n_entries ):
                            if arg_type == myint:
                                entries[i] = myint( entries[i] )
                            elif arg_type == myfloat:
                                entries[i] = myfloat( entries[i] )

                        if ':' in arg and ( arg_type == myint or argtype == myfloat ):
                            entries = np.linspace( myfloat( entries[0] ) , 
                                                   myfloat( entries[1] ) ,
                                                   myfloat( entries[2] ) )

                        if arg_type == myint:
                            entries = entries.astype( myint )

                        entries = entries.tolist()
        
        except:
            sys.exit( '\nERROR ( SYML -- _parse_arg ): issue with ' + var + '!\n\n' )

        return entries



    def _is_single_variable( self , arg ):
        if isinstance( arg , myint ) or \
            isinstance( arg , myfloat ) or \
                isinstance( arg , myfloat2 ) or \
                    isinstance( arg , bool )   or \
                        ( isinstance( arg , str ) and ( ',' not in arg ) and ( ':' not in arg ) ): 
            return True
        
        else:
            return False



    def _get_type( self , var ):
        try:
            num = myfloat( var )

            if '.' in var:
                arg_type = myfloat
            else:
                arg_type = myint

        except:
            if var == 'None':
                arg_type = None
            elif ( var is True ) or ( var == 'True' ) or ( var is False ) or ( var == 'False' ):
                arg_type = bool
            else:
                arg_type = str

        return arg_type



    # ===========================================
    # Get outcome variable 
    # ===========================================

    def _get_outcome( self ):
        if self._col_out in self._df.keys():
            self._y = self._df[ self._col_out ].values
            
            if self._task_type == 'classification':
                self._n_classes = len( np.unique( self._y ) )

                if self._n_classes > 2:
                    sys.exit( '\nERROR ( SYML -- _get_outcome ): outcome column ' + \
                                self._col_out + ' contains more than 2 classes, impossible to do classification!\n\n' )
            
        else:
            sys.exit( '\nERROR ( SYML -- _get_outcome ): outcome column ' + \
                      self._col_out + ' is not in input data frame!\n'    + \
                      'Available Keys: (' + ','.join( self._df.keys() ) + ')\n\n' )    

       
    
    # ===========================================
    # Get feature variables
    # ===========================================   

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
            self._feats = self._df.keys()[:].tolist()


        # Remove outcome variable if present
        if self._col_out in self._feats:
            self._feats.remove( self._col_out )


        # Exclude columns
        self._exclude_feats = []

        if self._col_constr is not None:
            self._feats.remove( self._col_constr ) 
            self._exclude_feats.append( self._col_constr )

        if self._exclude is not None:
            if '*' in self._exclude:
                exclude = self._exclude.strip( '*' )
                
                for i in range( len( self._feats ) ):
                    if exclude in self._feats[i]:
                        self._feats.remove( self._feats[i] )
                        self._exclude_feats.append( self._feats[i] )
                
                if len( self._feats ) == 0:
                    sys.exit( '\nERROR ( SYML -- _get_features ): no feature columns ' + \
                              'were selected with ' + self._exclude + '!\n'    + \
                              'Available Keys: (' + ','.join( self._df.keys() ) + ')\n\n' )    

            else:
                for i in range( len( self._exclude ) ):
                    if self._exclude[i] in self._df.keys():
                        self._feats.remove( self._exclude[i] )
                        self._exclude_feats.append( self._exclude[i] )
                    else:
                        sys.exit( '\nERROR ( SYML -- _get_features ): feature columns ' + \
                                  self._exclude[i] + ' is not in input data frame!\n'    + \
                                  'Available Keys: (' + ','.join( self._df.keys() ) + ')\n\n' )    
           


    # ===========================================
    # Check feature and outcome variables
    # ===========================================

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

       

    # ===========================================
    # Slicing data frame
    # ===========================================

    def _slice_data_frame( self ):
        if self._col_constr is None:
            self._df_enc = self._df[ self._feats + [ self._col_out ] ]
        else:
            self._df_enc = self._df[ self._feats + [ self._col_constr ] + [ self._col_out ] ]


    # ===========================================
    # Convert non-numeric to 1-hot
    # ===========================================

    def _1hot_encoding( self ):
        # Do 1-hot encoding
        keys = self._df_enc.keys().tolist()
        keys.remove( self._col_out )

        keys_str = []
        for i in range( len( keys ) ):
            if self._is_numeric(  keys[i] ) is False:
                keys_str.append( keys[i] )    

        if len( keys_str ):
            self._df_enc   = pd.get_dummies( self._df , columns=keys_str )
            self._keys_enc = keys_str 
        else:
            self._df_enc   = self._df
            self._keys_enc = keys_str


        # Do label encoding
        if self._is_numeric( self._col_out ) is False:
            from sklearn import preprocessing
                
            label_enc = preprocessing.LabelEncoding()           
            arr       = self._df_enc[ self._col_out ].values
            
            label_enc.fit( arr )
            
            arr                           = label_enc.transform( arr )
            self._df_enc[ self._col_out ] = arr
            
            if self._keys_enc is None:
                self._keys_enc = [ self._col_out ]
            else:
                self._keys_enc += [ self._col_out ]


        # New keys to use for training
        if len( self._keys_enc ):
            self._key_enc_feat = self._df_enc.keys().tolist()
            self._key_enc_feat.remove( self._col_out )

            print( self._exclude_feats )
            if len( self._exclude_feats ):
                for feat in self._exclude_feats:
                    self._key_enc_feat.remove( feat )

        else:
            self._key_enc_feat = self._feats[:]

        self._key_enc_out = self._col_out   


    def _is_numeric( self , key ):
        if self._df[ key ].dtype == myint or \
            self._df[ key ].dtype == myfloat or \
            self._df[ key ].dtype == myfloat2:
            return True

        else:
            return False



    # ===========================================
    # Create output label
    # ===========================================

    def _create_output_label( self ):
        from time import gmtime, strftime
        import random        

        str_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        hash_out = str( random.getrandbits( 64 ) )

        if self._add_label is None:
            self._label_out = self._alg + '_' + str_time + '_' + hash_out
        else:
            self._label_out = self._alg + '_' + self._add_label + '_' + str_time + '_' + hash_out



    # ===========================================
    # Create param grid
    # ===========================================

    def _create_param_grid( self ):
        if 'rf-' in self._alg or 'xgboost-' in self._alg:
            self._param_combs = list( itertools.product( self._params[0] , self._params[1] ,
                                                         self._params[2] , self._params[3] ,
                                                         self._params[4] , self._params[5] ,
                                                         self._params[6] , self._params[7] ) )  
        


    # ===========================================
    # Training
    # ===========================================

    def _train( self ):
        # Split dataset
        print( '\n\nSplitting input data frame ....' )
        self._split_data( save=True )


        # Run training 
        print( '\n\nTraining model ....' )
        self._run_train()



    # ===========================================
    # Split data
    # ===========================================

    def _split_data( self , save=False ):
        from sklearn.model_selection import StratifiedKFold

        # Get constrained column
        if self._col_constr is None:
            inds  = np.arange( self._df.shape )
            y_sel = self._df[ self._col_out ].values
            
            print( '\t.... splitting occurring with respect to no column' )

        else:
            constr              = self._df[ self._col_constr ].values
            constr_un , inds_un = np.unique( constr , return_index=True )
            
            inds  = np.arange( len( inds_un ) )
            y     = self._df[ self._col_out ].values
            y_sel = y[ inds_un ]

            print( '\t.... splitting occurring with respect to column ', self._col_constr,
                   ' that contains ', len( inds ),' unique values' )

       
        # Initialize lists
        inds_train = [];  inds_test = []


        # Case multiple splits to do k-fold cross-validation
        if self._cv_type == 'k-fold':
            strat_kfold = StratifiedKFold( n_splits = self._n_splits )

            for train_index, test_index in strat_kfold.split( inds , y_sel ):
                inds_train.append( train_index )
                inds_test.append( test_index )


        # Case random resampling cross-validation or single split
        else:
            if self._testing < 1.0:
                mem             = self._test_perc
                self._test_perc = 20
                print( 'Warning ( SYML -- _split_data ): selected percentage for testing is too low ' + \
                       '(' + str( mem ) + ')!\nTesting percentage set to ' + str( self._test_perc ) )

            if self._cv_type is None:
                n_test     = myint( self._df.shape[0] * self._test_perc / 100.0 )
            
                inds_test  = random.sample( inds , n_test )
                inds_train = np.setdiff1d( inds , inds_test )
            
                inds_test  = [ inds_test.tolist() ] 
                inds_train = [ inds_train.tolist() ]

            elif self._cv_type == 'resampling':
                for i in range( self._n_splits ):
                    inds_test  = random.sample( inds , n_test )
                    inds_train = np.setdiff1d( inds , inds_test )
            
                inds_test.append( inds_test.tolist() )
                inds_train.append( inds_train.tolist() )


        # Get real indices
        if self._col_constr is None:
            inds_real_train = inds_train[:]
            inds_real_test  = inds_test[:]

        else:
            inds_real_train = [];  inds_real_test = [] 
            
            for i in range( len( inds_train ) ):
                inds_all_train = []

                for j in range( len( inds_train[i] ) ):
                    pid            = constr_un[ inds_train[i][j] ]
                    inds_all_pid   = np.argwhere( constr == pid )[0]
            
                    for ind in inds_all_pid:  
                        inds_all_train.append( ind )

                inds_real_train.append( inds_all_train )

                inds_all_test = []

                for j in range( len( inds_test[i] ) ):
                    pid           = constr_un[ inds_test[i][j] ]
                    inds_all_pid  = np.argwhere( constr == pid )[0]
            
                    for ind in inds_all_pid:  
                        inds_all_test.append( ind )

                inds_real_test.append( inds_all_test )


        # Checkpoint
        for i in range( len( inds_real_train ) ):
            self._are_train_and_test_disjoint( inds_real_train[i] , inds_real_test[i] )
        
        if self._cv_type == 'k-fold':
            self._are_test_folds_disjoint( inds_real_test )


        # Split data frame
        self._df_train = [];  self._df_test = []
    
        for i in range( len( inds_real_train ) ):
            self._df_train.append( self._df_enc.iloc[ inds_real_train[i] ] )
            self._df_test.append( self._df_enc.iloc[ inds_real_test[i] ] )
        
        
        # Save splits
        if save:
            if len( inds_real_train ) == 1:
                df_aux  = self._df.iloc[ inds_real_train[0] ]
                fileout = os.path.join( self._path_out , 'syml_split_train_' + self._label_out + '.csv' )
                df_aux.to_csv( fileout , sep=SEP , index=False )

                df_aux  = self._df.iloc[ inds_real_test[0] ]
                fileout = os.path.join( self._path_out , 'syml_split_test_' + self._label_out + '.csv' )
                df_aux.to_csv( fileout , sep=SEP , index=False )

            else:
                for i in range( len( inds_real_train ) ):
                    if i < 10:
                        str_fold = '0' + str( i )
                    else:
                        str_fold = str( i )

                    df_aux  = self._df.iloc[ inds_real_train[i] ]
                    fileout = os.path.join( self._path_out , 'syml_split_train_' + str_fold + '_' + \
                                                self._label_out + '.csv' )
                    df_aux.to_csv( fileout , sep=SEP , index=False )

                    df_aux  = self._df.iloc[ inds_real_test[i] ]
                    fileout = os.path.join( self._path_out , 'syml_split_test_' + str_fold + '_' + \
                                                self._label_out + '.csv' )
                    df_aux.to_csv( fileout , sep=SEP , index=False )


    
    # ===========================================
    # Are training and testing sets disjoint?
    # ===========================================

    def _are_train_and_test_disjoint( self , inds_train , inds_test ):
        intersec = np.intersect1d( np.array( inds_train ) , np.array( inds_test ) )

        if len( intersec ):
            sys.exit( '\nERROR ( SYML -- _are_train_and_test_disjoint ): training and testing splits are not disjoint!\n\n' ) 

    
    
    # ===========================================
    # Are training and testing sets disjoint?
    # ===========================================

    def _are_test_folds_disjoint( self , inds_test_list ):
        n_el  = len( inds_test_list )
        ii    = np.arange( n_el ).tolist()
        combs = list( itertools.combinations( ii , 2 ) )

        for i in range( len( combs ) ):
            intersec = np.intersect1d( np.array( inds_test_list[combs[i][0]] ) , inds_test_list[combs[i][1]] )

            if len( intersec ):
                sys.exit( '\nERROR ( SYML -- _are_test_folds_disjoint ): testing splits of different folds are not disjoint!\n\n' ) 



    # ===========================================
    # Run algorithm training
    # ===========================================

    def _run_train( self ):
        # Initalize filenames for output files
        self._create_output_filenames()


        # Initialize monitoring metric
        self._init_monitor()


        # Load modules
        if 'rf' in self._alg:
            if self._task_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                Algorithm = RandomForestClassifier
            elif self._task_type == 'regression':
                from sklearn.ensemble import RandomForestRegressor
                Algorithm = RandomForestRegressor
            PARAMS = RF_PARAMS

        elif 'xg_boost' in self._alg:
            if self._task_type == 'classification':
                from xgboost import XGBClassifier
                Algorithm = XGBClassifier
            elif self._task_type == 'regression':
                from xgboost import XGBRegressor
                Algorithm = XGBRegressor
            PARAMS = XG_PARAMS


        # For loop of param grid search
        for i in range( len( self._param_combs ) ):
            print( '\n\t.... training at grid point n.', i,' out of ', len( self._param_combs ) )
           
            # Initialize list of metrics and probabilities
            metrics = [];  trues = [];  probs = []

            # For loop on k-fold for cross-validation
            for j in range( self._n_splits ):
                if self._cv_type == 'k-fold':
                    print( '\n\t\t.... training on fold n.', j )
                elif self._cv_type == 'resampling':
                    print( '\n\t\t.... training split n.', j )

                # Get training data
                x_train = self._df_train[ j ][ self._key_enc_feat ].values.astype( myfloat )
                y_train = self._df_train[ j ][ self._key_enc_out ].values.astype( myfloat )

                if self._verbose:
                    print( '\t\t.... using ', len( self._key_enc_feat ),'features ', self._key_enc_feat )
                    print( '\t\t.... predicting outcome ', self._key_enc_out )

                
                # Get grid point parameters
                kwargs    = {}

                for k in range( len( PARAMS ) ):
                    kwargs[ PARAMS[k] ] = self._param_combs[i][k]
                
                if self._verbose:
                    print( '\t\t.... using kwargs: \n', kwargs )


                # Initialize algorithm
                self._clf = Algorithm( **kwargs )


                # Fit algorithm to training data
                self._clf.fit( x_train , y_train )


                # Get testing data
                x_test = self._df_test[ j ][ self._key_enc_feat ].values
                y_test = self._df_test[ j ][ self._key_enc_out ].values
 
                
                # Predict on testing data
                y_prob = self._clf.predict_proba( x_test )


                # Compute testing metrics
                metrics.append( self._compute_metrics( y_test , y_prob ) )
                trues.append( y_test.tolist() )
                probs.append( y_prob.tolist() )

            
            # --------- Outside the loop on data splits for cross-validation
            
            print( '' )

            # Evaluate metrics
            save = self._evaluate_metrics( metrics ) 
            
            
            # Save update logger
            self._update_logger( kwargs , n_grid=i , save=save )


            if save:
                # Save model if selected metric has improved
                self._save_model()


                # Save model if selected metric has improved
                self._save_history( trues , probs )


                # Save config file
                self._save_config( kwargs )



    # ===========================================
    # Create output filenames
    # ===========================================

    def _create_output_filenames( self ):
        # CSV logger
        self._file_logger = os.path.join( self._path_out , 'syml_logger_' + self._label_out + '.csv' )

        # Model
        self._file_model = os.path.join( self._path_out , 'syml_model_' + self._label_out + '.pkl' )

        # Predictions on testing
        self._file_histo = os.path.join( self._path_out , 'syml_history_' + self._label_out + '.json' )
        
        # Predictions on testing
        self._file_yaml = os.path.join( self._path_out , 'syml_config_' + self._label_out + '.yml' )
        
    
    
    # ===================================
    # Initializing monitoring metric.
    # ===================================

    def _init_monitor( self ):
        if self._metric == 'mse':
            self._metric_monitor = 1e10
        else:
            self._metric_monitor = 0


   
    # ===================================
    # Compute metrics
    # ===================================
    
    def _compute_metrics( self , y_true , y_prob ):
        # Common print
        print( '\t\tTesting metrics:' )

        
        # Case --> Classification
        if self._task_type == 'classification':

            # Transform probabilities in 1-class prediction
            y_class = y_prob.argmax( axis=1 ).astype( myint )


            # Compute metrics for 2-class problem
            if self._n_classes == 2:
                if len( np.unique( y_class ) ) == 1:
                    self._accuracy    = self._precision   = self._recall    = \
                    self._f1score     = self._cohen_kappa = self._auc       = \
                    self._sensitivity = self._specificity = self._threshold = 0.0
                
                else:
                    self._accuracy    = accuracy_score( y_true , y_class )
                    self._precision   = precision_score( y_true , y_class )
                    self._recall      = recall_score( y_true , y_class )
                    self._f1score     = f1_score( y_true , y_class )
                    self._cohen_kappa = cohen_kappa_score( y_true , y_class )
                    fpr , tpr , thres = roc_curve( y_true , y_prob[:,1] )
                    self._auc         = auc( fpr , tpr )
                    self._calc_sens_and_spec( tpr , fpr , thres )
            
                print( '\t\t\taccuracy: %s - precision: %s - recall: %s - f1score: %s - cohen_kappa: %s' % \
                    ( self._num2str( self._accuracy ) , self._num2str( self._precision ) ,
                      self._num2str( self._recall ) , self._num2str( self._f1score ) , 
                      self._num2str( self._cohen_kappa ) ) )
                print( '\t\t\tauc: %s - sensitivity( %s ): % s - specificity( %s ): %s' % \
                    ( self._num2str( self._auc ) , self._num2str( self._threshold ) , 
                      self._num2str( self._sensitivity ) , self._num2str( self._threshold ) , 
                      self._num2str( self._specificity ) ) )
        
            else:
                self._accuracy    = accuracy_score( y_true , y_class )
                self._cohen_kappa = cohen_kappa_score( y_true , y_class )
                print( '\t\t\taccuracy: %s - cohen_kappa: %s ' % \
                       ( self._num2str( self._accuracy ) , self._num2str( self._cohen_kappa ) ) )    
    

        # Case --> Regression
        elif self._task_type == 'regression':
            self._mse = mean_squared_error( y_true , y_prob )
            self._r2  = r2_score( y_true , y_pred )
            print( '\t\t\tmse: %s - r_squared: %s ' % \
                  ( self._num2str( self._accuracy ) , self._num2str( self._cohen_kappa ) ) ) 

        return getattr( self , '_' + self._metric )



    # ===================================
    # Calculate sensitivity and specificity
    # using Youden's operating point
    # ===================================
        
    def _calc_sens_and_spec( self , tpr , fpr  , thres ):
        # Compute best operating point on ROC curve, define as 
        # the one maximizing the difference ( TPR - FPR ), also
        # known as Youden's operating point
        diff             = tpr - fpr
        i_max            = np.argwhere( diff == np.max( diff ) )
        self._threshold  = thres[ i_max ][0]
        fpr_best         = fpr[i_max]
        tpr_best         = tpr[i_max]
        
        
        # Compute sensitivity and specificity at optimal operating point
        self._sensitivity = tpr[ i_max ][0]
        self._specificity = 1 - fpr[ i_max ][0]
        
    

    # ===================================
    # Evaluate metrics
    # ===================================
    
    def _evaluate_metrics( self , metrics ):
        metrics = np.array( metrics )
        mean    = np.mean( metrics )
        std     = np.std( metrics )

        if self._metric == 'mse':
            if mean < self._metric_monitor:
                self._metric_monitor = mean
                print( '\t\t--------> Testing mean ', self._metric.upper() ,' has improved' )            
                return True
            else:
                print( '\t\t--------> Testing mean ', self._metric.upper() ,' has not improved,  mean: ', 
                        mean, '  monitor: ', self._metric_monitor )            
                return False
            
        else:
            if mean > self._metric_monitor:
                print( '\t\t--------> Testing mean ', self._metric.upper() ,' has improved from ', 
                            self._metric_monitor,' to ', mean )            
                self._metric_monitor = mean
                return True
            else:
                print( '\t\t--------> Testing mean ', self._metric.upper() ,' has not improved,  mean: ', 
                        mean, '  monitor: ', self._metric_monitor )            
                return False

 

    # ===================================
    # Update logger
    # ===================================
    
    def _update_logger( self , df_params , n_grid=0 , save=False ):
        # Common data frame
        df = pd.DataFrame( { 'hash'      : self._label_out   ,
                             'task'      : self._task_type   ,
                             'n_grid'    : n_grid            ,
                             'n_splits'  : self._n_splits    ,
                             'save_model': save              } , index=[0] )


        # Distinguish the various cases        
        if self._task_type == 'classification':
            if self._n_classes == 2:
                aux = pd.DataFrame( { 'test_accuracy'   : self._accuracy    ,
                                      'test_precision'  : self._precision   ,
                                      'test_recall'     : self._recall      , 
                                      'test_f1score'    : self._f1score     , 
                                      'test_auc'        : self._auc         ,
                                      'test_sensitivity': self._sensitivity ,
                                      'test_specificity': self._specificity ,
                                      'test_cohen_kappa': self._cohen_kappa ,
                                      'threshold'       : self._threshold } , index=[0] )
                                 
            else:
                aux = pd.DataFrame( { 'test_accuracy'   : self._accuracy ,
                                      'test_cohen_kappa': self._cohen_kappa }  , index=[0] )

        elif self._task_type == 'regression':
                aux = pd.DataFrame( { 'test_mse'   : self._mse  ,
                                      'test_r2'    : self._r2 } , index=[0] )
   
        df = pd.concat( [ df , aux ] , axis=1 )


        # Add parameters of the grid point
        for i in range( len( df_params.keys() ) ):
            df[ df_params.keys()[i] ] = df_params[ df_params.keys()[i] ] 
 

        # Write to CSV file
        if os.path.isfile( self._file_logger ):
            df.to_csv( self._file_logger , mode='a' , 
                       header=False , sep=SEP , index=False )
        else:
            df.to_csv( self._file_logger , 
                       sep=SEP , index=False )

        print( '\t\t--------> Updated logger ', self._file_logger )

    
    
    # ===================================
    # Save model according to a selected metric
    # ===================================
    
    def _save_model( self ):
        joblib.dump( self._clf , self._file_model )
        print( '\t\t--------> Best model saved to ' , self._file_model )            


    
    # ===================================
    # Save best model according to a selected metric
    # ===================================
    
    def _save_history( self , trues , probs ):
        df = dict( { 'y_true'      : trues               ,
                     'y_prob'      : probs               ,
                     'file_in'     : self._file_in       ,
                     'task'        : self._task_type     ,
                     'algorithm'   : self._alg           ,
                     'feature_cols': self._feats         ,
                     'outcome_col' : self._col_out       ,
                     'constr_col'  : self._col_constr    ,
                     'metric'      : self._metric        ,
                     'peak_value'  : self._metric_monitor } )

        with open( self._file_histo , 'w' ) as fp:
            json.dump( df , fp , sort_keys=True )
            
        print( '\t\t--------> Testing probabilities saved to ', self._file_histo )
       
    
    
    # ===================================
    # Print number as string with a certain precision 
    # ===================================
    
    def _num2str( self , num ):
        return str( round( num , 4 ) )


    
    # ===================================
    # Write config file
    # ===================================
    
    def _save_config( self , kwargs ):
        dict = { 'model': { 'algorithm': self._alg     ,
                            'metric'   : self._metric  ,
                            'col_out'  : self._col_out ,
                            'select'   : self._select  ,
                            'exclude'  : self._exclude } ,
                 'validation': { 'testing'   : self._test_perc  ,
                                 'col_constr': self._col_constr , 
                                 'cv_type'   : self._cv_type    ,
                                 'n_folds'   : self._n_splits    } }

        if 'rf' in self._alg:
            aux = { 'rf_params': kwargs }
        elif 'xg' in self._alg:
            aux = { 'xg_params': kwargs }

        dict.update( aux )

        with open( self._file_yaml , 'w' ) as outfile:
            yaml.dump( dict , outfile , default_flow_style=False )
        
        print( '\t\t--------> Config file has been saved to ', self._file_yaml )     
