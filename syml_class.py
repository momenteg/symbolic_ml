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
# ALGORITHM PARAMETER LIST 
# =============================================================================

RF_PARAMS = [ 'n_estimators' , 'criterion' , 'max_depth' , 'min_samples_split' ,
              'min_samples_leaf' , 'bootstrap' , 'oob_score' , 'class_weights' ]

XG_PARAMS = [ 'booster' , 'num_feature' , 'eta' , 'gamma' , 'max_depth' , 
              'lambda' , 'alpha' , 'tree_method' ] 




# =============================================================================
# CLASS SYMBOLIC ML 
# =============================================================================

class SYML:
    
    # ===================================
    # Init 
    # ===================================
    
    def __init__( self , file_in , file_cfg , label=None ):
        # Assign to class
        self._file_in   = file_in
        self._file_cfg  = file_cfg
        self._add_label = label

        
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
        self._kfold_cv   = cfg[ 'validation' ][ 'kfold_cv' ]
        self._n_folds    = cfg[ 'validation' ][ 'n_folds' ]

        if self._col_constr == 'None':
            self._col_constr = None


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
   

    
    # ===================================
    # Parse argument from config file 
    # ===================================
 
    def _parse_arg( self , arg , var='variable' ):
        try:
            arg = str( arg )

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

            # Case single entry
            else:
                entries   = arg
                n_entries = 1
                arg_type  = self._get_type( entries )

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

        
        except:
            sys.exit( '\nERROR ( SYML -- _parse_arg ): issue with ' + var + '!\n\n' )

        return entries


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
        self._df_enc = self._df[ self._feats + [ self._col_out ] ]



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
            self._dl_enc = pd.get_dummies( self._df , columns=keys_str )
        else:
            self._dl_enc = self._df


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
        import itertools

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
        self._split_data( save=True )


        # Run training 
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
        
        else:
            constr              = self._df[ self._col_constr ].values
            constr_un , inds_un = np.unique( constr , return_index=True )
            
            inds  = np.arange( len( inds_un ) )
            y     = self._df[ self._col_out ].values
            y_sel = y[ inds_un ]


        # Get number of labels
        self._n_labels = len( np.unique( y_sel ) )

        
        # Case multiple splits to do k-fold cross-validation
        if self._kfold_cv:
            strat_kfold = StratifiedKFold( n_splits = self._n_folds )

            inds_train = [];  inds_test = []

            for train_index, test_index in strat_kfold.split( inds , y_sel ):
                inds_train.append( train_index )
                inds_test.append( test_index )                          


        # Case single split
        else:
            if self._testing < 1.0:
                mem             = self._test_perc
                self._test_perc = 20
                print( 'Warning ( SYML -- _split_data ): selected percentage for testing is too low ' + \
                       '(' + str( mem ) + ')!\nTesting percentage set to ' + str( self._test_perc ) )

            n_test     = myint( self._df.shape[0] * self._test_perc / 100.0 )
            
            inds_test  = random.sample( inds , n_test )
            inds_train = np.setdiff1d( inds , inds_test )
            
            inds_test  = [ inds_test.tolist() ] 
            inds_train = [ inds_train.tolist() ] 


        # Get real indices
        inds_real_train = [];  inds_real_test = [] 
         
        for i in range( len( inds_train ) ):
            if self._col_constr is None:
                inds_real_train.append( inds_train )
                inds_real_test.append( inds_test )

            else:
                for j in range( len( inds_train[i] ) ):
                    inds_all_train = []
                    pid            = constr_un[ inds_train[i][j] ]
                    inds_all_pid   = np.argwhere( constr == pid )[0]
            
                    for ind in inds_all_pid:  
                        inds_all_train.append( ind )

                    inds_real_train.append( inds_all_train )

                for j in range( len( inds_test[i] ) ):
                    inds_all_test = []
                    pid           = constr_un[ inds_test[i][j] ]
                    inds_all_pid  = np.argwhere( constr == pid )[0]
            
                    for ind in inds_all_pid:  
                        inds_all_test.append( ind )

                    inds_real_test.append( inds_all_test )


        # Split data frame
        self._df_train = [];  self._df_test = [];  self._nsplit = 0
    
        for i in range( len( inds_real_train ) ):
            self._df_train.append( self._df_enc.iloc[ inds_real_train[i] ] )
            self._df_test.append( self._df_enc.iloc[ inds_real_test[i] ] )
            self._n_split += 1


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
    # Run algorithm training
    # ===========================================

    def _run_train():
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

            kwargs = pd.DataFrame( columns = RF_PARAM_LIST )

        elif 'xg_boost' in self._alg:
            if self._task_type == 'classification':
                from xgboost import XGBClassifier
                Algorithm = XGBClassifier
            elif self._task_type == 'regression':
                from xgboost import XGBRegressor
                Algorithm = XGBRegressor

            kwargs = pd.DataFrame( columns = XG_PARAM_LIST )


        # For loop of param grid search
        for i in range( len( self._param_combs ) ):
            print( '\t.... training at grid point n.', i,' out of ', len( self._param_combs ) )
            
            # For loop on k-fold for cross-validation
            for j in range( self._n_split ):
                if self._kfold_cv:
                    print( '\t\t .... using fold n.', j )

                # Get training data
                x_train = self._df_train[ j ][ self._feats ].values
                y_train = self._df_train[ j ][ self._col_out ].values
 
                
                # Get grid point parameters
                row = []
                for param in self._param_combs[i]:
                    row.append( param )
                    
                kwargs.iloc[0] = row


                # Initialize algorithm
                self._clf = Algorithm( **kwargs )


                # Fit algorithm to training data
                self._clf.fit( x_train , y_train )


                # Get testing data
                x_test = self._df_test[ j ][ self._feats ].values
                y_test = self._df_test[ j ][ self._col_out ].values
 
                
                # Predict on testing data
                y_prob = self._clf._predict_proba( x_test )


                # Compute testing metrics
                self._compute_testing_metrics( y_test , y_prob )


                # Save model if selected metric has improved
                self._save_model()


                # Save model if selected metric has improved
                self._save_history( y_test , y_prob )


                # Save config file
                self._save_config( kwargs )


                # Save update logger
                self._update_logger( kwargs , n_grid=i , n_fold=j )



    # ===========================================
    # Create output filenames
    # ===========================================

    def _create_output_filenames( self ):
        # CSV logger
        self._csv_logger = os.path.join( self._path_out , 'syml_logger_' + self._label_out + '.csv' )

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
            if self._num_classes == 2:
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
            
                print( '\t\taccuracy: %s - precision: %s - recall: %s - f1score: %s - cohen_kappa: %s' % \
                    ( self._num2str( self._accuracy ) , self._num2str( self._precision ) ,
                      self._num2str( self._recall ) , self._num2str( self._f1score ) , 
                      self._num2str( self._cohen_kappa ) ) )
                print( '\t\tauc: %s - sensitivity( %s ): % s - specificity( %s ): %s' % \
                    ( self._num2str( self._auc ) , self._num2str( self._threshold ) , 
                      self._num2str( self._sensitivity ) , self._num2str( self._threshold ) , 
                      self._num2str( self._specificity ) ) )
        
            else:
                self._accuracy    = accuracy_score( y_true , y_class )
                self._cohen_kappa = cohen_kappa_score( y_true , y_class )
                print( '\t\taccuracy: %s - cohen_kappa: %s ' % \
                       ( self._num2str( self._accuracy ) , self._num2str( self._cohen_kappa ) ) )    
    

        # Case --> Regression
        elif self._task_type == 'regression':
            self._mse = mean_squared_error( y_true , y_prob )
            self._r2  = r2_score( y_true , y_pred )
            print( '\t\tmse: %s - r_squared: %s ' % \
                  ( self._num2str( self._accuracy ) , self._num2str( self._cohen_kappa ) ) ) 



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
    # Save best model according to a selected metric
    # ===================================
    
    def _save_best_model( self ):
        # Case 1 --> model improvement corresponds to a metric decreasing in value
        if self._metric == 'mse':
            str_metric     = '_' + self._metric
            metric_current = getattr( self , str_metric )  
            
            if metric_current < self._metric_monitor:
                self._metric_monitor = metric_current
                self._save_model     = True
                joblib.dump( self._clf , self._file_model )
                print( '\t\tTesting ', self._metric.upper() ,' has improved: saving best model to ' , self._file_model )            
            else:
                self._save_model = False


        # Case 2 --> model improvement corresponds to a metric increasing in value 
        else:
            str_metric     = '_' + self._metric
            metric_current = getattr( self , str_metric )  
            
            if metric_current > self._metric_monitor:
                self._metric_monitor = metric_current
                self._save_model     = True
                joblib.dump( self._clf , self._file_model )
                print( '\t\tTesting ', self._metric.upper() ,' has improved: best model saved to ' , self._file_model )            
            else:
                self._save_model = False


    
    # ===================================
    # Save best model according to a selected metric
    # ===================================
    
    def _save_history( self , y_true , y_prob ):  
        if self._saved_model:
            df = pd.DataFrame( { 'y_true'      : y_true.tolist()      ,
                                 'y_prob'      : y_prob.tolist()      ,
                                 'file_csv'    : self._file_csv       ,
                                 'task'        : self._task_type      ,
                                 'algorithm'   : self._alg            ,
                                 'feature_cols': self._feats.tolist() ,
                                 'outcome_col' : self._col_out        ,
                                 'constr_col'  : self._col_constr     ,
                                 'metric'      : self._metric         ,
                                 'peak_value'  : self._metric_monitor } )

            with open( self._file_preds , 'w' ) as fp:
                json.dump( df , fp , sort_keys=True )
            
            print( '\t\tTesting probabilities saved to ', self._file_histo )
       
    
    
    # ===================================
    # Print number as string with a certain precision 
    # ===================================
    
    def _num2str( self , num ):
        return str( round( num , 4 ) )


    
    # ===================================
    # Write config file
    # ===================================
    
    def _save_config( kwargs ):
        if self._saved_model:     
            dict = { 'model': { 'algorithm': self._alg     ,
                                'metric'   : self._metric  ,
                                'col_out'  : self._col_out ,
                                'select'   : select        ,
                                'exclude'  : exclude       } ,
                     'validation': { 'testing'   : self._perc_test  ,
                                     'col_constr': self._col_constr , 
                                     'kfold_cv'  : self._kfold_cv   ,
                                     'n_folds'   : self._n_folds    } }

            if 'rf' in self._alg:
                aux = { 'rf_params': kwargs }
            elif 'xg' in self._alg:
                aux = { 'xg_params': kwargs }

            dict.update( aux )

            with open( fileout , 'w' ) as outfile:
                yaml.dump( dict , outfile , default_flow_style=False )
        
            print( '\t\tConfig file has been saved to ', self._file )     

  

    # ===================================
    # Update logger
    # ===================================
    
    def _update_logger( self , df_params , n_grid=0 , n_fold=0 ):
        # Common data frame
        df = pd.DataFrame( { 'hash'      : self._label_out   ,
                             'task'      : self._task_type   ,
                             'n_grid'    : self._n_grid      ,
                             'n_fold'    : self._n_fold      ,
                             'save_model': self._saved_model ,
                             } )


        # Distinguish the various cases        
        if self._task_type == 'classification':
            if self._num_classes == 2:
                aux = pd.DataFrame( { 'test_accuracy'   : self._accuracy    ,
                                      'test_precision'  : self._precision   ,
                                      'test_recall'     : self._recall      , 
                                      'test_f1score'    : self._f1score     , 
                                      'test_auc'        : self._auc         ,
                                      'test_sensitivity': self._sensitivity ,
                                      'test_specificity': self._specificity ,
                                      'test_cohen_kappa': self._cohen_kappa ,
                                      'threshold'       : self._threshold } )
                                 
            else:
                aux = pd.DataFrame( { 'test_accuracy'   : self._accuracy ,
                                     'test_cohen_kappa': self._cohen_kappa } )

        elif self._task_type == 'regression':
                aux = pd.DataFrame( { 'test_mse'   : self._mse ,
                                      'test_r2'    : self._r2 } )
    
        df.update( aux )


        # Add parameters of the grid point
        df = pd.concat( [ df , df_params ] , axis=1 ) 
 
            
        # Write to CSV file
        if os.path.isfile( self._file_logger ):
            df.to_csv( self._file_logger , mode='a' , 
                       header=False , sep=SEP , index=False )
        else:
            df.to_csv( self._file_logger , 
                       sep=SEP , index=False )

