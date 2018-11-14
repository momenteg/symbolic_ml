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




# =============================================================================
# MY VARIABLE FORMAT 
# =============================================================================

myint    = np.int
myfloat  = np.float32
myfloat2 = np.float64




# =============================================================================
# CSV SEPARATOR 
# =============================================================================

SEP = ';'




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


    
    # ===================================
    # Read input table 
    # ===================================

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
        self._select  = cfg[ 'model' ][ 'select' ]
        self._exclude = cfg[ 'model' ][ 'exclude' ]


        # Get validation parameters
        self._alg     = cfg[ 'validation' ][ 'testing' ]
        self._metric  = cfg[ 'validation' ][ 'col_constr' ]
        self._exclude = cfg[ 'validation' ][ 'kfold_cv' ]
        self._n_folds = cfg[ 'validation' ][ 'n_folds' ]


        # Get type of task
        if '-class' in self._alg:
            self._task_type = 'classification'
        elif '-regr' in self._alg:
            self._task_type = 'regression'


        # Get random forest parameters
        self._params = []

        if self._alg == 'rf':
            chapter = 'rf_params'

            if chapter in cfg.keys():
                for i in range( len( RF_PARAMS ) ):
                    if RF_PARAMS[i] in cfg[ chapter ].keys():
                        self._params.append( self._parse_arg( cfg[ chapter ][ RF_PARAMS[i] ] ,
                                                              arg_type = 'int'                     ,
                                                              var      = chapter + ':' + RF_PARAMS[i] ) )
                    else:
                        sys.exit( 'ERROR ( SYML -- _read_config_file ): param ' + RF_PARAMS[i] + \
                                  ' is not contained inside ' + self._file_cfg + '!\n\n' )

            else:
                sys.exit( '\nERROR ( SYML -- _read_config_file ): < rf_params > not found in ' + \
                          self._file_cfg + '!\n\n' ) 

        
         # Get random forest parameters
        if self._alg == 'xg':
            chapter = 'xg_params'

            if chapter in cfg.keys():
                for i in range( len( XG_PARAMS ) ):
                    if XG_PARAMS[i] in cfg[ chapter ].keys():
                        self._params.append( self._parse_arg( cfg[ chapter ][ XG_PARAMS[i] ] ,
                                                              arg_type = 'int'                     ,
                                                              var      = chapter + ':' + XG_PARAMS[i] ) )
                    else:
                        sys.exit( 'ERROR ( SYML -- _read_config_file ): param ' + XG_PARAMS[i] + \
                                  ' is not contained inside ' + self._file_cfg + '!\n\n' )

            else:
                sys.exit( '\nERROR ( SYML -- _read_config_file ): < xg_params > not found in ' + \
                          self._file_cfg + '!\n\n' ) 
   

    
    # ===================================
    # Parse argument from config file 
    # ===================================
 
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


    
    # ===================================
    # Get outcome variable 
    # ===================================
 
    def _get_outcome( self ):
        if self._col_out in self._df.keys():
            self._y = self._df[ self._col_out ].values

        else:
            sys.exit( '\nERROR ( SYML -- _get_outcome ): outcome column ' + \
                      self._col_out + ' is not in input data frame!\n'    + \
                      'Available Keys: (' + ','.join( self._df.keys() ) + ')\n\n' )    

       
    
    # ===================================
    # Get feature variables 
    # ===================================

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
            


    # ===================================
    # Check feature and outcome variables 
    # ===================================

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

       

    # ===================================
    # Slice data frame
    # ===================================
 
    def _slice_data_frame( self ):
        self._df_enc = self._df[ self._feats + [ self._col_out ] ]


    
    # ===================================
    # Convert non-numeric to 1-hot encoded
    # ===================================

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



    # ===================================
    # Create output label 
    # ===================================

    def _create_output_label( self ):
        from time import gmtime, strftime
        import random        

        str_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        hash_out = random.getrandbits( 64 )

        if self._alg == 'rf':
            string_alg = self._alg
        elif self._alg == 'xgboost':
            string_alg = 'xg'

        if self._add_label is None:
            self._label_out = string_alg + '_' + str_time + '_' + hash_out
        else:
            self._label_out = string_alg + '_' + self._add_label + '_' + str_time + '_' + hash_out



    # ===================================
    # Create param grid 
    # ===================================
 
    def _create_param_grid( self ):
        import itertools

        if self._alg == 'rf' or self._alg == 'xgboost':
            self._param_combs = list( itertools.product( self._params[0] , self._params[1] ,
                                                         self._params[2] , self._params[3] ,
                                                         self._params[4] , self._params[5] ,
                                                         self._params[6] , self._params[7] ) )  
        


    # ===================================
    # Training 
    # ===================================
 
    def _train():
        # Split dataset
        self._split_data( save=True )


        # Run training 
        self._run_train()



    # ===================================
    # Split data 
    # ===================================
 
    def _split_data( save=False ):
        from sklearn.model_selection import StratifiedKFold

        # Get constrained column
        if self._col_constr is None:
            inds  = np.arange( self._df.shape )
            y_sel = self._df[ self._col_out ].values
        
        else:
            constr              = self._df[ self._constr ].values
            constr_un , inds_un = np.unique( constr , return_index=True )
            
            inds  = np.arange( len( inds_un ) )
            y     = self._df[ self._col_out ].values
            y_sel = y[ inds_un ]


        # Get number of labels
        self._n_labels = len( np.unique( y_sel ) )

        
        # Case multiple splits to do k-fold cross-validation
        if self._kfold_cv:
            strat_kfold = StratifiedKFold( n_splits = self._nfolds )

            inds_train = [];  inds_test = []

            for train_index, test_index in skf.split( inds , y_sel ):
                inds_train.append( train_index )
                inds_test.append( test_index )                          


        # Case single split
        else:
            if self._testing < 1.0:
                mem           = self._testing
                self._testing = 20
                print( 'Warning ( SYML -- _split_data ): selected percentage for testing is too low ' + \
                       '(' + str( mem ) + ')!\nTesting percentage set to ' + str( self._testing ) )

            n_test     = myint( self._df.shape[0] * self._testing / 100.0 )
            
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


    
    # ===================================
    # Run algorihm training 
    # ===================================

    def _run_train():
        # Initalize filenames for output files
        self._create_output_filenames()


        # Initialize monitoring metric
        self._init_monitor()


        # Load modules
        if self._alg == 'rf':
            if self._task_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                Algorithm = RandomForestClassifier
            elif self._task_type == 'regression':
                from sklearn.ensemble import RandomForestRegressor
                Algorithm = RandomForestRegressor

        elif self._alg == 'xg_boost':
            if self._task_type == 'classification':
                from xgboost import XGBClassifier
                Algorithm = XGBClassifier
            elif self._task_type == 'regression':
                from xgboost import XGBRegressor
                Algorithm = XGBRegressor


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
 
                
                # Initialize algorithm
                clf = Algorithm( n_estimators      = self._param_combs[i][0] , 
                                 criterion         = self._param_combs[i][1] ,
                                 max_depth         = self._param_combs[i][2] ,
                                 min_samples_split = self._param_combs[i][3] ,     
                                 min_samples_leaf  = self._param_combs[i][4] ,
                                 bootstrap         = self._param_combs[i][5] , 
                                 oob_score         = self._param_combs[i][6] ,
                                 class_weights     = self._param_combs[i][7] )


                # Fit algorithm to training data
                clf.fit( x_train , y_train )


                # Get testing data
                x_test = self._df_test[ j ][ self._feats ].values
                y_test = self._df_test[ j ][ self._col_out ].values
 
                
                # Predict on testing data
                y_prob = clf._predict_proba( x_test )


                # Compute testing metrics
                self._compute_testing_metrics( y_test , y_prob )


                # Save model if selected metric has improved
                self._save_model()


                # Save update logger
                self._update_logger( n_iter=i , n_fold=j )



    # ===================================
    # Create output filenames 
    # ===================================
 
    def _create_output_filenames( self ):
        # CSV logger
        self._csv_logger = os.path.join( self._path_out , 'syml_logger_' + self._label_out + '.csv' )

        # Model
        self._file_model = os.path.join( self._path_out , 'syml_model_' + self._label_out + '.pkl' )

        # Predictions on testing
        self._file_preds = os.path.join( self._path_out , 'syml_test_preds_' + self._label_out + '.json' )
        
    
    
    # ===================================
    # Initializing monitoring metric.
    # ===================================

    def _init_monitor( self ):
        if self._metric == 'val_mse':
            self._metric_monitor = 1e10
        else:
            self._metric_monitor = 0


   
    # ===================================
    # Compute metrics
    # ===================================
    
    def _compute_metrics( self , y_true , y_prob ):
        # Case --> Classification
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
            
                print( '\rval_accuracy: %s - val_precision: %s - val_recall: %s - val_f1score: %s - val_cohen_kappa: %s' % \
                    ( self._num2str( self._accuracy ) , self._num2str( self._precision ) ,
                      self._num2str( self._recall ) , self._num2str( self._f1score ) , 
                      self._num2str( self._cohen_kappa ) ) )
                print( '\rval_auc: %s - val_sensitivity( %s ): % s - val_specificity( %s ): %s' % \
                    ( self._num2str( self._auc ) , self._num2str( self._threshold ) , 
                      self._num2str( self._sensitivity ) , self._num2str( self._threshold ) , 
                      self._num2str( self._specificity ) ) )
        
            else:
                self._accuracy    = accuracy_score( y_true , y_class )
                self._cohen_kappa = cohen_kappa_score( y_true , y_class )
                print( '\rval_accuracy: %s - val_cohen_kappa: %s ' % \
                       ( self._num2str( self._accuracy ) , self._num2str( self._cohen_kappa ) ) )    
    

        # Case --> Regression
    
    
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
        if self._metric is not None:
            if self._metric == 'accuracy':
                if self._accuracy > self._metric_monitor:
                    self._metric_monitor = self._accuracy
                    self.model.save( self._file_best_model )
                    print( 'val_accuracy has improved: saving best model to ' , self._file_best_model )            
        
            if self._metric == 'precision':
                if self._precision > self._metric_monitor:
                    self._metric_monitor = self._precision
                    self.model.save( self._file_best_model )
                    print( 'val_precision has improved: saving best model to ' , self._file_best_model )
                    
            elif self._metric == 'recall':
                if self._recall > self._metric_monitor:
                    self._metric_monitor = self._recall
                    self.model.save( self._file_best_model )
                    print( 'val_recall has improved: saving best model to ' , self._file_best_model )
                    
            elif self._metric == 'f1score':
                if self._f1score > self._metric_monitor:
                    self._metric_monitor = self._f1score
                    self.model.save( self._file_best_model )
                    print( 'val_f1score has improved: saving best model to ' , self._file_best_model )                    

            elif self._metric == 'auc':
                if self._auc > self._metric_monitor:
                    self._metric_monitor = self._auc
                    self.model.save( self._file_best_model )
                    print( 'val_auc has improved: saving best model to ' , self._file_best_model )
                    
            elif self._metric == 'sensitivity':
                if self._sensitivity > self._metric_monitor:
                    self._metric_monitor = self._sensitivity
                    self.model.save( self._file_best_model )
                    print( 'val_sensitivity has improved: saving best model to ' , self._file_best_model )

            elif self._metric == 'specificity':
                if self._specificity > self._metric_monitor:
                    self._metric_monitor = self._specificity
                    self.model.save( self._file_best_model )
                    print( 'val_specificity has improved: saving best model to ' , self._file_best_model )                    
        
            elif self._metric == 'cohen_kappa':
                if self._cohen_kappa > self._metric_monitor:
                    self._metric_monitor = self._cohen_kappa
                    self.model.save( self._file_best_model )
                    print( 'val_cohen_kappa has improved: saving best model to ' , self._file_best_model )        
    
    
    
    # ===================================
    # Print number as string with a certain precision 
    # ===================================
    
    def _num2str( self , num ):
        return str( round( num , 4 ) )


    
    # ===================================
    # Update logger
    # ===================================
    
    def _update_logger( self , epoch ):    
        # Distinguish between case with 2 classes and > 2 classes        
        if self._num_classes == 2:
            df = pd.DataFrame( { 'val_accuracy'   : self._accuracy    ,
                                 'val_precision'  : self._precision   ,
                                 'val_recall'     : self._recall      , 
                                 'val_f1score'    : self._f1score     , 
                                 'val_auc'        : self._auc         ,
                                 'val_sensitivity': self._sensitivity ,
                                 'val_specificity': self._specificity ,
                                 'val_cohen_kappa': self._cohen_kappa ,
                                 'threshold'      : self._threshold   ,
                                 'phase'          : self._phase       } , index=[epoch] )
                                 
        else:
            df = pd.DataFrame( { 'val_accuracy'   : self._accuracy ,
                                 'val_cohen_kappa': self._cohen_kappa } , index=[epoch] )
            
        
        # Write to CSV file
        if os.path.isfile( self._file_metric_logger ):
            df.to_csv( self._file_metric_logger , mode='a' , 
                       header=False , sep=SEP , index=False )
        else:
            df.to_csv( self._file_metric_logger , 
                       sep=SEP , index=False )


