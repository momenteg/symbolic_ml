'''
    COMPUTE METRICS                                                                 
'''
 
# Author   : Filippo Arcadu
# Date     : 10/11/2018


from __future__ import division , print_function
import numpy as np
from sklearn.metrics import accuracy_score , precision_score , recall_score , r2_score , \
                            f1_score , cohen_kappa_score , roc_curve , auc , mean_squared_error 




# =============================================================================
# MY VARIABLE FORMAT 
# =============================================================================

myint    = np.int
myfloat  = np.float32
myfloat2 = np.float64




# =============================================================================
# LIST OF AVAILABLE METRICS
# =============================================================================

METRICS = [ 'accuracy' , 'precision' , 'recall' , 'sensitivity' , 'specificity' ,
            'auc' , 'f1score' , 'cohen_kappa' , 'mse' ]




# =============================================================================
# CLASS METRICS
# =============================================================================

class Metrics:
    
    # ===================================
    # Init 
    # ===================================
    
    def __init__( self , task='classification' , n_classes=2 , metric='auc' ):
        # Assign to class
        self._task      = task
        self._n_classes = n_classes
        self._metric    = metric


        # Checkpoint
        self._check_inputs()
    
    
    
    # ===================================
    # Check input 
    # ===================================
 
    def _check_inputs( self ):
        # Check whether metric is available
        if self._metric not in METRICS:
            sys.exit( '\nERROR ( Metrics -- _check_inputs ): selected metric ' + \
                      self._metric + ' is not available!\nSelect among ' + ','.join( METRICS )+ '\n\n')

        
        # Check whether metric is compatible with number of class 
        if self._task == 'classification':
            if self._n_classes > 2 and ( self._metric != 'accuracy' and self._metric != 'cohen_kappa' ):
                sys.exit( '\nERROR ( Metrics -- _check_inputs ): not possible to use metric ' + \
                          self._metric + ' with a ' + str( self._n_classes ) + ' classification problem!\n\n')



    # ===================================
    # Compute metrics
    # ===================================
    
    def _compute_metrics( self , y_true , y_prob ):
        # Common print
        print( '\t\tTesting metrics:' )

        
        # Case --> Classification
        if self._task == 'classification':

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
                    self._tpr         = tpr
                    self._fpr         = fpr
            
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
        self._fpr_best   = fpr[i_max]
        self._tpr_best   = tpr[i_max]
        
        
        # Compute sensitivity and specificity at optimal operating point
        self._sensitivity = tpr[ i_max ][0]
        self._specificity = 1 - fpr[ i_max ][0]
    
    
    

    # ===================================
    # Print number as string with a certain precision 
    # ===================================
    
    def _num2str( self , num ):
        return str( round( num , 4 ) )

    
    
    # =============================================================================
    # Get confidence interval
    # =============================================================================

    def _ci_bootstrap_roc( y_true , y_score , level=95 , n_bootstraps=1000 ):
        # Compute metrics through bootstrapping
        bootstrap_auc  = []
        bootstrap_sens = []
        bootstrap_spec = []

        ind_all = np.arange( len( y_true ) )
        
        for i in range( n_bootstraps ):
            ind = np.random.choice( ind_all , len( ind_all ) - 1 )
        
            try:
                self._compute_metrics( y_true[ind] , y_score[ind] )
                bootstrap_auc.append( self._auc )
                bootstrap_sens.append( self._sensitivity )
                bootstrap_spec.append( self._specificity )
            
            except:
                pass
            
        
        # Convert to array and sort
        bootstrap_auc = np.array( bootstrap_auc ).reshape( -1 )
        bootstrap_auc.sort()
        
        bootstrap_sens = np.array( bootstrap_sens ).reshape( -1 )
        bootstrap_sens.sort()
        
        bootstrap_spec = np.array( bootstrap_spec ).reshape( -1 )
        bootstrap_spec.sort()
            
        
        # Get confidence interval
        thres = ( 100 - level ) / 100.0 * 0.5
    
        try:
            auc_conf_up   = bootstrap_auc[ myint( ( 1 - thres ) * n_bootstraps ) ]
            auc_conf_down = bootstrap_auc[ myint( thres * n_bootstraps ) ]
        
            sens_conf_up   = bootstrap_sens[ myint( ( 1 - thres ) * n_bootstraps ) ]
            sens_conf_down = bootstrap_sens[ myint( thres * n_bootstraps ) ]

            spec_conf_up   = bootstrap_spec[ myint( ( 1 - thres ) * n_bootstraps ) ]
            spec_conf_down = bootstrap_spec[ myint( thres * n_bootstraps ) ]
    
            return [ auc_conf_up , auc_conf_down ,
                    sens_conf_up , sens_conf_down ,
                    spec_conf_up , spec_conf_down ]
    
        except:
            return [ -1 , -1 , -1 , -1 , -1 , -1 ] 
 
