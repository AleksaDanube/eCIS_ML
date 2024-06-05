# eCIS_ML
codes and the final model for the Genome-wide discovery of toxins associated with the extracellular contractile injection system"
the package XGBoost was used to train and evaluate the Boosting Machine Model for the eCIS associated toxins classification.
the the_codes_eCIS contains the used imports and functions as well as the script that runs them to create and save the final model
the functions that are included are:
prepare_features - data preprocessing
div_train_and_test - splits for train and test
trainXGBoost - trains one model and tests it, returns the model and the evaluations
xgBoostClassifierKfoldWithFixedDp - trains N models by N-fold cross validations and returns the evalueations
saveModel - saves the model
lines 186,187 are creating the model based on all the data, the previous functions were used to evaluate the model. 
