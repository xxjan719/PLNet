import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Import from config (path is already set up in config.py)
from config import DIR_PROJECT, DIR_DATA

from p2p_bio import load_training_features

# Set OpenMP environment variable to handle runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from p2p_bio.constant import param_combination

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
from p2p_bio.utils import setup_logging
np.random.seed(1234)


print('Now we can test the model. Before that, make sure you have already run the dataset_generation.')
print('Available datasets: V2020, WT, MT, V2020+WT, V2020+MT, SKempi_V2(WT+MT), P2P(V2020+WT+MT)[For these two, you just need to input SKempi_V2 or P2P], Antibody-Antigen')
DATASET_INPUT = input('Which dataset do you want to use? ').strip()
# Validate user input
VALID_DATASET = ['V2020', 'WT', 'MT', 'V2020+WT','V2020+MT','SKempi_V2', 'P2P','Antibody-Antigen']
if DATASET_INPUT not in VALID_DATASET:
    print(f"Invalid dataset choice. Valid options are: {', '.join(VALID_DATASET)}")
    sys.exit(1)
METHOD_INPUT = input('Which method do you want to use? (biophysics, esm, PPI, All) ').strip()
if METHOD_INPUT not in ['biophysics', 'esm', 'PPI', 'All']:
    print("Invalid method choice. Valid options are: biophysics, esm, PPI, All")
    sys.exit(1)
print(f'Okay! You have selected dataset: {DATASET_INPUT} and method: {METHOD_INPUT}. Now we can start the model training and prediction.')

kf = KFold(n_splits=10, shuffle=True, random_state=1234)
scaler = StandardScaler()
PLOT_OPEN = False
for params in param_combination:
    DATASET_SELECTION = DATASET_INPUT
    TABLE = pd.read_csv(os.path.join(DIR_DATA, 'P2P.csv'))
    wt = TABLE[TABLE['source'] == 'Skempi_WT']
    ID_train = wt['ID'].values.flatten()
    print(f'ID_train shape: {ID_train.shape}')
    input_dataset_V2020,target_label_V2020,ID_V2020 = load_training_features(
            method=METHOD_INPUT,  # Just pass the string directly
            data_type='V2020')
    input_dataset_WT, target_label_WT, ID_WT = load_training_features(
            method=METHOD_INPUT,  # Just pass the string directly
            data_type='WT')
    input_dataset_MT, target_label_MT, ID_MT = load_training_features(
            method=METHOD_INPUT,  # Just pass the string directly
            data_type='MT')
    if DATASET_SELECTION == 'P2P':
        # For 2D arrays (features):
        input_dataset = np.vstack([input_dataset_V2020, input_dataset_WT, input_dataset_MT])
        
        # For 1D arrays (labels and IDs):
        target_label = np.concatenate([target_label_V2020, target_label_WT, target_label_MT])
        ID_train = np.concatenate([ID_V2020, ID_WT, ID_MT])
        
        print(f'for now, P2P dataset shape: {input_dataset.shape}, target shape: {target_label.shape}, ID shape: {ID_train.shape}')
    elif DATASET_SELECTION == 'SKempi_V2':
        input_dataset = np.vstack([input_dataset_WT, input_dataset_MT])
        target_label = np.concatenate([target_label_WT, target_label_MT])
        ID_train = np.concatenate([ID_WT, ID_MT])
        print(f'for now, SKempi_V2 dataset shape: {input_dataset.shape}, target shape: {target_label.shape}, ID shape: {ID_train.shape}')
    elif DATASET_SELECTION == 'V2020':
        input_dataset = input_dataset_V2020
        target_label = target_label_V2020
        ID_train = ID_V2020
        print(f'for now, V2020 dataset shape: {input_dataset.shape}, target shape: {target_label.shape}, ID shape: {ID_train.shape}')
        
    elif DATASET_SELECTION == 'WT':
        input_dataset = input_dataset_WT
        target_label = target_label_WT
        ID_train = ID_WT
        print(f'for now, WT dataset shape: {input_dataset.shape}, target shape: {target_label.shape}, ID shape: {ID_train.shape}')
    elif DATASET_SELECTION == 'MT':
        input_dataset = input_dataset_MT
        target_label = target_label_MT
        ID_train = ID_MT
        print(f'for now, MT dataset shape: {input_dataset.shape}, target shape: {target_label.shape}, ID shape: {ID_train.shape}')
    elif DATASET_SELECTION == 'V2020+WT':
        input_dataset = np.vstack([input_dataset_V2020, input_dataset_WT])
        target_label = np.concatenate([target_label_V2020, target_label_WT])
        ID_train = np.concatenate([ID_V2020, ID_WT])
        print(f'for now, V2020+WT dataset shape: {input_dataset.shape}, target shape: {target_label.shape}, ID shape: {ID_train.shape}')
    elif DATASET_SELECTION == 'V2020+MT':
        input_dataset = np.vstack([input_dataset_V2020, input_dataset_MT])
        target_label = np.concatenate([target_label_V2020, target_label_MT])
        ID_train = np.concatenate([ID_V2020, ID_MT])
        print(f'for now, V2020+MT dataset shape: {input_dataset.shape}, target shape: {target_label.shape}, ID shape: {ID_train.shape}')
    
    else:
        raise ValueError(f"Unsupported method: {DATASET_SELECTION} for now. Please choose from {VALID_DATASET}.")
    
    logger, log_filepath = setup_logging(DATASET_SELECTION, METHOD_INPUT)
    logger.info(f'Running GBDT for {DATASET_SELECTION} dataset with {METHOD_INPUT} features')
    logger.info(f'Saving results to: {log_filepath}')
    logger.info(f"=== GBDT Results for {DATASET_SELECTION} dataset using {METHOD_INPUT} features ===")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset shape: {input_dataset.shape}, Target shape: {target_label.shape}")
    logger.info(f"Parameters: {params}\n")
    if DATASET_SELECTION == 'V2020' or DATASET_SELECTION == 'WT' or DATASET_SELECTION == 'V2020+WT':
        # Create log file for GBDT results specific to this dataset
        
        
        try:
            # Open log file with explicit flushing  
            print('Processing data and starting cross-validation...')
            feature_all_scaled = scaler.fit_transform(input_dataset)
            y = target_label
            for fold, (index_train, index_test) in enumerate(kf.split(feature_all_scaled)):
                X_train, y_train = feature_all_scaled[index_train], y[index_train]
                X_test, y_test = feature_all_scaled[index_test], y[index_test]        
                gbdt = GradientBoostingRegressor(**params)
                gbdt.fit(X_train, y_train)
                y_pred = gbdt.predict(X_test)
                mse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                pearson_corr, _ = pearsonr(y_test, y_pred)
                    
                # Print to console
                fold_result = f"Fold {fold}: MAE: {mae:.3f}, RMSE: {mse:.3f}, Pearson Correlation: {pearson_corr:.3f}"
                print(fold_result)
                    
                # Write to log file
                logger.info(fold_result)
                    
                if fold == 0:
                    y_preds = y_pred
                    y_tests = y_test
                else:
                    y_preds = np.concatenate([y_preds,y_pred],axis=0)
                    y_tests = np.concatenate([y_tests,y_test],axis=0)
                
            y_preds = np.array(y_preds)
            y_tests = np.array(y_tests)  
            mae = mean_absolute_error(y_tests, y_preds) 
            rmse = np.sqrt(mean_squared_error(y_tests, y_preds))
            pearson_corr, _ = pearsonr(y_tests, y_preds)
                
            # Print overall results
            overall_result = f"\nOverall Results:\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, Pearson Correlation: {pearson_corr:.3f}"
            print(overall_result)
                
            # Write overall results to log file
            logger.info(overall_result)
                
            
                
            logger.info(f"\nGBDT training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"\nResults saved to {log_filepath}")
        except Exception as e:
            print(f"ERROR writing to log file: {str(e)}")
            # Try to write to a different location as a fallback
            with open(os.path.join(DIR_PROJECT, 'gbdt_error.log'), 'w') as error_log:
                error_log.write(f"Error: {str(e)}\n")
    elif DATASET_SELECTION == 'MT':
        print('Processing data and starting cross-validation...')
        ID_series = pd.Series(ID_train)
        feature_all_scaled = scaler.fit_transform(input_dataset)
        y = target_label
        for fold, (train_index, test_index) in enumerate(kf.split(ID_train)):
            ID_train_train, ID_train_test = ID_train[train_index], ID_train[test_index]
            index_train = [ID_series[ID_series.str.startswith(ID_part)].index.tolist() for ID_part in ID_train_train]
            index_train = [item for sublist in index_train for item in sublist]  # Flatten the list

            index_test = [ID_series[ID_series.str.startswith(ID_part)].index.tolist() for ID_part in ID_train_test]
            index_test = [item for sublist in index_test for item in sublist]  # Flatten the list

            # Create training and testing sets
            X_train, y_train = feature_all_scaled[index_train], y[index_train]
            X_test, y_test = feature_all_scaled[index_test], y[index_test]
                
            gbdt = GradientBoostingRegressor(**params)
            gbdt.fit(X_train, y_train)
            y_pred = gbdt.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            pearson_corr, _ = pearsonr(y_test, y_pred)
            # Print to console
            fold_result = f"Fold {fold}: MAE: {mae:.3f}, RMSE: {mse:.3f}, Pearson Correlation: {pearson_corr:.3f}"
            print(fold_result)
                    
            # Write to log file
            logger.info(fold_result)
            if fold == 0:
                y_preds = y_pred
                y_tests = y_test
            else:
                y_preds = np.concatenate([y_preds,y_pred],axis=0)
                y_tests = np.concatenate([y_tests,y_test],axis=0)
        y_preds = np.array(y_preds)
        y_tests = np.array(y_tests)   
        mae = mean_absolute_error(y_tests, y_preds)
        rmse = np.sqrt(mean_squared_error(y_tests, y_preds))
        pearson_corr, _ = pearsonr(y_tests, y_preds)
        # Print overall results
        overall_result = f"\nOverall Results:\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, Pearson Correlation: {pearson_corr:.3f}"
        print(overall_result)        
        # Write overall results to log file
        logger.info(overall_result)       
        logger.info(f"\nGBDT training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"\nResults saved to {log_filepath}")

    elif DATASET_SELECTION == 'P2P':
        X = np.concatenate([input_dataset_V2020, input_dataset_MT, input_dataset_WT], axis=0)
        y = np.concatenate([target_label_V2020, target_label_MT, target_label_WT], axis=0)
        ID_series = pd.Series(ID_train)
        X_scaled = scaler.fit_transform(X)
        X_v2020_scaled = X_scaled[:len(input_dataset_V2020)]
        X_mt_scaled = X_scaled[len(input_dataset_V2020):len(input_dataset_V2020) + len(input_dataset_MT)]
        X_wt_scaled = X_scaled[len(input_dataset_V2020) + len(input_dataset_MT):]

        unique_ids = np.unique(ID_train)
        np.random.shuffle(unique_ids)  # Shuffle IDs
        mt_folds = np.array_split(unique_ids, 10)
    
        v2020_indices = np.arange(len(X_v2020_scaled))
        np.random.shuffle(v2020_indices)
        v2020_folds = np.array_split(v2020_indices, 10)
        
        wt_indices = np.arange(len(X_wt_scaled))
        np.random.shuffle(wt_indices)
        wt_folds = np.array_split(wt_indices, 10)
        print('Processing data and starting cross-validation...')
        for fold in range(10):
            # MT train-test split based on IDs
            test_ids = mt_folds[fold]
            train_ids = np.concatenate([mt_folds[i] for i in range(10) if i != fold])
            
            index_train_mt = [ID_series[ID_series.str.startswith(ID_part)].index.tolist() for ID_part in train_ids]
            index_train_mt = [item for sublist in index_train_mt for item in sublist]  # Flatten the list
            
            index_test_mt = [ID_series[ID_series.str.startswith(ID_part)].index.tolist() for ID_part in test_ids]
            index_test_mt = [item for sublist in index_test_mt for item in sublist]  # Flatten the list

            #print(f'index_test_mt is {index_test_mt}, index_train_mt is {index_train_mt}')
            X_mt_train, y_mt_train = X_mt_scaled[index_train_mt], y_mt[index_train_mt]
            X_mt_test, y_mt_test = X_mt_scaled[index_test_mt], y_mt[index_test_mt]
            #print(f'X mt train shape is {X_mt_train.shape}, X mt test shape is {X_mt_test.shape}')
            
            index_train_v2020 = np.concatenate([v2020_folds[i] for i in range(10) if i != fold])
            index_test_v2020 = v2020_folds[fold]
            X_v2020_train, y_v2020_train = X_v2020_scaled[index_train_v2020], y_v2020[index_train_v2020]
            X_v2020_test, y_v2020_test = X_v2020_scaled[index_test_v2020], y_v2020[index_test_v2020]
            #print(f'X v2020 train shape is {X_v2020_train.shape}, X v2020 test shape is {X_v2020_test.shape}')
            
            index_train_wt = np.concatenate([wt_folds[i] for i in range(10) if i != fold])
            index_test_wt = wt_folds[fold]
            X_wt_train, y_wt_train = X_wt_scaled[index_train_wt], y_wt[index_train_wt]
            X_wt_test, y_wt_test = X_wt_scaled[index_test_wt], y_wt[index_test_wt]
            #print(f'X wt train shape is {X_wt_train.shape}, X wt test shape is {X_wt_test.shape}')
            
            X_train = np.concatenate([X_v2020_train,X_mt_train,X_wt_train],axis=0)
            X_test = np.concatenate([X_v2020_test,X_mt_test,X_wt_test],axis=0)
            y_train = np.concatenate([y_v2020_train,y_mt_train,y_wt_train],axis=0)
            y_test = np.concatenate([y_v2020_test,y_mt_test,y_wt_test],axis=0)
            
            gbdt = GradientBoostingRegressor(**params)
            gbdt.fit(X_train, y_train)
            y_pred = gbdt.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            pearson_corr, _ = pearsonr(y_test, y_pred)
            # Print to console
            fold_result = f"Fold {fold}: MAE: {mae:.3f}, RMSE: {mse:.3f}, Pearson Correlation: {pearson_corr:.3f}"
            print(fold_result)
                    
            # Write to log file
            logger.info(fold_result)
            if fold == 0:
                y_preds = y_pred
                y_tests = y_test
            else:
                y_preds = np.concatenate([y_preds,y_pred],axis=0)
                y_tests = np.concatenate([y_tests,y_test],axis=0)
        y_preds = np.array(y_preds)
        y_tests = np.array(y_tests)   
        mae = mean_absolute_error(y_tests, y_preds)
        rmse = np.sqrt(mean_squared_error(y_tests, y_preds))
        pearson_corr, _ = pearsonr(y_tests, y_preds)
        # Print overall results
        overall_result = f"\nOverall Results:\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, Pearson Correlation: {pearson_corr:.3f}"
        print(overall_result)        
        # Write overall results to log file
        logger.info(overall_result)       
        logger.info(f"\nGBDT training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"\nResults saved to {log_filepath}")
        
    
    elif DATASET_SELECTION == 'WT+MT':
        X = np.concatenate([input_dataset_MT, input_dataset_WT], axis=0)
        y = np.concatenate([target_label_MT, target_label_WT], axis=0)
        ID_series = pd.Series(ID_train)
        X_scaled = scaler.fit_transform(X)
        X_mt_scaled = X_scaled[:len(input_dataset_MT)]
        X_wt_scaled = X_scaled[len(input_dataset_MT):]

        unique_ids = np.unique(ID_train)
        np.random.shuffle(unique_ids)  # Shuffle IDs
        mt_folds = np.array_split(unique_ids, 10)
    
        wt_indices = np.arange(len(X_wt_scaled))
        np.random.shuffle(wt_indices)
        wt_folds = np.array_split(wt_indices, 10)
        print('Processing data and starting cross-validation...')
        for fold in range(10):
            # MT train-test split based on IDs
            test_ids = mt_folds[fold]
            train_ids = np.concatenate([mt_folds[i] for i in range(10) if i != fold])
            
            index_train_mt = [ID_series[ID_series.str.startswith(ID_part)].index.tolist() for ID_part in train_ids]
            index_train_mt = [item for sublist in index_train_mt for item in sublist]  # Flatten the list
            
            index_test_mt = [ID_series[ID_series.str.startswith(ID_part)].index.tolist() for ID_part in test_ids]
            index_test_mt = [item for sublist in index_test_mt for item in sublist]  # Flatten the list

            #print(f'index_test_mt is {index_test_mt}, index_train_mt is {index_train_mt}')
            X_mt_train, y_mt_train = X_mt_scaled[index_train_mt], y_mt[index_train_mt]
            X_mt_test, y_mt_test = X_mt_scaled[index_test_mt], y_mt[index_test_mt]
            #print(f'X mt train shape is {X_mt_train.shape}, X mt test shape is {X_mt_test.shape}')
            
            
            index_train_wt = np.concatenate([wt_folds[i] for i in range(10) if i != fold])
            index_test_wt = wt_folds[fold]
            X_wt_train, y_wt_train = X_wt_scaled[index_train_wt], y_wt[index_train_wt]
            X_wt_test, y_wt_test = X_wt_scaled[index_test_wt], y_wt[index_test_wt]
            #print(f'X wt train shape is {X_wt_train.shape}, X wt test shape is {X_wt_test.shape}')
            
            X_train = np.concatenate([X_mt_train,X_wt_train],axis=0)
            X_test = np.concatenate([X_mt_test,X_wt_test],axis=0)
            y_train = np.concatenate([y_mt_train,y_wt_train],axis=0)
            y_test = np.concatenate([y_mt_test,y_wt_test],axis=0)
            # print(f'overall train shape is {X_train.shape}, X v2020 test shape is {X_test.shape}')
            
            gbdt = GradientBoostingRegressor(**params)
            gbdt.fit(X_train, y_train)
            y_pred = gbdt.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            pearson_corr, _ = pearsonr(y_test, y_pred)
            # Print to console
            fold_result = f"Fold {fold}: MAE: {mae:.3f}, RMSE: {mse:.3f}, Pearson Correlation: {pearson_corr:.3f}"
            print(fold_result)
                    
            # Write to log file
            logger.info(fold_result)
            if fold == 0:
                y_preds = y_pred
                y_tests = y_test
            else:
                y_preds = np.concatenate([y_preds,y_pred],axis=0)
                y_tests = np.concatenate([y_tests,y_test],axis=0)
        y_preds = np.array(y_preds)
        y_tests = np.array(y_tests)   
        mae = mean_absolute_error(y_tests, y_preds)
        rmse = np.sqrt(mean_squared_error(y_tests, y_preds))
        pearson_corr, _ = pearsonr(y_tests, y_preds)
        # Print overall results
        overall_result = f"\nOverall Results:\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, Pearson Correlation: {pearson_corr:.3f}"
        print(overall_result)        
        # Write overall results to log file
        logger.info(overall_result)       
        logger.info(f"\nGBDT training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"\nResults saved to {log_filepath}")
        


