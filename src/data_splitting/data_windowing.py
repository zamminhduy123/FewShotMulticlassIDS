import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding
import argparse

import random
np.random.seed(2022)
random.seed(2022)

# ================================ ATTACK TYPES ================================
ATTACK_TYPES_MAS = {
    "benign": 0,
    "correlated": 7,
    "max_speedometer": 8,
    "reverse_light_off" : 9,
    "reverse_light_on": 10,
    "max_engine" : 11,
}

FAB_MAS_ATTACK_TYPES = {
    "benign": 0,
    "fabrication_correlated": 1,
    "masquerade_correlated": 2,
    "fabrication_max_speedometer": 3,
    "masquerade_max_speedometer": 4,
    "fabrication_reverse_light_off": 5,
    "masquerade_reverse_light_off": 6,
    "fabrication_reverse_light_on": 7,
    "masquerade_reverse_light_on": 8,
    "fabrication_max_engine": 9,
    "masquerade_max_engine": 10,
    "fuzzing": 11,
}

ATTACK_TYPES_EX_COR = {
    "max_speedometer": 1,
    "reverse_light_off" : 2,
    "reverse_light_on": 3,
    "max_engine" : 4,
    "fuzzing" : 5,

    "correlated": 6,
}

FIXED_ATTACK_TYPES = {
    "fuzzing": 3,
    "mar": 2,
    "fab": 1,
}

ATTACK_TYPES_CHD = {
    "Fuzzy": 1,
    "gear": 2,
    "RPM" : 3,
    "DoS": 4,
}

ATTACK_TYPES_BINARY = {
    "correlated": 1,
    "fuzzing" : 1,
    "max_engine" : 1,
    "max_speedometer": 1,
    "reverse_light_off" : 1,
    "reverse_light_on": 1,
}

ATTACK_TYPES = {
    "benign": 0,
    "correlated": 1,
    "max_speedometer": 2,
    "reverse_light_off" : 3,
    "reverse_light_on": 4,
    "max_engine" : 6,
    "fuzzing" : 6,
}

ROAD_LABEL_MAP_UNIQUE_IDS = {
    "correlated_signal": 1, #3
    "max_speedometer": 2, # 3
    "reverse_light_on": 3, # 3
    "reverse_light_off": 4, # 3
    "fuzzing": 6, # 3
    "max_engine_coolant_temp": 5,
}


ROAD_MAS_LABEL_MAP_UNIQUE_IDS = {
    "correlated_signal": 1, #3
    "max_speedometer": 2, # 3
    "reverse_light_on": 3, # 3
    "reverse_light_off": 4, # 3
    "max_engine_coolant_temp": 5,
}

ROAD_FULL_MAS_LABEL_MAP_UNIQUE_IDS = {
    "correlated_signal": 7, #3
    "max_speedometer": 8, # 3
    "reverse_light_on": 9, # 3
    "reverse_light_off": 10, # 3
    "max_engine_coolant_temp": 11,
}

# ================================ UTILITIES ================================
def split_hex_string_column(df, column_name):
    # Number of characters to group together from the hex string
    chunk_size = 2
    
    # Create the new columns from the hexadecimal string column
    for i in range(0, 16, chunk_size):
        new_col_name = f'Data{i//2 + 1}'
        # Slice the string into chunks of 2, convert to integer from base 16
        df[new_col_name] = df[column_name].str[i:i+chunk_size].apply(lambda x: int(x, 16))
    
    return df

def decimal_to_fixed_length_binary(number, bit_length=29):
    """
    Converts a decimal number into a binary string with a fixed length.
    """
    if number >= 2**bit_length:
        raise ValueError(f"The number {number} cannot be represented in {bit_length} bits without loss of information.")
    
    binary_str = bin(number)[2:].zfill(bit_length)
    return binary_str

def expand_binary_to_columns(df, decimal_column):
    """
    Expands a decimal column in a DataFrame to 29 binary bit columns.
    """
    # Convert decimal column to binary with fixed length
    df['binary'] = df[decimal_column].apply(lambda x: decimal_to_fixed_length_binary(x))
    
    # Split binary column into separate bit columns
    for i in range(29):
        df[f'can_{i+1}'] = df['binary'].apply(lambda x: int(x[i]))
    
    # Optionally, drop the temporary binary column
    df.drop(columns=['binary'], inplace=True)
    
    return df

def split_decimal_to_digits(df, column_name):
    """
    Splits a column of decimal numbers into multiple single-digit columns.
    
    :param df: The DataFrame containing the column to split.
    :param column_name: The name of the column containing decimal numbers.
    :return: A DataFrame with the original data and new columns for each digit.
    """
    # Convert the column to strings to easily access individual digits
    df[column_name] = df[column_name].astype(str)
    
    # Find the maximum length of any number in the column
    max_length = df[column_name].str.len().max()
    
    # Pad the numbers with leading zeros so they all have the same length
    df[column_name] = df[column_name].apply(lambda x: x.zfill(max_length))
    
    # Split each number into its digits and create a new column for each digit
    for i in range(max_length):
        df[f'can_{i+1}'] = df[column_name].apply(lambda x: int(x[i]))
    
    return df


def fill_flag(sample):
    if not isinstance(sample['label'], str):
        col = 'Data' + str(sample['DLC'])
        sample['label'], sample[col] = sample[col], sample['label']
    return sample

#turn raw data into more understandable format
def process_data(df, label = 1, decimal = False, isCHD=False):
    print("Starting processing data")
    
    df = df.fillna(256)

    print("Number of rows in df:", len(df))

    def calculate_tslm(group):
        group['TSLM'] = group.groupby('id')['time'].diff().fillna(0)
        return group

    if not isCHD:
        df = split_hex_string_column(df, "data")
        df = df.drop("data", axis = 1)
        # df = df.drop("time_diffs", axis = 1)
        # df = df.groupby('id').apply(calculate_tslm)

        df = calculate_tslm(df)
        df['TSLM'] = df['TSLM'].fillna(0)
        # df["IAT"] = df['Time'

        # # Fill NA 


        # print("Number of rows in df:", len(df))
        
        # df['label'] = df['label'].apply(lambda x: True if x != 0 else False)
    else:
        #Fill missing data in payload as 0
        df = df.apply(fill_flag, axis=1)
        # convert payload into int from hex
        num_data_bytes = 8
        for x in range(num_data_bytes):
            df['Data'+str(x)] = df['Data'+str(x)].map(lambda t: int(str(t), 16))
        df = df.fillna(256)

        df = calculate_tslm(df)
        df['TSLM'] = df['TSLM'].fillna(0)

        df = df.drop("DLC", axis = 1)
        df['label'] = df['label'].apply(lambda x: True if x == "T" else False)

        df['time_diffs'] = df['time'].diff().fillna(0)
        # df["IAT"] = df['Time']

    if (decimal):
        df['id'] = df['id'].astype(str)
        df['id'] = df['id'].apply(lambda x: int(x, 16))

        #Seperate digits
        # df = split_decimal_to_digits(df, 'id')
        # df = df.drop("id", axis = 1)

        # Not seperate digits
        can_id_unique = df['id'].unique()
        can_id_to_idx = {can_id: idx for idx, can_id in enumerate(can_id_unique)}
        df['id'] = df['id'].map(can_id_to_idx)
        # df['id'] = df['id'].map(lambda x: x/len(can_id_unique))
        # df['id'] = embed_features(df, df['id'], len(df['id'].unique()), 8)
    else:
        df = expand_binary_to_columns(df, 'id')



    # df['label'] = df['label'].astype(int)


    # Assuming 'data' column contains hex strings
    # df['data_length'] = df['data'].apply(len)
    
    # Convert hex data to numerical values
    # df['data_numeric'] = df['data'].apply(lambda x: int(x, 16))
    
    # # Calculate statistical properties
    # df['data_mean'] = df['data_numeric'].rolling(window=15).mean()
    # df['data_variance'] = df['data_numeric'].rolling(window=15).var()

    # df = df.drop("time", axis = 1)
    # df = df.drop("id", axis = 1)
    # df = df.drop("label", axis = 1)
    # df = df.drop("data_numeric", axis = 1)


    # Apply MinMaxScaler to each column
    scaler = StandardScaler()
    scaler_cols = ['Data0' if (isCHD) else 'Data8', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7',
                    'TSLM',
                    'time_diffs',
                    'id']
    df[scaler_cols] = scaler.fit_transform(df[scaler_cols])

    feature_col = [col for col in df.columns if col not in ['time', 'label', 
                                                            # 'TSLM', 'time_diffs'
                                                            ]]
    # Step 2: Add this array as a new column to the original DataFrame
    df['features'] = df[feature_col].apply(lambda row: [row[col] for col in feature_col], axis=1)
    
    print('Processing: DONE')

    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] != 0].shape[0])
    print("feature num: ", df.columns)

    return df

def embed_features(df, data, input_dim, output_dim):
    """
    Embeds the input data using an Embedding layer.
    
    :param data: The input data to be embedded.
    :param input_dim: The size of the vocabulary, i.e., the maximum integer index + 1.
    :param output_dim: The dimension of the dense embedding.
    :return: The embedded features.
    """
    embedding_layer = Embedding(input_dim, output_dim)
    embedded_features = embedding_layer(data.values.astype("int64"))
    for i in range(8):
        df[f'canId{i+1}'] = embedded_features[:, i]
    
    return embedded_features

# ================================ DATA WINDOWING & SPLITTING ================================

def parse_args():
    parser = argparse.ArgumentParser(description='Data windowing and preprocessing for CAN bus data')
    
    # Basic configuration
    parser.add_argument('--window-type', type=str, default='2d', help='Type of windowing to use')
    parser.add_argument('--chd', action='store_true', help='Use car-hacking dataset')
    parser.add_argument('--is-mar', action='store_true', help='Use MAR dataset')
    parser.add_argument('--exclude-something', action='store_true', help='Whether to exclude certain data')
    parser.add_argument('--exclude-name', type=str, default='fuzzing', help='Name of data to exclude')
    parser.add_argument('--process-type', type=str, default='ss_11_2d_no', help='Type of processing')
    
    # Window parameters
    parser.add_argument('--window-size', type=int, default=16, help='Size of the sliding window')
    parser.add_argument('--step', type=int, default=1, help='Step size for sliding window')
    parser.add_argument('--feature', type=int, default=11, help='Number of features')
    
    # Split parameters
    parser.add_argument('--test-size', type=float, default=0.8, help='Proportion of test split')
    parser.add_argument('--val-split', default=True, action='store_true', help='Whether to create validation split')
    parser.add_argument('--val-size', type=float, default=0.5, help='Proportion of validation split')
    
    # Data configuration
    parser.add_argument('--data-type', type=str, default='dec', help='Type of data processing')
    parser.add_argument('--extra', type=str, default='_90test', help='Extra suffix for file names')
    
    # Paths
    parser.add_argument('--road-data-path', type=str, 
                       default='/home/ntmduy/CANET/CICIDS2017/data/road/',
                       help='Base path for dataset')
    parser.add_argument('--car-hacking-path', type=str,
                       default='/home/ntmduy/car-ids/raw-data/car-hacking/',
                       help='Path for car hacking dataset')
    parser.add_argument('--save-path', type=str,
                    #    default='/home/ntmduy/CANET/CICIDS2017/data',
                       default='/home/ntmduy/FewShotCanBus/test_Data/',
                       help='Base path for saving processed data')
    
    args = parser.parse_args()
    
    # Derive additional paths
    args.fab_path = os.path.join(args.road_data_path, "fab_dataset")
    args.mar_path = os.path.join(args.road_data_path, "mar_dataset")
    
    return args

DATA_PROPERTY = ['time', 'id', 'data', 'time_diffs', 'label']
CHD_DATA_PROPERTY = ['time', 'id', 'DLC', 
                    'Data0', 'Data1', 'Data2', 
                    'Data3', 'Data4', 'Data5', 
                    'Data6', 'Data7', 'label']
DATA_META = {'time': "float64", 'time_diffs': "float64", 'id': 'string', 'data': 'string', 'label' : "bool"}

def make_data_2d(path, args):
    xtrain = np.empty((0, args.window_size, args.feature))
    ytrain = np.empty((0))
    xtest = np.empty((0, args.window_size, args.feature))
    ytest = np.empty((0))

    time_tr = np.empty((0, args.window_size))
    time_te = np.empty((0, args.window_size))

    xval = np.empty((0, args.window_size, args.feature))
    yval = np.empty((0))

    ex_data = np.empty((0, args.window_size, args.feature))
    ex_label = np.empty((0))
    ex_time = np.empty((0, args.window_size))
    
    unique_ids = ROAD_LABEL_MAP_UNIQUE_IDS if (not args.is_mar) else ROAD_MAS_LABEL_MAP_UNIQUE_IDS
    mas_str = "_masquerade" if (args.is_mar) else ""
    
    num_data = 0
    # INCREASE range(1, 3) if want to merge both fabrication and masquerade
    for num_data in range(1, 2):
        for entry in unique_ids.keys():
            if (args.chd and not entry.endswith('.csv')):
                continue
            if not args.chd:
                df = pd.DataFrame()
                if (entry != "max_engine_coolant_temp"):
                    for i in range(3):
                        temp_df = pd.read_csv(f"{path}/{entry}_attack_{i+1}{mas_str}_dataset.csv", header=None, skiprows=1, names=DATA_PROPERTY, dtype=DATA_META)
                        print(temp_df.head())
                        df = pd.concat([df, temp_df], ignore_index=True)
                else:
                    df = pd.read_csv(f"{path}/{entry}_attack{mas_str}_dataset.csv", header=None, skiprows=1, names=DATA_PROPERTY, dtype=DATA_META)

                df = df.sort_values('time', ascending=True)

                label = unique_ids[entry]
                df = process_data(df, label, decimal=True, isCHD=args.chd)
                
                print("Number of Labels:", len(df.label.unique()))
                print("Label Count:", df.label.value_counts())

                print("===", df.head(2))
                as_strided = np.lib.stride_tricks.as_strided
                output_shape = ((len(df) - args.window_size) // args.step + 1, args.window_size)
                features = as_strided(df.features, output_shape, (8*args.step, 8))
                timestamp = as_strided(df.time, output_shape, (8*args.step, 8))
                l = as_strided(df.label, output_shape, (1*args.step, 1)) 
                
                dft = pd.DataFrame({
                    'features': pd.Series(features.tolist()), 
                    'time': pd.Series(timestamp.tolist()), 
                    'label': pd.Series(l.tolist())
                }, index= range(len(l)))

                def map_true_false(x: np.ndarray) -> np.ndarray:
                    # Initialize an empty list to store the result
                    result = []

                    # Loop through each row in the array
                    for row in x:
                        # Check if there is at least one True in the row
                        if np.any(row):  # np.any checks if there's any True value in the row
                            result.append(label)
                        else:
                            result.append(0)
                    
                    # Convert the result to a numpy array and return
                    return np.array(result)

                dft['label'] = map_true_false(l)
                
                data_3d = np.array(dft['features'].tolist())
                time_3d = np.array(dft['time'].tolist())
                labels_cur = dft['label'].values

                if (args.exclude_something and entry.startswith(args.exclude_name)):
                    # Change max_engine to 2
                    ex_data = np.concatenate((ex_data, data_3d))
                    ex_label = np.concatenate((ex_label, labels_cur))
                    ex_time = np.concatenate((ex_time, time_3d))
                    
                    # Still keep the benign data
                    data_3d = data_3d[labels_cur == 0]
                    time_3d = time_3d[labels_cur == 0]
                    labels_cur = labels_cur[labels_cur == 0]

                    benign_train, benign_test, benign_labels_train, benign_labels_test, time_train, time_test = train_test_split(data_3d, labels_cur, time_3d, test_size=0.2, random_state=42, shuffle=True)

                    xtrain = np.concatenate((xtrain, benign_train))
                    ytrain = np.concatenate((ytrain, benign_labels_train))
                    xtest = np.concatenate((xtest, benign_test))
                    ytest = np.concatenate((ytest, benign_labels_test))                                 
                    time_tr = np.concatenate((time_tr, time_train))
                    time_te = np.concatenate((time_te, time_test))
                    continue

                attack_idx = np.where(labels_cur != 0)[0]
                benign_idx = np.where(labels_cur == 0)[0]

                data_train_a, data_test_a, labels_train_a, labels_test_a, time_train_a, time_test_a = train_test_split(data_3d[attack_idx], labels_cur[attack_idx], time_3d[attack_idx], test_size=args.test_size, random_state=42, shuffle=True)
                data_train_b, data_test_b, labels_train_b, labels_test_b, time_train_b, time_test_b = train_test_split(data_3d[benign_idx], labels_cur[benign_idx], time_3d[benign_idx], test_size=args.test_size, random_state=42, shuffle=True)

                if args.val_split:
                    data_train_a, data_val_a, labels_train_a, labels_val_a, time_train_a, time_val_a = train_test_split(data_train_a, labels_train_a, time_train_a, test_size=args.val_size, random_state=42, shuffle=True)
                    data_train_b, data_val_b, labels_train_b, labels_val_b, time_train_a, time_val_b = train_test_split(data_train_b, labels_train_b, time_train_b, test_size=args.val_size, random_state=42, shuffle=True)
                    
                    data_val = np.concatenate((data_val_a, data_val_b))
                    labels_val = np.concatenate((labels_val_a, labels_val_b))
                    time_val = np.concatenate((time_val_a, time_val_b))

                data_train = np.concatenate((data_train_a, data_train_b))
                data_test = np.concatenate((data_test_a, data_test_b))
                labels_train = np.concatenate((labels_train_a, labels_train_b))
                labels_test = np.concatenate((labels_test_a, labels_test_b))
                time_train = np.concatenate((time_train_a, time_train_b))
                time_test = np.concatenate((time_test_a, time_test_b))
            
                print("entry: ", entry, np.shape(data_train), np.shape(data_test))
                print("DATA DISTRIBUTION", Counter(labels_train), Counter(labels_test))
                
                xtrain = np.concatenate((xtrain, data_train))
                ytrain = np.concatenate((ytrain, labels_train))
                if args.val_split:    
                    xval = np.concatenate((xval, data_val))
                    yval = np.concatenate((yval, labels_val))
                xtest = np.concatenate((xtest, data_test))
                ytest = np.concatenate((ytest, labels_test))

                time_tr = np.concatenate((time_tr, time_train))
                time_te = np.concatenate((time_te, time_test))

                print("done ", entry)
        
        unique_ids = ROAD_MAS_LABEL_MAP_UNIQUE_IDS
        mas_str = "_masquerade" 
        path = args.mar_path
        num_data+=1
        
    print("=========")
    print(f"saved xtrain-road-{args.window_type}-{args.data_type}-{file}-{args.window_size}-{args.step}.npy") 

    folder = 'car-hacking' if args.chd else 'road'
    save_dir = os.path.join(args.save_path, folder)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    if (args.exclude_something):
        np.save(os.path.join(save_dir, f"{args.exclude_name}_{args.process_type}_data.npy"), ex_data)
        np.save(os.path.join(save_dir, f"{args.exclude_name}_{args.process_type}_label.npy"), ex_label)
        np.save(os.path.join(save_dir, f"{args.exclude_name}_{args.process_type}_time.npy"), ex_time)

    print("DATA DISTRIBUTION", Counter(ytrain), Counter(yval), Counter(ytest))
    np.save(os.path.join(save_dir, f"xtrain-{folder}-{args.window_type}-{args.data_type}-{file}-{args.window_size}-{args.step}.npy"), xtrain)
    np.save(os.path.join(save_dir, f"timetr-{folder}-{args.window_type}-{args.data_type}-{file}-{args.window_size}-{args.step}.npy"), time_tr)
    np.save(os.path.join(save_dir, f"ytrain-{folder}-{args.window_type}-{args.data_type}-{file}-{args.window_size}-{args.step}.npy"), ytrain) 
    if args.val_split:
        np.save(os.path.join(save_dir, f"xval-{folder}-{args.window_type}-{args.data_type}-{file}-{args.window_size}-{args.step}.npy"), xval)
        np.save(os.path.join(save_dir, f"yval-{folder}-{args.window_type}-{args.data_type}-{file}-{args.window_size}-{args.step}.npy"), yval) 
    np.save(os.path.join(save_dir, f"xtest-{folder}-{args.window_type}-{args.data_type}-{file}-{args.window_size}-{args.step}.npy"), xtest)
    np.save(os.path.join(save_dir, f"timete-{folder}-{args.window_type}-{args.data_type}-{file}-{args.window_size}-{args.step}.npy"), time_te)
    np.save(os.path.join(save_dir, f"ytest-{folder}-{args.window_type}-{args.data_type}-{file}-{args.window_size}-{args.step}.npy"), ytest)
    print("DONE PROCESS") 

    return xtrain, xval, xtest, ytrain, yval, ytest

if __name__ == "__main__":
    args = parse_args()
    
    # Generate file name based on arguments
    if args.is_mar:
        file = f"mar_{args.process_type}_split{args.extra}"
    else:
        file = f"fab_{args.process_type}_split{args.extra}"
    
    if args.chd:
        file = f"chd_{args.process_type}_split{args.extra}"
    if args.exclude_something:
        file = f"{file}_{args.exclude_name}"
    
    folder = 'car-hacking' if args.chd else 'road'
    
    # Select appropriate data path
    if args.chd:
        data_path = args.car_hacking_path
    else:
        data_path = args.mar_path if args.is_mar else args.fab_path

    make_data_2d(data_path, args)
