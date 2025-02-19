import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding

# EDIT THE CONFIG ADD THE END OF THE FILE TO SUIT THE DATASET

ATTACK_TYPES = {
    "benign": 0,
    "correlated": 1,
    "max_speedometer": 2,
    "reverse_light_off" : 3,
    "reverse_light_on": 4,
    "max_engine" : 5,
    "fuzzing" : 6,
}

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

def get_attack_full(isFab, attack_name):
    map = ATTACK_TYPES if (isFab) else ATTACK_TYPES_MAS
    for attack_type in map.keys():
        if attack_type in attack_name:
            return map[attack_type]
    return None

def get_attack_type_number(attack_name, bi=False, isCHD=False):
    """
    Returns the number corresponding to an attack type if the given attack_name
    contains any of the property names of ATTACK_TYPES as a substring.
    
    :param attack_name: The name of the attack which may contain an attack type.
    :return: The number associated with the first matching attack type, or None if not found.
    """
    map = ATTACK_TYPES if not isCHD else ATTACK_TYPES_CHD
    for attack_type in map.keys():
        if attack_type in attack_name:
            return 1 if (bi) else map[attack_type]
    return None

def split_hex_string_column(df, column_name):
    # Number of characters to group together from the hex string
    chunk_size = 2
    
    # Create the new columns from the hexadecimal string column
    for i in range(0, 16, chunk_size):
        new_col_name = f'Data{i//2 + 1}'
        # Slice the string into chunks of 2, convert to integer from base 16
        df[new_col_name] = df[column_name].str[i:i+chunk_size].apply(lambda x: int(x, 16))
    
    return df

def split_bit_string_column(df, column_name):
    # Convert hex to binary
    df['binary'] = df[column_name].apply(lambda x: bin(int(str(x), 16))[2:].zfill(16))
    
    # Get the maximum bit length
    max_bit_length = df['binary'].str.len().max()
    
    # Split binary string into separate bit columns
    for i in range(max_bit_length):
        new_col_name = f'bit_{i+1}'
        df[new_col_name] = df['binary'].apply(lambda x: int(x[i]) if i < len(x) else 0)
    
    # Drop the intermediate 'binary' column
    df = df.drop(columns=['binary'])
    
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
    scaler_cols = ['Data0' if (chd) else 'Data8', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7',
                    'TSLM',
                    'time_diffs',
                    'id']
    df[scaler_cols] = scaler.fit_transform(df[scaler_cols])

    feature_col = [col for col in df.columns if col not in ['time', 'label']]
    # Step 2: Add this array as a new column to the original DataFrame
    df['features'] = df[feature_col].apply(lambda row: [row[col] for col in feature_col], axis=1)
    
    print('Processing: DONE')

    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] != 0].shape[0])
    print("feature num: ", df.columns)

    return df

def sliding_window_df_to_ndarray_with_labels(df, label, window_size=38, step=38):
    num_samples = (df.shape[0] - window_size) // step + 1
    data_3d = np.empty((num_samples, window_size, df.shape[1] - 1))
    labels = np.empty(num_samples, dtype="int64")

    for i in range(num_samples):
        start_row = i * step
        end_row = start_row + window_size
        window_df = df.iloc[start_row:end_row]
       
        features = window_df.drop(columns=['Label']).to_numpy()
        # scaler = MinMaxScaler()
        # features = scaler.fit_transform(features)

        # Exclude the label column for the 12x12 matrix
        data_3d[i, :, :] = features
        
        # Determine the label for the current window
        labels[i] = label if window_df['Label'].any() else 0

    return data_3d, labels


def list_files_in_directory(directory_path):
    """
    Reads all filenames in a given directory path and returns them as a list.

    :param directory_path: The path of the directory from which to read filenames.
    :return: A list of filenames found in the directory.
    """
    # List everything in the directory
    all_entries = os.listdir(directory_path)
    
    # Filter out directories, keep only files
    files = [entry for entry in all_entries if os.path.isfile(os.path.join(directory_path, entry))]
    
    return files

def chunking(df, label):
    n = 12  # Chunk size
    final_arrays = []
    final_labels = []



    # Process each chunk
    for start in range(0, len(df), n):
        chunk = df.iloc[start:start+n]
        # Check if the chunk is smaller than n and pad if necessary
        padded_chunk = np.pad(chunk.drop(columns=['Label']).values, ((0, max(0, n - len(chunk))), (0, 0)), mode='constant', constant_values=0)
        # Flatten the chunk and store
        data_array = padded_chunk.flatten()
        # Store label (True if any True in the chunk)
        l = label if chunk['Label'].any() else 0
        
        final_arrays.append(data_array)
        final_labels.append(l)

    # Convert the list of arrays to a 2D numpy array
    return np.array(final_arrays), np.array(final_labels)

def split_hex_to_bits(df, hex_column, col_num=64):
    # Convert hex to binary
    df['binary'] = df[hex_column].apply(lambda x: bin(int(x, 16))[2:].zfill(col_num))

    # Split binary string into separate columns
    df_binary = df['binary'].apply(lambda x: pd.Series(list(x)))

    # Rename columns
    df_binary.columns = [f'bit_{i}' for i in range(col_num)]

    # Concatenate the original dataframe with the new binary columns
    df = pd.concat([df, df_binary], axis=1)

    # Drop the intermediate 'binary' column
    df = df.drop(columns=['binary'])

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

def make_data_2d(path):
    WINDOW_SIZE = 16
    STEP = 16
    FEATURE = 11
    TEST_SIZE = 0.2
    
    val_split = True
    VAL_SIZE = 0.1

    xtrain = np.empty((0, WINDOW_SIZE, FEATURE))
    ytrain = np.empty((0))
    xtest = np.empty((0, WINDOW_SIZE, FEATURE))
    ytest = np.empty((0))

    time_tr = np.empty((0, WINDOW_SIZE))
    time_te = np.empty((0, WINDOW_SIZE))

    xval = np.empty((0, WINDOW_SIZE, FEATURE))
    yval = np.empty((0))

    ex_data = np.empty((0, WINDOW_SIZE, FEATURE))
    ex_label = np.empty((0))
    ex_time = np.empty((0, WINDOW_SIZE))
    
    all_entries = os.listdir(path)
    for entry in all_entries:
        if (chd and not entry.endswith('.csv')):
            continue
        if os.path.isfile(os.path.join(path, entry)):
            df = pd.read_csv(path+'/'+entry, header=None, skiprows=1, names=CHD_DATA_PROPERTY if chd else DATA_PROPERTY)
            df = df.sort_values('time', ascending=True)

            label = get_attack_type_number(entry, isCHD=chd)
            df = process_data(df, label, decimal=True, isCHD=chd)

            print("===", df.head(2))

            # df.to_csv(f"/home/ntmduy/CANET/CICIDS2017/csv/{entry}", index=False)

            # data_3d, labels_cur = sliding_window_df_to_ndarray_with_labels(df.drop("features", axis=1), label, window_size=WINDOW_SIZE, step=STEP)
            as_strided = np.lib.stride_tricks.as_strided
            output_shape = ((len(df) - WINDOW_SIZE) // STEP + 1, WINDOW_SIZE)
            features = as_strided(df.features, output_shape, (8*STEP, 8))
            timestamp = as_strided(df.time, output_shape, (8*STEP, 8))
            l = as_strided(df.label, output_shape, (1*STEP, 1)) 
            
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

            if (exclude_something and entry.startswith(exclude_name)):
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

            data_train_a, data_test_a, labels_train_a, labels_test_a, time_train_a, time_test_a = train_test_split(data_3d[attack_idx], labels_cur[attack_idx], time_3d[attack_idx], test_size=TEST_SIZE, random_state=42, shuffle=True)
            data_train_b, data_test_b, labels_train_b, labels_test_b, time_train_b, time_test_b = train_test_split(data_3d[benign_idx], labels_cur[benign_idx], time_3d[benign_idx], test_size=TEST_SIZE, random_state=42, shuffle=True)

            if val_split:
                data_train_a, data_val_a, labels_train_a, labels_val_a, time_train_a, time_val_a = train_test_split(data_train_a, labels_train_a, time_train_a, test_size=VAL_SIZE, random_state=42, shuffle=True)
                data_train_b, data_val_b, labels_train_b, labels_val_b, time_train_a, time_val_b = train_test_split(data_train_b, labels_train_b, time_train_b, test_size=VAL_SIZE, random_state=42, shuffle=True)
                
                data_val = np.concatenate((data_val_a, data_val_b))
                labels_val = np.concatenate((labels_val_a, labels_val_b))
                time_val = np.concatenate((time_val_a, time_val_b))

            data_train, data_test, labels_train, labels_test, time_train, time_test = np.concatenate((data_train_a, data_train_b)), np.concatenate((data_test_a, data_test_b)), np.concatenate((labels_train_a, labels_train_b)), np.concatenate((labels_test_a, labels_test_b)), np.concatenate((time_train_a, time_train_b)), np.concatenate((time_test_a, time_test_b))
        
            
            print("entry: ", entry, np.shape(data_train), np.shape(data_test))
            print("DATA DISTRIBUTION", Counter(labels_train), Counter(labels_test))
            
            xtrain = np.concatenate((xtrain, data_train))
            ytrain = np.concatenate((ytrain, labels_train))
            if val_split:    
                xval = np.concatenate((xval, data_val))
                yval = np.concatenate((yval, labels_val))
            xtest = np.concatenate((xtest, data_test))
            ytest = np.concatenate((ytest, labels_test))

            time_tr = np.concatenate((time_tr, time_train))
            time_te = np.concatenate((time_te, time_test))

            print("done ", entry)
        
    print("=========")
    print(f"saved xtrain-road-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy") 

    if (exclude_something):
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/{exclude_name}_{process_type}_data.npy', ex_data)
        np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/{exclude_name}_{process_type}_label.npy", ex_label) 
        np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/{exclude_name}_{process_type}_time.npy", ex_time) 
    
    if (split_finetune):
        xtrain, x_ft, ytrain, y_ft, time_tr, time_ft = train_test_split(xtrain, ytrain, time_tr, test_size=0.5, random_state=42, shuffle=True)
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/x-ft-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', x_ft)
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/y-ft-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', y_ft)
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/time-ft-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', time_ft)
        print("FINETUNE DISTRIBUTION", Counter(y_ft))

    print("DATA DISTRIBUTION", Counter(ytrain), Counter(yval), Counter(ytest))
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xtrain-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', xtrain)
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/timetr-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', time_tr)
    np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/ytrain-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy", ytrain) 
    if val_split:
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xval-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', xval)
        np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/yval-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy", yval) 
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xtest-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', xtest)
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/timete-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', time_te)
    np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/ytest-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy", ytest)
    print("DONE PROCESS") 

    return xtrain, xval, xtest, ytrain, yval, ytest

def make_data_2d_by_attack(path):
    WINDOW_SIZE = 16
    STEP = 16
    FEATURE = 9
    
    val_split = True

    xtrain = np.empty((0, WINDOW_SIZE, FEATURE))
    ytrain = np.empty((0))
    xtest = np.empty((0, WINDOW_SIZE, FEATURE))
    ytest = np.empty((0))

    time_tr = np.empty((0, WINDOW_SIZE))
    time_te = np.empty((0, WINDOW_SIZE))

    xval = np.empty((0, WINDOW_SIZE, FEATURE))
    yval = np.empty((0))

    ex_data = np.empty((0, WINDOW_SIZE, FEATURE))
    ex_label = np.empty((0))
    ex_time = np.empty((0, WINDOW_SIZE))
    
    # label = 1
    isFab = True
    for i in range(2):
        all_entries = os.listdir(path)
        for entry in all_entries:
            if (chd and not entry.endswith('.csv')):
                continue
            if os.path.isfile(os.path.join(path, entry)):
                df = pd.read_csv(path+'/'+entry, header=None, skiprows=1, names=CHD_DATA_PROPERTY if chd else DATA_PROPERTY)
                df = df.sort_values('time', ascending=True)

                label = get_attack_full(isFab, entry)
                df = process_data(df, label, decimal=True, isCHD=chd)

                print("===", df.head(2))

                # df.to_csv(f"/home/ntmduy/CANET/CICIDS2017/csv/{entry}", index=False)

                # data_3d, labels_cur = sliding_window_df_to_ndarray_with_labels(df.drop("features", axis=1), label, window_size=WINDOW_SIZE, step=STEP)
                as_strided = np.lib.stride_tricks.as_strided
                output_shape = ((len(df) - WINDOW_SIZE) // STEP + 1, WINDOW_SIZE)
                features = as_strided(df.features, output_shape, (8*STEP, 8))
                timestamp = as_strided(df.time, output_shape, (8*STEP, 8))
                l = as_strided(df.label, output_shape, (1*STEP, 1)) 
                
                dft = pd.DataFrame({
                    'features': pd.Series(features.tolist()), 
                    'time': pd.Series(timestamp.tolist()), 
                    'label': pd.Series(l.tolist())
                }, index= range(len(l)))
                dft['label'] = dft['label'].apply(lambda x: label if any(x) else 0)
                
                data_3d = np.array(dft['features'].tolist())
                time_3d = np.array(dft['time'].tolist())
                labels_cur = dft['label'].values

                if (exclude_something and entry.startswith(exclude_name)):
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

                data_train_a, data_test_a, labels_train_a, labels_test_a, time_train_a, time_test_a = train_test_split(data_3d[attack_idx], labels_cur[attack_idx], time_3d[attack_idx], test_size=0.2, random_state=42, shuffle=True)
                data_train_b, data_test_b, labels_train_b, labels_test_b, time_train_b, time_test_b = train_test_split(data_3d[benign_idx], labels_cur[benign_idx], time_3d[benign_idx], test_size=0.2, random_state=42, shuffle=True)
                

                data_train, data_test, labels_train, labels_test, time_train, time_test = np.concatenate((data_train_a, data_train_b)), np.concatenate((data_test_a, data_test_b)), np.concatenate((labels_train_a, labels_train_b)), np.concatenate((labels_test_a, labels_test_b)), np.concatenate((time_train_a, time_train_b)), np.concatenate((time_test_a, time_test_b))
                
                if val_split:
                    data_train, data_val, labels_train, labels_val, time_train, time_val = train_test_split(data_train, labels_train, time_train, test_size=0.2, random_state=42, shuffle=True)
                
                print("entry: ", entry, np.shape(data_train), np.shape(data_test))
                if val_split:
                    print(np.shape(data_val))
                print("DATA DISTRIBUTION", Counter(labels_train), Counter(labels_test))
                if val_split:
                    print(Counter(labels_train))
                xtrain = np.concatenate((xtrain, data_train))
                ytrain = np.concatenate((ytrain, labels_train))
                if val_split:    
                    xval = np.concatenate((xval, data_val))
                    yval = np.concatenate((yval, labels_val))
                xtest = np.concatenate((xtest, data_test))
                ytest = np.concatenate((ytest, labels_test))

                time_tr = np.concatenate((time_tr, time_train))
                time_te = np.concatenate((time_te, time_test))

                print("done ", entry)
        
    print("=========")
    print(f"saved xtrain-road-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy") 

    if (exclude_something):
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/{exclude_name}_{process_type}_data.npy', ex_data)
        np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/{exclude_name}_{process_type}_label.npy", ex_label) 
        np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/{exclude_name}_{process_type}_time.npy", ex_time) 
    
    if (split_finetune):
        xtrain, x_ft, ytrain, y_ft, time_tr, time_ft = train_test_split(xtrain, ytrain, time_tr, test_size=0.5, random_state=42, shuffle=True)
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/x-ft-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', x_ft)
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/y-ft-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', y_ft)
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/time-ft-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', time_ft)
        print("FINETUNE DISTRIBUTION", Counter(y_ft))

    print("DATA DISTRIBUTION", Counter(ytrain), Counter(yval), Counter(ytest))
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xtrain-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', xtrain)
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/timetr-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', time_tr)
    np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/ytrain-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy", ytrain) 
    if val_split:
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xval-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', xval)
        np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/yval-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy", yval) 
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xtest-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', xtest)
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/timete-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy', time_te)
    np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/ytest-{folder}-{type}-{data_type}-{file}-{WINDOW_SIZE}-{STEP}.npy", ytest)
    print("DONE PROCESS") 

    return xtrain, xval, xtest, ytrain, yval, ytest

def make_data(path, bi=False):
    all_entries = os.listdir(path)
    
    data = pd.DataFrame()
    labels = []
    c = 0
    for entry in all_entries:
        if os.path.isfile(os.path.join(path, entry)):
            df = pd.read_csv(path+'/'+entry, header=None, skiprows=1, names=DATA_PROPERTY)
            #print("on", entry)
            label = get_attack_type_number(entry, bi)
            df = process_data(df, label, decimal=True)

            data = pd.concat([data, df], axis=0, ignore_index=True, sort=False)
            labels.append(label)
            c+=1

    #

    data_ = data.drop(['Label'], axis=1)
    data_ed = data_.values
    # StandardScaler/
    # data_ed = MinMaxScaler().fit_transform(data_ed)

    L = data['Label']
    L = L.values
    i = 0
    labels = L.reshape(L.shape[0], 1)

    print("data sample", data_ed[0], labels[0])

    return data_ed, labels

def make_data_extend(path):
    all_entries = os.listdir(path)
    
    data = np.empty((0, 144))
    labels = None
    for entry in all_entries:
        if os.path.isfile(os.path.join(path, entry)):
            df = pd.read_csv(path+'/'+entry, header=None, skiprows=1, names=DATA_PROPERTY)
            
            label = get_attack_type_number(entry)
            df = process_data(df, label, decimal=True)

            data_3d, labels_cur = chunking(df, label)
            print('=======', np.shape(data), np.shape(data_3d), np.unique(labels_cur))

            data = np.concatenate((data, data_3d), axis=0)
            if (labels is None): 
                labels = labels_cur 
            else:
                labels = np.concatenate((labels, labels_cur), axis=0)

    return data, labels   

import torch
def only_id_image(path):
    STEP = 16
    val_split = True

    xtrain = np.empty((0, STEP, STEP))
    ytrain = np.empty((0))
    xtest = np.empty((0, STEP, STEP))
    ytest = np.empty((0))
    xval = np.empty((0, STEP, STEP))
    yval = np.empty((0))
    ex_data = np.empty((0, STEP, STEP))
    ex_label = np.empty((0))

    all_entries = os.listdir(path)
    for entry in all_entries:
        if (chd and not entry.endswith('.csv')):
            continue
        if os.path.isfile(os.path.join(path, entry)):
            df = pd.read_csv(path+'/'+entry, header=None, skiprows=1, names=CHD_DATA_PROPERTY if chd else DATA_PROPERTY)
            df = df.sort_values('time', ascending=True)

            label = get_attack_type_number(entry, isCHD=chd)
            
            #GET only ID and label
            df = df[['id', 'label']]
            df['label'] = df['label'].apply(lambda x: label if x == "T" else 0)

            # gray scale ID
            df['id'] = df['id'].apply(lambda x: int(x, 16))
            def normalize_id(can_id):
                return int((can_id / 2047) * 255)

            # group a consecutive sequence of 48 CAN IDs into a grayscale CAN image
            # Step 2: Create 48x48 Numpy Array from 2304 CAN IDs
            def create_np_array_from_ids(df, start_idx, LABEL):
                """
                Create a 48x48 numpy array from 2304 consecutive CAN IDs.
                Args:
                    df (pd.DataFrame): DataFrame containing CAN IDs and Labels
                    start_idx (int): Starting index for the sequence of CAN IDs
                Returns:
                    np.array: 48x48 numpy array representing the data
                    int: Corresponding label (0 for normal, >0 for attack)
                """
                # Extract 2304 consecutive CAN IDs
                can_ids = df['id'].iloc[start_idx:start_idx+STEP*STEP].values
                
                # Normalize CAN IDs to values (0-255)
                pixel_values = [normalize_id(can_id) for can_id in can_ids]
                
                # Reshape the list of 2304 pixel values into a 48x48 numpy array
                np_array = np.array(pixel_values).reshape(STEP, STEP)
                
                # Check if any label in the 2304 rows is > 0 (indicating an attack)
                if (df['label'].iloc[start_idx:start_idx+STEP*STEP] > 0).any():
                    label = LABEL  # Attack label
                else:
                    label = 0  # Normal label
                
                return np_array, label

            arrays ,labels = [], []
            for start_idx in range(0, len(df) - 2304 + 1, STEP):
                np_array, l = create_np_array_from_ids(df, start_idx, label)
                arrays.append(np_array)
                labels.append(l)

            arrays = np.array(arrays)
            labels = np.array(labels)

            if (exclude_something and entry.startswith(exclude_name)):
                # Change max_engine to 2
                ex_data = np.concatenate((ex_data, arrays))
                ex_label = np.concatenate((ex_label, labels))
                
                # Still keep the benign data
                arrays = arrays[labels == 0]
                labels = labels[labels == 0]

                benign_train, benign_test, benign_labels_train, benign_labels_test = train_test_split(arrays, labels, test_size=0.2, random_state=42, shuffle=True)

                xtrain = np.concatenate((xtrain, benign_train))
                ytrain = np.concatenate((ytrain, benign_labels_train))
                xtest = np.concatenate((xtest, benign_test))
                ytest = np.concatenate((ytest, benign_labels_test))                                 
                continue

            # TRAIN TEST SPLIT            
            attack_idx = np.where(np.array(labels) != 0)[0]
            benign_idx = np.where(np.array(labels) == 0)[0]
            data_train_a, data_test_a, labels_train_a, labels_test_a = train_test_split(arrays[attack_idx], labels[attack_idx], test_size=0.2, random_state=42, shuffle=True)
            data_train_b, data_test_b, labels_train_b, labels_test_b = train_test_split(arrays[benign_idx], labels[benign_idx], test_size=0.2, random_state=42, shuffle=True)
            

            data_train, data_test, labels_train, labels_test = np.concatenate((data_train_a, data_train_b)), np.concatenate((data_test_a, data_test_b)), np.concatenate((labels_train_a, labels_train_b)), np.concatenate((labels_test_a, labels_test_b))
            
            if val_split:
                data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train, test_size=0.2, random_state=42, shuffle=True)

            xtrain = np.concatenate((xtrain, data_train))
            ytrain = np.concatenate((ytrain, labels_train))
            xtest = np.concatenate((xtest, data_test))
            ytest = np.concatenate((ytest, labels_test))
            if val_split:    
                xval = np.concatenate((xval, data_val))
                yval = np.concatenate((yval, labels_val))
            print("DATA DISTRIBUTION", Counter(labels_train), Counter(labels_val)  if (val_split) else "", Counter(labels_test))

    if (exclude_something):
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/{exclude_name}_{process_type}-{STEP}-{STEP}_data.npy', ex_data)
        np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/{exclude_name}_{process_type}-{STEP}-{STEP}_label.npy", ex_label)
    
    # SAVE
    print("DATA DISTRIBUTION", Counter(ytrain), Counter(yval) if (val_split) else "", Counter(ytest))
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xtrain-{folder}-{type}-{data_type}-{file}-{STEP}-{STEP}.npy', xtrain)
    np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/ytrain-{folder}-{type}-{data_type}-{file}-{STEP}-{STEP}.npy", ytrain) 
    if val_split:
        np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xval-{folder}-{type}-{data_type}-{file}-{STEP}-{STEP}.npy', xval)
        np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/yval-{folder}-{type}-{data_type}-{file}-{STEP}-{STEP}.npy", yval) 
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xtest-{folder}-{type}-{data_type}-{file}-{STEP}-{STEP}.npy', xtest)
    np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/ytest-{folder}-{type}-{data_type}-{file}-{STEP}-{STEP}.npy", ytest)


def just_process_data(path):
    FEATURE = 11

    xtrain = np.empty((0, FEATURE))
    ytrain = np.empty((0))
    xval = np.empty((0, FEATURE))
    yval = np.empty((0))
    xtest = np.empty((0, FEATURE))
    ytest = np.empty((0))

    all_entries = os.listdir(path)
    for entry in all_entries:
        if os.path.isfile(os.path.join(path, entry)):
            df = pd.read_csv(path+'/'+entry, header=None, skiprows=1, names=CHD_DATA_PROPERTY if chd else DATA_PROPERTY)
            df = df.sort_values('time', ascending=True)

            label = get_attack_type_number(entry, isCHD=chd)
            df = process_data(df, label, decimal=True, isCHD=chd)

            df['label'] = df['label'].apply(lambda x: label if x else 0)

            x = df['features'].values
            y = df['label'].values

            attack_idx = np.where(y != 0)[0]
            benign_idx = np.where(y == 0)[0]

            data_train_a, data_test_a, labels_train_a, labels_test_a = train_test_split(x[attack_idx], y[attack_idx], test_size=0.2, random_state=42, shuffle=True)
            data_train_b, data_test_b, labels_train_b, labels_test_b = train_test_split(x[benign_idx], y[benign_idx], test_size=0.2, random_state=42, shuffle=True)

            data_train, data_test, labels_train, labels_test = np.concatenate((data_train_a, data_train_b)), np.concatenate((data_test_a, data_test_b)), np.concatenate((labels_train_a, labels_train_b)), np.concatenate((labels_test_a, labels_test_b))

            data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train, test_size=0.2, random_state=42, shuffle=True)

            xtrain = np.concatenate((xtrain, data_train))
            ytrain = np.concatenate((ytrain, labels_train))
            xval = np.concatenate((xval, data_val))
            yval = np.concatenate((yval, labels_val))
            xtest = np.concatenate((xtest, data_test))
            ytest = np.concatenate((ytest, labels_test))
    
    print("DATA DISTRIBUTION", Counter(ytrain), Counter(yval), Counter(ytest))
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xtrain-raw-{data_type}.npy', xtrain)
    np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/ytrain-raw-{data_type}.npy", ytrain)
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xval-raw-{data_type}.npy', xval)
    np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/yval-raw-{data_type}.npy", yval)
    np.save (f'/home/ntmduy/CANET/CICIDS2017/data/{folder}/xtest-raw-{data_type}.npy', xtest)
    np.save(f"/home/ntmduy/CANET/CICIDS2017/data/{folder}/ytest-raw-{data_type}.npy", ytest)



import random
torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)

# ========================= Config to window the processed data =========================
# Change the path to your CSV folder
DATA_PATH = '/home/ntmduy/CANET/CICIDS2017/data/road/'
FAB_PATH = DATA_PATH + "fab_dataset"
MAR_PATH = DATA_PATH + "mar_dataset"

DATA_PROPERTY = ['time', 'id', 'data', 'time_diffs', 'label']
CHD_DATA_PROPERTY = ['time', 'id', 'DLC', 
                           'Data0', 'Data1', 'Data2', 
                           'Data3', 'Data4', 'Data5', 
                           'Data6', 'Data7', 'label']
DATA_META = {'time': "float64", 'time_diffs': "int64", 'id': 'object', 'data': 'object', 'label' : "bool"}

type = '2d'                     # turn single message to window of messages
chd = False                     # use car-hacking or ROAD dataset
is_mar = False                  # use fabrication or masquerade dataset in ROAD
exclude_something = True        # exclude some attack type
exclude_name = "fuzzing"        # exclude attack type name
process_type="ss_11_2d_no"      # process type name "11 feature 2d"
split_finetune = False          # split data for finetuning
 
extra = "_96test"
file = f"mar_{process_type}_split{extra}" if (is_mar) else f"fab_{process_type}_split{extra}"
if (chd):
    file = f"chd_{process_type}_split{extra}"
if (exclude_something):
    file = f"{file}_{exclude_name}"
data_type = 'dec'


folder = 'car-hacking' if chd else 'road'
if chd:
    DATA_PATH = '/home/ntmduy/car-ids/raw-data/car-hacking/'
    make_data_2d(DATA_PATH)
    # only_id_image(DATA_PATH)
    # just_process_data(DATA_PATH)
else:
    if type == 'ext':
        data, labels = make_data_extend(MAR_PATH if is_mar else FAB_PATH)
    elif type == '2d':
        make_data_2d(MAR_PATH if is_mar else FAB_PATH)
    else:
        data, labels = make_data(MAR_PATH if is_mar else FAB_PATH, bi=False)

# np.save(f'./data/road/data-road.npy', data)
# np.save(f"./data/road/label-road.npy", labels) 

# ==================== TRAIN_TEST_SPLIT ====================
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=False)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, shuffle=False)

# print("DATA:", np.shape(X_train), np.shape(X_test), data.shape, labels.shape, y_train.shape, y_test.shape)

# # Calculate the ratio of data with label = 0 compared to others
# ratio = len(labels[labels != 0]) / len(labels[labels == 0])
# print("Ratio of data with label = 0 compared to others:", ratio * 100, "%")

# # Calculate the ratio of data with label = 0 compared to others
# ratio_tr = len(y_train[y_train != 0]) / len(y_train[y_train == 0])
# print("Ratio of train data with label = 0 compared to others:", ratio_tr* 100, "%")

# # Calculate the ratio of data with label = 0 compared to others
# ratio_t = len(y_test[y_test != 0]) / len(y_test[y_test == 0])
