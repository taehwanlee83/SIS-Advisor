## This is an automated report generation program to create a file named "exp_result.csv".
print('Initialization...', end='')
import os
import pandas as pd
import numpy as np
import timeit

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from imblearn import over_sampling as OS
from imblearn import under_sampling as us
from imblearn import ensemble as es
from imblearn import combine as cmb
from imblearn import metrics as imb_metrics    # Only if you want to get more info on class imbalance

# Global variables
RANDOM_SEED = 10
MAX_ITER = 100

scores = 'roc_auc'
param_grid_original = {
    'n_estimators': [3,5,10,15,20,50,100],
    'max_depth': [3,5,10,15],
    'max_leaf_nodes':[5,10,15,20],
}

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')

# Function declaration
def pseudo_labeling(input_data):
    df_in = pd.DataFrame(input_data)
    df_in_set = df_in.iloc[:,29]
    df_in = df_in.drop(columns=df_in.columns[29])
    df_in.set_index('Model Time', inplace=True)
    df_in = df_in.apply(pd.to_numeric)
    df_in.reset_index(inplace=True)
    df_in['Normal Status'] = np.all(df_in.iloc[:, 8:20]==1, axis=1)
    df_in = df_in.drop(columns=df_in.columns[24:28])
    df_in = df_in.drop(columns=df_in.columns[8:20])
    df_in_label = df_in.iloc[:,13]
    df_in_input = df_in.drop(columns=['Model Time', df_in.columns[13]])
    return df_in_input, df_in_label, df_in_set

def resampling(rs_method, data, RANDOM_SEED):
    if rs_method == 'us':
        rs = us.OneSidedSelection(random_state=RANDOM_SEED)
        x_res, y_res = rs.fit_resample(data.iloc[:,:12], data.iloc[:,12]); data_res = pd.concat([x_res, y_res], axis=1)
        rfc = RandomForestClassifier(random_state=RANDOM_SEED)
        return data_res, rfc
    elif rs_method == 'os':
        rs = OS.SMOTE(random_state=RANDOM_SEED)
        x_res, y_res = rs.fit_resample(data.iloc[:,:12], data.iloc[:,12]); data_res = pd.concat([x_res, y_res], axis=1)
        rfc = RandomForestClassifier(random_state=RANDOM_SEED)
        return data_res, rfc
    elif rs_method == 'osus':
        rs = cmb.SMOTEENN(random_state=RANDOM_SEED)
        x_res, y_res = rs.fit_resample(data.iloc[:,:12], data.iloc[:,12]); data_res = pd.concat([x_res, y_res], axis=1)
        rfc = RandomForestClassifier(random_state=RANDOM_SEED)
        return data_res, rfc
    elif rs_method == 'usos':
        rs_1 = us.OneSidedSelection(random_state=RANDOM_SEED)
        rs_2 = OS.SMOTE(random_state=RANDOM_SEED)
        x_res, y_res = rs_1.fit_resample(data.iloc[:,:12], data.iloc[:,12])
        x_res, y_res = rs_2.fit_resample(x_res, y_res); data_res = pd.concat([x_res, y_res], axis=1)
        rfc = RandomForestClassifier(random_state=RANDOM_SEED)
        return data_res, rfc
    elif rs_method == 'es':
        rfc = es.BalancedRandomForestClassifier(random_state=RANDOM_SEED)
        return data, rfc
    elif rs_method == 'bw':
        rfc = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED)
        return data, rfc
    elif rs_method == 'none':
        rfc = RandomForestClassifier(random_state=RANDOM_SEED)
        return data, rfc
    elif rs_method == 'test':
        rfc = RandomForestClassifier(random_state=RANDOM_SEED)
        return data, rfc
    else:
        print("Check again the resampling method chosen.")
        raise SystemExit

def RFC(rs_method, *training):

    for i in range(10,MAX_ITER,10):
        RANDOM_SEED = i
        trial_count = i/10
        f_exp.write('Trial %i (Random Seed: %i) (resampled by %s)' % (trial_count,RANDOM_SEED, rs_method))
        print('Trial %i %s' % (trial_count, rs_method))
        for data in training:
            start_time = timeit.default_timer()
            data, rfc = resampling(rs_method, data, RANDOM_SEED)
            CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_original, scoring = scores, cv=10, verbose=1, n_jobs=4)
            x_train, y_train = data.iloc[:,:12], data.iloc[:,12]
            CV_rfc.fit(x_train, y_train)
            best_rfc = CV_rfc.best_estimator_
            best_rfc.fit(x_train, y_train)

            y_pred = best_rfc.predict(x_train)

            fpr, tpr, _ = metrics.roc_curve(y_train, y_pred)
            cm = metrics.confusion_matrix(y_train, y_pred)

            print(metrics.auc(fpr,tpr),'\n', cm)
            f_exp.write(', %f, %f, %f, %f, %f' % (metrics.auc(fpr,tpr), cm[0,0], cm[0,1], cm[1,0], cm[1,1]))

            y_pred = best_rfc.predict(x_test)

            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
            cm = metrics.confusion_matrix(y_test, y_pred)

            print(metrics.auc(fpr,tpr),'\n', cm)
            f_exp.write(', %f, %f, %f, %f, %f \n' % (metrics.auc(fpr,tpr), cm[0,0], cm[0,1], cm[1,0], cm[1,1]))
            terminate_time = timeit.default_timer()
            print("Fitting time: %f secs" % (terminate_time-start_time))
    return None
print('Done')
## File import

print('File loading...', end='')
start_time = timeit.default_timer()

# Emulation data set
set_emul_0 = pd.read_csv(data_path + 'emul_0.csv', sep=',')
set_emul_0_small_slug = pd.read_csv(data_path+'emul_0_small_slug.csv', sep=',')
set_emul_0_small_slug2 = pd.read_csv(data_path + 'emul_0_small_slug2.csv', sep=',')
set_emul_1 = pd.read_csv(data_path + 'emul_1.csv', sep=',')
set_emul_1_small_slug = pd.read_csv(data_path + 'emul_1_small_slug.csv', sep=',')
set_emul_1_small_slug2 = pd.read_csv(data_path + 'emul_1_small_slug2.csv', sep=',')
set_emul_5 = pd.read_csv(data_path + 'emul_2.csv', sep=',')
set_emul_5_small_slug = pd.read_csv(data_path + 'emul_2_small_slug.csv', sep=',')
set_emul_5_small_slug2 = pd.read_csv(data_path + 'emul_2_small_slug2.csv', sep=',')
set_emul_10 = pd.read_csv(data_path + 'emul_3.csv', sep=',')
set_emul_10_small_slug = pd.read_csv(data_path + 'emul_3_small_slug.csv', sep=',')
set_emul_10_small_slug2 = pd.read_csv(data_path + 'emul_3_small_slug2.csv', sep=',')
set_emul_15 = pd.read_csv(data_path + 'emul_4.csv', sep=',')
set_emul_15_small_slug = pd.read_csv(data_path + 'emul_4_small_slug.csv', sep=',')
set_emul_15_small_slug2 = pd.read_csv(data_path + 'emul_4_small_slug2.csv', sep=',')


# simulation data set
set1 = pd.read_csv(data_path + '1.csv', sep=',')
set2 = pd.read_csv(data_path + '2.csv', sep=',')
set3 = pd.read_csv(data_path + '3.csv', sep=',')
set4 = pd.read_csv(data_path + '4.csv', sep=',')
set5 = pd.read_csv(data_path + '5.csv', sep=',')
set6 = pd.read_csv(data_path + '6.csv', sep=',')
set7 = pd.read_csv(data_path + '7.csv', sep=',')
set8 = pd.read_csv(data_path + '8.csv', sep=',')
set9 = pd.read_csv(data_path + '9.csv', sep=',')
set10 = pd.read_csv(data_path + '10.csv', sep=',')
set11 = pd.read_csv(data_path + '11.csv', sep=',')

set_complex = pd.read_csv(data_path + 'comp_slug.csv', sep=',')

set1['set'] = 'sep inlet plug'
set2['set'] = 'sep inlet leak'
set3['set'] = 'sep inlet mv closed'
set4['set'] = 'malf on well +'
set5['set'] = 'malf on well -'
set6['set'] = 'Slug'
set7['set'] = 'sep wtr out plug'
set8['set'] = 'sep oil out plug'
set9['set'] = 'sep oil out leak'
set10['set'] = 'sep gas out plug'
set11['set'] = 'sep gas out leak'

set_complex['set'] = 'complex slug'

set_emul_0 = pd.concat([set_emul_0, set_emul_0_small_slug, set_emul_0_small_slug2], ignore_index=True)
set_emul_0['set'] = 'emulated 0% noise'
del set_emul_0_small_slug, set_emul_0_small_slug2
set_overall_sim_1 = pd.concat([set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11], ignore_index=True)
set_overall_sim_2 = pd.concat([set1, set2, set3, set4, set5, set7, set8, set9, set10, set11], ignore_index=True) # simulation data set without scenario 6
set_overall_sim_3 = set6

del set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11

set_diff_op = set_complex
del set_complex
print('Done')

print('Labeling...', end='')
# To remove the rows containing unit (First row contains unit info, and Model Time col value is NaN. By dropna method, this row can be dropped.)
source_sim_clean_1 = set_overall_sim_1.dropna()
source_sim_clean_2 = set_overall_sim_2.dropna()
source_sim_clean_3 = set_overall_sim_3.dropna()
sim_diff_op = set_diff_op.dropna()

target_0_clean = set_emul_0.dropna()

del set_emul_0, set_emul_1, set_overall_sim_1, set_overall_sim_2

# Labeling
df_src_input, df_src_label, df_src_set = pseudo_labeling(source_sim_clean_1); df_src = pd.concat([df_src_input, df_src_label], axis=1)
df_src_wo6_input, df_src_wo6_label, df_src_wo6_set = pseudo_labeling(source_sim_clean_2); df_src_wo6 = pd.concat([df_src_wo6_input, df_src_wo6_label], axis=1)
df_src_custom_input, df_src_custom_label, df_src_custom_set = pseudo_labeling(source_sim_clean_3);
# df_src_custom = pd.concat([df_src_custom_input, df_src_custom_label], axis=1)

df_diff_op_input, df_diff_op_label, df_diff_op_set = pseudo_labeling(sim_diff_op); df_diff_op = pd.concat([df_diff_op_input, df_diff_op_label], axis=1)

df_tgt_0_input, df_tgt_0_label, df_tgt_0_set = pseudo_labeling(target_0_clean); df_tgt_0 = pd.concat([df_tgt_0_input, df_tgt_0_label], axis=1)

terminate_time = timeit.default_timer()
print('Done')



# Training, test set definition

data_1 = df_tgt_0
data_2 = pd.concat([df_tgt_0, df_src_wo6], ignore_index=True)
data_3 = pd.concat([df_tgt_0, df_src], ignore_index=True)
# data_4 = pd.concat([df_tgt_0, df_src_custom], ignore_index=True)
x_test = df_diff_op_input; y_test = df_diff_op_label
rs_method_list = ['none','us', 'os', 'osus', 'usos', 'es', 'bw']

# Fitting & recording

f_exp = open(path+'exp_result.csv','w')
f_exp.write('Loading and preprocessing completed in %f secs.\n' % (terminate_time - start_time))
f_exp.write(', Training, , , , , Test, , , , \n')
f_exp.write(', ROC_AUC, TN, FP, FN, TP, ROC_AUC, TN, FP, FN, TP\n')

for rs_method in rs_method_list:
    RFC(rs_method, *[data_1, data_2, data_3])

f_exp.close()
