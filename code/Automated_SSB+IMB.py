## This is an automated report generation program to create a file named "exp_result.csv".
print('Initialization...', end='')
import os
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn import preprocessing

from imblearn import over_sampling as OS
from imblearn import under_sampling as us
from imblearn import ensemble as es 
from imblearn import combine as cmb
from imblearn.pipeline import Pipeline
from imblearn import metrics as imb_metrics     # Only if you want to get more info on class imbalance

# Global variables
MAX_ITER = 100

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

scores = 'roc_auc'
param_grid_original = {
    'rfc__n_estimators': [3,5,10,15,20,50,100],
    'rfc__max_depth': [3,5,10,15],
    'rfc__max_leaf_nodes':[5,10,15,20],
}

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data') + os.sep

# Function declarations
def pseudo_labeling(input_data):
    df_in = pd.DataFrame(input_data)
    df_in_set = df_in.iloc[:,29]
    df_in = df_in.drop(columns=df_in.columns[29])
    df_in.set_index('Model Time', inplace=True)
    df_in = df_in.apply(pd.to_numeric)
    df_in.reset_index(inplace=True)
    df_in['Normal Status'] = np.all(df_in.iloc[:, 8:20]==1, axis=1)
    df_in['Normal Status'] = (~df_in['Normal Status']).astype(int) 
    df_in.rename(columns={'Normal Status': 'Fault Status (1=fault)'}, inplace=True)     # 1: fault, 0: normal
    df_in = df_in.drop(columns=df_in.columns[24:28])
    df_in = df_in.drop(columns=df_in.columns[8:20])
    df_in_label = df_in.iloc[:,13]
    df_in_input = df_in.drop(columns=['Model Time', df_in.columns[13]])
    return df_in_input, df_in_label, df_in_set

def _smote_strategy(y, minority_label=1):
    from collections import Counter
    c = Counter(y)
    return {minority_label: c.get(minority_label, 0) + 2500}

def resampling(rs_method, data, RANDOM_SEED):
    rfc = RandomForestClassifier(random_state=RANDOM_SEED)
    if rs_method == 'us':
        rs = us.OneSidedSelection(random_state=RANDOM_SEED)
        estimator = Pipeline([('rs', rs), ('rfc', rfc)])
        return data, estimator
    elif rs_method == 'os':
        rs = OS.SMOTE(
            random_state=RANDOM_SEED,
            sampling_strategy=lambda y: _smote_strategy(y, minority_label=1)
        )
        estimator = Pipeline([('rs', rs), ('rfc', rfc)])
        return data, estimator
        
    elif rs_method == 'osus':
        rs_1 = OS.SMOTE(
            random_state=RANDOM_SEED,
            sampling_strategy=lambda y: _smote_strategy(y, minority_label=1)
        )
        rs = cmb.SMOTEENN(
            random_state=RANDOM_SEED,
            smote=rs_1
        )
        estimator = Pipeline([('rs', rs), ('rfc', rfc)])
        return data, estimator
    
    elif rs_method == 'usos':
        rs_1 = us.OneSidedSelection(random_state=RANDOM_SEED)
        minority_count = sum(data.iloc[:,12] == 1)
        target_minority = min(4000, minority_count * 50)
        rs_2 = OS.SMOTE(
            random_state=RANDOM_SEED,
            sampling_strategy=lambda y: _smote_strategy(y, minority_label=1)
        )
        estimator = Pipeline([('oss', rs_1), ('smote', rs_2), ('rfc', rfc)])
        return data, estimator
    elif rs_method == 'es':
        rfc = es.BalancedRandomForestClassifier(random_state=RANDOM_SEED)
        estimator = Pipeline([('rfc', rfc)])
        return data, estimator
    elif rs_method == 'bw':
        rfc = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED)
        estimator = Pipeline([('rfc', rfc)])
        return data, estimator
    elif rs_method == 'none':
        estimator = Pipeline([('rfc', rfc)])
        return data, estimator
    else:
        raise ValueError(f"Unknown rs_method: {rs_method}")

def RFC(rs_method, *training):

    for i in range(1, MAX_ITER+1):
        RANDOM_SEED = i * 10
        trial_count = i
        f_exp.write('Trial %i (Random Seed: %i) (resampled by %s)' % (trial_count,RANDOM_SEED, rs_method))
        print('Trial %i %s' % (trial_count, rs_method))
        for data in training:
            start_time = timeit.default_timer()
            data, estimator = resampling(rs_method, data, RANDOM_SEED)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
            CV_rfc = GridSearchCV(estimator=estimator, param_grid=param_grid_original, scoring=scores, cv=cv, verbose=1, n_jobs=4)
            x_train, y_train = data.iloc[:,:12], data.iloc[:,12]
            CV_rfc.fit(x_train, y_train)
            best_rfc = CV_rfc.best_estimator_
            best_rfc.fit(x_train, y_train)

            y_prob_tr = best_rfc.predict_proba(x_train)[:, 1]
            y_pred_tr = (y_prob_tr > 0.5).astype(int)
            auc_tr = metrics.roc_auc_score(y_train, y_prob_tr)
            cm_tr = metrics.confusion_matrix(y_train, y_pred_tr, labels=[0,1])
            f_exp.write(', %f, %f, %f, %f, %f' % (auc_tr, cm_tr[0,0], cm_tr[0,1], cm_tr[1,0], cm_tr[1,1]))

            y_prob_te = best_rfc.predict_proba(x_test)[:, 1]
            y_pred_te = (y_prob_te > 0.5).astype(int)
            auc_te = metrics.roc_auc_score(y_test, y_prob_te)
            cm_te = metrics.confusion_matrix(y_test, y_pred_te, labels=[0,1])
            print(auc_te,'\n', cm_te)
            f_exp.write(', %f, %f, %f, %f, %f \n' % (auc_te, cm_te[0,0], cm_te[0,1], cm_te[1,0], cm_te[1,1]))

            terminate_time = timeit.default_timer()
            print("Fitting time: %f secs" % (terminate_time-start_time))
    return None
print('Done.')

def plot_pca_projection(original, sim, sim_wo6, test, save_path="pca_projection.png"):
    """Create a 1x3 PCA panel: (a) Orig vs Test, (b) Orig+Sim(w/6) vs Test, (c) Orig+Sim(w/o6) vs Test."""
	# 1) Extract feature matrices (first 12 columns)
    X_orig = original.iloc[:, :12].to_numpy()
    X_sim = sim.iloc[:, :12].to_numpy()          # with scenario 6
    X_sim_wo6 = sim_wo6.iloc[:, :12].to_numpy()  # without scenario 6
    X_test = test.iloc[:, :12].to_numpy()

    # 2) Fit a single PCA on all samples to ensure a common projection basis
    X_all = np.vstack([X_orig, X_sim, X_sim_wo6, X_test])
    pca = PCA(n_components=2, random_state=0)
    Z_all = pca.fit_transform(X_all)

    # 3) Split back to each dataset
    n1, n2, n3, n4 = len(X_orig), len(X_sim), len(X_sim_wo6), len(X_test)
    Z_orig, Z_sim, Z_sim_wo6, Z_test = np.split(Z_all, [n1, n1 + n2, n1 + n2 + n3])

    # 4) Shared axis limits across panels for fair visual comparison
    xmin = np.min(Z_all[:, 0]); xmax = np.max(Z_all[:, 0])
    ymin = np.min(Z_all[:, 1]); ymax = np.max(Z_all[:, 1])
    xpad = 0.05 * (xmax - xmin + 1e-12)
    ypad = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    # 5) Marker/alpha scheme:
    #    - Original, Test: typically fewer samples → larger & darker
    #    - Simulation sets: typically many samples → smaller & lighter
    style = {
        "orig":  dict(c="blue",  marker="o", s=40, alpha=0.9, label="Original"),
        "test":  dict(c="red",   marker="x", s=60, alpha=0.95, label="Test (complex_slug)"),
        "sim0":  dict(c="teal",  marker="s", s=30, alpha=0.8, label="Sim (w/o6)"),
        "sim":   dict(c="green",marker="^", s=60, alpha=0.5, label="Sim (w/6)")
    }

    # 6) Plot panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)

    # (a) Original vs Test
    ax = axes[0]
    ax.scatter(Z_orig[:, 0], Z_orig[:, 1], **style["orig"])
    ax.scatter(Z_test[:, 0], Z_test[:, 1], **style["test"])
    ax.set_title("(a) Original vs Test")
    ax.set_xlabel("PCA Component 1"); ax.set_ylabel("PCA Component 2")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # (b) Original + Sim (w/o6) vs Test
    ax = axes[1]
    ax.scatter(Z_sim_wo6[:, 0], Z_sim_wo6[:, 1], **style["sim0"])
    ax.scatter(Z_orig[:, 0], Z_orig[:, 1], **style["orig"])
    ax.scatter(Z_test[:, 0], Z_test[:, 1], **style["test"])
    ax.set_title("(b) Original + Simulation (w/o6) vs Test")
    ax.set_xlabel("PCA Component 1"); ax.set_ylabel("PCA Component 2")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # (c) Original + Sim (w/6) vs Test
    ax = axes[2]
    ax.scatter(Z_sim[:, 0], Z_sim[:, 1], **style["sim"])
    ax.scatter(Z_orig[:, 0], Z_orig[:, 1], **style["orig"])
    ax.scatter(Z_test[:, 0], Z_test[:, 1], **style["test"])
    ax.set_title("(c) Original + Simulation (w/6) vs Test")
    ax.set_xlabel("PCA Component 1"); ax.set_ylabel("PCA Component 2")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


## File import

print('File loading...', end='')
start_time = timeit.default_timer()

# Emulation data set
set_emul_0 = pd.read_csv(data_path + 'emul_0.csv', sep=',')
set_emul_0_small_slug = pd.read_csv(data_path+'emul_0_small_slug.csv', sep=',')
set_emul_0_small_slug2 = pd.read_csv(data_path + 'emul_0_small_slug2.csv', sep=',')

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
set_emul_0['set'] = 'original'
del set_emul_0_small_slug, set_emul_0_small_slug2

set_overall_sim_1 = pd.concat([set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11], ignore_index=True)
set_overall_sim_2 = pd.concat([set1, set2, set3, set4, set5, set7, set8, set9, set10, set11], ignore_index=True) # simulation data set without scenario 6

del set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11

set_test = set_complex
del set_complex
print('Done')

print('Labeling...', end='')
# To remove the rows containing unit (First row contains unit info, and Model Time col value is NaN. By dropna method, this row can be dropped.)
source_sim_clean_1 = set_overall_sim_1.dropna()
source_sim_clean_2 = set_overall_sim_2.dropna()

sim_test = set_test.dropna()

original = set_emul_0.dropna()

del set_emul_0, set_overall_sim_1, set_overall_sim_2

# Labeling
df_sim_input, df_sim_label, df_sim_set = pseudo_labeling(source_sim_clean_1); df_sim = pd.concat([df_sim_input, df_sim_label], axis=1)
df_sim_wo6_input, df_sim_wo6_label, df_sim_wo6_set = pseudo_labeling(source_sim_clean_2); df_sim_wo6 = pd.concat([df_sim_wo6_input, df_sim_wo6_label], axis=1)

df_test_input, df_test_label, df_test_set = pseudo_labeling(sim_test); df_test = pd.concat([df_test_input, df_test_label], axis=1)

df_original_input, df_original_label, df_original_set = pseudo_labeling(original); df_original = pd.concat([df_original_input, df_original_label], axis=1)

terminate_time = timeit.default_timer()
print('Done')

# Training, test set definition
original = df_original
sim_wo6 = pd.concat([df_original, df_sim_wo6], ignore_index=True)
sim = pd.concat([df_original, df_sim], ignore_index=True)

x_test = df_test_input; y_test = df_test_label
rs_method_list = ['none','us', 'os', 'osus', 'usos', 'es', 'bw']

del df_original, df_sim_wo6, df_sim, df_test_input, df_test_label


# Fitting & recording

f_exp = open(path + os.sep + 'exp_result.csv','w')
f_exp.write('Loading and preprocessing completed in %f secs.\n' % (terminate_time - start_time))
f_exp.write(', Training, , , , , Test, , , , \n')
f_exp.write(', ROC_AUC, TN, FP, FN, TP, ROC_AUC, TN, FP, FN, TP\n')

for rs_method in rs_method_list:
    if rs_method == 'none':
        RFC('none', original)      # Baseline
        RFC('none', *[sim_wo6, sim])        # SIM-GEN
    else:
		# Resampling/weighted methods ONLY on original (no simulation mixes)
        RFC(rs_method, original)
        
f_exp.close()

# PCA projection plot
plot_pca_projection(original, sim, sim_wo6, df_test)