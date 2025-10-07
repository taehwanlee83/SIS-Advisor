## This is an automated report generation program to create file named "exp_result.csv".
## Strongly recommend to check and change 'MAX_ITER' value before running the program. (Recommended value: 1~10)
## Each iteration contains 8 model training/testings: 3 none (on origianl, original+sim w/o 6, original+sim w/6) + 5 resampling/weighted (on original only) experiments.

# MIT License
# 
# Copyright (c) 2025 [Tae Hwan Lee / NTNU]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

print('Initialization...', end='')
import os
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
from sklearn import metrics
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn import preprocessing

from imblearn import ensemble as es 
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

# Global variables
MAX_ITER = 100

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

scores = 'roc_auc'

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data') + os.sep

print('Done.')

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

def get_resampled_counts(best_est, X, y, rs_method):
	if rs_method in ('es', 'bw', 'none'):
		c = Counter(y)
		return len(y), len(y), c.get(1,0), c.get(0,0)

	if rs_method in ('smote', 'adasyn', 'us', 'osus'):
		sampler = best_est.named_steps.get('sampler', None)
		if sampler is None:
			c = Counter(y)
			return len(y), len(y), c.get(1,0), c.get(0,0)
		Xr, yr = sampler.fit_resample(X, y)
		c = Counter(yr)
		return len(y), len(yr), c.get(1,0), c.get(0,0)

	if rs_method == 'usos':
		oss   = best_est.named_steps['oss']
		smote = best_est.named_steps['smote']
		X1, y1 = oss.fit_resample(X, y)
		X2, y2 = smote.fit_resample(X1, y1)
		c = Counter(y2)
		return len(y), len(y2), c.get(1,0), c.get(0,0)

	# Fallback
	c = Counter(y)
	return len(y), len(y), c.get(1,0), c.get(0,0)

def build_pipeline(rs_method, RANDOM_SEED):
	rfc = RandomForestClassifier(random_state=RANDOM_SEED)

	if rs_method == 'smote':
		sampler = SMOTE(random_state=RANDOM_SEED)
		return Pipeline([('sampler', sampler), ('rfc', rfc)]), 'single'

	if rs_method == 'adasyn':
		sampler = ADASYN(random_state=RANDOM_SEED)
		return Pipeline([('sampler', sampler), ('rfc', rfc)]), 'single'

	if rs_method == 'us':
		sampler = OneSidedSelection(random_state=RANDOM_SEED)
		return Pipeline([('sampler', sampler), ('rfc', rfc)]), 'single'

	if rs_method == 'osus':
		sampler = SMOTEENN(
			smote=SMOTE(random_state=RANDOM_SEED),
			enn=EditedNearestNeighbours(),
			random_state=RANDOM_SEED
		)
		return Pipeline([('sampler', sampler), ('rfc', rfc)]), 'single'

	if rs_method == 'usos':
		oss = OneSidedSelection(random_state=RANDOM_SEED)
		smt = SMOTE(sampling_strategy=1.0, random_state=RANDOM_SEED)
		return Pipeline([('oss', oss), ('smote', smt), ('rfc', rfc)]), 'double'

	if rs_method == 'es':   # Balanced RF
		from imblearn.ensemble import BalancedRandomForestClassifier
		rfc = BalancedRandomForestClassifier(random_state=RANDOM_SEED)
		return Pipeline([('rfc', rfc)]), 'none'

	if rs_method == 'bw':   # class_weight RF
		rfc = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED)
		return Pipeline([('rfc', rfc)]), 'none'

	if rs_method == 'none':
		return Pipeline([('rfc', rfc)]), 'none'

	raise ValueError(f'Unknown rs_method: {rs_method}')

def build_param_grid_stage1(rs_method):
	if rs_method == 'smote':
		return [{
			'sampler__sampling_strategy': [0.6, 0.8, 1.0],
			'sampler__k_neighbors': [3, 5, 7],
		}]

	if rs_method == 'adasyn':
		return [{
			'sampler__sampling_strategy': [0.6, 0.8, 1.0],
			'sampler__n_neighbors': [3, 5, 7],
		}]
	
	if rs_method == 'us':
		return [{
			'sampler__n_neighbors': [1, 3, 5],
			'sampler__n_seeds_S': [1, 3, 5],
		}]

	if rs_method == 'osus':
		return [{
			'sampler__sampling_strategy': [0.6, 0.8, 1.0],
			'sampler__smote__k_neighbors': [3, 5, 7],
			'sampler__enn__n_neighbors': [3, 5, 7],
		}]

	if rs_method == 'usos':
		return [{
			'oss__n_neighbors': [1, 3, 5],
			'oss__n_seeds_S': [1, 3, 5],
			# 'smote__sampling_strategy': [0.6, 0.8, 1.0],     # disabled (auto) to ensure the stability of two-step resampling
			'smote__k_neighbors': [3, 5, 7],
		}]

	# es/bw/none
	return []

def build_param_grid_stage2():
	return [{
		'rfc__n_estimators': [3,5,10,15,20,50,100],
		'rfc__max_depth': [3,5,10,15],
		'rfc__max_leaf_nodes':[5,10,15,20]
	}]

def RFC(rs_method, *training):

	for i in range(1, MAX_ITER+1):
		RANDOM_SEED = i * 10
		trial_count = i
		f_exp.write('Trial %i (Random Seed: %i) (resampled by %s)' % (trial_count,RANDOM_SEED, rs_method))
		print('Trial %i %s' % (trial_count, rs_method))
		for data in training:
			start_time = timeit.default_timer()
			estimator, sampler_kind = build_pipeline(rs_method, RANDOM_SEED)
			cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
			x_train, y_train = data.iloc[:,:12], data.iloc[:,12]

			# Stage 1 grid search (resampling)
			if sampler_kind in ('single', 'double'):
				grid_stage1 = build_param_grid_stage1(rs_method)
				gs1 = GridSearchCV(estimator=estimator,
								   param_grid=grid_stage1,
								   scoring=scores, cv=cv,
								   verbose=1, n_jobs=4)
				gs1.fit(x_train, y_train)
				est_with_best_sampler = gs1.best_estimator_
			else:
				est_with_best_sampler = estimator

			# stage 2 grid search (RF)
			grid_stage2 = build_param_grid_stage2()
			gs2 = GridSearchCV(estimator=est_with_best_sampler,
							   param_grid=grid_stage2,
							   scoring=scores, cv=cv,
							   verbose=1, n_jobs=4)
			gs2.fit(x_train, y_train)
			best_rfc = gs2.best_estimator_
			best_rfc.fit(x_train, y_train)

			n_orig, n_res, n_min_res, n_maj_res = get_resampled_counts(best_rfc, x_train, y_train, rs_method)
			f_exp.write(', %d, %d, %d, %d' % (n_orig, n_res, n_min_res, n_maj_res))

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

def make_resampler(method: str):
	if method.upper() == 'NONE':
		return None
	if method.upper() == 'SMOTE':
		return SMOTE(random_state=42, sampling_strategy=1.0)
	if method.upper() == 'ADASYN':
		return ADASYN(random_state=42, sampling_strategy=1.0)
	if method.upper() == 'OSS':
		return OneSidedSelection(random_state=42)
	if method.upper() == 'SMOTEENN':
		return SMOTEENN(smote=SMOTE(random_state=42, sampling_strategy=1.0))
	if method.upper() == 'OSUS':
		return Pipeline([('smote', SMOTE(random_state=42, sampling_strategy=1.0)),
			('oss', OneSidedSelection(random_state=42))])
	if method.upper() == 'USOS':
		return Pipeline([('oss', OneSidedSelection(random_state=42)),
			('smote', SMOTE(random_state=42, sampling_strategy=1.0))])
	raise ValueError(f"Unknown resampling method: {method}")

def _xy(df: pd.DataFrame):
	return df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()

def pca_visualization(original_df, test_df, sim_wo6_df,	sim_w6_df):
	Xo, yo = _xy(original_df); Xt, yt = _xy(test_df)
	scaler = preprocessing.StandardScaler().fit(Xo)
	Xo_s = scaler.transform(Xo); Xt_s = scaler.transform(Xt)
	# Fix coordinate/directions
	pca = PCA(n_components=2, svd_solver='full', random_state=42).fit(Xo_s)

	comp = pca.components_.copy()
	if np.sum(comp[0]) < 0: comp[0] *= -1
	if np.sum(comp[1]) < 0: comp[1] *= -1
	pca.components_ = comp

	Zt = pca.transform(Xt_s)

	panels = [
		("Original + Test",                Xo, yo, "NONE"),
		("US(OSS) + Test",                 Xo, yo, "OSS"),
		("SMOTE + Test",                   Xo, yo, "SMOTE"),
		("ADASYN + Test",                  Xo, yo, "ADASYN"),
		("OSUS(SMOTE→ENN) + Test",         Xo, yo, "SMOTEENN"),
		("USOS(OSS→SMOTE) + Test",         Xo, yo, "USOS"),
	]

	Xs0, ys0 = _xy(sim_wo6_df); panels.append(("Original + Sim w/o6 + Test", np.vstack([Xo, Xs0]), np.hstack([yo, ys0]), "NONE"))

	Xs1, ys1 = _xy(sim_w6_df); panels.append(("Original + Simulation (w6) + Test", np.vstack([Xo, Xs1]), np.hstack([yo, ys1]), "NONE"))

	Zr_list, yrs_list, titles = [], [], []
	for title, Xtr, ytr, method in panels:
		Xtr_s = scaler.transform(Xtr)
		sampler = make_resampler(method)
		if sampler is None: Xrs, yrs = Xtr_s, ytr
		else:               Xrs, yrs = sampler.fit_resample(Xtr_s, ytr)
		Zr = pca.transform(Xrs)
		Zr_list.append(Zr); yrs_list.append(yrs); titles.append(title)

	all_z = np.vstack([Zt] + Zr_list)
	xmin, ymin = all_z.min(axis=0)
	xmax, ymax = all_z.max(axis=0)
	dx, dy = xmax - xmin, ymax - ymin
	margin = 0.05
	xlim = (xmin - dx*margin, xmax + dx*margin)
	ylim = (ymin - dy*margin, ymax + dy*margin)

	fig, axes = plt.subplots(2, 4, figsize=(12, 24), sharex=True, sharey=True)
	axes = axes.ravel()
	plt.tight_layout(pad=0.5)

	for i, ax in enumerate(axes):
		ax.set_box_aspect(1)
		Zr, yrs, title = Zr_list[i], yrs_list[i], titles[i]
		for cls in np.unique(yrs):
			ax.scatter(Zr[yrs == cls, 0], Zr[yrs == cls, 1], s=10, alpha=0.6, marker='o', label=f"train y={cls}")
		for cls in np.unique(yt):
			ax.scatter(Zt[yt == cls, 0],  Zt[yt == cls, 1],  s=12, alpha=0.9, marker='x', label=f"test y={cls}")
		ax.set_title(title, fontsize=14)
		ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
		ax.set_xlim(*xlim); ax.set_ylim(*ylim)

		h, l = ax.get_legend_handles_labels()
		uniq = dict(zip(l, h))
		ax.legend(uniq.values(), uniq.keys(), fontsize=12, loc='best')

	return fig


def main():
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
	set6['set'] = 'sluggish flow'
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
	original_df = set_emul_0.dropna()

	del set_emul_0, set_overall_sim_1, set_overall_sim_2

	# Labeling
	df_sim_input, df_sim_label, df_sim_set = pseudo_labeling(source_sim_clean_1); df_sim = pd.concat([df_sim_input, df_sim_label], axis=1)
	df_sim_wo6_input, df_sim_wo6_label, df_sim_wo6_set = pseudo_labeling(source_sim_clean_2); df_sim_wo6 = pd.concat([df_sim_wo6_input, df_sim_wo6_label], axis=1)

	df_test_input, df_test_label, df_test_set = pseudo_labeling(sim_test); df_test = pd.concat([df_test_input, df_test_label], axis=1)

	df_original_input, df_original_label, df_original_set = pseudo_labeling(original_df); df_original = pd.concat([df_original_input, df_original_label], axis=1)

	terminate_time = timeit.default_timer()
	print('Done')

	# Training, test set definition
	original = df_original
	sim_wo6 = pd.concat([df_original, df_sim_wo6], ignore_index=True)
	sim = pd.concat([df_original, df_sim], ignore_index=True)
	test = df_test

	global x_test, y_test
	x_test = df_test_input; y_test = df_test_label

	rs_method_list = ['none','us', 'smote', 'adasyn', 'osus', 'usos', 'es', 'bw']
	# rs_method_list = ['smote', 'adasyn']

	del df_original, df_sim_wo6, df_sim, df_test_input, df_test_label, df_test

	# Fitting & recording
	global f_exp
	f_exp = open(path + os.sep + 'exp_result.csv','w')
	f_exp.write('Loading and labeling completed in %f secs.\n' % (terminate_time - start_time))
	f_exp.write(', Samples, , , , Training, , , , , Test, , , , \n')
	f_exp.write(', n_orig, n_res, n_min_res, n_maj_res, ROC_AUC, TN, FP, FN, TP, ROC_AUC, TN, FP, FN, TP\n')

	for rs_method in rs_method_list:
		if rs_method == 'none':
			RFC('none', original)                 # Baseline
			RFC('none', *[sim_wo6, sim])          # SIM-GEN
		else:
			RFC(rs_method, original)
		
	f_exp.close()

	# PCA projection plot
	fig = pca_visualization(original, test, sim_wo6, sim)
	plt.show()

if __name__ == "__main__":
	main()