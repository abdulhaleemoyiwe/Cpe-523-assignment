# ============================================================
# P8 ENERGY EFFICIENCY - COMPLETE EXPERIMENT
# 4 MODELS | 2 ENCODINGS | 5-FOLD x 3 REPEATS
# ============================================================

!pip -q install openpyxl statsmodels

import os, time, zipfile, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

FILE = "/content/ENB2012_data.xlsx"
OUT = "/content/P8_results"

os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)
os.makedirs(f"{OUT}/tables", exist_ok=True)

FEATURES = ["X1","X2","X3","X4","X5","X6","X7","X8"]
CATS = ["X6","X8"]
NUMS = ["X1","X2","X3","X4","X5","X7"]

TARGETS = {
    "Heating Load":"Y1",
    "Cooling Load":"Y2"
}

MODELS = {
    "OLS": (
        LinearRegression(),
        {}
    ),

    "Ridge": (
        Ridge(),
        {"model__alpha":[0.01,0.1,1,10,100]}
    ),

    "KNN": (
        KNeighborsRegressor(),
        {
            "model__n_neighbors":[3,5,7,11,15],
            "model__weights":["uniform","distance"],
            "model__p":[1,2]
        }
    ),

    "Random Forest": (
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators":[100,200],
            "model__max_depth":[None,15,30],
            "model__min_samples_leaf":[1,2],
            "model__max_features":["sqrt"]
        }
    )
}

ALGORITHMS = list(MODELS.keys())
SEEDS = [42,123,2026]

# ------------------------------------------------------------
# 2. LOAD + CLEAN DATA
# ------------------------------------------------------------

xls = pd.ExcelFile(FILE)

sheet = None
for s in xls.sheet_names:
    temp = pd.read_excel(FILE, sheet_name=s)
    if len(temp) > 0:
        sheet = s
        data = temp.copy()
        break

print("="*70)
print("DATASET AUDIT")
print("="*70)
print("Sheet:", sheet)
print("Original shape:", data.shape)
print("Missing values:", data.isna().sum().sum())
print("Duplicate rows:", data.duplicated().sum())

data = data.dropna(how="all").drop_duplicates().reset_index(drop=True)

print("Cleaned shape:", data.shape)

data.describe().T.to_csv(
    f"{OUT}/tables/dataset_descriptive_statistics.csv"
)

data.isna().sum().to_csv(
    f"{OUT}/tables/missing_values.csv"
)

data.nunique().to_csv(
    f"{OUT}/tables/unique_values.csv"
)

data.corr(numeric_only=True).to_csv(
    f"{OUT}/tables/correlation_matrix.csv"
)

# ------------------------------------------------------------
# 3. DATASET CORRELATION FIGURE
# ------------------------------------------------------------

corr = data.corr(numeric_only=True)

plt.figure(figsize=(9,7))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)),corr.columns,rotation=45)
plt.yticks(range(len(corr.columns)),corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j,i,f"{corr.iloc[i,j]:.2f}",
                 ha="center",va="center")

plt.title("Energy Efficiency Correlation Matrix")
plt.tight_layout()
plt.savefig(
    f"{OUT}/figures/correlation_matrix.png",
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 4. REPEATED CV SPLITS
# ------------------------------------------------------------

splits = []

for repeat, seed in enumerate(SEEDS,1):

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for fold,(tr,te) in enumerate(cv.split(data),1):

        splits.append(
            (repeat,fold,tr,te)
        )

print("\nOuter CV folds:",len(splits))

# ------------------------------------------------------------
# 5. STORAGE
# ------------------------------------------------------------

results = []
oof = []
times = []
params = []

# ------------------------------------------------------------
# 6. PIPELINE FUNCTION
# ------------------------------------------------------------

def make_pipeline(model_name, encoding):

    model = clone(MODELS[model_name][0])

    if encoding == "Integer-coded":

        prep = ColumnTransformer([
            ("num",
             StandardScaler(),
             FEATURES)
        ])

    else:

        prep = ColumnTransformer([
            ("num",
             StandardScaler(),
             NUMS),

            ("cat",
             OneHotEncoder(
                 handle_unknown="ignore",
                 sparse_output=False
             ),
             CATS)
        ])

    return Pipeline([
        ("prep",prep),
        ("model",model)
    ])

# ------------------------------------------------------------
# 7. MAIN EXPERIMENT
# ------------------------------------------------------------

for encoding in ["Integer-coded","One-hot"]:

    for target_name,target_col in TARGETS.items():

        y = data[target_col].values

        for model_name in ALGORITHMS:

            print(
                f"\nRunning: {encoding} | "
                f"{target_name} | {model_name}"
            )

            for repeat,fold,tr,te in splits:

                Xtr = data.iloc[tr][FEATURES]
                Xte = data.iloc[te][FEATURES]

                ytr = y[tr]
                yte = y[te]

                pipe = make_pipeline(
                    model_name,
                    encoding
                )

                grid = MODELS[model_name][1]

                start = time.perf_counter()

                if grid:

                    inner = KFold(
                        n_splits=3,
                        shuffle=True,
                        random_state=1000+repeat+fold
                    )

                    search = GridSearchCV(
                        pipe,
                        grid,
                        scoring="neg_root_mean_squared_error",
                        cv=inner,
                        n_jobs=-1,
                        refit=True
                    )

                    search.fit(Xtr,ytr)

                    fitted = search.best_estimator_
                    best = search.best_params_

                else:

                    pipe.fit(Xtr,ytr)
                    fitted = pipe
                    best = {}

                train_time = (
                    time.perf_counter()-start
                )

                start = time.perf_counter()

                pred = fitted.predict(Xte)

                pred_time = (
                    time.perf_counter()-start
                )

                rmse = np.sqrt(
                    mean_squared_error(yte,pred)
                )

                mae = mean_absolute_error(
                    yte,pred
                )

                r2 = r2_score(
                    yte,pred
                )

                results.append([
                    encoding,model_name,target_name,
                    repeat,fold,rmse,mae,r2
                ])

                times.append([
                    encoding,model_name,target_name,
                    repeat,fold,train_time,
                    pred_time/len(te)*1000
                ])

                params.append([
                    encoding,model_name,target_name,
                    repeat,fold,str(best)
                ])

                for actual,p in zip(yte,pred):

                    oof.append([
                        encoding,model_name,target_name,
                        repeat,fold,actual,p,actual-p
                    ])

# ------------------------------------------------------------
# 8. DATAFRAMES
# ------------------------------------------------------------

cols = [
    "Encoding","Algorithm","Target",
    "Repeat","Fold","RMSE","MAE","R2"
]

results = pd.DataFrame(
    results,columns=cols
)

times = pd.DataFrame(
    times,
    columns=[
        "Encoding","Algorithm","Target",
        "Repeat","Fold",
        "Training_Time_s",
        "Prediction_ms_per_sample"
    ]
)

params = pd.DataFrame(
    params,
    columns=[
        "Encoding","Algorithm","Target",
        "Repeat","Fold","Best_Parameters"
    ]
)

oof = pd.DataFrame(
    oof,
    columns=[
        "Encoding","Algorithm","Target",
        "Repeat","Fold",
        "Actual","Predicted","Residual"
    ]
)

# ------------------------------------------------------------
# 9. MEAN PREDICTOR BASELINE
# ------------------------------------------------------------

baseline = []

for encoding in ["Integer-coded","One-hot"]:

    for target_name,target_col in TARGETS.items():

        y = data[target_col].values

        for repeat,fold,tr,te in splits:

            pred = np.repeat(
                y[tr].mean(),
                len(te)
            )

            baseline.append([
                encoding,
                "Mean Predictor",
                target_name,
                repeat,
                fold,
                np.sqrt(
                    mean_squared_error(y[te],pred)
                ),
                mean_absolute_error(y[te],pred),
                r2_score(y[te],pred)
            ])

baseline = pd.DataFrame(
    baseline,
    columns=cols
)

# ------------------------------------------------------------
# 10. MAIN SUMMARY: MEAN +/- SD
# ------------------------------------------------------------

summary = pd.concat(
    [baseline,results]
).groupby(
    ["Encoding","Algorithm","Target"]
)[["RMSE","MAE","R2"]].agg(
    ["mean","std"]
)

summary.to_csv(
    f"{OUT}/tables/main_results_mean_sd.csv"
)

print("\n")
print("="*70)
print("MAIN RESULTS: MEAN ± SD")
print("="*70)

for idx,row in summary.iterrows():

    print(
        idx,
        "| RMSE:",
        f"{row[('RMSE','mean')]:.3f} ± {row[('RMSE','std')]:.3f}",
        "| MAE:",
        f"{row[('MAE','mean')]:.3f} ± {row[('MAE','std')]:.3f}",
        "| R²:",
        f"{row[('R2','mean')]:.3f} ± {row[('R2','std')]:.3f}"
    )

# ------------------------------------------------------------
# 11. FRIEDMAN + MEAN RANKS
# ------------------------------------------------------------

friedman = []
ranks_out = []

for encoding in ["Integer-coded","One-hot"]:

    for target in TARGETS:

        sub = results[
            (results.Encoding==encoding) &
            (results.Target==target)
        ]

        for metric in ["RMSE","MAE","R2"]:

            pvt = sub.pivot_table(
                index=["Repeat","Fold"],
                columns="Algorithm",
                values=metric
            )[ALGORITHMS].dropna()

            stat,p = friedmanchisquare(
                *[
                    pvt[a].values
                    for a in ALGORITHMS
                ]
            )

            friedman.append([
                encoding,target,metric,
                len(pvt),stat,p
            ])

            ranks = pvt.rank(
                axis=1,
                ascending=(metric!="R2")
            )

            for a in ALGORITHMS:

                ranks_out.append([
                    encoding,target,metric,
                    a,ranks[a].mean()
                ])

friedman = pd.DataFrame(
    friedman,
    columns=[
        "Encoding","Target","Metric",
        "N","Friedman_statistic","p_value"
    ]
)

ranks_out = pd.DataFrame(
    ranks_out,
    columns=[
        "Encoding","Target","Metric",
        "Algorithm","Mean_Rank"
    ]
)

friedman.to_csv(
    f"{OUT}/tables/friedman_results.csv",
    index=False
)

ranks_out.to_csv(
    f"{OUT}/tables/friedman_mean_ranks.csv",
    index=False
)

print("\n")
print("="*70)
print("FRIEDMAN TEST")
print("="*70)
print(friedman.to_string(index=False))

# ------------------------------------------------------------
# 12. WILCOXON + HOLM + EFFECT SIZE
# Planned comparisons against OLS
# ------------------------------------------------------------

wilcox = []

for encoding in ["Integer-coded","One-hot"]:

    for target in TARGETS:

        sub = results[
            (results.Encoding==encoding) &
            (results.Target==target)
        ]

        for metric in ["RMSE","MAE","R2"]:

            pvt = sub.pivot_table(
                index=["Repeat","Fold"],
                columns="Algorithm",
                values=metric
            )[ALGORITHMS].dropna()

            raw = []
            temp = []

            for alg in [
                "Ridge",
                "KNN",
                "Random Forest"
            ]:

                x = pvt[alg].values
                y = pvt["OLS"].values

                diff = x-y
                nz = diff[diff!=0]
                n = len(nz)

                if n == 0:
                    W,p,r = np.nan,np.nan,np.nan

                else:

                    test = wilcoxon(
                        x,y,
                        alternative="two-sided"
                    )

                    W,p = test.statistic,test.pvalue

                    z = (
                        (W-n*(n+1)/4)
                        /
                        np.sqrt(
                            n*(n+1)*(2*n+1)/24
                        )
                    )

                    r = abs(z)/np.sqrt(n)

                raw.append(p)

                temp.append([
                    encoding,target,metric,
                    f"{alg} vs OLS",
                    W,p,n,r
                ])

            reject,adjusted,_,_ = multipletests(
                raw,
                alpha=0.05,
                method="holm"
            )

            for i,row in enumerate(temp):

                wilcox.append(
                    row+[adjusted[i],reject[i]]
                )

wilcox = pd.DataFrame(
    wilcox,
    columns=[
        "Encoding","Target","Metric",
        "Comparison","Wilcoxon_W",
        "Raw_p","N","Effect_size_r",
        "Holm_adjusted_p",
        "Significant_after_Holm"
    ]
)

wilcox.to_csv(
    f"{OUT}/tables/wilcoxon_holm_effect_sizes.csv",
    index=False
)

print("\n")
print("="*70)
print("WILCOXON + HOLM + EFFECT SIZE")
print("="*70)
print(wilcox.to_string(index=False))

# ------------------------------------------------------------
# 13. COMPUTATIONAL COST
# ------------------------------------------------------------

cost = times.groupby(
    ["Encoding","Algorithm","Target"]
)[
    ["Training_Time_s",
     "Prediction_ms_per_sample"]
].agg(["mean","std"])

cost.to_csv(
    f"{OUT}/tables/computational_cost.csv"
)

print("\n")
print("="*70)
print("COMPUTATIONAL COST")
print("="*70)
print(cost.round(4).to_string())

# ------------------------------------------------------------
# 14. ENCODING COMPARISON
# ------------------------------------------------------------

encoding_comparison = results.groupby(
    ["Encoding","Algorithm","Target"]
)[["RMSE","MAE","R2"]].mean().reset_index()

encoding_comparison.to_csv(
    f"{OUT}/tables/encoding_comparison.csv",
    index=False
)

print("\n")
print("="*70)
print("ENCODING COMPARISON")
print("="*70)
print(encoding_comparison.round(4).to_string(index=False))

# ------------------------------------------------------------
# 15. RESIDUAL PLOTS
# ------------------------------------------------------------

for target in TARGETS:

    for alg in ALGORITHMS:

        d = oof[
            (oof.Encoding=="One-hot") &
            (oof.Target==target) &
            (oof.Algorithm==alg)
        ]

        plt.figure(figsize=(6,5))
        plt.scatter(
            d.Predicted,
            d.Residual,
            alpha=.4
        )
        plt.axhline(0,linestyle="--")
        plt.xlabel("Predicted Load (kWh/m²)")
        plt.ylabel("Residual")
        plt.title(f"{alg} - {target}")
        plt.tight_layout()

        plt.savefig(
            f"{OUT}/figures/{alg.replace(' ','_')}_{target.replace(' ','_')}_residuals.png",
            dpi=250
        )

        plt.close()

# ------------------------------------------------------------
# 16. SAVE RAW RESULTS
# ------------------------------------------------------------

results.to_csv(
    f"{OUT}/raw_fold_results.csv",
    index=False
)

baseline.to_csv(
    f"{OUT}/baseline_results.csv",
    index=False
)

oof.to_csv(
    f"{OUT}/out_of_fold_predictions.csv",
    index=False
)

times.to_csv(
    f"{OUT}/fold_computational_times.csv",
    index=False
)

params.to_csv(
    f"{OUT}/best_parameters.csv",
    index=False
)

# ------------------------------------------------------------
# 17. ZIP EVERYTHING
# ------------------------------------------------------------

zip_path = "/content/P8_energy_efficiency_results.zip"

with zipfile.ZipFile(
    zip_path,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for root,dirs,files_ in os.walk(OUT):

        for f in files_:

            path = os.path.join(root,f)

            z.write(
                path,
                os.path.relpath(
                    path,
                    OUT
                )
            )

print("\n")
print("="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print("Results folder:",OUT)
print("ZIP file:",zip_path)

# ------------------------------------------------------------
# 18. DOWNLOAD
# ------------------------------------------------------------

from google.colab import files
files.download(zip_path)
