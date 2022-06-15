import pandas as pd
import numpy as np
import xgboost as xgb

np.random.seed(2018)

## データの前処理

trn = pd.read_csv("../dataset/train_ver2.csv")
tst = pd.read_csv("../dataset/test_ver2.csv")

prods = trn.columns[24:].tolist()
trn[prods] = trn[prods].fillna(0.0).astype(np.int8)

no_product = trn[prods].sum(axis=1) == 0
trn = trn[~no_product]

for col in trn.columns[24:]:
    tst[col] = 0
df = pd.concat([trn, tst], axis=0)

features = []

categorical_cols = ["ind_empleado", "pais_residencia", "sexo", "tiprel_1mes", "indresi", "indext", "conyuemp", "canal_entrada", "indfall", "tipodom", "nomprov", "segmento"]
for col in categorical_cols:
    df[col], _ = df[col].factorize(na_sentinel=-99)
features += categorical_cols

df["age"] = df["age"].str.replace(" ", "")
df["age"] = df["age"].str.replace("NA", -99)
df["age"] = df["age"].astype(np.int8)

df["antiguedad"] = df["antiguedad"].str.replace(" ", "")
df["antiguedad"] = df["antiguedad"].str.replace("NA", -99)
df["antiguedad"] = df["antiguedad"].astype(np.int8)

df["renta"] = df["renta"].str.replace(" ", "")
df["renta"] = df["renta"].str.replace("NA", -99)
df["renta"] = df["renta"].fillna(-99, inplace=True)
df["renta"] = df["renta"].astype(float).astype(np.int8)

df["indrel_1ms"] = df["indrel_1ms"].str.replace("P", 5)
df["indrel_1ms"] = df["indrel_1ms"].fillna(-99, inplace=True)
df["indrel_1ms"] = df["indrel_1ms"].astype(float).astype(np.int8)
features += ["age", "antiguedad", "renta", "ind_nuevo", "indrel", "indrel_1mes", "ind_actividad_cliente"]

df["fecha_alta_month"] = df["fecha_alta"].map(lambda x: 0.0 if x.__class__ is float else float(x.split("-")[1])).astype(np.int8)
df["fecha_alta_year"] = df["fecha_alta"].map(lambda x: 0.0 if x.__class__ is float else float(x.split("-")[0])).astype(np.int16)
features += ["fecha_alta_month", "fecha_alta_year"]

df["ult_fec_cli_1t_month"] = df["ult_fec_cli_1t"].map(lambda x: 0.0 if x.__class__ is float else float(x.split("-")[1])).astype(np.int8)
df["ult_fec_cli_1t_year"] = df["ult_fec_cli_1t"].map(lambda x: 0.0 if x.__class__ is float else float(x.split("-")[0])).astype(np.int16)
features += ["ult_fec_cli_1t_month", "ult_fec_cli_1t_year"]

df.fillna(-99, inplace=True)

def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")]
    int_date = (int(Y) - 2015) * 12 * int(M)
    return int_date

trn["int_date"] = trn["fecha_dato"].map(date_to_int).astype(np.int8)

df_lag = df.copy()
df_lag.columns = [col + "_prev" if col not in ["ncodpers", "int_date"] else col for col in df.columns]
df_lag["int_date"] += 1

df_trn = df.merge(df_lag, on=["ncodepers", "int_date"], how="left")

del df, df_lag

for prod in prods:
    prev = prod + "_prev"
    df_trn[prev].fillna(0, inplace=True)
df_trn.fillna(-99, inplace=True)

features += [feature + "_prev" for feature in features]
features += [prod + "_prev" for prod in prods]

## モデル学習

use_dates = ["2016-01-28", "2016-02-28", "2016-03-28", "2016-04-28"]
trn = df_trn[df_trn["fecha_dato"].isin(use_dates)]
tst = df_trn[df_trn["fecha_dato"] == "2016-05-28"]

del df_trn

X = []
Y = []
for i, prod in enumarate(prods):
    prev = prod + "_prev"
    prX = trn[(trn[prod] == 1) & (trn[prev] == 0)]
    prY = np.zeros(prX.shape[0], dtype=np.int8) + i
    X.append(prX)
    Y.append(prY)
XY = pd.concat(X)
Y = np.hstack(Y)
XY["y"] = Y

vld_date = XY[XY["fecha_dato"] != vld_date]
vld_vld = XY[XY["fecha_dato"] == vld_date]

param = {
    "booster": "gbtree",
    "max_depth": 8,
    "nthread": 4,
    "num_class": len(prods),
    "objective": "multi:softprob",
    "silent": 1,
    "eval_metric": "mlogloss",
    "eta": 0.1,
    "min_child_weight": 10,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.9,
    "seed": 2018,
}

X_trn = XY_trn.as_matrix(columns=features)
Y_trn = XY_trn.as_matrix(columns=["y"])
dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)

X_vld = XY_vld.as_matrix(columns=features)
Y_vld = XY_vld.as_matrix(columns=["y"])
dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)

watch_list = [(dtrn, "train"), (dvld, "eval")]
model = xgb.train(param, dtrn, num_boost_round=1000, evals=watch_list, early_stopping_rounds=20)

import pickle
pickle.dump(model, open("model/xgb.baseline.pkl", "wb"))
best_ntree_limit = model.best_ntree_limit
