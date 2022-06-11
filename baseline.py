import pandas as pd
import numpy as np
import xgboost as xgb

np.random.seed(2018)

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

df["age"].replace(" NA", -99, inplace=True)
df["age"] = df["age"].astype(np.int8)

df["antiguedad"].replace(" NA", -99, inplace=True)
df["antiguedad"].df["antiguedad"].astype(np.int8)

df["renta"].replace(" NA", -99, inplace=True)
df["renta"].fillna(-99, inplace=True)
df["renta"] = df["renta"].astype(float).astype(np.int8)

df["indrel_1ms"].replace("P", 5, inplace=True)
df["indrel_1ms"].fillna(-99, inplace=True)
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