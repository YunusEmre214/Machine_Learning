import pandas as pd

df=pd.read_csv("oznitelik_muhendisligi_pratik.csv")
pd.set_option('display.max_columns', None)
print(df.head())

df["deneyim_orani"]=df["deneyim_yili"]/df["yas"]
df["yillik_harcama_tahmini"]=df["aylik_harcama"]*12

print(df)

numeric_df=df.drop("sehir",axis=1)
korelasyon=numeric_df.corr(numeric_only=True)["performans_puani"].sort_values(ascending=False)
print(korelasyon)

secilen_ozniteliler = korelasyon[abs(korelasyon) > 0.75].index.tolist()
secilen_ozniteliler.remove("performans_puani")

print(secilen_ozniteliler)