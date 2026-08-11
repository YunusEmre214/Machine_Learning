import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler

df = pd.read_csv('musteri_verisi_ml_pratik.csv')

print(df.head())
print("------------------------------")
print(df.info())

print(df.isnull().sum())

df_dropna=df.dropna()
print(f"Eksik veriler çıktıktan sonra :\n{df_dropna}")

df_field=df.copy()

sayisal_sutunlar=['yas','maas','deneyim_yili']

for sutun in sayisal_sutunlar:
    medyan_degeri=df_field[sutun].median()
    df_field[sutun]=df_field[sutun].fillna(medyan_degeri)

df_field["egitim"]=df_field["egitim"].fillna(df_field["egitim"].mode()[0])

print(f"Eksik veriler doldurulduktan sonra: \n{df_field}")


aykiri_deger_maskesi = pd.Series(False, index = df_field.index)

for sutun in sayisal_sutunlar:

    q1 = df_field[sutun].quantile(0.25)
    q3 = df_field[sutun].quantile(0.75)

    iqr = q3 - q1

    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr

    sutun_maskesi = (
        (df_field[sutun] < alt_sinir) | (df_field[sutun] > ust_sinir)
    )

    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi

    print(f"Aykırı değer sayısı: {sutun_maskesi.sum()}")

    if sutun_maskesi.any():
        print(f"Aykırı değerler: \n{df_field.loc[sutun_maskesi, sutun]}")

print(f"En az bir aykırı değer içeren satırlar \n{df_field.loc[aykiri_deger_maskesi]}")

df_clean=df_field.loc[~aykiri_deger_maskesi].copy()
df_clean.reset_index(drop=True, inplace=True)
print(f"Aykırı değerler temizlendikten sonra: \n{df_clean}")

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(df_clean["satin_aldi"])

print(f"Hedef değişkenin sınıf etiketleri: {label_encoder.classes_}")
print(f"Hedef değişkenin değerleri: {y}")

X=df_clean.drop(columns=["satin_aldi"])
X=pd.get_dummies(X,columns=["egitim"],drop_first=True,dtype=int)

print(f"Özellikler: \n{X}")

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # val = %80, test = %20

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.4, random_state=42, stratify=y_train_val)

print(f"X_train boyutu: {X_train.shape}")
print(f"X_val boyutu: {X_val.shape}")
print(f"X_test boyutu: {X_test.shape}")

standard_scaler = StandardScaler()

X_train_standard = X_train.copy()
X_val_standard = X_val.copy()
X_test_standard = X_test.copy()

X_train_standard[sayisal_sutunlar] = standard_scaler.fit_transform(X_train[sayisal_sutunlar])
X_val_standard[sayisal_sutunlar] = standard_scaler.transform(X_val[sayisal_sutunlar])
X_test_standard[sayisal_sutunlar] = standard_scaler.transform(X_test[sayisal_sutunlar])

print(f"X_train_standard: \n{X_train_standard}")

minmax_scaler = MinMaxScaler()

X_train_normalized = X_train.copy()
X_val_normalized = X_val.copy()
X_test_normalized = X_test.copy()

X_train_normalized[sayisal_sutunlar] = minmax_scaler.fit_transform(X_train[sayisal_sutunlar])
X_val_normalized[sayisal_sutunlar] = minmax_scaler.transform(X_val[sayisal_sutunlar])
X_test_normalized[sayisal_sutunlar] = minmax_scaler.transform(X_test[sayisal_sutunlar]) 

print(f"X_train_normalized: \n{X_train_normalized}")