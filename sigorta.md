```python
import pandas as pd
from scipy.stats import zscore
```


```python
df = pd.read_csv("C:\\Users\\Umut\\Desktop\\insurance.csv")
```


```python
age_bins = [18, 30, 40, 50, 60, 100]
age_labels =['18-29', '30-39', '40-49', '50-59', '60+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)
```


```python
age_group_premiums = df.groupby('age_group')['charges'].mean().reset_index()
```


```python
print(age_group_premiums)
```

      age_group       charges
    0     18-29   9822.837599
    1     30-39  11639.308653
    2     40-49  14782.043077
    3     50-59  17062.292763
    4       60+  21063.163398
    


```python
smoker_premiums = df.groupby('smoker')['charges'].mean().reset_index()
```


```python
print(smoker_premiums)
```

      smoker       charges
    0     no   8434.268298
    1    yes  32050.231832
    


```python
# Z-score hesapla
df['zscore'] = zscore(df['charges']) 
```


```python
# Belirli bir Z-score eşiği üzerindeki değerleri seç
outliers_zscore = df[df['zscore'].abs() > 2]
```


```python
# IQR hesapla
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1
```


```python
# Belirli bir IQR eşiği üzerindeki değerleri seç
outliers_iqr = df[(df['charges'] < Q1 - 1.5 * IQR) | (df['charges'] > Q3 + 1.5 * IQR)]
```


```python
# Aykırı değerleri görüntüle
print("Aykırı Değerler (Z-score):")
print(outliers_zscore)
```

    Aykırı Değerler (Z-score):
          age     sex     bmi  children smoker     region      charges age_group  \
    14     27    male  42.130         0    yes  southeast  39611.75770     18-29   
    23     34  female  31.920         1    yes  northeast  37701.87680     30-39   
    29     31    male  36.300         2    yes  southwest  38711.00000     30-39   
    34     28    male  36.400         1    yes  southwest  51194.55914     18-29   
    38     35    male  36.670         1    yes  northeast  39774.27630     30-39   
    ...   ...     ...     ...       ...    ...        ...          ...       ...   
    1288   20    male  39.400         2    yes  southwest  38344.56600     18-29   
    1300   45    male  30.360         0    yes  southeast  62592.87309     40-49   
    1301   62    male  30.875         3    yes  northwest  46718.16325       60+   
    1303   43    male  27.800         0    yes  southwest  37829.72420     40-49   
    1323   42  female  40.370         2    yes  southeast  43896.37630     40-49   
    
            zscore  
    14    2.175983  
    23    2.018214  
    29    2.101574  
    34    3.132806  
    38    2.189409  
    ...        ...  
    1288  2.071304  
    1300  4.074389  
    1301  2.763024  
    1303  2.028775  
    1323  2.529924  
    
    [108 rows x 9 columns]
    


```python
pd.set_option('display.max_rows', None)  # Tüm satırları göster
print(outliers_zscore)

```

          age     sex     bmi  children smoker     region      charges age_group  \
    14     27    male  42.130         0    yes  southeast  39611.75770     18-29   
    23     34  female  31.920         1    yes  northeast  37701.87680     30-39   
    29     31    male  36.300         2    yes  southwest  38711.00000     30-39   
    34     28    male  36.400         1    yes  southwest  51194.55914     18-29   
    38     35    male  36.670         1    yes  northeast  39774.27630     30-39   
    39     60    male  39.900         0    yes  southwest  48173.36100     50-59   
    49     36    male  35.200         1    yes  southeast  38709.17600     30-39   
    53     36    male  34.430         0    yes  southeast  37742.57570     30-39   
    55     58    male  36.955         2    yes  northwest  47496.49445     50-59   
    84     37  female  34.800         2    yes  southwest  39836.51900     30-39   
    86     57  female  31.160         0    yes  northwest  43578.93940     50-59   
    94     64  female  31.300         2    yes  southwest  47291.05500       60+   
    109    63    male  35.090         0    yes  southeast  47055.53210       60+   
    123    44    male  31.350         1    yes  northeast  39556.49450     40-49   
    146    46    male  30.495         3    yes  northwest  40720.55105     40-49   
    175    63  female  37.700         0    yes  southwest  48824.45000       60+   
    185    36    male  41.895         3    yes  northeast  43753.33705     30-39   
    240    23  female  36.670         2    yes  northeast  38511.62830     18-29   
    251    63  female  32.200         2    yes  southwest  47305.30500       60+   
    252    54    male  34.210         2    yes  southeast  44260.74990     50-59   
    254    50    male  31.825         0    yes  northeast  41097.16175     40-49   
    256    56    male  33.630         0    yes  northwest  43921.18370     50-59   
    265    46    male  42.350         3    yes  southeast  46151.12450     40-49   
    271    50    male  34.200         2    yes  southwest  42856.83800     40-49   
    281    54    male  40.565         3    yes  northeast  48549.17835     50-59   
    288    59  female  36.765         1    yes  northeast  47896.79135     50-59   
    292    25    male  45.540         2    yes  southeast  42112.23560     18-29   
    298    31    male  34.390         3    yes  northwest  38746.35510     30-39   
    312    43    male  35.970         3    yes  southeast  42124.51530     40-49   
    327    45    male  36.480         2    yes  northwest  42760.50220     40-49   
    328    64  female  33.800         1    yes  southwest  47928.03000       60+   
    330    61  female  36.385         1    yes  northeast  48517.56315       60+   
    338    50    male  32.300         1    yes  northeast  41919.09700     40-49   
    377    24    male  40.150         0    yes  southeast  38126.24650     18-29   
    381    55    male  30.685         0    yes  northeast  42303.69215     50-59   
    420    64    male  33.880         0    yes  southeast  46889.26120       60+   
    421    61    male  35.860         0    yes  southeast  46599.10840       60+   
    422    40    male  32.775         1    yes  northeast  39125.33225     30-39   
    488    44  female  38.060         0    yes  southeast  48885.13561     40-49   
    524    42    male  26.070         1    yes  southeast  38245.59327     40-49   
    530    57    male  42.130         1    yes  southeast  48675.51770     50-59   
    543    54  female  47.410         0    yes  southeast  63770.42801     50-59   
    549    43  female  46.200         0    yes  southeast  45863.20500     40-49   
    558    35  female  34.105         3    yes  northwest  39983.42595     30-39   
    569    48    male  40.565         2    yes  northwest  45702.02235     40-49   
    577    31  female  38.095         1    yes  northeast  58571.07448     30-39   
    587    34  female  30.210         1    yes  northwest  43943.87610     30-39   
    609    30    male  37.800         2    yes  southwest  39241.44200     18-29   
    615    47  female  36.630         1    yes  southeast  42969.85270     40-49   
    621    37    male  34.100         4    yes  southwest  40182.24600     30-39   
    629    44  female  38.950         0    yes  northwest  42983.45850     40-49   
    665    43    male  38.060         2    yes  southeast  42560.43040     40-49   
    667    40  female  32.775         2    yes  northwest  40003.33225     30-39   
    668    62    male  32.015         0    yes  northeast  45710.20785       60+   
    674    44  female  43.890         2    yes  southeast  46200.98510     40-49   
    677    60    male  31.350         3    yes  northwest  46130.52650     50-59   
    682    39    male  35.300         2    yes  southwest  40103.89000     30-39   
    697    41    male  35.750         1    yes  southeast  40273.64550     40-49   
    706    51  female  38.060         0    yes  southeast  44400.40640     50-59   
    725    30  female  39.050         3    yes  southeast  40932.42950     18-29   
    736    37  female  38.390         0    yes  southeast  40419.01910     30-39   
    739    29    male  35.500         2    yes  southwest  44585.45587     18-29   
    742    53    male  34.105         0    yes  northeast  43254.41795     50-59   
    803    18  female  42.240         0    yes  southeast  38792.68560       NaN   
    819    33  female  35.530         0    yes  northwest  55135.40209     30-39   
    826    56    male  31.790         2    yes  southeast  43813.86610     50-59   
    828    41    male  30.780         3    yes  northeast  39597.40720     40-49   
    845    60  female  32.450         0    yes  southeast  45008.95550     50-59   
    852    46  female  35.530         0    yes  northeast  42111.66470     40-49   
    856    48  female  33.110         0    yes  southeast  40974.16490     40-49   
    860    37  female  47.600         2    yes  southwest  46113.51100     30-39   
    883    51  female  37.050         3    yes  northeast  46255.11250     50-59   
    893    47    male  38.940         2    yes  southeast  44202.65360     40-49   
    901    60    male  40.920         0    yes  southeast  48673.55880     50-59   
    947    37    male  34.200         1    yes  northeast  39047.28500     30-39   
    951    51    male  42.900         2    yes  southeast  47462.89400     50-59   
    953    44    male  30.200         2    yes  southwest  38998.54600     40-49   
    956    54    male  30.800         1    yes  southeast  41999.52000     50-59   
    958    43    male  34.960         1    yes  northeast  41034.22140     40-49   
    1022   47    male  36.080         1    yes  southeast  42211.13820     40-49   
    1031   55  female  35.200         0    yes  southeast  44423.80300     50-59   
    1036   22    male  37.070         2    yes  southeast  37484.44930     18-29   
    1037   45  female  30.495         1    yes  northwest  39725.51805     40-49   
    1047   22    male  52.580         1    yes  southeast  44501.39820     18-29   
    1049   49    male  30.900         0    yes  southwest  39727.61400     40-49   
    1062   59    male  41.140         1    yes  southeast  48970.24760     50-59   
    1070   37    male  37.070         1    yes  southeast  39871.70430     30-39   
    1090   47    male  36.190         0    yes  southeast  41676.08110     40-49   
    1096   51  female  34.960         2    yes  northeast  44641.19740     50-59   
    1111   38    male  38.390         3    yes  southeast  41949.24410     30-39   
    1118   33    male  35.750         1    yes  southeast  38282.74950     30-39   
    1122   53  female  36.860         3    yes  northwest  46661.44240     50-59   
    1124   23  female  42.750         1    yes  northeast  40904.19950     18-29   
    1146   60    male  32.800         0    yes  southwest  52590.82939     50-59   
    1152   43  female  32.560         3    yes  southeast  40941.28540     40-49   
    1156   19    male  44.880         0    yes  southeast  39722.74620     18-29   
    1207   36    male  33.400         2    yes  southwest  38415.47400     30-39   
    1218   46  female  34.600         1    yes  southwest  41661.60200     40-49   
    1230   52    male  34.485         3    yes  northwest  60021.39897     50-59   
    1240   52    male  41.800         2    yes  southeast  47269.85400     50-59   
    1241   64    male  36.960         2    yes  southeast  49577.66240       60+   
    1249   32    male  33.630         1    yes  northeast  37607.52770     30-39   
    1284   61    male  36.300         1    yes  southwest  47403.88000       60+   
    1288   20    male  39.400         2    yes  southwest  38344.56600     18-29   
    1300   45    male  30.360         0    yes  southeast  62592.87309     40-49   
    1301   62    male  30.875         3    yes  northwest  46718.16325       60+   
    1303   43    male  27.800         0    yes  southwest  37829.72420     40-49   
    1323   42  female  40.370         2    yes  southeast  43896.37630     40-49   
    
            zscore  
    14    2.175983  
    23    2.018214  
    29    2.101574  
    34    3.132806  
    38    2.189409  
    39    2.883233  
    49    2.101424  
    53    2.021576  
    55    2.827319  
    84    2.194550  
    86    2.503701  
    94    2.810349  
    109   2.790893  
    123   2.171418  
    146   2.267578  
    175   2.937018  
    185   2.518108  
    240   2.085105  
    251   2.811526  
    252   2.560024  
    254   2.298689  
    256   2.531973  
    265   2.716182  
    271   2.444050  
    281   2.914279  
    288   2.860387  
    292   2.382541  
    298   2.104495  
    312   2.383555  
    327   2.436092  
    328   2.862967  
    330   2.911667  
    338   2.366586  
    377   2.053270  
    381   2.398357  
    420   2.777158  
    421   2.753189  
    422   2.135801  
    488   2.942031  
    524   2.063128  
    530   2.924715  
    543   4.171663  
    549   2.692398  
    558   2.206686  
    569   2.679083  
    577   3.742159  
    587   2.533848  
    609   2.145393  
    615   2.453386  
    621   2.223110  
    629   2.454510  
    665   2.419565  
    667   2.208330  
    668   2.679759  
    674   2.720301  
    677   2.714481  
    682   2.216637  
    697   2.230660  
    706   2.571560  
    725   2.285080  
    736   2.242669  
    739   2.586847  
    742   2.476893  
    803   2.108322  
    819   3.458348  
    826   2.523108  
    828   2.174798  
    845   2.621831  
    852   2.382494  
    856   2.288528  
    860   2.713075  
    883   2.724772  
    893   2.555224  
    901   2.924553  
    947   2.129354  
    951   2.824544  
    953   2.125328  
    956   2.373230  
    958   2.293489  
    1022  2.390711  
    1031  2.573493  
    1036  2.000253  
    1037  2.185381  
    1047  2.579903  
    1049  2.185554  
    1062  2.949062  
    1070  2.197457  
    1090  2.346511  
    1096  2.591451  
    1111  2.369077  
    1118  2.066198  
    1122  2.758338  
    1124  2.282748  
    1146  3.248148  
    1152  2.285812  
    1156  2.185152  
    1207  2.077162  
    1218  2.345315  
    1230  3.861966  
    1240  2.808597  
    1241  2.999239  
    1249  2.010420  
    1284  2.819669  
    1288  2.071304  
    1300  4.074389  
    1301  2.763024  
    1303  2.028775  
    1323  2.529924  
    


```python
import pandas as pd

# Veri setini oku
df = pd.read_csv("C:\\Users\\Umut\\Desktop\\insurance.csv")

# Yaş gruplarını belirle
age_bins = [18, 30, 40, 50, 60, 100]  # Örnek yaş grupları
age_labels = ['18-29', '30-39', '40-49', '50-59', '60+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# Aykırı değerleri dışla
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1
filtered_df = df[(df['charges'] >= Q1 - 1.5 * IQR) & (df['charges'] <= Q3 + 1.5 * IQR)]

# Yaş gruplarına göre ortalama primleri hesapla
age_group_premiums = filtered_df.groupby('age_group')['charges'].mean().reset_index()

# Sonuçları göster
print("Yaş Gruplarına Göre Ortalama Primler:")
print(age_group_premiums)

# Aykırı değerleri dışla
filtered_df = df[(df['charges'] >= Q1 - 1.5 * IQR) & (df['charges'] <= Q3 + 1.5 * IQR)]

# Sigara içen ve içmeyen müşterilere göre ortalama primleri hesapla
smoker_premiums = filtered_df.groupby('smoker')['charges'].mean().reset_index()

# Sonuçları göster
print("\nSigara İçen ve İçmeyen Müşterilere Göre Ortalama Primler:")
print(smoker_premiums)

```

    Yaş Gruplarına Göre Ortalama Primler:
      age_group       charges
    0     18-29   6647.461892
    1     30-39   8063.435151
    2     40-49  10773.284895
    3     50-59  13398.518684
    4       60+  16409.341283
    
    Sigara İçen ve İçmeyen Müşterilere Göre Ortalama Primler:
      smoker       charges
    0     no   8355.712011
    1    yes  22014.245543
    


```python
# Cinsiyete göre ortalama primleri hesapla
gender_premiums = df.groupby('sex')['charges'].mean().reset_index()
```


```python
#cinsiyete göre ortalama 
print(gender_premiums)
```

          sex       charges
    0  female  12569.578844
    1    male  13956.751178
    


```python
# NaN değerleri her bir cinsiyet ve yaş grubu kombinasyonu için ortalama ile doldur
gender_age_premiums['charges'].fillna(gender_age_premiums.groupby('sex')['charges'].transform('mean'), inplace=True)

# Sonuçları göster
print("Cinsiyet ve Yaş Gruplarına Göre Ortalama Primler (NaN Değerleri Doldurulmuş):")
print(gender_age_premiums)

```

    Cinsiyet ve Yaş Gruplarına Göre Ortalama Primler (NaN Değerleri Doldurulmuş):
          sex age_group       charges
    0  female     18-29  16884.924000
    1  female     30-39   5519.063600
    2  female     40-49   8240.589600
    3  female     50-59  14891.928530
    4  female       60+  28923.136920
    5    male     18-29   3087.507150
    6    male     30-39  10752.578837
    7    male     40-49   6920.042993
    8    male     50-59   6920.042993
    9    male       60+   6920.042993
    


```python
import pandas as pd

# Veri setini oku
df = pd.read_csv("C:\\Users\\Umut\\Desktop\\insurance.csv")

# Sigara içmeyen kişilerdeki tutarsızlıkları kontrol et
inconsistent_smoker_rows = df[(df['smoker'] == 'no') & (df['charges'] < 15000)]

# Sonuçları göster
print("Tutarsız Smoker Değerleri:")
print(inconsistent_smoker_rows)

```

    Tutarsız Smoker Değerleri:
          age     sex     bmi  children smoker     region      charges
    62     64    male  24.700         1     no  northwest  30166.61817
    115    60    male  28.595         0     no  northeast  30259.99556
    242    55  female  26.800         1     no  southwest  35160.13457
    387    50    male  25.365         2     no  northwest  30284.64294
    573    62  female  36.860         1     no  northeast  31620.00106
    599    52  female  37.525         2     no  northwest  33471.97189
    936    44    male  29.735         2     no  northeast  32108.66282
    1012   61  female  33.330         4     no  southeast  36580.28216
    1206   59  female  34.800         2     no  southwest  36910.60803
    1258   55    male  37.715         3     no  northwest  30063.58055
    


```python
import pandas as pd

# Veri setini oku
df = pd.read_csv("C:\\Users\\Umut\\Desktop\\insurance.csv")

# Yaş gruplarını belirle
age_bins = [18, 30, 40, 50, 60, 100]
age_labels = ['18-29', '30-39', '40-49', '50-59', '60+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# Yaş gruplarına göre ortalama primleri hesapla
age_group_premiums = df.groupby('age_group')['charges'].mean().reset_index()

# Belirli bir yaş grubuna odaklan
target_age_group = '30-39'
target_group_data = df[df['age_group'] == target_age_group]

# Pazarlama stratejisi geliştirme
if target_age_group == '30-39':
    print(f"Pazarlama Stratejisi: {target_age_group} yaş grubundaki müşterilere özel avantajlı teklifler sunulabilir.")
    print("Bu yaş grubundaki müşterilere yönelik özel kampanyalar düzenlenebilir.")
else:
    print(f"Pazarlama Stratejisi: {target_age_group} yaş grubu için özel bir strateji geliştirilmedi.")

# İsteğe bağlı olarak diğer demografik gruplara da benzer analiz ve stratejiler ekleyebilirsiniz.

```

    Pazarlama Stratejisi: 30-39 yaş grubundaki müşterilere özel avantajlı teklifler sunulabilir.
    Bu yaş grubundaki müşterilere yönelik özel kampanyalar düzenlenebilir.
    
