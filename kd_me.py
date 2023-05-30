import pandas as pd

df1 = pd.read_csv("FIVE_230101_230521 (1).csv")
df2 = pd.read_csv("KRSB_230101_230521 (2).csv")
df3 = pd.read_csv("MRKS_230101_230521 (1).csv")
df4 = pd.read_csv("PMSB_230101_230521 (2).csv")
df5 = pd.read_csv("RKKE_230101_230521 (2).csv")
print(df1.dtypes)
df1['DateTime'] = df1["<DATE>"].astype(str) + ' ' + df1["<TIME>"].astype(str)
df2['DateTime'] = df2["<DATE>"].astype(str) + ' ' + df2["<TIME>"].astype(str)
df3['DateTime'] = df3["<DATE>"].astype(str) + ' ' + df3["<TIME>"].astype(str)
df4['DateTime'] = df4["<DATE>"].astype(str) + ' ' + df4["<TIME>"].astype(str)
df5['DateTime'] = df5["<DATE>"].astype(str) + ' ' + df5["<TIME>"].astype(str)
print(df1['DateTime'][len(df1['DateTime'])-1],df2['DateTime'][len(df2['DateTime'])-1],df3['DateTime'][len(df3['DateTime'])-1],df4['DateTime'][len(df4['DateTime'])-1],df5['DateTime'][len(df5['DateTime'])-1],sep='\n')
arr = df1['<CLOSE>'].to_list()
x = 100
y = x
datetime = arr
close = arr

K = []
D = []
for i in range(x, len(datetime)):
  minx = min(close[i - x:i + 1])
  maxx = max(close[i - x:i + 1])
  K.append(100 * (close[i] - minx) / (maxx - minx))

for i in range(y, len(K)):
  D.append(0)
  for j in range(y + 1):
    D[i - x] += (y + 1 - j) * K[i - j]
  D[i - x] /= (y + 2) * (y + 1) / 2


a=100
med=[]
for i in range(a,len(close)):
  med.append(0)
  for j in range(a+1):
    med[i-a]+=(y+1-j)*close[i-j]
  med[i-a]/=(y+2)*(y+1)/2

# Example arrays


# Create a DataFrame
DD=[0]*x*2
DD.extend(D)

KK=[0]*x
KK.extend(K)

MED=[0]*a
MED.extend(med)
# print(c)
print(len(df1["DateTime"]),len(KK),len(DD),len(MED))







arr = df2['<CLOSE>'].to_list()
K2 = []
D2 = []
datetime = arr
close = arr
for i in range(x, len(datetime)):
  minx = min(close[i - x:i + 1])
  maxx = max(close[i - x:i + 1])
  if(maxx-minx==0):
    K2.append(K2[i-1-x])
  else:
    K2.append(100 * (close[i] - minx) / (maxx - minx))
    
        

for i in range(y, len(K2)):
  D2.append(0)
  for j in range(y + 1):
    D2[i - y] += (y + 1 - j) * K2[i - j]
  D2[i - y] /= (y + 2) * (y + 1) / 2

a=100
med2=[]
for i in range(a,len(close)):
  med2.append(0)
  for j in range(a+1):
    med2[i-a]+=(y+1-j)*close[i-j]
  med2[i-a]/=(y+2)*(y+1)/2
# Example arrays


# Create a DataFrame
DD2=[0]*x*2
DD2.extend(D2)

KK2=[0]*x
KK2.extend(K2)

MED2=[0]*a
MED2.extend(med2)
# print(c)





#3


arr = df3['<CLOSE>'].to_list()
K3 = []
D3 = []
datetime = arr
close = arr
for i in range(x, len(datetime)):
  minx = min(close[i - x:i + 1])
  maxx = max(close[i - x:i + 1])
  if(maxx-minx==0):
    K3.append(K3[i-1-x])
  else:
    K3.append(100 * (close[i] - minx) / (maxx - minx))

for i in range(y, len(K3)):
  D3.append(0)
  for j in range(y + 1):
    D3[i - x] += (y + 1 - j) * K3[i - j]
  D3[i - x] /= (y + 2) * (y + 1) / 2

a=100
med3=[]
for i in range(a,len(close)):
  med3.append(0)
  for j in range(a+1):
    med3[i-a]+=(y+1-j)*close[i-j]
  med3[i-a]/=(y+2)*(y+1)/2
# Example arrays


# Create a DataFrame
DD3=[0]*x*2
DD3.extend(D3)

KK3=[0]*x
KK3.extend(K3)

MED3=[0]*a
MED3.extend(med3)


#4


arr = df4['<CLOSE>'].to_list()
K4 = []
D4 = []
datetime = arr
close = arr
for i in range(x, len(datetime)):
  minx = min(close[i - x:i + 1])
  maxx = max(close[i - x:i + 1])
  K4.append(100 * (close[i] - minx) / (maxx - minx))

for i in range(y, len(K4)):
  D4.append(0)
  for j in range(y + 1):
    D4[i - x] += (y + 1 - j) * K4[i - j]
  D4[i - x] /= (y + 2) * (y + 1) / 2

a=100
med4=[]
for i in range(a,len(close)):
  med4.append(0)
  for j in range(a+1):
    med4[i-a]+=(y+1-j)*close[i-j]
  med4[i-a]/=(y+2)*(y+1)/2
# Example arrays


# Create a DataFrame
DD4=[0]*x*2
DD4.extend(D4)

KK4=[0]*x
KK4.extend(K4)

MED4=[0]*a
MED4.extend(med4)



#5


arr = df5['<CLOSE>'].to_list()
K5 = []
D5 = []
datetime = arr
close = arr
for i in range(x, len(datetime)):
  minx = min(close[i - x:i + 1])
  maxx = max(close[i - x:i + 1])
  if(maxx-minx==0):
    K5.append(K5[i-1-x])
  else:
    K5.append(100 * (close[i] - minx) / (maxx - minx))

for i in range(y, len(K5)):
  D5.append(0)
  for j in range(y + 1):
    D5[i - x] += (y + 1 - j) * K5[i - j]
  D5[i - x] /= (y + 2) * (y + 1) / 2

a=100
med5=[]
for i in range(a,len(close)):
  med5.append(0)
  for j in range(a+1):
    med5[i-a]+=(y+1-j)*close[i-j]
  med5[i-a]/=(y+2)*(y+1)/2
# Example arrays


# Create a DataFrame
DD5=[0]*x*2
DD5.extend(D5)

KK5=[0]*x
KK5.extend(K5)

MED5=[0]*a
MED5.extend(med5)


print(len(df1["DateTime"]),len(KK),len(DD),len(MED),len(KK2),len(DD2),len(MED2),len(KK3),len(DD3),len(MED3),len(KK4),len(DD4),len(MED4),len(KK5),len(DD5),len(MED5))

data1 = {"Datetime": df1["DateTime"], "K": KK, "D": DD, "med": MED}
odf1 = pd.DataFrame(data1)

data2 = {"Datetime": df2["DateTime"], "K": KK2, "D2": DD2, "med2": MED2,}
odf2 = pd.DataFrame(data2)

data3 = {"Datetime": df3["DateTime"], "K": KK3, "D": DD3, "med": MED3,}
odf3 = pd.DataFrame(data3)

data4 = {"Datetime": df4["DateTime"], "K": KK4, "D": DD4, "med": MED4,}
odf4 = pd.DataFrame(data4)

data5 = {"Datetime": df5["DateTime"], "K": KK5, "D": DD5, "med": MED5,}
odf5 = pd.DataFrame(data5)
# Save DataFrame to a CSV file






# df1 = pd.read_csv("FIVE_230101_230521 (1).csv")
# df2 = pd.read_csv("KRSB_230101_230521 (2).csv")
# df3 = pd.read_csv("MRKS_230101_230521 (1).csv")
# df4 = pd.read_csv("PMSB_230101_230521 (2).csv")
# df5 = pd.read_csv("RKKE_230101_230521 (2).csv")



odf1.to_csv("FIVE.csv", index=False)
odf2.to_csv("KRSB.csv", index=False)
odf3.to_csv("MRKS.csv", index=False)
odf4.to_csv("PMSB.csv", index=False)
odf5.to_csv("RKKE.csv", index=False)




