---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Binom ve Poisson Dağılımları

FİZ302 - İstatistik Fizik Lab IV "Binom ve Poisson Dağılımları" deneyinin simülasyonu.

Dr. Emre S. Taşcı, emre.tasci@hacettepe.edu.tr  
Fizik Mühendisliği Bölümü  
Hacettepe Üniversitesi

Haziran 2024

```{code-cell} ipython3
import numpy as np
from scipy import special
import pandas as pd
import matplotlib.pyplot as plt
```

## Binom Dağılımı

$$P_{N,p} = p^n q^{N-n}\begin{bmatrix}N\\n\end{bmatrix}=p^n q^{N-n}\frac{N!}{(N-n)!n!}$$

+++

>Genelleştirilmiş olasılık formülünün basit bir uygulaması olarak birkaç tane yirmi yüzlü _(0'dan 9'a her rakamın iki kere yer aldığı)_ zarın atıldığını düşünelim: Örneğin böyle 3 zarı attığımız zaman, iki 7 elde etme olasılığı nedir? Her ayrı olay için p olasılığı 1/10, N = 3 ve n = 2'dir. Buna göre olasılık 0.027'dir.

```{code-cell} ipython3
p = 1/10
q = 1 - p
N = 3
n = 2
special.comb(N,n)*p**n*q**(N-n)
```

Binom dağılımda:

$$\bar n = Np\\
\sigma^2 = Npq$$

+++

## Poisson Dağılımı

$a = Np$ ve $N\gg$ ile $p\ll$ olmak üzere:

$$P_a(n) = \frac{a^ne^{-a}}{n!}$$

$$\bar n = a\\
\sigma^2 = a$$

+++

## Deney

+++

>Yirmi yüzlü üç zardan elde edilen sayıları Çizelge 1 ‘e geçirerek 360 rakamlı bir gelişigüzel sayılar çizelgesi kurun.

```{code-cell} ipython3
n_satir = 15
n_sutun = 24
N = n_satir * n_sutun
rakamlar = np.random.randint(0,10,(n_satir,n_sutun))
rakamlar
```

### Çizelge 1 (Olasılık Dağılımı Deneyi)

```{code-cell} ipython3
for i in range(n_satir):
    print("| ",end="")
    for j in range(n_sutun):
        print("{:d}".format(rakamlar[i,j]),end=" ")
        if((j+1)%3==0):
            print("|",end=" ")
    if((i+1)%3 == 0):
        print("")
        print("-"*(n_sutun*2+(int)(n_sutun/2)+5))
    else:
        print("")
    
```

### Çizelge 2A (Olasılık Dağılımı Deneyi)

```{code-cell} ipython3
print("{:^6s} {:^8s} {:^10s}".format("Sayı","Frekans","Olasılık"))
print("-"*25)
for i in range(10):
    toplam = np.sum(i==rakamlar)
    print("{:^6d} {:^8d} {:^10.5f}".format(i,toplam,toplam/(n_satir*n_sutun)))
```

```{code-cell} ipython3
# Hazır fonksiyonlar kullanarak
ortalama = rakamlar.mean()
varyans = rakamlar.var()

print("Ortalama: {:.4f}".format(ortalama))
print("Varyans : {:.4f}".format(varyans))
```

```{code-cell} ipython3
# Formül kullanarak
rakamlar_tum = rakamlar.flatten()
toplam_aux = 0
for n_i in rakamlar_tum:
    toplam_aux += n_i
ortalama_formul = toplam_aux / N

varyans_aux = 0
for n_i in rakamlar_tum:
    varyans_aux += (n_i - ortalama_formul)**2
varyans_formul = varyans_aux / N

print("Ortalama: {:.4f} (formul)".format(ortalama_formul))
print("Varyans : {:.4f} (formul)".format(varyans_formul))
```

```{code-cell} ipython3
plt.hist(rakamlar_tum,bins=np.arange(11),edgecolor="red")
plt.xticks(np.arange(10)+0.5,range(10)) # x değerleri tam ortada çıksın diye
                                        # yarım birim sağa öteliyoruz
plt.xlabel("Rakam")
plt.ylabel("Frekans")
plt.show()
```

## Deney

>1. Deney 4 için hazırlanan gelişigüzel sayılar çizelgesi binom ve Poisson dağılımları için
ilginç örnekler verir. Yirmi yüzlü zar ile elde edilen üç-rakamlı gelişigüzel sayıları alarak her üç gruptaki 7'lerin sayısı için bir frekans sayımı yapın. Yani, üç rakamlı sayılardan kaç tanesinde hiç 7 yoktur; bir tane 7, iki 7 veya üç 7 kaç tanesinde var? Sonuçlarınızı ana binom dağılımına göre beklediklerinizle karşılaştırın. Örnek ortalamasını ve variansı hesaplayıp ana binom dağılımındaki değerlerle karşılaştırın.

```{code-cell} ipython3
sayilar = rakamlar_tum.reshape((int)(N/3),3)
sayilar[:10,:]
```

```{code-cell} ipython3
k = 7
sayi =  sayilar[4,:]
sayi_str = "".join(sayi.astype("str"))
print("{:3s} sayısındaki {:d} adedi: {:d}".format(sayi_str,k,np.sum(k==sayi)))
```

```{code-cell} ipython3
k_toplam = np.zeros(4,int)

for i in range(sayilar.shape[0]):
    sayi =  sayilar[i,:]
    k_toplam[np.sum(k==sayi)] += 1

for i in range(4):
    print("İçinde {:d} tane {:d} olan sayı adedi: {:d}".format(i,k,k_toplam[i]),end="\t")
    print("(Olasılık: {:.5f})".format(k_toplam[i]/sayilar.shape[0]))
    
print("\nToplam sayı adedi: {:d}".format(sayilar.shape[0]))
```

### Sonuçların binom dağılımına göre karşılaştırılması

Binom dağılımına göre karşılaştırırken sormamız gereken soru: 

"Rastgele 3 rakam çekilirse, bunların $n$ $(n=0,1,2,3)$ tanesinin 7 olma olasılığı nedir?"

şeklindedir.

```{code-cell} ipython3
N = 3
p = 1/10
q = 1 - p

for n in range(4):
    olasilik_n = special.comb(N,n)*p**n*q**(N-n)
    print("Rastgele çekilen 3 rakamdan {:d} tanesinin {:d} olma ihtimali: {:.5f}"\
         .format(n,k,olasilik_n))
    
    
```

### 9 rakamlı gruplar

+++

>2. Şimdi de gelişigüzel sayılar çizelgesindeki dokuz rakamlı grupları alalım. 9 rakamlı
grupların her birindeki 7'lerin sayısı için frekans sayımı yapın; çizelgede böyle 40 grup bulunmaktadır. Sonuçlarınızı Çizelge 1’ e geçirin.

```{code-cell} ipython3
sayilar_9 = rakamlar.reshape(40,9)
sayilar_9[:10,:]
```

```{code-cell} ipython3
# Her bir sayıyı oluşturan rakamlardan kaçar tane
# olduğunu tutan çizelge

# Örneğin cizelge_1_aux.loc[4,6]: 4 indisli sayıda kaç adet 
# 6 olduğunu vermekte

# Benzer şekilde, cizelge_1_aux.loc[6,0]: 6 indisli sayıda 
# kaç adet 0 olduğunu vermekte

sayilar_9_adedi = sayilar_9.shape[0]
cizelge_1_aux_np = np.zeros((sayilar_9_adedi,10),int)
for i in range(sayilar_9_adedi):
    for rakam in range(10):
        cizelge_1_aux_np[i,rakam] = np.sum(rakam==sayilar_9[i,:])
        
cizelge_1_aux = pd.DataFrame(cizelge_1_aux_np)
cizelge_1_aux.index.names=["Sayı"]
cizelge_1_aux.loc[:10,:]
```

```{code-cell} ipython3
# Bu da hangi rakamın bir sayıda kaç kere çıkmış
# olduğunu tutan çizelge (Çizelge 1)

# Örneğin cizelge_1.loc[4,6]: 4 rakamının kaç tane sayıda
# 6 kere olduğunu vermekte;

# Benzer şekilde, cizelge_1.loc[6,0]: 6 rakamını 
# hiç içermeyen kaç tane sayı olduğunu vermekte

cizelge_1_np = np.zeros((10,10),int)
for rakam in range(10):
    for n in range(10):
        rakamdan_n_tane_var = np.sum(cizelge_1_aux[rakam] == n)
        cizelge_1_np[rakam,n] = rakamdan_n_tane_var
cizelge_1_np

cizelge_1 = pd.DataFrame(cizelge_1_np)
cizelge_1.index.names = ["Rakam"]
```

```{code-cell} ipython3
# Çizelge 1'in sütunlarını föydeki gibi sıralayalım
# (0 en sonda olacak şekilde)
cizelge_1 = cizelge_1.loc[:,[1,2,3,4,5,6,7,8,9,0]]
cizelge_1
```

>Son olarak, olasılıkları elde etmek için bunları toplam ölçme sayısına bölün

```{code-cell} ipython3
cizelge_1_olasiliklar = cizelge_1/40
cizelge_1_olasiliklar
```

```{code-cell} ipython3
# Her rakamın olasılığı üzerinden
# (yani sütunlar boyunca) ortalama
# değerlerini alalım
cizelge_1_olasiliklar_ortalama = cizelge_1_olasiliklar.mean()

cizelge_1_olasiliklar_ortalama
```

>Sonuçlarınızı $P_{N,1/10} (n)$ binom dağılımı değerleriyle karşılaştırın.

```{code-cell} ipython3
N = 9
p = 1/10
q = 1-p

print("Seçilen bir rakamın 9 rakamlı bir sayıda:\n")
for n in range(10):
    olasilik_n_binom = special.comb(N,n)*p**n*q**(N-n)
    print( "\t{:d} kere çıkma ihtimali: {:.9f}"\
         .format(n,olasilik_n_binom))
```

>Bu dağılımın değerlerini hesaplarken P(n+1)'i P(n) cinsinden veren bir geri götürme (özyineleme / _recurrence_) bağıntısı kullanmak fazla işlem yapmayı engeller.
>
>Binom dağlımı için uygun olan geri götürme bağıntısı (...) aşağıdaki bağıntıdır.
>
>$$P_{N,p}(N+1)=\frac{p(N-n)}{q(n+1)}P_{N,p}(n)$$
>
>Buna göre sadece $P_{N,p}(0)$'ı hesaplamak ve bu bağıntıyı kullanmak yeterlidir. Geri götürme bağıntısını kullanırken başta yapılan bir hata süregideceğinden genel olarak bu yol biraz sakıncalıdır. Bununla birlikte bu özel durumda artan n'ler için P’nin değerleri çok çabuk küçüldüğünden bir sakınca yoktur. Gerçekten, $n\gt4$ için değerlerin $10^{-5}$'ten küçük olduğunu, böylece daha ileri gitmenin anlamsızlığını görmelisiniz.

```{code-cell} ipython3
N = 9
p = 1/10
q = 1-p

print("Seçilen bir rakamın 9 rakamlı bir sayıda n kere çıkma ihtimali:\n")

print("{:^4s} {:^11s} | {:^11s} | {:^11s}"\
      .format("n","Binom","Özyineleme","Örnek"))
print("-"*44)
n = 0
olasilik_n_binom = special.comb(N,n)*p**n*q**(N-n)
P_Np_n = olasilik_n_binom
print("{:^4d} {:.9f} | {:11s} | {:.9f}"\
         .format(n,olasilik_n_binom,"",cizelge_1_olasiliklar_ortalama.loc[n]))

for n in range(1,10):
    olasilik_n_binom = special.comb(N,n)*p**n*q**(N-n)
    olasilik_n_ozyineleme = p * (N-(n-1)) / (q*((n-1)+1))*P_Np_n
    print("{:^4d} {:.9f} | {:.9f} | {:.9f}"\
         .format(n,olasilik_n_binom,olasilik_n_ozyineleme,\
                 cizelge_1_olasiliklar_ortalama.loc[n]))
    P_Np_n = olasilik_n_ozyineleme
```

>3. Şimdi 9 rakamlı gelişigüzel sayı gruplarındaki sayıların dağılımı için Poisson yaklaşıklığını ele alalım. $N = 9$, $p = 1/10$ olup $a = Np = 0,9$'dur. Poisson dağılımını kullanarak $n$'nin 0'dan 9'a kadar olan değerleri için olasılıkları yeniden hesaplayın. Hiç kuşkusuz $N = 9$, $N = \infty$'dan çok uzak olduğundan kesin bir uyuşma beklememeliyiz, fakat yine de karşılaştırma ilgi çekicidir.

```{code-cell} ipython3
N = 9
p = 1/10
a = N*p

for n in range(10):
    poisson = a**n*np.exp(-a) / special.factorial(n)
    print(n,poisson)
```

```{code-cell} ipython3
N = 9
p = 1/10
q = 1-p

print("Seçilen bir rakamın 9 rakamlı bir sayıda n kere çıkma ihtimali:\n")

print("{:^4s} {:^11s} | {:^11s} | {:^11s}"\
      .format("n","Binom","Örnek","Poisson"))
print("-"*44)

dizi_binom = []
dizi_ornek = []
dizi_poisson = []

for n in range(0,10):
    olasilik_n_binom = special.comb(N,n)*p**n*q**(N-n)
    poisson = a**n*np.exp(-a) / special.factorial(n)
    print("{:^4d} {:.9f} | {:.9f} | {:.9f}"\
         .format(n,olasilik_n_binom,\
                 cizelge_1_olasiliklar_ortalama.loc[n],poisson))
    dizi_binom.append(olasilik_n_binom)
    dizi_ornek.append(cizelge_1_olasiliklar_ortalama.loc[n])
    dizi_poisson.append(poisson)
```

```{code-cell} ipython3
n = range(10)
plt.plot(n,dizi_binom,"*",markerfacecolor="none")
plt.plot(n,dizi_ornek,"x")
plt.plot(n,dizi_poisson,"o",markerfacecolor="none")
plt.legend(["Binom","Örnek","Poisson"])
plt.show()
```

```{code-cell} ipython3

```
