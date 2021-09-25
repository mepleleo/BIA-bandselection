# BIA-bandselection, hyperspectral band selection

## band influence algorithm (Classification-of-Hyperspectral-Moldy-Peanut)

Update date: 2021.09.25

This code is the implemention of band influence algorithm (BIA)


The effects of 5 band selection methods on 4 classification models were compared.

band selection methods: BSNET, EGCSR-R GSM,MVPCA SpaBS, BIA.

classification models: DT, KNN,  SVM, ShuffleNet V2.

![image](https://github.com/mepleleo/BIA-bandselection/blob/main/BIA_.png)


required package:

```
python 3.6
scipy
sklearn
skimage
matplotlib
pandas==1.0.5
# BIA(shuff):
pytorch==1.9.0

```



