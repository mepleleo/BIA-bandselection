# BIA-bandselection, hyperspectral band selection

## band influence algorithm (Classification-of-Hyperspectral-Moldy-Peanut)
paper: A Band Influence Algorithm for Hyperspectral Band Selection to Classify Moldy Peanuts  

Update date: 2022.05.29

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

对比了BSNET, EGCSR-R GSM,MVPCA SpaBS, BIA.这几个波段选择方法  
在 DT, KNN,  SVM, ShuffleNet V2.分类模型上做了验证  
BIA方法思路：  
原始全波段128波段  
（1）使用原始全波段数据训练分类模型  
（2）对原始数据从第一个波段开始的连续3个波段（或5个）置零，其他波段不变，  
（3）输入到第一步训练好的模型，得到一个分类精度，作为波段2的精度，  
（4）以1为步长，依次滑动（循环）置零原始数据，得到 128-2 个波段对应的精度，第一和最后一个波段精度设为0，  
（5）得到一个精度曲线集合，将精度集合分成3部分（或者更多），每部分取精度的局部最小值作为初始特征波段  
（6）假设需要提取8个特征波段，根据每部分精度曲线的面积计算每部分提取的波段数，  
（7.1）例如第一部分需要提取3个波段，如果初始特征波段数如果多于3个就提取3个值最小的波段作为特征波段，  
（7.2）如果少于3个就以2为步长，从精度曲线中从小到大提取少的波段数  
（8）其他部分同理。合并每部分提取的波段作为最终的特征波段集合。  



