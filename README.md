# E.ML-toolbox-Pruning Progress Report
## week 7/6/2020 - 7/12/2020
### work this week
* Introduce work done so far to Maki
* create new repo and clean up previous code
* finish Rewinding/late-resetting and reinitialzing
* Capable of accepting progressive pruning

### TODO
* conduct more testing with arbitrary input modules
* cross validtion
* continue improving accuracy to SOTA

### rewinding and progressive pruning
```
prune more than 85%
pruning iteration: 0 iter
Pruning 0.25 of current weight; num of non-zero weights 3750 ; remaining weights = 75.00%
pruning iteration: 1 iter
Rewinding params to epoch 1
Pruning 0.25 of current weight; num of non-zero weights 2812 ; remaining weights = 56.24%
pruning iteration: 2 iter
Rewinding params to epoch 1
Pruning 0.25 of current weight; num of non-zero weights 2110 ; remaining weights = 42.20%
pruning iteration: 3 iter
Rewinding params to epoch 1
Pruning 0.25 of current weight; num of non-zero weights 1582 ; remaining weights = 31.64%
pruning iteration: 4 iter
Rewinding params to epoch 1
Pruning 0.25 of current weight; num of non-zero weights 1187 ; remaining weights = 23.74%
pruning iteration: 5 iter
Rewinding params to epoch 1
Pruning 0.25 of current weight; num of non-zero weights 890 ; remaining weights = 17.80%
pruning iteration: 6 iter
Rewinding params to epoch 1
Pruning 0.25 of current weight; num of non-zero weights 668 ; remaining weights = 13.36%
Stopping criterion met.
```
### Old testing data

#### testing result
![testing result](https://github.com/wth6618/E.ML-toolbox-Pruning/blob/master/images/model_overview.PNG)

#### prune result raw
![Result before refine](https://github.com/wth6618/E.ML-toolbox-Pruning/blob/master/images/exp_result_cifar10_resnet164.png)
#### pruning result with finetuning
![result after refine](https://github.com/wth6618/E.ML-toolbox-Pruning/blob/master/images/exp_result_cifar10_resnet164_refined.png)
