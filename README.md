# E.ML-toolbox-Pruning Progress Report
## week 7/6/2020 - 7/12/2020
### work this week
* Introduce work done so far to Maki
* create new repo and clean up previous code
* finish Rewinding/late-resetting and reinitialzing
* capable of accepting progressive pruning
* finish iterative pruning

### TODO
* conduct more testing with arbitrary input modules
* combine and encapsulate implementations
* cross validtion
* continue improving accuracy to SOTA
## week 7/13/2020 - 7/19/2020
### work this week
* Maki:
  * Adapted open-source pruning implementations to toolbox interface
  * compare modified pruning method's result to paper's result
* Tianhao: 
  * modify and testing iterative pruning 
  * reproduce [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307) [__tensorflow__] as pytorch
  * Adding enforce_isometry as an option for pruning method to enforce approximate dynamical isometry in the sparse network (has unsolved issues)
  

### TODO
* continue to modify the pruning method, the goal is to accept arbitrary input network
* adapt orthognal to reinitialization and test
* continue working on pytorch version of [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307); including  jacobian_singular_value  

### Rethinking-network-pruning result comparsion
![](https://github.com/wth6618/E.ML-toolbox-Pruning/blob/master/images/comparisons.PNG)
![](https://github.com/wth6618/E.ML-toolbox-Pruning/blob/master/images/cifar50.PNG)
### Iterative Pruning
![iterative pruning](https://github.com/wth6618/E.ML-toolbox-Pruning/blob/master/images/iterative%20pruning.png)
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

