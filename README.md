# SchwabBandit


> We examine the exploration - exploitation tradeoff in a multi-armed bandit setting.

## `bandit.py`
Contains functions for a two-armed bandit simulation. Thompson and Infomax algorithms are implemented.


## `entropy_estimators.py`
Code from [NPEET](https://github.com/gregversteeg/NPEET) for numeric entropy estimation.


## `entropy_est_testing.py`
Tests the timing and accuracy of various entropy estimation functions for use in `bandit.py`

N.B. The original Jimmy estimator uses 2 bins and 10,000 samples. The modified estimator uses 50 bins, and also 10,000 samples.

### Results
#### Accuracy (10,000 Trials)
Entropy Estimator	| Mean		| MSE
---			| :---:		| :---:
Integration ("True")	| -0.5143	| N/A
Integration II		| -0.5143	| N/A
Integration III		| -0.5143	| N/A
Jimmy's Estimation	| 0.2978	| 0.6594
Jimmy's Estimation II	| -0.7450	| 0.7169
NPEET			| -28.275	| 770.63
#### Timing (1,000 Trials)
Entropy Estimator	| Mean Time
---			| :---:
Integration ("True")	| 1.6534 E-1
Integration II		| 8.0553 E-2
Integration III		| 8.0726 E-2
Jimmy's Estimation	| 3.8440 E-4
Jimmy's Estimation II	| 1.8905 E-3
NPEET			| 2.4456 E-2
