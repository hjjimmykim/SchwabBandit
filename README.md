# SchwabBandit

---
> We examine the exploration - exploitation tradeoff in a multi-armed bandit setting.
---
## bandit
Contains functions for a two-armed bandit simulation. Thompson and Infomax algorithms are implemented.


## entropy\_est\_testing
Tests the timing and accuracy of various entropy estimation functions for use in *bandit*

Results (10000 Trials)
Entropy Estimator	| Mean		| MSE
---			| :---:		| :---:
Integration ("True")	| -0.5143	| N/A
Jimmy's Estimation	| 0.2978	| 0.6594
Jimmy's Estimation II	| -0.7450	| 0.7169
NPEET			| ~-28.273	| ~770.
