# SchwabBandit


> We examine the exploration - exploitation tradeoff in a multi-armed bandit setting.

## `bandit.py`
Contains functions for a two-armed bandit simulation. Thompson and Infomax algorithms are implemented.


## `entropy_est_testing.py`
Tests the timing and accuracy of various entropy estimation functions for use in `bandit.py`

Monte Carlo integration with quasi-random sampling far outperforms all other integration methods tested. 

## `bandit2.py`
Compare using correct beta distribution prior to using a Gaussian to approximate
