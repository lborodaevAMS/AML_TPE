# AML_TPE

Evaluation of Meta-learning initialization(MI) for Sequential Model-Based Optimization(SMBO) with Tree Parzen Estimator(TPE).
Roughly follows [Feurer et al. 2015].

Works as follows:
1. Downloads the datasets used in the paper from OpenML
2. Extracts metafeatures
3. Random Search for best hyperparameters
4. Evaluate MI-TPE and TPE in leave-one-dataset-out fashion with warm starting MI-TPE on the data from step 3
