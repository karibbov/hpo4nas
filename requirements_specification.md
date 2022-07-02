# How can we use HPO techniques to find optimal architectures?
## The search space problem:

Let’s try to use SMAC, an automatic generic algorithm configuration tool, which can also optimize the hyperparameters for deep neural networks, to pick optimal architectures instead of the optimal hyperparameters for a specific architecture and dataset combination.

Let’s also use NASBench201, a tabular benchmark of already trained architectures and their performances, to be able to fairly compare optimization techniques and to speed up the search. This way SMAC will not have to train the picked architectures from scratch to see how that architecture would perform. In NASLib the benchmark itself defines the search space, a set of architectures, we would like to optimize over.

There is a problem, however.

SMAC uses ConfigSpace, a package for keeping track of hyperparameters in a search space, and samples hyperparameters from this space based on what it thinks is the most optimal choice. Such a sampled hyperparameter is used to evaluate the performance of a given static architecture that uses those hyperparameters.

However, we are trying to use SMAC for sampling architectures and NOT hyperparameters. So somehow, we have to find a way to convert the description of architectures into a description of hyperparameters written in the configuration space mentioned before. These converted architectures that are now treated as regular hyperparameters, can then be used as the search space for SMAC.
By running SMAC on this search space, we pick an architecture that is optimal according to SMAC. We would then like to evaluate the picked architecture’s performance. Since we do not want to train these architectures from scratch, but instead query them on NASBench201 inside of NASLib, we have to somehow define a way to convert them from a hyperparameter setting in the ConfigSpace into descriptions of architectures that the benchmark accepts as a query argument.

These would be the steps for using SMAC or some other HPO method utilizing config space, to search the optimal architecture:
1. Create a function for converting the set of architectures inside of the NASBench201 search space into hyperparameter settings of a ConfigSpace object. After this step, all architectures can be sampled from a ConfigSpace object by sampling a hyperparameter configuration (setting) from the space. The architectural choices would be represented as categorical hyperparameters. For example, for each edge we could define a categorical hyperparameter. Then add all possible operational choices for that edge as an entry in this categorical hyperparameter
2. Choose an HPO technique that can deal with categorical hyperparameters. We will use SMAC (The default optimizer in SMAC)
3. Search with the chosen HPO technique from step 2 on the ConfigSpace defined in step 1. This will return an architecture configuration encoded as a hyperparameter setting sampled from the ConfigSpace
4. Convert this sampled architecture into a representation that can be used to query the NASBench201 benchmark
5. Query the benchmark with the selected architecture
6. The result of the query will be the train and validation performance that this architecture would have if we trained it for a certain number of epochs.
7. Repeat from step 3. until our budget runs out. Budget could be as simple as the maximum amount of time (~epochs) we would like to spend on searching for a good architecture.

To summarize, our goal is to use HPO to optimize a neural network’s architecture on a certain dataset. This poses a few problems, which can be tackled by defining two functions for converting between two different data structures. In short, there is a type mismatch in the search space that NASBench201 defines and in the one that the optimizer searches over.
The dataset we will use is CIFAR10 at first. The found architecture should perform relatively well on another similar dataset, like ImageNet.

## Comparing several optimizers and interpreting their results

In the following, the optimizers DEHB, SMAC, and RE will be compared. To help better understand the final results, we will use DeepCAVE.

SMAC is the only optimizer that works out of the box with DeepCAVE. For the others we need to find a way to convert the outputs produced by these optimizers into a format interpretable by DeepCAVE.
