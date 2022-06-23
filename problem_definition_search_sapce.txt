How can we use HPO techniques to find optimal architectures?

Let’s try to use SMAC, an automatic generic algorithm configuration technique, which can also optimize the hyperparameters for any deep neural network, for picking optimal architectures instead of hyperparameters.

Let’s also use NASBench201, a tabular benchmark for deep neural network architectures, that given an architecture returns the performance of it, to speed up the optimization procedure of SMAC. This way SMAC will not have to train the picked architectures from scratch to see how that architecture would perform. This benchmark would also define the search space we would like to optimize over.

There are some challenges that we will have to face in such a setup.

First of all, SMAC uses ConfigSpace, a package for keeping track of hyperparameters in a configuration space, and samples hyperparameters from this space based on what it thinks is the most optimal choice. Such a sampled hyperparameter is used to evaluate the performance of a given static architecture that uses those hyperparameters.

However, we are trying to use SMAC for sampling architectures and NOT hyperparameters. So somehow, we have to find a way to convert the description of architectures into a description of hyperparameters written in the configuration space mentioned before. These converted architectures that are now treated as hyperparameters, can then be used as the search space for SMAC.
By running SMAC on this search space, we pick an architecture that is optimal according to SMAC. We would then like to evaluate the picked architecture’s performance, and here is where the second challenge comes into play. Namely, since we do not want to train these architectures from scratch, but instead query them on NASBench201 inside of NASlib, we have to somehow define a way to convert them from a hyperparameters setting in the ConfigSpace into descriptions of architectures that the benchmark accepts as a query argument. So basically the reverse of what we did before.

These would be the steps for using SMAC or some other HPO method utilizing config space, for searching the optimal architecture:
    1. Create a function for converting the set of architectures inside of the NASBench201 search space into hyperparameter settings inside of a ConfigSpace object. After this step, all architectures can be sampled from a ConfigSpace object by sampling a hyperparameter configuration (setting) from the space. The architectural choices would be represented as categorical hyperparameters. For example, for each edge we could define a categorical hyperparameter. Then add all possible operational choices for that edge as an entry in this categorical hyperparameter
    2. Choose a HPO technique that can deal with categorical hyperparameters. We will use one defined in SMAC
    3. Search with the chosen HPO technique from step 2 on the ConfigSpace defined in step 1. This will return an architecture configuration encoded as a hyperparameter setting sampled from the ConfigSpace
    4. Convert this sampled architecture into a representation that can be used to query the NASBench201 benchmark
    5. Query the benchmark with the selected architecture
    6. The result of the query will be the train and validation performance that this architecture would have if we trained it for a certain number of epochs.
    7. Repeat from step 3. until our budget runs out. Budget could be as simple as the maximum amount of time (~epochs) we would like to spend on searching for the best architecture.

To summarize, our goal is to use HPO to optimize a neural network’s architecture for a certain dataset. This poses a few problems, which can be tackled by defining two functions for converting between two different data structures. In short, there is a type mismatch in the search space, which is the main source of the issue.
The dataset we will use is CIFAR10. The found architecture should perform well on another similar dataset, like ImageNet.


