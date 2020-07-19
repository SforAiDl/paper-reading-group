# Confidence-Aware Learning for Deep Neural Networks

[https://arxiv.org/abs/2007.01458](https://arxiv.org/abs/2007.01458)

Jooyoung Moon     Jihyo Kim     Younghak Shin     Sangheum Hwang

---

## Problem Statement

Deep neural networks (DNNs) have an 'overconfidence' issue. They have a tendency to produce predictive probabilities with high confidence, even for incorrect predictions. This makes DNNs unreliable for safety-critical applications.

> The model should know what it does not know.

The model should not only be accurate, but should also indicate when and how likely it is to be wrong.

### An Example:

Modern DNNs are far more accurate relative to their older counterparts (such as LeNet). However, we are much less confident in the confidence of these modern DNNs. To visualize this: 

![https://miro.medium.com/max/625/1*fhvkdS0OebldzKps2soGxA.png](https://miro.medium.com/max/625/1*fhvkdS0OebldzKps2soGxA.png)

On Calibration of Modern Neural Networks by Guo et al.(2017)

## Past Approaches

There are two perspectives with which the problem of evaluating the quality of confidence estimates:

### Confidence Calibration

We attempt to predict probability estimates that directly reflect the true correctness likelihood. In other words, we want to output predictive probabilities that can be directly interpreted as a predictions' confidence levels.

### Ordinal Ranking (focus of the work presented in this paper)

Ordinal ranking aims to estimate confidence values whose ranking among samples are effective to distinguish correct from incorrect predictions. This is often cast into other tasks such as out of distribution detection and failure prediction. In other words, it is the problem of ranking among samples to distinguish correct from incorrect predictions according to confidence estimates. More formally, the following relationship should hold:

![Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled.png](Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled.png)

where  is some confidence function (such as maximum class probability or margin).

Other approaches include Bayesian methods such as Markov Chain Monte Carlo (MCMC) and variational inference. 

**Problem:** Computationally expensive for larger neural networks. Bayesian methods that can get around this still need multiple forward passes for inference.

Non Bayesian methods include directly learning confidence intervals by augmenting  a network's architecture with the relevant values and ensembles of neural networks. 

**Problem:** Also computationally demanding.

## Confidence-Aware Learning

### Hypothesis

The authors hypothesize that the probability of being correct is *roughly* proportional to the frequency of correct predictions during training with SGD-based optimizers. They also observe that easy-to-classify samples are learned earlier than hard-to-classify samples during training. 

![Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%201.png](Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%201.png)

 Based on the above observations, the authors attempt to estimate the probability of being classified correctly through the frequency of correct prediction events during training.

### Correctness Ranking Loss (CRL)

The authors propose CRL , so that the classifier learns the ordinal ranking relationship stated above. More formally, it is defined as:

![Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%202.png](Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%202.png)

where c_i is the proportion of correct event predictions of x_i over the total number of examinations and **g** is defined as:

![Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%203.png](Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%203.png)

Then, the total loss function is defined as:

![Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%204.png](Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%204.png)

where *lambda* is a scaling factor to control the influence of CRL.

### Practical Considerations

The CRL loss should technically be computed over all sample pairs at each model update. To make it computationally feasible, the  authors do it only over each mini-batch at model update.

For the confidence function K, the authors consider three simple functions - maximum class probability, normalized negative entropy, and the margin.

## Experiments and Results

![Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%205.png](Confidence%20Aware%20Learning%20for%20Deep%20Neural%20Networks%205080dd3c7bff4d6bbd44632448b1de31/Untitled%205.png)

AURC - Area Under Risk-Coverage Curve NLL - Negative Log Likelihood FPR - False Positive Rate ECE - Expected Calibration Error

They also conduct experiments on related tasks such as out of distribution detection and show positive results.

## Some Thoughts on The Paper

- Considering the fact that the application of the techniques in this paper are geared most towards safety-critical applications, experiments should have been conducted on real world datasets in place of standard datasets. On the flip side, the authors have conducted plenty of experiments.
- A plus point for the proposed technique is that is computationally much cheaper than its predecessors, and is also very simple to interpret and implement.
- The proposed technique is model agnostic and task agnostic, as it is simply an additional loss function to be added to the standard loss function being used.