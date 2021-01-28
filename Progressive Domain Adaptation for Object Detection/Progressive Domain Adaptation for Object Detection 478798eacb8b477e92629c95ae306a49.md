# Progressive Domain Adaptation for Object Detection

Author:  Chun-Han Yao, Han-Kai Hsu, Hung-Yu Tseng, Maneesh Singh, Ming-Hsuan Yang, Wei-Chih Hung, Yi-Hsuan Tsai
Link: https://arxiv.org/pdf/1910.11319v1.pdf
Publishing/Release Date: Oct 24, 2019
Status: Finished

## Motivation **and basic approach**

- Recent deep learning models for object detection rely on large amount of bounding box annotations , which is time consuming.
- Supervised models do not generalize well when testing on images from a different distribution. ( scenes, weather, lighting conditions and camera settings )
- Unsupervised Domain Adaptation.  Adapting existing labels to target testing data.

      But, what if there is a large gap between domains? ( domain-shift problem)

- Bridge the domain gap with an **intermediate domain ( translating source image to mimic target )**  and progressively solve easier adaptation sub-tasks.
- To tackle the domain-shift problem, adopt adversarial learning to align distributions at the feature level.
- In addition, **a weighted task loss** is applied to deal with unbalanced image quality in the intermediate domain.To reduce the outlier impact of the low-quality translated images, we
propose a weighted version in our adaptation method, where the weight is determined based on the distance to the target distribution
- we obtain the distance from the discriminator in the image translation model and incorporate it into the detection framework as a weight in the task loss.

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled.png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled.png)

**Note**: Conventional domain adaptation problem : From source to target.

**New approach** : intermediate synthetic domain that allows us to gradually solve separate sub-tasks with smaller gaps (shown as L: S→F and L : F→T). In addition, we treat each image in the synthetic domain unequally based on its quality with respect to the target domain, where the size of the yellow triangles stand for their weights (i.e., the closer to target, the higher of the weight).

1) we introduce an intermediate domain in the proposed adaptation framework to achieve progressive feature alignment for object detection

 2) we develop a weighted task loss during domain alignment based on the importance of the samples in the intermediate domain.

 3) we conduct extensive adaptation experiments under various object detection scenarios and achieve state-of-the-art performance.

## Previous Work on object detection

- **Object detection based on deep CNN** , categorized into Region proposal-based and single-shot detectors. Fast R-CNN , **Faster R-CNN (RPNs)** and single-shot approaches.
- Trade off : Achieve state-of-the-art performance but require large amount of labelled data and might over-fit to training domain.
- **Domain Adaptation for object detection.** To target domains with unlabeled or weakly labeled images in the target domain.
- Domain Adversarial Neural Network (**DANN**), numerous works have been
proposed to utilize adversarial learning for the feature distribution alignment between two domains.

[https://arxiv.org/pdf/1409.7495.pdf](https://arxiv.org/pdf/1409.7495.pdf)

### DANN

- neural network architectures that are trained on labeled data from the source domain and unlabeled data from the target domain .
- As the training progresses, the approach promotes the emergence of features that are (i) discriminative for the main learning task on the source domain and (ii) indiscriminate with respect to the shift between the domains.
- this adaptation behaviour can be achieved in almost any feed-forward model by augmenting it with few standard layers and a new gradient reversal layer.

## GRL ( Gradient reversal layer )

For training the feature extractor in order to maximize the classification loss of domain predictor, Gradient Reversal layer was place between Feature extractor and domain classifier. 

The Gradient Reversal Layer basically acts as an identity function (outputs is same as input) during forward propagation but during back propagation it multiplies its input by -1. — Leads to Gradient ascent.

- PixelDA - synthesizes additional images in the target domain by learning one-to-many mapping.
- CyCADA  and AugGAN  both design a **CycleGAN** - like network to transform images from the source domain to the target one. These transformed images are treated are training image for target domain. — Does not perform alignment in semantic/pixel levels.

### CycleGAN -  [https://arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593)

- Previous GANs - utilized supervised training, where we have access to (x, y) pairs of corresponding images from the two domains we want to learn to translate between.
- CycleGAN- unpaired image to image translation and unsupervised.
- CycleGAN is a Generative Adversarial Network (GAN) that uses two generators and two discriminators.
- CycleGAN objective function, an **adversarial loss [ generator fools discriminator ]** and a **cycle consistency loss [ F(G(x)) ≈ x and G(F(y)) ≈ y ].**

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%201.png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%201.png)

Some results by using CycleGAN:

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%202.png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%202.png)

- Kim et al. utilize image translation network to generate multiple domains and use a multi-domain discriminator to adapt all domains simultaneously, but this method does not consider the distance between the generated ones and the final target.
- In this work, we observe that simply applying image translation without knowing the distance between each generated sample and the target domain may result in less effective adaptation.
- Hence ,an intermediate domain to reduce the effort of mapping two significantly different distributions and then adopt a two stage alignment strategy with sample weights to account for the sample quality.

## Progressive Domain Adaptation

### Adaptation in the Feature Space

To align distributions in the feature space, we propose a deep model which consists of two components; a detection network and a discriminator network for feature alignment via adversarial learning.

- Detection Network

    [](https://arxiv.org/pdf/1506.01497.pdf)

    - Faster R-CNN is used for object detection.

    ![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%203.png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%203.png)

    - Faster Rnn  - a base encoder network E to extract image features. Given an image I, the feature map E(I) is extracted and then fed into two branches: Region Proposal Network
    (RPN) and Region of Interest (ROI) classifier.
    - **L_detection(E(I)) = L_rpn + L_cls + L_reg**

    where L_rpn, L_cls, and L_reg are the loss functions for the RPN, classifier and bounding box regression, respectively.

- Domain Discriminator
    - Discriminate whether the feature E(I) is from the source or the target domain.
    - The probability of each pixel belonging to the target domain is obtained as
    P = D(E(I)) ∈ R (H×W).
    - d=0 : source distribution. d=1: target.
    - Discriminator loss function , (compare with min-max loss in GAN ) —> ( binomial cross-entropy with number of classes 2).

        ![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(43).png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(43).png)

- Adversarial learning
    - Using the Gradient Reverse Layer (GRL) proposed in [6] to learn the domain-invariant feature E(I)
    - GRL is placed in between the discriminator and the detection network, only
    affecting the gradient computation in the backward pass.
    - During backpropagation, GRL negates the gradients that flow through. As a result, the encoder E receives gradients that force it to update in an opposite direction which
    maximizes the discriminator loss. This allows E to produce features that fools the discriminator D while D tries to distinguish the domain of the features.

    - overall min-max loss function of the adaptive detection model-

    ![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(44).png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(44).png)

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(41).png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(41).png)

### **Intermediate Domain**

- Intermediate domain is created using an image to image translation network, CycleGAN to learn a function that maps the source domain images to the target ones.
- synthetic target images - translation from source images to the target domain
- Often this synthetic target images are used to augment training data, but in this paper it is used as a separate domain F, to connect S and T (unlabeled target) domains.

    ![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(45).png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(45).png)

    Similarity between source domain S and F is the image content, only diverging in the visual appearances, while F and the target domain T are different in image details but have similar distributions on the pixel level.

### Adaptation Process

- At the first stage, we use S as the labeled domain, adapting to F without labels. Due to the underlying similarity between S and F in image contents, the network focuses on aligning the feature distributions with respect to the appearance difference on the pixel-level
- After aligning pixel discrepancies between S and F, we take F as the source domain for supervision and adapts to T as stage two in the proposed method. During this step, the model can take advantage of the appearance invariant features from the first step and focus on adapting the object and context distributions.

### Weighted Supervision

- some images fail to preserve details of objects or contain artifacts when translated, and these failure cases may have a larger distance to the target distribution.
- For eg: check t-sne above , to notice outliers far from both source and target.
- the paper propose an important weighting strategy for synthetic samples based
on their distances to the target distribution.
- We obtain the weights by taking the predicted output scores from the target domain discriminator D_cycle.

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(47).png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(47).png)

I - synthetic target image, p(T) - probability of being in target

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(48).png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(48).png)

The final weighted loss function - 

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(51).png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(51).png)

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(52).png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Screenshot_(52).png)

## Implementation details

- Adopt VGG16  as the backbone for the Faster R-CNN detection network.
- We design the discriminator network D using 4 convolution layers with filters of size 3 × 3. The first 3 convolution layers have 64 channels, each followed by a leaky ReLU .
- final domain classification layer has 1 channel that outputs the binary label prediction. Our synthetic domain is generated by training CycleGAN on the source and target domain images.

## Datasets

- KITTI
- Cityscapes - We use Cityscapes with then KITTI dataset is used to evaluate the cross camera adaptation
- Foggy Cityscapes - for cross weather adaptation.
- BDD 100K - to test adaptation to a larger dataset

## Results

We show a baseline Faster R-CNN result trained on the source data without applying
domain adaptation, and a supervised model trained fully on the target domain data (oracle) to illustrate the existing gap between domains.

Cross camera adaptation

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%204.png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%204.png)

Weather Adaptation

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%205.png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%205.png)

Adaptation from small to larger dataset

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%206.png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%206.png)

The result -

![Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%207.png](Progressive%20Domain%20Adaptation%20for%20Object%20Detection%20478798eacb8b477e92629c95ae306a49/Untitled%207.png)

## **Discussion**

To improve precision, could we use a image translation network to generate multiple domains and use a multi-domain discriminator to adapt all domains simultaneously [ Kim et al. ] , and add to this a weighted loss function that considers the distance between the synthetic and target domain?