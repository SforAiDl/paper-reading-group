### *A new method that views object detection as a direct set prediction problem. The approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel.  DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster RCNN baseline on the challenging COCO object detection dataset. 
Training code and pre-trained models are available at:
[https://github.com/facebookresearch/detr.](https://github.com/facebookresearch/detr)*

# About previous detectors:

Most of the current works including Faster RCNNs and YOLO consist of some post-processing techniques such as removing near duplicated with NMS (Non-Max suppression).

They also consist of initial pre-computation of anchor boxes.

DETR comes up with an alternative pipeline for object detecting removing the need for anchor boxes and non max suppression. 

# DETR Pipeline:

**Learning Objective :**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/89760352-23f5-47de-9441-fd341a295f96/Screenshot_from_2020-06-07_01-17-55.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/89760352-23f5-47de-9441-fd341a295f96/Screenshot_from_2020-06-07_01-17-55.png)

Here N denotes the total predictions that your model will make and is chosen to be a typically large value ( For example N=100 ).  

**Summary of pipeline:**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1380873d-b598-4af0-9817-128704f306c7/Screenshot_from_2020-06-07_01-11-53.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1380873d-b598-4af0-9817-128704f306c7/Screenshot_from_2020-06-07_01-11-53.png)

# Bipartite Matching:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/421bb627-b1f3-4f54-8924-08c346f2c904/Screenshot_from_2020-06-07_01-27-59.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/421bb627-b1f3-4f54-8924-08c346f2c904/Screenshot_from_2020-06-07_01-27-59.png)

# Hungarian algorithm:

Hungarian algorithm is used to solve the assignment problem. Suppose there are 3 potholes and 3 workers and we know the cost it takes for each worker to reach every pothole, what would be the best assignment such that the total cost is minimized? This can be solved with the Hungarian algorithm.

# Implementation details:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/88dd8f96-fa3b-470f-822e-fd5ffde639b6/Screenshot_from_2020-06-07_01-11-59.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/88dd8f96-fa3b-470f-822e-fd5ffde639b6/Screenshot_from_2020-06-07_01-11-59.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/67a27105-41ed-46d3-9258-407ff0cd6efe/Screenshot_from_2020-06-07_01-34-03.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/67a27105-41ed-46d3-9258-407ff0cd6efe/Screenshot_from_2020-06-07_01-34-03.png)

### **Loss function :**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4046e895-bc2c-4ea2-8dc9-b0128e42f056/Screenshot_from_2020-06-07_01-15-34.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4046e895-bc2c-4ea2-8dc9-b0128e42f056/Screenshot_from_2020-06-07_01-15-34.png)

where L_box is:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a1f98ff-ee61-49ce-a7d2-163fb342bbe7/Screenshot_from_2020-06-07_01-16-02.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a1f98ff-ee61-49ce-a7d2-163fb342bbe7/Screenshot_from_2020-06-07_01-16-02.png)

# Observations:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c8cb115d-fcaa-4222-9614-5cb497a5ef44/Screenshot_from_2020-06-07_01-12-11.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c8cb115d-fcaa-4222-9614-5cb497a5ef44/Screenshot_from_2020-06-07_01-12-11.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a2fb6871-5770-4506-98ac-7daed9d94dd3/Screenshot_from_2020-06-07_01-12-19.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a2fb6871-5770-4506-98ac-7daed9d94dd3/Screenshot_from_2020-06-07_01-12-19.png)

# Results:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7446973e-a0e2-4a42-8862-b6c7b186a573/Screenshot_from_2020-06-07_01-12-05.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7446973e-a0e2-4a42-8862-b6c7b186a573/Screenshot_from_2020-06-07_01-12-05.png)

### Concluding Remarks:

DETR doesn't perform as great as compared to Faster RCNN on smaller objects. However it performs better on larger objects.

Larger training time and computationally heavy.

Attention in object detection pipeline improves performance as can be looked at from the Non-Local neural networks example.

There are plenty of other drawbacks and a lot of room for improvement.