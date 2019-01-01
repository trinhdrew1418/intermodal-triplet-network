# Intermodal Triplet Learning for Cross-modal Retrieval

A PyTorch implementation for an inter-modal triplet network to learn the joint embedding space of both text and
images. An application is cross-modal retrieval where given an image, we obtain the most relevant words and
vice versa.

This particular implementation was trained on the NUSWIDE dataset that contains 81 groundtruth tags for each
image along with noisy user-made tags. 

# Image to Text Example:

For each given image (on the bottom of each list), the 10 nearest words are retrieved using FAISS

![](images/image_to_text/1.png) ![](images/image_to_text/2.png) ![](images/image_to_text/3.png)
![](images/image_to_text/4.png) ![](images/image_to_text/5.png) ![](images/image_to_text/6.png)

# Text to Image Example:

For each text query, the nearest 3 images are retrieved using FAISS

![](images/text_to_image/1.png) ![](images/text_to_image/7.png) ![](images/text_to_image/3.png)
![](images/text_to_image/4.png) ![](images/text_to_image/5.png) ![](images/text_to_image/6.png)
