# Prediction of color tags

Convolutional Neural Networks (CNNs) can be used to predict color tags. Automated tagging of image files is important for various industries such as e-commerce and sport. 

For training and validation, the [Vehicles Color Recognition (VCoR) dataset](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset) was used, which has 15 colors classes. A separate unseen testset has overall accuracy of about 84%.

One goal of this project was to see if the model can generalizes to other kind of products and objects. For this purpose I have used [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) and several test images show promising results. I am planning to test this more extensively.

Please check also at [Kaggle](https://www.kaggle.com/code/rrighart/the-prediction-of-color-tags) for updates. A [blog](https://www.rrighart.com/blog-color-tags.html) will be published soon.

# An App classifying images on different colors 

An App is running at [HuggingFace](https://huggingface.co/spaces/rrighart/color-tags). 
Image files can be uploaded and the App returns the most probable color(s) in the image.
The App works best for single objects and products. Since the [underlying model](https://www.kaggle.com/code/rrighart/the-prediction-of-color-tags/data) was trained on vehicle data, I am currently testing if it generalizes to different object categories.

![shoes](https://www.rrighart.com/uploads/8/3/7/7/83774724/shoes-cnn-classification_orig.png)

# Contact

Ruthger Righart (PhD)

Self-employed data scientist in machine learning and computer vision

Email: rrighart@googlemail.com

Web: https://www.rrighart.com

