# AIorReality: Image Classifier for Recognizing AI-Generated Images Based on SVM



https://github.com/123sghn/AIorReality/assets/119744172/661fe822-389f-4fa2-9f59-792e669472fd



## Source Code location Guide：

To facilitate users, we provide a brief guide that users can use to identify the location of the given part of the source code in AIorReality. The table is as follows:

<div align=center><b>Table 1: Source Code Location Guide for AIorReality</b></div>


| Main Directory        | Subdirectory      | Content                                                      |
| --------------------- | ----------------- | ------------------------------------------------------------ |
| datasets              | AI_from_kaggle    | Store AiArtData and RealArt                                  |
| model_saved           | --                | Pre-trained models are used for real-time detection          |
| Result                | Grid_search       | Stores the results of the grid search, including logs and heat maps |
|                       | Train             | Stores the results of model training, including logs, loss trends, and confusion matrices |
| temp_picture          | --                | Used to store the images captured in the video and predict the results of the pictures |
| test                  | --                | Used to get footage for still video                          |
| util                  | image_to_video.py | Generate a still video from the image                        |
|                       | logo.ico          | UI logo                                                      |
|                       | output.mp4        | Generated static video                                       |
|                       | rename.py         | Used for recoding data sets                                  |
| Classifier.py         | --                | It is used for classifier construction and model training    |
| Feature_extraction.py | --                | 4 ways of feature extraction                                 |
| Grid_search.py        | --                | Perform a grid search to obtain optimal hyperparameters      |
| main.py               | --                | UI interface: Used for visual classification effect display  |
