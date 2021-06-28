# :brain: Psychological Analysis of Handwriting using Machine Learning
The objective of this project is to establish a system which takes an input image of the handwriting and outputs his/her personality traits selected from a few handwriting features.

# Motivation
Graphology is defined as the analysis of the physical characteristics and patterns of the handwriting of an individual to understand his/her psychological state at
the time of writing. Accuracy of the predictions of personality depends on how much skilled the analyst is. 
Human intervention has proven to be effective but it is time consuming and error prone so this method focuses on developing a tool to analysis personality of an individual using computer vision and machine learning.

# Tech/Framework used
- Python
- OpenCV
- Machine Learning/Deep Learning

# Input & Output
![](/uploads/I-O.PNG)
# Features
The proposed methodology extracts seven handwriting features namely:
- top margin
- pen pressure
- baseline angle
- letter size
- line spacing
- word spacing 
- slant angle

# Visualizations
![](/uploads/results.PNG)

The combination of above mentioned seven features/vectors are used to predict the personality of the individual from the eight selected personality traits:
- Emotional Stability
- Mental Energy or Will Power
- Modesty
- Personal Harmony and Flexibility
- Lack of Discipline
- Poor Concentration Power
- Non-communicativeness
- Social Isolation

# Dataset Reference
The [dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) was taken from the Research Group on Computer Vision and Artificial Intelligence. The IAM Handwriting Database contains forms of handwritten English text which can be used to train and test handwritten text recognizers. The database contains forms of unconstrained handwritten text, which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.

