"Deep Learning-based Analysis of Basins of Attraction"

This repository holds the necessary code to calculate several metrics, such as the fractal dimension, basin entropy, boundary basin entropy, and the Wada property, from a set of previously computed basins of attraction.

The file Basin_predictor.py opens a GUI where a .csv file with the route of the basins must be opened. Then, the program will calculate the basin metrics and create a new .csv file with the basin names and the basin metrics. Finally, the basins can also be visualized in the application. A video demonstrating this process is attached below.

Basin_Metric_Training.py contains the necessary code to train a new ResNet50. Although it is not recommended due to the large number of basins needed. We have provided trained weights for this neural network in the folder results/checkpoints.

The file basin_conda_env and req.txt provide the necessary conda environment and packages to run the code.



