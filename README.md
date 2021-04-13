# Spatio-Spectral Feature Representation for Motor Imagery Classification using Convolutional Neural Networks

## Journal Publication

**J.-S. Bang, M.-H. Lee, S. Fazli, C. Guan, S.-W. Lee, "Spatio-Spectral Feature Representation for Motor Imagery Classification using Convolutional Neural Networks," IEEE Trans. on Neural Networks and Learning Systems, 2021**
https://ieeexplore.ieee.org/document/9325918



## Citing
When using this code in a scientific publication, please cite us as:

>@article{bang2021spatio,

  title={Spatio-spectral feature representation for motor imagery classification using convolutional neural networks},
  
  author={Bang, Ji-Seon and Lee, Min-Ho and Fazli, Siamac and Guan, Cuntai and Lee, Seong-Whan},
  
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  
  year={2021},
  
  publisher={IEEE}
}


## Modules
 We provide Matlab and Python implementation for generating feature representation and classification.

 Belows are the description of each file.
 

 - 'Filterset_OpenBMI.m' : Code for generating filter-set. 

 - 'FeatureRepresentation_OpenBMI.m' : Code for generating feature representation.

 - 'Classification_OpenBMI.py' : Code for classification.

 - 'func_mutual_information2.m', 'prep_filterbank2.m' : The following two files are slightly modified versions from the original toolbox.

 - 'electrode_position.mat' : Data indicating the location of channels. Used for local average reference function.

## Requirements
Matlab R2017a or later

Python 3 

 - tensorflow-gpu == 1.11.0
 - numpy >= 1.17.5
 - scipy >= 1.1.0
 - scikit-learn >= 0.22.2

## Dataset Reference
Please refer to http://gigadb.org/dataset/100542 to download OpenBMI dataset.


## Toolbox Reference
Please refer to https://github.com/PatternRecognition/OpenBMI and 
https://github.com/bbci/bbci_public to download the toolbox that we used.


## License
[Apache License 2.0]

