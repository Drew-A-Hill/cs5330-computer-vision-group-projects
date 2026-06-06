# Project 2: Contnet Based Image Retrieval
**Drew Hill, Abhiram Banda**    
CS5330 Pattern Recognition & Computer Vision


## Files Submitted
matcher.cpp: Main matching logic.
csv_helper.cpp: Generates the CSV files for features. 
features.cpp: Feature extraction functions.
features.h: Header file for features functions.
features_helper.cpp: Computes sobel x sobel y and magnitude.
features_helper.h: the header file for the features_helper functions.

## Build
To run this program you need to compile both make and csv_helper then you must run the programs with required arguments. For all except designated you must first must run csv_helper to generate a csv of feature values then run matcher. Below will provide compile and run instructions for each.  

**CSV_Helper**
Compile with: make csv_helper  
Run with the following:
./csv_helper olympus features.csv b  
./csv_helper olympus histogram_features.csv h    
./csv_helper olympus histogram_features_3.csv h3  
./csv_helper olympus mult_hist_1.csv mh  
./csv_helper olympus texture.csv tx  


**Matcher:**  
Compile with: make matcher  
Run with: ./matcher image_path csv_file N method 

**Methods:**  
  b    - Baseline  
  h    - Color histogram matching  
  h3   - 3D RGB histogram matching  
  mh   - Multi histogram matching  
  mh2  - Full and center histogram matching  
  tx   - Texture and color matching  
  rc   - DNN cosine distance   
  dvf  - Compare DNN vs classic  
  lp   - Custom flower matching  
  res2 - DNN SSD distance (Extension 1)   

## Extensions
**Extension1 (res2):** Uses DNN embeding with SSD distance metric rather than using the cosine distance metric. Run with te csv file path ResNet18_olym.csv and method res2. The different outcomes are then compared between using SSD and cosine difference.  

**Extension 2 (GUI):** All methods will automatically display the target image and the top matches in a window. Press any key to close the window.


