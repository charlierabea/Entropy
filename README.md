# Entropy
The orininal code for project "AMD Neovascular activity prediction using OCT-
angiography based on entropy and deep learning" under the College Student Research Scholarship(National Science Council, Taiwan)

The results now assists ophthalmologists at TPEVGH in interpreting OCTA images(patent pending), significantly advancing the field.
> the OCTA_Net model is modified from https://github.com/Luodian/Otter](https://github.com/iMED-Lab/OCTA-Net-OCTA-Vessel-Segmentation-Network.git)https://github.com/iMED-Lab/OCTA-Net-OCTA-Vessel-Segmentation-Network.git

# AMD Neovascular activity prediction using OCT-angiography based on entropy and deep learning
Purpose 
The study aims to explore the potential of incorporating the information science concept of entropy in the classification of eyes with active and inactive age- related macular degeneration (AMD).

Methods 
A total of 35 reactive events and 59 treatment events from 97 follow-ups with AMD were analyzed using OCTA vascular density maps, centerline maps, and foveal avascular zone (FAZ) masks at the superficial capillary plexus (SCP) level. We assessed OCTA metrics, including entropy, vessel density, vessel calibre, vessel tortuosity, FAZ area, and FAZ circularity. Additionally, a supervised machine learning algorithm called the eXtreme Gradient Boost (XGBoost) classifier was developed to categorize images into inactive and active AMD groups.

Results 
Our analysis revealed that the entropy and vessel density of central vessels increased significantly in reactive events. In treatment events, entropy, vessel density, vessel calibre, and vessel tortuosity primarily showed high significance increases. FAZ area and circularity, however, did not reach statistical significance in either event type. The XGBoost classifier demonstrated excellent performance, achieving an accuracy of 0.967, AUROC of 0.967, sensitivity of 0.93, and specificity of 1.00. When the model was constructed without entropy inputs, its performance declined, with an accuracy of 0.867, AUROC of 0.837, sensitivity of 0.95, and specificity of 0.72.

Conclusions 
Our study indicate that incorporating entropy into the evaluation of OCTA metrics may enhance the classification of active and inactive AMD. This improvement could contribute to more accurate diagnoses and better management of the condition.
<img width="731" alt="Screenshot 2024-01-03 at 7 10 39 PM" src="https://github.com/charlierabea/Entropy/assets/100334852/25871da9-2c64-45eb-aae0-d7d8856f7ac5">
