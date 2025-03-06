# Sentinel2TS
Data and code for pre-processed Sentinel-2 time series

You can find the numpy files corresponding to the time series [here](https://drive.google.com/drive/folders/1doHnjryCMptkzxYFfw-ILwAD0tOK3LGH?usp=sharing).

**Associated papers:**

Frion, A., Drumetz, L., Tochon, G., Dalla Mura, M. & Aïssa El Bey, A. (2023). Learning Sentinel-2 reflectance dynamics for data-driven assimilation and forecasting. EUSIPCO 2023. arXiv:2305.03743.

Frion, A., Drumetz, L., Dalla Mura, M., Tochon, G., & Aïssa El Bey, A. (2023). Neural Koopman prior for data assimilation. IEEE Transactions on Signal Processing. arXiv:2309.05317.

Frion, A., Drumetz, L., Tochon, G., Dalla Mura, M., & Aïssa El Bey, A. (2024). Koopman Ensembles for Probabilistic Time Series Forecasting. EUSIPCO 2024. arXiv:2403.06757. 

**Organisation of the repository:**

File "KoopmanAE.py" contains the implementation of the Koopman auto-encoder model discussed in the papers.

Files "Fontainebleau_trained_model.pt" and "Fontainebleau_trained_K.pt" contain weights for the model which gave Sentinel-2 results in the EUSIPCO 2023 paper.

The notebook "Fontainebleau_forecasting.ipynb" shows how the Fontainebleau data can be loaded and forecasted in various ways by our trained model.

The notebook "Fontainebleau_forecasting_benchmark.ipynb" enables to re-train the models and/or reproduce the results associated to the main benchmark of the TSP paper.

The directory "ensembles" contains everything related to the EUSIPCO 2024 paper.

If you like working with scripts rather than notebooks, you can check out this fork: https://github.com/tevkhieu/Sentinel2TS
