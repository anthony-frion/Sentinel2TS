# Sentinel2TS
Data and code for pre-processed Sentinel-2 time series

You can find the numpy files corresponding to the time series here : https://drive.google.com/drive/folders/1doHnjryCMptkzxYFfw-ILwAD0tOK3LGH?usp=sharing

Associated papers :

Frion, A., Drumetz, L., Tochon, G., Dalla Mura, M. & Aïssa El Bey, A. (2023). Learning Sentinel-2 reflectance dynamics for data-driven assimilation and forecasting. EUSIPCO 2023. arXiv:2305.03743.

Frion, A., Drumetz, L., Dalla Mura, M., Tochon, G. & Aïssa El Bey, A. (2023). Neural Koopman Prior for Data Assimilation (submitted pre-print for IEEE Transactions on Signal Processing, not yet publicly available).

File "KoopmanAE.py" contains the implementation of the Koopman auto-encoder model discussed in the papers.

Files "Fontainebleau_trained_model.pt" and "Fontainebleau_trained_K.pt" contain weights for the model which gave Sentinel-2 results in the papers.
