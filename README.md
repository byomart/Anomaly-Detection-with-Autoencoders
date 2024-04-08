# Anomaly Detection with Autoencoders

The objective pursued by this project is anomaly detection, a crucial task in cybersecurity aimed at identifying suspicious or malicious activities within network traffic data. To accomplish this, we utilize the UNSW-NB15 dataset, which comprises raw network packets captured by the IXIA PerfectStorm tool in the Cyber Range Lab of UNSW Canberra.

The UNSW-NB15 dataset offers a unique blend of real modern normal activities and synthetic contemporary attack behaviors, making it an invaluable resource for training anomaly detection models. It captures a diverse range of network traffic, encompassing nine types of attacks: Fuzzers, Analysis, Backdoors, DoS (Denial of Service), Exploits, Generic, Reconnaissance, Shellcode, and Worms.

Our project involves developing anomaly detection algorithms capable of discerning normal network activities from potentially malicious ones. By leveraging machine learning and deep learning techniques, we aim to build robust models capable of accurately identifying and flagging anomalies in network traffic data, thereby enhancing cybersecurity measures.

## Index

### Exploratory Data Analysis (EDA) and Preprocessing
The initial step involves exploring the dataset to gain insights into its structure and characteristics. The log file contains the results of this exploration, providing valuable information for further analysis. Additionally, categorical variable encoding and dataset normalization are performed to prepare the data for subsequent processing steps. Moreover, a correlation study is conducted to examine the relationships between different variables.

As an illustration, the images below showcase both the distribution of variables using a pie chart and the correlation analysis based on the correlation matrix. These visualizations offer valuable insights into the dataset's composition and inter-variable relationships, aiding in the understanding of its underlying patterns and potential anomalies.

<p align="center">
  <img src="https://github.com/fbayomartinez/Anomaly-Detection-with-Autoencoders/blob/e899ee658cc16572d42ff9bea114d2b86c55a0e5/images/attack_pie_chart.png" alt="Texto alternativo" width="550">
</p>

<p align="center">
  <img src="https://github.com/fbayomartinez/Anomaly-Detection-with-Autoencoders/blob/2caf3d74640907e09dd5f5f65ffc0fb826e4a432/images/correlation_heatmap.png" alt="Texto alternativo">
</p>

<p align="center">
  <img src="https://github.com/fbayomartinez/Anomaly-Detection-with-Autoencoders/blob/2caf3d74640907e09dd5f5f65ffc0fb826e4a432/images/attack_cat_correlation_matrix.png" alt="Texto alternativo">
</p>

### Autoencoder for Anomaly Detection

As you may already known, an autoencoder is a type of neural network used for unsupervised learning of data representations. It consists of two main parts: the encoder and the decoder. The encoder takes an input and compresses it into a lower-dimensional representation called a latent code. Then, the decoder takes this latent code and tries to reconstruct the original input.

In the context of anomaly detection, the autoencoder is primarily **trained on normal or non-anomalous data**. Once trained, the autoencoder can be used to reconstruct new samples. If the difference between the original input and the reconstruction is significant, it can be inferred that the sample is an anomaly.

The autoencoder adapts to the normal data characteristics during training and therefore cannot accurately reconstruct anomalies that significantly differ from normal samples. Hence, **anomalies tend to have higher reconstruction errors**, allowing them to be detected.


<p align="center">
  <img src="https://github.com/fbayomartinez/Anomaly-Detection-with-Autoencoders/blob/2caf3d74640907e09dd5f5f65ffc0fb826e4a432/images/loss.png" alt="Texto alternativo">
</p>

<p align="center">
  <img src="https://github.com/fbayomartinez/Anomaly-Detection-with-Autoencoders/blob/2caf3d74640907e09dd5f5f65ffc0fb826e4a432/images/loss_distributions.png" alt="Texto alternativo">
</p>

<p align="center">
  <img src="https://github.com/fbayomartinez/Anomaly-Detection-with-Autoencoders/blob/2caf3d74640907e09dd5f5f65ffc0fb826e4a432/images/combined_loss_distributions.png" alt="Texto alternativo">
</p>

<p align="center">
  <img src="https://github.com/fbayomartinez/Anomaly-Detection-with-Autoencoders/blob/2caf3d74640907e09dd5f5f65ffc0fb826e4a432/images/anomaly_detect.png" alt="Texto alternativo">
</p>

