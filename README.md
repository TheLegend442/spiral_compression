# Spiral Compresssion Analysis

This repository contains the code and data relevant to final version of my diploma thesis titled: **"Training an Explainable Model of Spiral Compression in Spirography"**.

**Abstract:**<br>
Parkinson’s disease is a slowly progressive neurological disorder in which brain cells gradually die off. An accurate assessment of a patient’s condition is essential for appropriate therapy. One test used to evaluate symptom severity is digital spirography, where the subject is asked to draw an Archimedean spiral. Any anomalies that appear during drawing (e.g., partially straight segments, tightening/compression, jaggedness) reflect symptoms of parkinsonism. In this thesis, we focus on detecting spiral compression in an explainable, human-understandable way. We developed a transformer-based model. As is often the case in medicine, we did not have access to a sufficient amount of labeled data. Therefore, we generated a synthetic dataset and parameterized the anomalies based on their interpretations. We evaluated the model on a test set and via an interactive web interface developed for this purpose, which enables spiral drawing and spiral-compression assessment using our model.

## Repository content:
Repository is organized into 4 folders:
<ul>
<li><i>/src</i> - contains all fundamental scripts used to generate data, train the model and analyse the results.</li>
<li><i>/data</i> - contains train and test data (generated spirals as '.npz' files).</li>
<li><i>/models</i> - contains '.md' file with training descriptions and trained model weights.</li>
<li><i>/results</i> - contains plots of final results.</li>
</ul>

Scripts in <i>/src</i> folder:
<ul>
<li><i>spiral_generation.py</i> - script used to generate artificial spirals - it intakes spiral type, amount to generate and save path. Then it generates the artificial spirals used as my train and test data.</li>
<li><i>chronos_emb.py</i> - helper script to perform spiral embedding using Chronos-2 model.</li>
<li><i>transformer.py</i> - script containing transformer model definition, dataset definition and training process. It is used to train the model with specified parameters on specified dataset.</li>
<li><i>inference.py</i> - script that performs the inference with specified model and saves some basic plots. It can also calculate loss on a test set.</li>
<li><i>helper_functions.py</i> - helper script containing some functions common to multiple scripts to avoid mismatching versions.</li>
</ul>

### Note:
The code in this repository is an assembly of scripts that were used to research and write my diploma thesis. I cannot guarantee it's coherency but it should work with minimal corrections (paths, script names, etc.)<br>
To start training the model data should be generated as it was not uploaded here due to it's size.