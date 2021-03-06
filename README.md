# HPO for NAS
Repository for the deep learning project.

The aim of this project is to find answers and solutions to the following 
questions and tasks related to Hyper Parameter Optimization (HPO) and Neural Architecture Search (NAS).

<ul>

<li> How well can traditional HPO optimizers
like SMAC perform for NAS. Can we identify
under what conditions and properties it can
outperform other methods provided by NASLib?

<li> Main Task:
Identify where HPO based NAS
is strongest, using SMAC on the
NASBench201 search space
for CIFAR10.
Use DeepCave to analyse and
Highlight these strengths.

</ul>

*Other questions raised during the project will also be discussed.*

## Setup

Install python 3.9 inside of a virtual environment (Conda recommended)

Clone NASLib inside this directory with the following command:<br>
git clone https://github.com/automl/NASLib.git

Switch to the NASLib directory that you just cloned and follow the setup guide of NASLib:<br>
https://github.com/automl/NASLib

Preferably switch to the Develop tree inside of NASLib
