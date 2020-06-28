# build env for nlp tasks

conda create -y -n nlp_env python=3.8 
source activate nlp_env
conda install -y -c anaconda nltk 
conda install -y -c anaconda pandas 
conda install -y -c anaconda scikit-learn
conda install -y -c anaconda beautifulsoup4
conda install -y -c anaconda lxml 
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge wordcloud 
conda install -y -c conda-forge spacy




conda activate nlp_env
