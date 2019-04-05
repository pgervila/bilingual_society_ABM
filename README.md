## SIMULATION OF BILINGUAL SOCIETIES
#### An agent-based model to simulate, explore and understand long-term linguistic trends in bilingual societies
#### A tool to predict sustainability of minority languages

##### How to use it:
##### 1) Set following env variable in command line (in order to ensure repeatability of unordered data structures): 
#####    _export PYTHONHASHSEED=0_
##### 2) From root directory in a python session:
#####    _from bilangsim import BiLangModel_
#####    _model = BiLangModel(400, num_clusters=2)_
#####    _model.run_model(5)_, to run just 5 steps
