The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using  
`pip install -r requirements.txt`  
To start the program  
`python3 main.py`  
Config file contains  
```
model_transformer = 'all-mpnet-base-v2'
num_of_cores = 4
path = 'SZPJ_SP1_collection/documents'
output_name = "output.txt"
path_query = 'SZPJ_SP1_collection/query_devel.xml'
```
