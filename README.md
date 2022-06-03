# Rice_Classification

### Final notebooks
You can find final notebooks in the following files:
* Images: `code/prediction_from_images/eval.ipynb`
* Tables: `code/prediction_from_table/Zajęcia_26_05.ipynb`
###  Environment
Once in virtual environment
```bash
pip install -r requirements.txt
```

### Training
```bash
cd code
python ./prediction_from_images/train.py
```
You can pass different than default arguments while running command above. To list 
them run
```bash
python ./prediction_from_images/train.py -h