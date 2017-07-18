## Preprocessing

Since the model doesn't waste memory on loading the full dataset in memory, it is not practical to perform advanced preprocessing via tensorflow methods. That's why all files must be preprocessed in advance before being fed to the model. 

The model currently supports processing of the ConLL-2003 dataset

The model should automatically apply the needed preprocessing (applies regex, vocabulary generation, meta data, etc.) and generate "cleaned" files before training initiation.

Alternatively you can use the following predefined script

```
# Navigate back to the root directory
cd $PROJECT_DIR

# run the preprocessing script
python3.5 preprocess.py
```

If you make changes which impact the preprocessing (change in vocabulary size or max sentence length), you should manually remove all the previously preprocessed files as the model avoids recreation of already preprocessed files.

