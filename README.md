# striped-bass
The Striped Bass Clickbait Detector

## Used Libraries
* Python v 3.7.2

### Additional python libraries
* xgboost v. 0.82
* sklearn v. 0.20.3
* pandas v. 0.24.1


## Training
```python
python train.py trainDir
```
`trainDir` points to the directory with training data, containing both an `instances.jsonl` and a `truth.jsonl`.

## Test
```python
python runClassifier.py -i inputDir -o outputDir -c xgboost:randomforrest
```

To run the classsifier specify the `inputDir` containing `instances.jsonl`, the outputDir where `results.jsonl` will be created, and the one of the classifiers `xgboost` or `randomforrest`


## Local evaluation
```python
python evaluate.py dir
```
Evaluate both classifiers, `dir` is an optional input to compute the features from. If not specified the current available features file is used.