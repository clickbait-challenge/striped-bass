# striped-bass
The Striped Bass Clickbait Detector

## Used Libraries

* xgboost
* sklearn
* json_lines
* pandas


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