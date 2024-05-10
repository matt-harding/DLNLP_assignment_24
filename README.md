# DLNLP_assignment_24
DLNLP_assignment_24 submission

## Project Structure

The project is broken up into six folders

* **Datasets** : Contains the cropped training images in a folder called train_images and the labelling data in a file called train.csv
* **Reference**: Documentation on the assignment task
* **Report**: LaTex files for report
* **Utils**: Holds the custom Torch Dataset class WhaleDataset. Could be expanded to hold other util classes

**main.py** is the entry point to run the data preprocessing and . 




## Run Locally 

Poetry was for dependency management. Build using Python 3.10. Not tested on other versions of Python.

``` poetry shell ```

``` poetry install ```

``` python main.py ```


Don't have Poetry? I have generated a requirements.txt via 
```poetry export --without-hashes --format=requirements.txt > requirements.txt ``` 

