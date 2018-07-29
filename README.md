# Supervised Machine Learning for Click Fraud Detection


Shan Dou<br>
MLND capstone project<br>
July 2018

Link to proposal review: https://review.udacity.com/#!/reviews/1314525

---
## 1. Data source:
Kaggle competion "[TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)".


----
## 2. Create conda environment with all dependencies
```
conda env create -f environment.yml
```

This operation will create a conda environment named `mlnd_clean`. If you wish to use a different name, please open the requirement file `environment.yml` and change the first line `name: mlnd_clean` into your preferred name.

---
## 3. Activate and deactivate conda environment
Once all the dependencies are installed, please run the following command in your shell terminal to activate the environment
```
source activate mlnd_clean
```

To deactivate, type
```
source deactivate
```

----
## 4. Key modules and availabilities
* The following modules are installed with `conda`

```
1. numpy
2. pandas 
3. seaborn
4. sklearn
5. xgboost
6. lightgbm
7. imblearn
8. notebook
```
* Module for stack ensemble is installed with `pip`:

```
9. mlens
```
For more information about mlens, please visit [its webiste](http://ml-ensemble.com/).

----
## 5. Key components of the project
1. Jupyter notebooks:
	* `MLNDcapstone_shandou_main.ipynb`: Main workbook
	* `MLNDcapstone_shandou_robustness.ipynb`: Companion workbook for models' robustness texts

2. Python models:
	* `preprocessing.py`: data processing
	* `modeling.py`: modeling
	* `utils.py`: miscellaneous tasks such as visualization and generating result summary tables

3. Dataset:<br>
The raw data `train.csv` can be directly download from [Kaggle](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data). Out of file size concerns, only downsized training data and the original testing data are included in this repo.
	* `train_sample.csv`: 0.1% of the raw click records
	* `train_sample_2.csv`: 0.2% of the raw click records
	* `test.csv`: First 10 lines of the orignal test data downloaded from Kaggle.
	NOTE that `test.csv` provided by Kaggle is only used for checking data fields. In the actual implementation, testing data is instead a portion of `train_sample.csv` or `train_sample_2.csv`


4. Proposals and reports:
	* `proposal.pdf`: Proposal of the capstone project
	* `proposal_review.pdf`: Comments from proposal review
	* `report.pdf`: Report of the capstone project
5. Others:
	* folder `images/`: contains all the images used in the report
	* matplotlib style sheet `stylelib/custom.mplstyle`: dataviz styler used throughout this project