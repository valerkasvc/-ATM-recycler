# bicc_recyclers

# End-to-end Pipeline


    DIS - demand - out
    CIN - offer  - in
____________________

* Preprocess
    * gather raw_data
    * preprocess raw_data -> data (separate for  CIN and for DIS)
    * validation data
    * train/test split for gathering_stats and validation 
    * gathering stats (from data) for CIN and DIS separatly
* Fit/Predict pipeline
    * Init/Train/Save model
    * Make future predictions (horizon = 28 days) based on prediction_date
* PostPreprocess pipeline:
    * Take model predictions
    * Create outputs based on customer needs 
* Validation postprocess pipeline
    * Validate only test predictions


# To get started

1. ```conda create -n env python=3.7.6```
2. ```conda activate env```
2. ```pip install -r requirements.txt```
3. ```bash main.py```
