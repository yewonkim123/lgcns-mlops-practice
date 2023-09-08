import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from deepchecks import SuiteResult
from deepchecks.core.suite import SuiteResult
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation, train_test_validation

from src.common.constants import (
    ARTIFACT_PATH,
    DATA_PATH,
    DRIFT_DETECTION_PATH,
    LOG_FILEPATH,
)
from src.common.logger import handle_exception, set_logger
from src.preprocess import CAT_FEATURES, preprocess_pipeline

logger = set_logger(os.path.join(LOG_FILEPATH, "logs.log"))
sys.excepthook = handle_exception
warnings.filterwarnings(action="ignore")


DATE = datetime.now().strftime("%Y%m%d")
LABEL_NAME = "rent"
model = joblib.load(os.path.join(ARTIFACT_PATH, "model.pkl"))


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA_PATH, filename),
        usecols=lambda x: x not in ["area_locality", "posted_on", "id"],
    )


def log_failed_check_info(suite_result: SuiteResult):
    for result in suite_result.get_not_passed_checks():
        logger.info(
            "The following test failed!\n"
            f"{result.header}: {result.conditions_results[0].details}"
        )


def get_drift_test(suite_result: SuiteResult, test_name: str) -> dict:
    result_json = json.loads(suite_result.to_json())
    test_result = [
        x
        for x in result_json.get("results")
        if x.get("check").get("name") == test_name
    ][0]
    conditions_results = test_result.get("conditions_results")[0]
    value = test_result.get("value").get("Drift score").get("value")

    conditions_results["value"] = value

    return conditions_results


def data_drift_detection(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    label: str,
    cat_features: str,
    save_as_html: bool = False,
) -> None:
    train_set = Dataset(train_df, label=label, cat_features=cat_features)
    new_set = Dataset(new_df, label=label, cat_features=cat_features)

    validation_suite = train_test_validation()
    suite_result = validation_suite.run(train_set, new_set)

    log_failed_check_info(suite_result=suite_result)

    if save_as_html:
        suite_result.save_as_html(
            os.path.join(DRIFT_DETECTION_PATH, f"{DATE}_data_drift.html")
        )


def model_drift_detection(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    label: str,
    cat_features: str,
    save_as_json: bool = True,
    save_as_html: bool = False,
) -> None:
    def get_xy(df: pd.DataFrame):
        y = np.log1p(df[label])
        x = preprocess_pipeline.fit_transform(X=df.drop([label], axis=1), y=y)

        return x, y

    x_train, y_train = get_xy(train_df)
    x_new, y_new = get_xy(new_df)

    train_set = Dataset(
        x_train,
        label=y_train,
        cat_features=cat_features,
    )
    new_set = Dataset(
        x_new,
        label=y_new,
        cat_features=cat_features,
    )

    evaluation_suite = model_evaluation()
    suite_result = evaluation_suite.run(train_set, new_set, model["regr"])

    log_failed_check_info(suite_result=suite_result)

    # Prediction Drift 정보만 저장
    if save_as_json:
        prediction_drift = get_drift_test(
            suite_result=suite_result, test_name="Prediction Drift"
        )
        json_obj = json.dumps(prediction_drift, indent=4)

        with open("./prediction_drift.json", "w") as file:
            file.write(json_obj)

    if save_as_html:
        suite_result.save_as_html(
            os.path.join(DRIFT_DETECTION_PATH, f"{DATE}_model_drift.html")
        )


def main():
    train_df = load_data(filename="house_rent_train.csv")
    new_df = load_data(filename="house_rent_new.csv")

    logger.debug(f"{train_df.info()}")
    logger.debug(f"{new_df.info()}")

    logger.info("Detect data drift")
    data_drift_detection(
        train_df=train_df,
        new_df=new_df,
        label=LABEL_NAME,
        cat_features=CAT_FEATURES,
        save_as_html=True,
    )

    logger.info("Detect model drift")
    model_drift_detection(
        train_df=train_df,
        new_df=new_df,
        label=LABEL_NAME,
        cat_features=CAT_FEATURES,
        save_as_json=True,
        save_as_html=True,
    )

    logger.info(
        "Detection results can be found in the following path:\n"
        f"{DRIFT_DETECTION_PATH}"
    )


if __name__ == "__main__":
    main()
