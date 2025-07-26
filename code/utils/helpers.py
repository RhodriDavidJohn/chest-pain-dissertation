### file which contains useful functions for the main scripts

import os
import pandas as pd
import joblib
import logging
from sklearn.pipeline import Pipeline



def logger_setup(filename: str) -> logging.Logger:
    
    os.makedirs('logs', exist_ok=True)
    filepath = os.path.join('logs', filename)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=filepath,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)

    LOGGER = logging.getLogger(filename)

    return LOGGER


def load_csv(filepath: str, LOGGER: logging.Logger) -> pd.DataFrame:

    try:
        df = pd.read_csv(filepath)
        LOGGER.info(f"Data loaded successfully: {filepath}")
        return df
    except Exception as e:
        LOGGER.error(f"Error loading {filepath}: {e}")
        raise(e)


def save_to_csv(df: pd.DataFrame, filepath: str, LOGGER: logging.Logger) -> None:

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        LOGGER.info(f"Data saved to {filepath}")
        return None
    except Exception as e:
        LOGGER.error(f"Error saving data to {filepath}: {e}")
        raise(e)


def save_model(model: Pipeline, filepath: str, LOGGER: logging.Logger):

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        LOGGER.info(f"Model saved successfully: {filepath}")
        return None
    except Exception as e:
        msg = f"Error saving the model {filepath}: {e}"
        LOGGER.error(msg)
        raise(e)


def load_model(filepath: str, LOGGER: logging.Logger):

    try:
        model = joblib.load(filepath)
        LOGGER.info(f"Model loaded successfully: {filepath}")
        return model
    except Exception as e:
        msg = f"Error loading the model {filepath}: {e}"
        LOGGER.error(msg)
        raise(e)


def remove_file(filepath: str, LOGGER: logging.Logger):

    if os.path.exists(filepath):
        os.remove(filepath)
        LOGGER.info(f"File deleted: {filepath}")
    else:
        LOGGER.warning(f"File does not exist: {filepath}")
        raise ValueError(f"Filepath does not exist: {filepath}")


def get_model_name(pipe: Pipeline) -> str:
    return list(pipe.named_steps.keys())[-1]


def map_model_name(model_name: str, LOGGER: logging.Logger) -> str:

    model_name_dict = {
        'lreg_model': 'Logistic Regression',
        'rfc_model': 'Random Forest',
        'xgb_model': 'XGBoost',
        'lgbm_model': 'LightGBM',
    }
    try:
        return model_name_dict[model_name]
    except:
        msg = (f'{model_name} not found. Model name must be one of '
               f'{model_name_dict.keys()}')
        LOGGER.error(msg)
        raise ValueError(msg)


def get_model_features(pipe: Pipeline, step_name: str) -> list:

    features = pipe.named_steps[step_name].get_feature_names_out()
    try:
        features_list = [feature.split('__')[1] for feature in features]
    except:
        pass
    
    return features_list
