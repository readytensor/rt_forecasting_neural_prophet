import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Union, Literal, Callable, Type
from neuralprophet import NeuralProphet, set_random_seed, set_log_level
from neuralprophet.utils import save, load
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from torch import cuda
import torch
from logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(task_name="model")

PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_FILE_NAME = "model_file"


class Forecaster:
    """A wrapper class for the NeuralProphet Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "NeuralProphet Forecaster"
    made_up_frequency = "S"  # by seconds
    made_up_start_dt = "2000-01-01 00:00:00"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: int = None,
        lags_forecast_ratio: int = None,
        growth: Literal["off", "linear", "discontinuous"] = "linear",
        seasonality_mode: Literal["additive", "multiplicative"] = "additive",
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        loss_func: Union[str, torch.nn.modules.loss._Loss, Callable] = "Huber",
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "AdamW",
        normalize: Literal[
            "auto", "soft", "soft1", "minmax", "standardize", "off"
        ] = "auto",
        random_state: int = 0,
        trainer_config: dict = {},
        use_exogenous: bool = True,
        early_stopping: bool = True,
        **kwargs,
    ):
        """Construct a new NeuralProphet Forecaster

        Args:

            data_schema (ForecastingSchema):
                Schema of the training data.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            lags_forecast_ratio (int):
                Sets the n_lags parameter depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the lags_forecast_ratio is 10,
                n_lags will be 20*10 = 200.

            growth (Literal["off", "linear", "discontinuous"]):
                Set use of trend growth type.

                Options:
                off: no trend.

                (default) linear: fits a piece-wise linear trend with n_changepoints + 1 segments
                discontinuous: For advanced users only - not a conventional trend,
                allows arbitrary jumps at each trend changepoint

            seasonality_mode (Literal["additive", "multiplicative"]):
                Specifies mode of seasonality
                Options
                (default) additive
                multiplicative

            learning_rate (Optional[float]):
                Maximum learning rate setting for 1cycle policy scheduler.
                Default None: Automatically sets the learning_rate based on a learning rate range test. For manual user input, (try values ~0.001-10).

            epochs (Optional[int]):
                Number of epochs (complete iterations over dataset) to train model.

            batch_size (Optional[int]):
                Number of samples per mini-batch.
                If not provided, batch_size is approximated based on dataset size.
                For manual values, try ~8-1024. For best results also leave epochs to None.

            loss_func (Union[str, torch.nn.modules.loss._Loss, Callable]):
                Type of loss to use:

                Options
                (default) Huber: Huber loss function
                MSE: Mean Squared Error loss function
                MAE: Mean Absolute Error loss function
                torch.nn.functional.loss.: loss or callable for custom loss, eg. L1-Loss

            optimizer (Union[str, Type[torch.optim.Optimizer]]):
                Optimizer used for training.

            normalize (Literal["auto", "soft", "soft1", "minmax", "standardize", "off"]):
                Type of normalization to apply to the time series.

            random_state (int):
                Sets the underlying random seed at model initialization time.

            use_exogenous (bool):
                Indicated if covariates are used or not.

            earlyy_stopping (bool):
                If true, early stopping is enabled.

            trainer_config (dict):
                Dictionary of additional trainer configuration parameters.
        """
        self.data_schema = data_schema
        self.history_forecast_ratio = history_forecast_ratio
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.normalize = normalize
        self.trainer_config = trainer_config
        self.use_exogenous = use_exogenous
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.kwargs = kwargs
        self._is_trained = False
        self.history_length = None
        self.lags_forecast_ratio = lags_forecast_ratio
        self.n_forecasts = self.data_schema.forecast_length

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )

        self.n_lags = int(self.data_schema.forecast_length * lags_forecast_ratio)

        self.trainer_config = {}

        if cuda.is_available():
            self.accelerator = "auto"
            self.trainer_config["accelerator"] = "gpu"
            logger.info("GPU training is available.")
        else:
            self.accelerator = None
            logger.info("GPU training not available.")
        self.model = None
        self.history = None
        self._is_trained = False
        self.series_length = None
        self.dropped_columns = []
        self.used_regressors = []

    def prepare_data(self, data: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Function to prepare the dataframe to use with Prophet.

        Prophet expects the following columns:
        - ds: datetime column
        - y: target column of numeric type (float or int)

        If the time column is of type int, we will update it to be datetime
        by creating artificial dates starting at '1/1/2023 00:00:00'
        that increment by 1 second for each row.

        The final dataframe will have the following columns:
        - <series_id>: name of column is kept as-is
        - ds: contains the datetime column. When passed integers, these are changed to
                datetimes as described above. We store the mapping of datetimes to
                original integer values.
                When passed date or datetimes, these are converted to datetime type
                and local time zone information is stripped (as per Prophet's
                requirements.)
        - y: contains the target series to forecasting

        Additionally, there may be 0, 1 or more future covariates as were originally
        passed. These are returned as-is.
        """
        time_col = self.data_schema.time_col
        id_col = self.data_schema.id_col
        time_col_dtype = self.data_schema.time_col_dtype

        groups_by_ids = data.groupby(id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [groups_by_ids.get_group(id_) for id_ in all_ids]

        if self.history_length:
            for index, series in enumerate(all_series):
                all_series[index] = series.iloc[-self.history_length :]

            data = pd.concat(all_series)

        # sort data
        data = data.sort_values(by=[id_col, time_col])

        if time_col_dtype == "INT":
            # Find the number of rows for each location (assuming all locations have
            # the same number of rows)
            series_val_counts = data[id_col].value_counts()
            series_len = series_val_counts.iloc[0]
            num_series = series_val_counts.shape[0]

            if is_train:
                # since prophet requires a datetime column, we will make up a timeline
                start_date = pd.Timestamp(self.made_up_start_dt)
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
                self.last_timestamp = datetimes[-1]
                self.timedelta = datetimes[-1] - datetimes[-2]
            else:
                start_date = self.last_timestamp + self.timedelta
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
            int_vals = sorted(data[time_col].unique().tolist())
            self.time_to_int_map = dict(zip(datetimes, int_vals))
            # Repeat the datetime range for each location
            data[time_col] = list(datetimes) * num_series
        else:
            data[time_col] = pd.to_datetime(data[time_col])
            data[time_col] = data[time_col].dt.tz_localize(None)

        # rename columns as expected by Prophet
        data = data.rename(
            columns={
                self.data_schema.target: "y",
                time_col: "ds",
                self.data_schema.id_col: "ID",
            }
        )

        reordered_cols = ["ID", "ds"]
        if is_train:
            reordered_cols += ["y"]
        other_cols = [c for c in data.columns if c not in reordered_cols]
        reordered_cols.extend(other_cols)
        data = data[reordered_cols]
        return data
    
    def handle_regressors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Set regressors if `use_exogenous and regresors that are not constant.
        Return data with appropriate columns
        """
        if self.use_exogenous:
            regressors = self.data_schema.future_covariates + \
                            self.data_schema.past_covariates
            for regressor in regressors:
                if data[regressor].nunique() >= 1:
                    self.used_regressors.append(regressor)
        else:
            self.used_regressors = []
        
        cols = ["ID", "ds", "y"] + self.used_regressors
        return data[cols]

    def add_regressors_to_model(self):
        for covariate in self.data_schema.future_covariates:
            if covariate in self.used_regressors:
                self.model.add_future_regressor(name=covariate)

        for covariate in (self.data_schema.past_covariates):
            if covariate in self.used_regressors:
                self.model.add_lagged_regressor(names=covariate)
    
    def _verify_adequate_series_length(self, history: pd.DataFrame) -> None:
        """
        Verify if given series length is adequate and n_lags can be 
        accomodated. 
        """
        series_length = self._get_series_length(history)

        if series_length < self.data_schema.forecast_length * 2:
            raise ValueError(
                f"Training series is too short. History should be "
                "at least double the forecast horizon. Given:\n"
                f"history_length = ({series_length}), "
                f"forecast horizon = ({self.data_schema.forecast_length})"
            )

    def _get_series_length(self, history: pd.DataFrame) -> int:
        """
        Get the length of the series.
        """
        if self.series_length is None:
            id_col = self.data_schema.id_col
            target_col = self.data_schema.target
            self.series_length  = history.groupby(id_col)[target_col].count().iloc[0]
        return self.series_length
    
    def _set_n_lags(self, history: pd.DataFrame) -> int:
        """
        If not, set the number of lags based on lags_forecast_ratio
        and the forecast horizon.
        """
        series_length = self._get_series_length(history)
        if series_length < self.n_forecasts + self.n_lags:
            logger.warning(
                "Dataframe has less than (n_forecasts + n_lags) rows."
                " Setting n_lags = (history_length - n_forecasts) = "
                f"{series_length - self.n_forecasts}"
            )
            self.n_lags = series_length - self.n_forecasts

    def fit(
        self,
        history: pd.DataFrame,
    ) -> None:
        """Fit the Forecaster to the training data.

        Args:
            history (pandas.DataFrame): The features of the training data.
        """
        np.random.seed(self.random_state)
        set_log_level("ERROR")
        set_random_seed(self.random_state)

        history = history.copy()

        self._verify_adequate_series_length(history)
        self._set_n_lags(history)

        self.model = NeuralProphet(
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            loss_func=self.loss_func,
            optimizer=self.optimizer,
            normalize=self.normalize,
            n_changepoints=3,
            **self.kwargs,
        )

        processed_history = self.prepare_data(history)
        processed_history = self.handle_regressors(processed_history)
        self.add_regressors_to_model()

        self.model.fit(df=processed_history, early_stopping=self.early_stopping)
        self._is_trained = True
        self.history = processed_history

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The prediction dataframe.
        """
        set_log_level("ERROR")
        set_random_seed(self.random_state)

        test_data = self.prepare_data(test_data.copy(), is_train=False)

        regressors_df = None
        covariates = self.used_regressors

        if self.use_exogenous and len(covariates) > 0:
            valid_covariates = [c for c in covariates
                                if c in self.data_schema.future_covariates]
            regressors_df = test_data[valid_covariates]

        future_data = self.model.make_future_dataframe(
            df=self.history,
            periods=self.data_schema.forecast_length,
            regressors_df=regressors_df,
        )

        all_forecasts = self.model.predict(df=future_data, raw=True, decompose=False)

        forecast = []
        for _, row in all_forecasts.iterrows():
            predictions_columns_names = [
                f"step{i}" for i in range(self.data_schema.forecast_length)
            ]
            forecast += row[predictions_columns_names].values.tolist()

        test_data[prediction_col_name] = forecast
        test_data.rename(
            columns={
                "ID": self.data_schema.id_col,
                "yhat": prediction_col_name,
                "ds": self.data_schema.time_col,
            },
            inplace=True,
        )
        # Change datetime back to integer
        if self.data_schema.time_col_dtype == "INT":
            test_data[self.data_schema.time_col] = \
                test_data[self.data_schema.time_col].map(
                    self.time_to_int_map
                )
        return test_data

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        save(self.model, os.path.join(model_dir_path, MODEL_FILE_NAME))
        self.model = None
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        forecaster = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        model = load(os.path.join(model_dir_path, MODEL_FILE_NAME))
        forecaster.model = model
        return forecaster

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.
        test_dataframe (pd.DataFrame): The testing data (needed only if the data contains future covariates).

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(data_schema=data_schema, **hyperparameters)
    model.fit(history=history)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
