import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Union, Literal, Callable, Type
from neuralprophet import NeuralProphet
from neuralprophet.utils import save, load
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from torch import cuda
import torch

warnings.filterwarnings("ignore")


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
        yearly_seasonality: Union[Literal["auto"], bool, int] = "auto",
        weekly_seasonality: Union[Literal["auto"], bool, int] = "auto",
        daily_seasonality: Union[Literal["auto"], bool, int] = "auto",
        seasonality_mode: Literal["additive", "multiplicative"] = "additive",
        seasonality_reg: float = 0,
        season_global_local: Literal["global", "local"] = "global",
        n_forecasts: int = 1,
        n_lags: int = 0,
        ar_layers: Optional[list] = [],
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


            yearly_seasonality (Union[Literal["auto"], bool, int]):
                Fit yearly seasonality.

                Options:
                True or False
                auto: set automatically
                value: number of Fourier/linear terms to generate


            weekly_seasonality (Union[Literal["auto"], bool, int]):
                Fit monthly seasonality.

                Options:
                True or False
                auto: set automatically
                value: number of Fourier/linear terms to generate

            daily_seasonality (Union[Literal["auto"], bool, int]):
                Fit daily seasonality.

                Options:
                True or False
                auto: set automatically
                value: number of Fourier/linear terms to generate


            seasonality_mode (Literal["additive", "multiplicative"]):
                Specifies mode of seasonality

                Options
                (default) additive
                multiplicative

            seasonality_reg (float):
                Parameter modulating the strength of the seasonality model.

            season_global_local (Literal["global", "local"]):
                Modelling strategy of the seasonality when multiple time series are present. Options:
                global: All the elements are modelled with the same seasonality.
                local: Each element is modelled with a different seasonality.


            n_forecasts (int):
                Number of steps ahead of prediction time step to forecast.


            n_lags (int):
                Previous time series steps to include in auto-regression. Aka AR-order


            ar_layers (Optional[list]):
                array of hidden layer dimensions of the AR-Net.
                Specifies number of hidden layers (number of entries) and layer dimension (list entry).

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
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_reg = seasonality_reg
        self.season_global_local = season_global_local
        self.n_forecasts = n_forecasts
        self.n_lags = n_lags
        self.ar_layers = ar_layers
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

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )

        if lags_forecast_ratio:
            self.n_lags = self.data_schema.forecast_length * lags_forecast_ratio

        self.trainer_config = {}

        if cuda.is_available():
            self.trainer_config["accelerator"] = "gpu"
            print("GPU training is available.")
        else:
            print("GPU training not available.")

    def prepare_data(self, data: pd.DataFrame, is_train=True) -> pd.DataFrame:
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
        target_col = self.data_schema.target
        # sort data
        data = data.sort_values(by=[id_col, time_col])

        if self.data_schema.time_col_dtype == "INT":
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
        data = data.rename(columns={target_col: "y", time_col: "ds"})
        reordered_cols = [id_col, "ds"]
        other_cols = [c for c in data.columns if c not in reordered_cols]
        reordered_cols.extend(other_cols)
        data = data[reordered_cols]

        if is_train:
            data.drop(columns=self.data_schema.past_covariates, inplace=True)
        else:
            data["y"] = 0
        return data

    def fit(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
        history_length: int = None,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate NeuralProphet model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.
            history_length (int): The length of the series used for training.
        """
        np.random.seed(self.random_state)

        history = self.prepare_data(history.copy())

        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.models = {}

        for id, series in zip(all_ids, all_series):
            if history_length:
                series = series.iloc[-history_length:]
            model = self._fit_on_series(history=series)
            self.models[id] = model

        self.all_ids = all_ids
        self._is_trained = True
        self.data_schema = data_schema

    def _fit_on_series(self, history: pd.DataFrame):
        """Fit Prophet model to given individual series of data"""
        model = NeuralProphet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_reg=self.seasonality_reg,
            season_global_local=self.season_global_local,
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            ar_layers=self.ar_layers,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            loss_func=self.loss_func,
            optimizer=self.optimizer,
            normalize=self.normalize,
            trainer_config=self.trainer_config,
            **self.kwargs,
        )

        future_covariates = self.data_schema.future_covariates

        if self.use_exogenous:
            for covariate in future_covariates:
                model.add_future_regressor(name=covariate)

        self.target_series = history["y"]

        model.fit(history, early_stopping=self.early_stopping)
        return model

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
        time_col = self.data_schema.time_col
        id_col = self.data_schema.id_col
        time_col_dtype = self.data_schema.time_col_dtype

        future_df = self.prepare_data(test_data.copy(), is_train=False)
        groups_by_ids = future_df.groupby(id_col)
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=id_col) for id_ in self.all_ids
        ]
        # for some reason, multi-processing takes longer! So use single-threaded.
        # forecast one series at a time
        all_forecasts = []
        for id_, series_df in zip(self.all_ids, all_series):
            forecast = self._predict_on_series(key_and_future_df=(id_, series_df))
            forecast.insert(0, id_col, id_)
            all_forecasts.append(forecast)

        # concatenate all series' forecasts into a single dataframe
        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)
        all_forecasts["yhat1"] = all_forecasts["yhat1"].round(4)
        all_forecasts.rename(
            columns={
                "yhat1": prediction_col_name,
                "ds": time_col,
            },
            inplace=True,
        )
        # Change datetime back to integer
        if time_col_dtype == "INT":
            all_forecasts[time_col] = all_forecasts[time_col].map(self.time_to_int_map)
        return all_forecasts

    def _predict_on_series(self, key_and_future_df):
        """Make forecast on given individual series of data"""
        key, future_df = key_and_future_df
        if self.models.get(key) is not None:
            forecast = self.models[key].predict(future_df)
            df_cols_to_use = ["ds", "yhat1"]
            cols = [c for c in df_cols_to_use if c in forecast.columns]
            forecast = forecast[cols]
        else:
            # no model found - key wasnt found in history, so cant forecast for it.
            forecast = None
        return forecast

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        for id, model in self.models.items():
            save(model, os.path.join(model_dir_path, f"{id}_{MODEL_FILE_NAME}"))
        models = self.models
        self.models = {}
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        self.models = models

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        forecaster = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        models = {}
        file_names = [i for i in os.listdir(model_dir_path) if i != PREDICTOR_FILE_NAME]
        for file in file_names:
            series_id = file.split(MODEL_FILE_NAME)[0][:-1]
            model = load(os.path.join(model_dir_path, file))
            models[series_id] = model

        forecaster.models = models
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

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(
        history=history,
        data_schema=data_schema,
        history_length=model.history_length,
    )
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
