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
        n_lags: int = 1,
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

            n_lags (int):
                Previous time series steps to include in auto-regression. Aka AR-order (n_lags >= 1)

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
        self.freq = self.map_frequency(self.data_schema.frequency)
        self.n_forecasts = self.data_schema.forecast_length

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

        self.model = NeuralProphet(
            n_forecasts=self.n_forecasts,
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_reg=self.seasonality_reg,
            season_global_local=self.season_global_local,
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
        other_cols = [c for c in data.columns if c not in reordered_cols]
        reordered_cols.extend(other_cols)
        data = data[reordered_cols]

        dropped_columns = self.data_schema.static_covariates.copy()

        if not self.use_exogenous:
            dropped_columns += (
                self.data_schema.future_covariates.copy()
                + self.data_schema.past_covariates.copy()
            )

        else:
            for covariate in self.data_schema.future_covariates:
                if data[covariate].nunique() > 1:
                    self.model.add_future_regressor(name=covariate)
                else:
                    dropped_columns.append(covariate)

            for covariate in self.data_schema.past_covariates:
                if data[covariate].nunique() > 1:
                    self.model.add_lagged_regressor(names=covariate)
                else:
                    dropped_columns.append(covariate)

        data.drop(columns=dropped_columns, inplace=True)
        self.dropped_columns = dropped_columns

        return data

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
        history = self.prepare_data(history.copy())
        self.model.fit(df=history, early_stopping=self.early_stopping)
        self._is_trained = True
        self.history = history

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
        time_col = self.data_schema.time_col
        id_col = self.data_schema.id_col
        original_time_col = test_data[time_col]

        regressors_df = None
        covariates = self.data_schema.future_covariates

        if self.use_exogenous and covariates:
            valid_covariates = [c for c in covariates if c not in self.dropped_columns]
            regressors_df = test_data[valid_covariates]

        test_data = self.model.make_future_dataframe(
            df=self.history,
            periods=self.data_schema.forecast_length,
            regressors_df=regressors_df,
        )

        all_forecasts = self.model.predict(df=test_data)
        columns = [f"yhat{i+1}" for i in range(self.n_forecasts)]

        groups_by_ids = all_forecasts.groupby("ID")
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [groups_by_ids.get_group(id_) for id_ in all_ids]

        for i, series in enumerate(all_series):
            series = series.iloc[-self.n_forecasts :]
            for col in columns:
                series["y"] = series["y"].combine_first(series[col])

            series["y"] = series["y"].round(4)
            all_series[i] = series

        all_forecasts = pd.concat(all_series)
        all_forecasts["ds"] = original_time_col.values

        all_forecasts.rename(
            columns={
                "ID": id_col,
                "y": prediction_col_name,
                "ds": time_col,
            },
            inplace=True,
        )

        all_forecasts = all_forecasts[[time_col, id_col, prediction_col_name]]
        return all_forecasts

    def map_frequency(self, frequency: str) -> str:
        """
        Maps the frequency in the data schema to the frequency expected by Prophet.

        Args:
            frequency (str): The frequency from the schema.

        Returns (str): The mapped frequency.
        """
        frequency = frequency.lower()
        frequency = frequency.split("frequency.")[1]
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency in ["secondly", "other"]:
            return "S"

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
