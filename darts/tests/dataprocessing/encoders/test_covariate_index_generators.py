import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.dataprocessing.encoders.encoder_base import (
    CovariatesIndexGenerator,
    FutureCovariatesIndexGenerator,
    PastCovariatesIndexGenerator,
)
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)


class CovariatesIndexGeneratorTestCase(DartsBaseTestClass):
    n_target = 24
    target_time = tg.linear_timeseries(length=n_target, freq="MS")
    cov_time_train = tg.datetime_attribute_timeseries(
        target_time, attribute="month", cyclic=True
    )
    cov_time_train_short = cov_time_train[1:]

    target_int = tg.linear_timeseries(length=n_target, start=2)
    cov_int_train = target_int
    cov_int_train_short = cov_int_train[1:]

    input_chunk_length = 12
    output_chunk_length = 6
    n_short = 6
    n_long = 8

    # pd.DatetimeIndex
    # expected covariates for inference dataset for n <= output_chunk_length
    cov_time_inf_short = TimeSeries.from_times_and_values(
        tg.generate_index(
            start=target_time.start_time(),
            length=n_target + n_short,
            freq=target_time.freq,
        ),
        np.arange(n_target + n_short),
    )
    # expected covariates for inference dataset for n > output_chunk_length
    cov_time_inf_long = TimeSeries.from_times_and_values(
        tg.generate_index(
            start=target_time.start_time(),
            length=n_target + n_long,
            freq=target_time.freq,
        ),
        np.arange(n_target + n_long),
    )

    # integer index
    # excpected covariates for inference dataset for n <= output_chunk_length
    cov_int_inf_short = TimeSeries.from_times_and_values(
        tg.generate_index(
            start=target_int.start_time(),
            length=n_target + n_short,
            freq=target_int.freq,
        ),
        np.arange(n_target + n_short),
    )
    # excpected covariates for inference dataset for n > output_chunk_length
    cov_int_inf_long = TimeSeries.from_times_and_values(
        tg.generate_index(
            start=target_int.start_time(),
            length=n_target + n_long,
            freq=target_int.freq,
        ),
        np.arange(n_target + n_long),
    )

    def helper_test_index_types(self, ig: CovariatesIndexGenerator):
        """test the index type of generated index"""
        # pd.DatetimeIndex
        idx = ig.generate_train_idx(self.target_time, self.cov_time_train)
        self.assertTrue(isinstance(idx, pd.DatetimeIndex))
        idx = ig.generate_inference_idx(
            self.n_short, self.target_time, self.cov_time_inf_short
        )
        self.assertTrue(isinstance(idx, pd.DatetimeIndex))
        idx = ig.generate_train_idx(self.target_time, None)
        self.assertTrue(isinstance(idx, pd.DatetimeIndex))

        # pd.RangeIndex
        idx = ig.generate_train_idx(self.target_int, self.cov_int_train)
        self.assertTrue(isinstance(idx, pd.RangeIndex))
        idx = ig.generate_inference_idx(
            self.n_short, self.target_int, self.cov_int_inf_short
        )
        self.assertTrue(isinstance(idx, pd.RangeIndex))
        idx = ig.generate_train_idx(self.target_int, None)
        self.assertTrue(isinstance(idx, pd.RangeIndex))

    def helper_test_index_generator_train(self, ig: CovariatesIndexGenerator):
        """
        If covariates are given, the index generators should return the covariate series' index.
        If covariates are not given, the index generators should return the target series' index.
        """
        # pd.DatetimeIndex
        # generated index must be equal to input covariate index
        idx = ig.generate_train_idx(self.target_time, self.cov_time_train)
        self.assertTrue(idx.equals(self.cov_time_train.time_index))
        # generated index must be equal to input covariate index
        idx = ig.generate_train_idx(self.target_time, self.cov_time_train_short)
        self.assertTrue(idx.equals(self.cov_time_train_short.time_index))
        # generated index must be equal to input target index when no covariates are defined
        idx = ig.generate_train_idx(self.target_time, None)
        self.assertEqual(idx[0], self.target_time.start_time())
        if isinstance(ig, PastCovariatesIndexGenerator):
            self.assertEqual(
                idx[-1],
                self.target_time.end_time()
                - self.output_chunk_length * self.target_time.freq,
            )
        else:
            self.assertEqual(idx[-1], self.target_time.end_time())

        # integer index
        # generated index must be equal to input covariate index
        idx = ig.generate_train_idx(self.target_int, self.cov_int_train)
        self.assertTrue(idx.equals(self.cov_int_train.time_index))
        # generated index must be equal to input covariate index
        idx = ig.generate_train_idx(self.target_int, self.cov_int_train_short)
        self.assertTrue(idx.equals(self.cov_int_train_short.time_index))
        # generated index must be equal to input target index when no covariates are defined
        idx = ig.generate_train_idx(self.target_int, None)
        self.assertEqual(idx[0], self.target_int.start_time())
        if isinstance(ig, PastCovariatesIndexGenerator):
            self.assertEqual(
                idx[-1],
                self.target_int.end_time()
                - self.output_chunk_length * self.target_int.freq,
            )
        else:
            self.assertEqual(idx[-1], self.target_int.end_time())

    def helper_test_index_generator_inference(self, ig, is_past=False):
        """
        For prediction (`n` is given) with past covariates we have to distinguish between two cases:
        1)  if past covariates are given, we can use them as reference
        2)  if past covariates are missing, we need to generate a time index that starts `input_chunk_length`
            before the end of `target` and ends `max(0, n - output_chunk_length)` after the end of `target`

        For prediction (`n` is given) with future covariates we have to distinguish between two cases:
        1)  if future covariates are given, we can use them as reference
        2)  if future covariates are missing, we need to generate a time index that starts `input_chunk_length`
            before the end of `target` and ends `max(n, output_chunk_length)` after the end of `target`
        """

        # check generated inference index without passing covariates when n <= output_chunk_length
        idx = ig.generate_inference_idx(self.n_short, self.target_time, None)
        if is_past:
            n_out = self.input_chunk_length
            last_idx = self.target_time.end_time()
        else:
            n_out = self.input_chunk_length + self.output_chunk_length
            last_idx = self.cov_time_inf_short.end_time()

        self.assertTrue(len(idx) == n_out)
        self.assertTrue(idx[-1] == last_idx)

        # check generated inference index without passing covariates when n > output_chunk_length
        idx = ig.generate_inference_idx(self.n_long, self.target_time, None)
        if is_past:
            n_out = self.input_chunk_length + self.n_long - self.output_chunk_length
            last_idx = (
                self.target_time.end_time()
                + (self.n_long - self.output_chunk_length) * self.target_time.freq
            )
        else:
            n_out = self.input_chunk_length + self.n_long
            last_idx = self.cov_time_inf_long.end_time()

        self.assertTrue(len(idx) == n_out)
        self.assertTrue(idx[-1] == last_idx)

        idx = ig.generate_inference_idx(
            self.n_short, self.target_time, self.cov_time_inf_short
        )
        self.assertTrue(idx.equals(self.cov_time_inf_short.time_index))
        idx = ig.generate_inference_idx(
            self.n_long, self.target_time, self.cov_time_inf_long
        )
        self.assertTrue(idx.equals(self.cov_time_inf_long.time_index))
        idx = ig.generate_inference_idx(
            self.n_short, self.target_int, self.cov_int_inf_short
        )
        self.assertTrue(idx.equals(self.cov_int_inf_short.time_index))
        idx = ig.generate_inference_idx(
            self.n_long, self.target_int, self.cov_int_inf_long
        )
        self.assertTrue(idx.equals(self.cov_int_inf_long.time_index))

    def test_past_index_generator_creation(self):
        # ==> test failures
        # one lag is >= 0 (not possible for past covariates)
        with pytest.raises(ValueError):
            _ = PastCovariatesIndexGenerator(
                1, 1, min_covariates_lag=1, max_covariates_lag=-1
            )
        with pytest.raises(ValueError):
            _ = PastCovariatesIndexGenerator(
                1, 1, min_covariates_lag=-1, max_covariates_lag=1
            )
        # max lag is smaller than min lag
        with pytest.raises(ValueError):
            _ = PastCovariatesIndexGenerator(
                1, 1, min_covariates_lag=-1, max_covariates_lag=-2
            )
        # one lag is given, the other isn't
        with pytest.raises(ValueError):
            _ = PastCovariatesIndexGenerator(
                1, 1, min_covariates_lag=None, max_covariates_lag=-1
            )
        with pytest.raises(ValueError):
            _ = PastCovariatesIndexGenerator(
                1, 1, min_covariates_lag=-1, max_covariates_lag=None
            )

        min_lag, max_lag = -2, -1
        ig = PastCovariatesIndexGenerator(
            1, 1, min_covariates_lag=min_lag, max_covariates_lag=max_lag
        )
        self.assertEqual(ig.shift_start, min_lag)
        self.assertEqual(ig.shift_end, max_lag)

        min_lag, max_lag = -1, -1
        ig = PastCovariatesIndexGenerator(
            1, 1, min_covariates_lag=min_lag, max_covariates_lag=max_lag
        )
        self.assertEqual(ig.shift_start, min_lag)
        self.assertEqual(ig.shift_end, max_lag)

    def test_future_index_generator_creation(self):
        # ==> test failures
        # max lag is smaller than min lag
        with pytest.raises(ValueError):
            _ = FutureCovariatesIndexGenerator(
                1, 1, min_covariates_lag=-1, max_covariates_lag=-2
            )
        # one lag is given, the other isn't
        with pytest.raises(ValueError):
            _ = FutureCovariatesIndexGenerator(
                1, 1, min_covariates_lag=None, max_covariates_lag=-1
            )
        with pytest.raises(ValueError):
            _ = FutureCovariatesIndexGenerator(
                1, 1, min_covariates_lag=-1, max_covariates_lag=None
            )

        # future covariates index generator (ig) can technically be used like a past covariates ig
        min_lag, max_lag = -2, -1
        ig = FutureCovariatesIndexGenerator(
            1, 1, min_covariates_lag=min_lag, max_covariates_lag=max_lag
        )
        self.assertEqual(ig.shift_start, min_lag)
        self.assertEqual(ig.shift_end, max_lag)

        min_lag, max_lag = -1, -1
        ig = FutureCovariatesIndexGenerator(
            1, 1, min_covariates_lag=min_lag, max_covariates_lag=max_lag
        )
        self.assertEqual(ig.shift_start, min_lag)
        self.assertEqual(ig.shift_end, max_lag)

        # different to past covariates ig, future ig can take positive and negative lags
        min_lag, max_lag = -2, 1
        ig = FutureCovariatesIndexGenerator(
            1, 1, min_covariates_lag=min_lag, max_covariates_lag=max_lag
        )
        self.assertEqual(ig.shift_start, min_lag)
        # when `max_lag` >= 0, we add one step to `shift_end`, as future lags start at 0 meaning first prediction step
        self.assertEqual(ig.shift_end, max_lag + 1)

    def test_past_index_generator(self):
        ig = PastCovariatesIndexGenerator(
            self.input_chunk_length, self.output_chunk_length
        )
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=True)

    def test_past_index_generator_with_lags(self):
        icl = self.input_chunk_length
        ocl = self.output_chunk_length
        freq = self.target_time.freq

        def test_routine_train(
            self, icl, ocl, min_lag, max_lag, start_expected, end_expected
        ):
            idxg = PastCovariatesIndexGenerator(
                icl, ocl, min_covariates_lag=min_lag, max_covariates_lag=max_lag
            )
            idx = idxg.generate_train_idx(self.target_time, None)
            self.assertEqual(idx[0], start_expected)
            self.assertEqual(idx[-1], end_expected)
            # check case 0: we give covariates, index will always be the covariate time index
            idx = idxg.generate_train_idx(self.target_time, self.cov_time_train)
            self.assertTrue(idx.equals(self.cov_time_train.time_index))
            return idxg

        def test_routine_inf(self, idxg, n, start_expected, end_expected):
            idx = idxg.generate_inference_idx(n, self.target_time, None)
            self.assertEqual(idx[0], start_expected)
            self.assertEqual(idx[-1], end_expected)
            # check case 0: we give covariates, index will always be the covariate time index
            idx = idxg.generate_inference_idx(
                n, self.target_time, self.cov_time_inf_short
            )
            self.assertTrue(idx.equals(self.cov_time_inf_short.time_index))

        # lags are required for RegressionModels
        # case 1: abs(min_covariates_lags) == input_chunk_length and abs(max_covariates_lag) == -1:
        # will give identical results as without setting lags
        expected_start = self.target_time.start_time()
        expected_end = self.target_time.end_time() - freq * self.output_chunk_length
        ig = test_routine_train(self, icl, ocl, -icl, -1, expected_start, expected_end)
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=True)
        # check inference for n <= output_chunk_length
        expected_start = self.target_time.end_time() - (icl - 1) * freq
        expected_end = self.target_time.end_time()
        test_routine_inf(self, ig, 1, expected_start, expected_end)
        test_routine_inf(self, ig, ocl, expected_start, expected_end)
        # check inference for n > output_chunk_length
        test_routine_inf(self, ig, ocl + 1, expected_start, expected_end + 1 * freq)

        # case 2: abs(min_covariates_lag) > input_chunk_length and abs(max_covariates_lag) == -1:
        # the start time of covariates begins before target start
        expected_start = self.target_time.start_time() - freq
        expected_end = self.target_time.end_time() - freq * self.output_chunk_length
        min_lag, max_lag = -13, -1
        ig = test_routine_train(
            self, icl, ocl, min_lag, max_lag, expected_start, expected_end
        )
        # check inference for n <= output_chunk_length
        expected_start = self.target_time.end_time() - 12 * freq
        expected_end = self.target_time.end_time()
        test_routine_inf(self, ig, 1, expected_start, expected_end)
        test_routine_inf(self, ig, ocl, expected_start, expected_end)
        # check inference for n > output_chunk_length
        test_routine_inf(self, ig, ocl + 1, expected_start, expected_end + 1 * freq)

        # case 3: abs(min_covariates_lag) > input_chunk_length and abs(max_covariates_lag) > -1:
        # the start time of covariates begins before target start
        expected_start = self.target_time.start_time() - freq
        expected_end = self.target_time.end_time() - freq * (
            1 + self.output_chunk_length
        )
        min_lag, max_lag = -13, -2
        ig = test_routine_train(
            self, icl, ocl, min_lag, max_lag, expected_start, expected_end
        )
        # check inference for n <= output_chunk_length
        expected_start = self.target_time.end_time() - 12 * freq
        expected_end = self.target_time.end_time() - freq
        test_routine_inf(self, ig, 1, expected_start, expected_end)
        test_routine_inf(self, ig, ocl, expected_start, expected_end)
        # check inference for n > output_chunk_length
        test_routine_inf(self, ig, ocl + 1, expected_start, expected_end + 1 * freq)

    def test_future_index_generator(self):
        ig = FutureCovariatesIndexGenerator(
            self.input_chunk_length, self.output_chunk_length
        )
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=False)

    def test_future_index_generator_with_lags(self):
        icl = self.input_chunk_length
        ocl = self.output_chunk_length
        freq = self.target_time.freq

        def test_routine_train(
            self, icl, ocl, min_lag, max_lag, start_expected, end_expected
        ):
            idxg = FutureCovariatesIndexGenerator(
                icl, ocl, min_covariates_lag=min_lag, max_covariates_lag=max_lag
            )
            idx = idxg.generate_train_idx(self.target_time, None)
            self.assertEqual(idx[0], start_expected)
            self.assertEqual(idx[-1], end_expected)
            # check case 0: we give covariates, index will always be the covariate time index
            idx = idxg.generate_train_idx(self.target_time, self.cov_time_train)
            self.assertTrue(idx.equals(self.cov_time_train.time_index))
            return idxg

        def test_routine_inf(self, idxg, n, start_expected, end_expected):
            idx = idxg.generate_inference_idx(n, self.target_time, None)
            self.assertEqual(idx[0], start_expected)
            self.assertEqual(idx[-1], end_expected)
            # check case 0: we give covariates, index will always be the covariate time index
            idx = idxg.generate_inference_idx(
                n, self.target_time, self.cov_time_inf_short
            )
            self.assertTrue(idx.equals(self.cov_time_inf_short.time_index))

        # lags are required for RegressionModels
        # case 1: abs(min_covariates_lags) == input_chunk_length and abs(max_covariates_lag) == -1:
        # will give identical results as without setting lags
        expected_start = self.target_time.start_time()
        expected_end = self.target_time.end_time() - freq * self.output_chunk_length
        ig = test_routine_train(self, icl, ocl, -icl, -1, expected_start, expected_end)
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=True)
        # check inference for n <= output_chunk_length
        expected_start = self.target_time.end_time() - (icl - 1) * freq
        expected_end = self.target_time.end_time()
        test_routine_inf(self, ig, 1, expected_start, expected_end)
        test_routine_inf(self, ig, ocl, expected_start, expected_end)
        # check inference for n > output_chunk_length
        test_routine_inf(self, ig, ocl + 1, expected_start, expected_end + 1 * freq)

        # case 2: abs(min_covariates_lag) > input_chunk_length and abs(max_covariates_lag) == -1:
        # the start time of covariates begins before target start
        expected_start = self.target_time.start_time() - freq
        expected_end = self.target_time.end_time() - freq * self.output_chunk_length
        min_lag, max_lag = -13, -1
        ig = test_routine_train(
            self, icl, ocl, min_lag, max_lag, expected_start, expected_end
        )
        # check inference for n <= output_chunk_length
        expected_start = self.target_time.end_time() - 12 * freq
        expected_end = self.target_time.end_time()
        test_routine_inf(self, ig, 1, expected_start, expected_end)
        test_routine_inf(self, ig, ocl, expected_start, expected_end)
        # check inference for n > output_chunk_length
        test_routine_inf(self, ig, ocl + 1, expected_start, expected_end + 1 * freq)

        # case 3: abs(min_covariates_lag) > input_chunk_length and abs(max_covariates_lag) > -1:
        # the start time of covariates begins before target start
        expected_start = self.target_time.start_time() - freq
        expected_end = self.target_time.end_time() - freq * (
            1 + self.output_chunk_length
        )
        min_lag, max_lag = -13, -2
        ig = test_routine_train(
            self, icl, ocl, min_lag, max_lag, expected_start, expected_end
        )
        # check inference for n <= output_chunk_length
        expected_start = self.target_time.end_time() - 12 * freq
        expected_end = self.target_time.end_time() - freq
        test_routine_inf(self, ig, 1, expected_start, expected_end)
        test_routine_inf(self, ig, ocl, expected_start, expected_end)
        # check inference for n > output_chunk_length
        test_routine_inf(self, ig, ocl + 1, expected_start, expected_end + 1 * freq)
