import numpy as np
import pandas as pd
from scipy import signal

class DataPreprocessor:
    def __init__(self, sampling_rate: int = 100):
        """
        Initialize preprocessing parameters
        
        Args:
            sampling_rate (int): Sensor sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Raw input dataset
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        # Remove NaN values
        df_cleaned = self._handle_missing_values(df)
        
        # Apply filtering
        df_filtered = self._apply_filters(df_cleaned)
        
        # Normalize sensor data
        df_normalized = self._normalize_data(df_filtered)
        
        return df_normalized

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataset
        
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        # Remove rows with too many missing values
        threshold = 0.3  # 30% missing data
        df_cleaned = df.dropna(thresh=len(df.columns) * (1 - threshold))
        
        # Interpolate remaining missing values
        df_cleaned = df_cleaned.interpolate(method='linear')
        
        return df_cleaned

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply digital filters to sensor data
        
        Args:
            df (pd.DataFrame): Input dataset
        
        Returns:
            pd.DataFrame: Filtered dataset
        """
        # Low-pass Butterworth filter
        def butter_lowpass_filter(data, cutoff=5, order=6):
            nyquist = 0.5 * self.sampling_rate
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            return signal.filtfilt(b, a, data)
        
        # Acceleration columns to filter
        acc_columns = [col for col in df.columns if 'acc1' in col]
        
        for col in acc_columns:
            df[f'{col}_filtered'] = butter_lowpass_filter(df[col])
        
        return df

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize sensor data using min-max scaling
        
        Args:
            df (pd.DataFrame): Input dataset
        
        Returns:
            pd.DataFrame: Normalized dataset
        """
        # Columns to normalize (sensor data)
        sensor_columns = [
            col for col in df.columns 
            if any(sensor in col for sensor in ['acc', 'gyro', 'mag']) 
            and 'filtered' in col
        ]
        
        # Min-max normalization
        for col in sensor_columns:
            df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        return df

    def segment_data(self, df: pd.DataFrame, window_size: int = 100, overlap: int = 50) -> list:
        """
        Segment time series data into windows
        
        Args:
            df (pd.DataFrame): Input dataset
            window_size (int): Number of samples in each window
            overlap (int): Number of overlapping samples
        
        Returns:
            List of segmented dataframes
        """
        segments = []
        start = 0
        while start + window_size <= len(df):
            segment = df.iloc[start:start+window_size]
            segments.append(segment)
            start += (window_size - overlap)
        
        return segments