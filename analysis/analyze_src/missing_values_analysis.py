from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Missing Value Analysis
# ----------------------------------------------
# This class defines a template for missing value analysis
# Subclasses must implement the methods to identify and visualize missing values
class MissingValueAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifiying and visualizing missing values.

        # Parameters:
        df(pd.DataFrame): The dataframe to be analyzed.

        # Returns:
        None: This method performs the analysis and visualizes the missing values.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        # Parameters:
        df(pd.DataFrame): The dataframe to be analyzed.

        # Returns:
        None: This method identifies the missing values in the dataframe.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes the missing values in the dataframe.

        # Parameters:
        df(pd.DataFrame): The dataframe to be analyzed.

        # Returns:
        None: This method visualizes the missing values in the dataframe.
        """
        pass


# Concrete Class for Missing Value Identification
# ----------------------------------------------
# This class implements methods to identify and visualize missing values in the dataframe.
class SimpleMissingValuesAnalysis(MissingValueAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Print the count of missing values for each column in the DataFrame.

        Parameters:
        df(pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values count to the console.
        """
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values>0])

    def visualize_missing_values(self, df: pd.DataFrame):
        """"
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df(pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Display a heatmap of missing values in the dataframe.
        """
        print("\nMissing Values Heatmap:")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage of the SimpleMissingValuesAnalysis class.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Missing Values Analysis
    # missing_values_analyzer = SimpleMissingValuesAnalysis()
    # missing_values_analyzer.analyze(df)
    pass