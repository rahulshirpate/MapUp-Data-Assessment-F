import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    id_1_values = df['id_1'].unique()
    id_2_values = df['id_2'].unique()

    car_matrix = pd.DataFrame(index=id_1_values, columns=id_2_values)

    for index, row in df.iterrows():
        car_matrix.loc[row['id_1'], row['id_2']] = row['car']

    return car_matrix


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Creating a new column 'car_type' based on conditions
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculating count of occurrences for each 'car_type' category
    car_type_counts = df['car_type'].value_counts().to_dict()

    # Sorting the dictionary alphabetically based on keys
    car_type_counts = dict(sorted(car_type_counts.items()))

    return car_type_counts


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Calculating the mean of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identifying indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    
    # Sorting the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Calculating the average 'truck' values for each route
    avg_truck_values = df.groupby('route')['truck'].mean()
    
    # Filtering routes where average 'truck' values are greater than 7
    filtered_routes = avg_truck_values[avg_truck_values > 7].index.tolist()

    # Sorting the routes in ascending order
    filtered_routes.sort()

    return filtered_routes


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    modified_matrix = matrix.copy()  # Creating a copy of the input DataFrame

    # Applying custom conditions to modify the values
    modified_matrix[matrix > 20] *= 0.75
    modified_matrix[matrix <= 20] *= 1.25

    # Rounding the modified values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Converting timestamp columns to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extracting day of the week and time from timestamp
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['time'] = df['timestamp'].dt.time

    # Group by 'id' and 'id_2' and check for incorrect timestamps
    grouped = df.groupby(['id', 'id_2'])

    # Checking if each group covers a full 24-hour period and spans all 7 days of the week
    completeness_check = grouped.apply(
        lambda x: (
            (x['time'].min() <= pd.Timestamp('00:00:00').time()) and
            (x['time'].max() >= pd.Timestamp('23:59:59').time()) and
            (sorted(x['day_of_week'].unique()) == list(range(7)))
        )
    )

    return completeness_check
