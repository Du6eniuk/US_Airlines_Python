# Overview
This analysis offers an in-depth exploration of the US domestic airline industry in 2022. Initially conceived as a project to demonstrate my data analysis skills, it evolved into a valuable resource for travelers seeking to minimize the risks of cancellations or delays during flights.

The data was sourced from a publicly available dataset on [Kaggle.com](https://www.kaggle.com/datasets/jl8771/2022-us-airlines-domestic-departure-data/data), which includes comprehensive information on flights, aircraft, airlines, weather conditions, and more. Using Python, I explored various aspects such as the most popular airports, major airlines, aircraft utilization, and the likelihood of cancellations or delays based on the time of year, day of the week, or choice of airline. You can find the full project, along with additional insights not included in this file, here: [Project_Airline](Project_Airline.ipynb).

# The Questions

These are the quiestion I answer in my project:

1. Which airports are the most popular based on airline traffic, and how does this change month by month?
2. What are the largest operating and marketing airlines, and how do these numbers vary between short and long flights?
3. Which aircraft are the most commonly used, and what is the average age of these machines?
4. What are the primary reasons for flight cancellations?
5. Which airline has the highest rate of successful flights?
6. What time of day and year offers the lowest risk of flight cancellations or delays?

# Tools I Used

For this particular project I have utilized:

- **Python:** The core of this analysis, enabling deep data exploration and insight generation through various libraries:
    - **Pandas Library:** For data manipulation and analysis.
    - **Matplotlib Library:** For creating foundational visualizations.
    - **Seaborn Library:** For producing advanced and aesthetically pleasing visualizations.
- **Jupyter Notebooks:** Used for running Python scripts and documenting the analysis process.
- **Visual Studio Code:** My preferred environment for writing and executing Python code.
- **Git & GitHub:** Employed for version control and sharing the project, facilitating collaboration and tracking progress.

# Data Preparation and Cleanup

This section outlines the steps taken to prepare the data for analysis.

## Import & Clean Up Data

First, I imported the necessary libraries and loaded the dataset. Following this, I conducted initial data cleaning tasks to optimize resource usage. This involved retaining only the essential columns needed for the analysis, ensuring a more efficient and focused approach to data exploration.

```python
# Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading Data
df_2022 = pd.read_csv('CompleteData.csv')

# Making Sure There Are No Duplicates
df_2022.shape
df_2022_ND = df_2022.drop_duplicates()
df_2022_ND.shape

# Dropping Unneccesery Columns
df_2022 = df_2022.drop(columns=['LOWEST_CLOUD_LAYER','N_CLOUD_LAYER','LOW_LEVEL_CLOUD','MID_LEVEL_CLOUD',
'HIGH_LEVEL_CLOUD','CLOUD_COVER','ACTIVE_WEATHER','WIND_SPD','WIND_GUST', 'LATITUDE','LONGITUDE','ELEVATION',
'VISIBILITY','TEMPERATURE','DEW_POINT','REL_HUMIDITY','ALTIMETER','WIND_DIR','RANGE','WIDTH','MESONET_STATION','TAXI_OUT']) 
```

# The Analysis

Each question in the project is broken down into four critical components:

1. Analysis
This section covers the approach and methods used to address the question. It includes the data manipulation, filtering, and calculations performed to derive meaningful insights from the dataset.
2. Visualization
Here, I present the graphical representation of the analyzed data. Visualizations are created using libraries like Matplotlib and Seaborn to make the data easier to interpret and to highlight key patterns and trends.
3. Results
This part provides a visualization.
4. Insights
In this section, I interpret the results, discussing their implications and relevance. Insights may also include recommendations or observations about broader trends or patterns identified during the analysis.

## 1. Which airports are the most popular based on airline traffic?

To identify the most popular airport in 2022, I grouped the data based on the destination or origin of each flight. To make the airport codes more understandable, I appended the corresponding city names (e.g., ATL became ATL (Atlanta)). This approach ensures clarity for everyone, not just those familiar with airport codes. Additionally, instead of focusing solely on the top airport, I expanded the analysis to visualize the top 5 airports, providing a broader view of the most active hubs in the US for 2022.

```python
# Grouping for DEST
df_top_5_destination = df_2022.groupby('DEST').size().sort_values(ascending=False).reset_index(name = 'Count').head(5)
# Grouping for ORIGIN
df_top_5_origin =  df_2022.groupby('ORIGIN').size().sort_values(ascending=False).reset_index(name = 'Count').head(5)
# Creating a Dictionary for Mapping
airport_name_mapping = {
    'ATL': 'ATL (Atlanta)', 
    'ORD' : 'ORD (Chicago)',
    'DEN' : 'DEN (Denver)',
    'DFW' : 'DFW (Dallas)',
    'CLT' : 'CLT (Charlotte)'
}
# Applying Our Mapping
df_top_5_destination['airport_full_name'] = df_top_5_destination['DEST'].map(airport_name_mapping)
df_top_5_origin['airport_full_name'] = df_top_5_origin['ORIGIN'].map(airport_name_mapping)
```
### Visualize Data
```python

# Using Subplots Since There are Two Groups
fig, ax = plt.subplots(2, 1)

# Visualizing Trip Destinations
sns.barplot(data=df_top_5_destination,  x = 'Count', y = 'airport_full_name', ax=ax[0], hue = 'Count')
ax[0].legend().remove()
ax[0].set_xlim([150_000, 320_000])
ax[0].set_xlabel('')
ax[0].set_ylabel('')
ax[0].set_title('Top 5 Airports as the Trips Destination in the US (2022)')
# Adding Major Formatter for Our Xaxis
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

# Visualizing Trip Origins
sns.barplot(data=df_top_5_origin,  x = 'Count', y = 'airport_full_name', ax=ax[1], hue = 'Count')
ax[1].legend().remove()
ax[1].set_xlim([150_000, 320_000])
ax[1].set_xlabel('')
ax[1].set_ylabel('')
ax[1].set_title('Top 5 Airports as the Trips Origin in the US (2022)')
# Adding Major Formatter for Our Xaxis
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
plt.tight_layout()
plt.show()
```
### Results
![Top 5 Aiports as the Trips Destionation and Origin in the US (2022)](Images\top_5_airports_in_2022.png)

*Bar graphs visualizing top 5 most popular trip origin and destination based on the count of trips.*

### Insights:

- Dominance of ATL (Atlanta): Whether considering trip origins or destinations, ATL (Hartsfield-Jackson Atlanta International Airport) is the busiest airport in the US, and indeed, the world, with over 600,000 trips recorded in 2022.
- Key Air Traffic Hubs: The next most popular airports in order are ORD (Chicago), DEN (Denver), DFW (Dallas), and CLT (Charlotte). These cities, though not as large as New York or Los Angeles, serve as major air traffic hubs in the US
- Surprising Positions for Major Cities: Notably, Los Angeles International Airport (LAX) ranks 6th, and LaGuardia Airport (LGA) in New York ranks 9th, indicating that some of the largest US cities are not necessarily the busiest in terms of air traffic.
- Consistency Across Metrics: The ranking remains consistent whether analyzing airports as trip origins or destinations, highlighting the stability of these hubs in the US air travel network.

## 1.1. How does this change month by month?

To analyze how airport traffic varies by month, the data must first be grouped accordingly. Before doing so, a new column must be created to represent the month extracted from the date information. Additionally, it's crucial to ensure that the data is sorted in the correct chronological order since string sorting defaults to alphabetical, which could misalign the months. This process will allow us to effectively visualize the total traffic for both arrivals and departures.

```python
# We Make Sure FL_DATE is in Datetime Format
df_2022['FL_DATE'] = pd.to_datetime(df_2022['FL_DATE'])
# Getting Month in a Correct String Format
df_2022['FL_MONTH_SHORT'] = df_2022['FL_DATE'].dt.strftime('%b')

# Creating Filter Based on Top 5 Aiports
top_5_airports_list_ORIGIN = list(df_2022.groupby('ORIGIN').size().sort_values(ascending=False).head(5).index)
# Creating Second List is Kind of Extra Since They are The Same
top_5_airports_list_DEST = list(df_2022.groupby('DEST').size().sort_values(ascending=False).head(5).index)

# Applying and Grouping
df_2022_top_5_ORIGIN = df_2022[df_2022['ORIGIN'].isin(top_5_airports_list_ORIGIN)]

df_2022_ORIGIN = df_2022_top_5_ORIGIN.groupby(['FL_MONTH_SHORT','ORIGIN']).size().reset_index(name = 'ORIGIN_COUNT')

df_2022_top_5_DEST = df_2022[df_2022['DEST'].isin(top_5_airports_list_DEST)]

df_2022_DEST = df_2022_top_5_DEST.groupby(['FL_MONTH_SHORT','DEST']).size().reset_index(name = 'DEST_COUNT')

# Merging to Find Total
df_merged = pd.merge(df_2022_ORIGIN, df_2022_DEST, left_on = ['FL_MONTH_SHORT', 'ORIGIN'], right_on = ['FL_MONTH_SHORT','DEST'], how = 'inner')

df_merged['TOTAL_FLIGHTS'] = df_merged['ORIGIN_COUNT'] + df_merged['DEST_COUNT']

# Define the Correct Order of Months
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

df_merged['FL_MONTH_SHORT'] = pd.Categorical(df_merged['FL_MONTH_SHORT'], categories=month_order, ordered=True)

# Once Again We Create and Apply Mapping
df_2022_to_plot = df_merged.sort_values('FL_MONTH_SHORT')
airport_name_mapping = {
    'ATL': 'ATL (Atlanta)', 
    'ORD' : 'ORD (Chicago)',
    'DEN' : 'DEN (Denver)',
    'DFW' : 'DFW (Dallas)',
    'CLT' : 'CLT (Charlotte)'
}
df_2022_to_plot['airport_full_name'] = df_2022_to_plot['ORIGIN'].map(airport_name_mapping)
```

### Visualize Data

```python
sns.set_style('ticks')
plt.figure(figsize=(10, 6))

sns.lineplot(data=df_2022_to_plot, x='FL_MONTH_SHORT',
 y='TOTAL_FLIGHTS', hue = 'airport_full_name',
 marker = 'o', linewidth=3, palette='deep')

ax = plt.gca() # used to apply formatter later
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
sns.despine()
plt.legend().remove()
plt.title('Top 5 Airports by Number of Flights in the US (2022)', fontsize = 14)
plt.ylabel('')
plt.xlabel('')

palette = sns.color_palette('deep') # We can access each color that seaborn uses for values and then assign it to the list. Now we can use indexes to get the same color for values as our lines are using.
colors = {
'ORD (Chicago)': palette[0],
'DFW (Dallas)': palette[1],
'DEN (Denver)': palette[2],
'CLT (Charlotte)': palette[3],
'ATL (Atlanta)': palette[4]
}

# Mapping Month to Index
month_to_index = {month: i for i, month in enumerate(df_2022_to_plot['FL_MONTH_SHORT'].unique())} # we basically required this part of the code to be able to adjust our x-value
# since FL_MONTH is non numerical value we mapped each month to its respective index, number we can now manipulate. Chat GPT helped me to code it. 

# Plotting Each Airport to Its respective position
for airport in df_2022_to_plot['airport_full_name'].unique():
    last_row = df_2022_to_plot[df_2022_to_plot['airport_full_name'] == airport].iloc[-1]
    x_value = month_to_index[last_row['FL_MONTH_SHORT']] + 0.2  # Small offset to the right
    plt.text(x_value, last_row['TOTAL_FLIGHTS'], airport, color=colors[airport], fontsize=12)

# these offset values are used to properly change the position of each number
offset_max_y = {
'ORD (Chicago)': 500,
'DFW (Dallas)': -1000,
'DEN (Denver)': 300,
'CLT (Charlotte)': 500,
'ATL (Atlanta)': 500
}
offset_max_x = {
'ORD (Chicago)': 0,
'DFW (Dallas)': -0.3,
'DEN (Denver)': 0,
'CLT (Charlotte)': 0,
'ATL (Atlanta)': 0
}
offset_min_y = {
'ORD (Chicago)': -700,
'DFW (Dallas)': -400,
'DEN (Denver)': -1000,
'CLT (Charlotte)': -700,
'ATL (Atlanta)': -800
}
offset_min_x = {
'ORD (Chicago)': -0.4,
'DFW (Dallas)': 0.2,
'DEN (Denver)': 0,
'CLT (Charlotte)': 0,
'ATL (Atlanta)': -0.5
}

# this loop is used to plot min and max values with respect to self-defined offsets. 
for airport in df_2022_to_plot['airport_full_name'].unique():
    # Filter data for the current airport
    airport_data = df_2022_to_plot[df_2022_to_plot['airport_full_name'] == airport]
    
    # Find min and max values
    max_value = airport_data['TOTAL_FLIGHTS'].max()
    min_value = airport_data['TOTAL_FLIGHTS'].min()

    # Get corresponding months for min and max values
    max_month = airport_data[airport_data['TOTAL_FLIGHTS'] == max_value]['FL_MONTH_SHORT'].values[0]
    min_month = airport_data[airport_data['TOTAL_FLIGHTS'] == min_value]['FL_MONTH_SHORT'].values[0]
    
    # Annotate max value
    plt.text(month_to_index[max_month] + offset_max_x[airport], max_value  + offset_max_y[airport], f'{int(max_value/1000)}K Max', color=colors[airport], ha='center', fontsize = 11) # we used the same dictionary
    # Min value
    plt.text(month_to_index[min_month] + offset_min_x[airport], min_value +  offset_min_y[airport], f'{int(min_value/1000)}K Min', color=colors[airport], ha='left', fontsize = 11)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12) 
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
plt.tight_layout()
plt.show()
```
### Results
![Top 5 Aiports Based on Count of Flights by Month](Images\top_5_airports_month_2022.png)

*Line graph visualizing the top 5 most popular airports based on a month.*

### Insights:

- The graph reveals that air traffic typically peaks during the summer months, from June through August, which aligns with the vacation season when most people travel.
- February stands out as the least busy month for air travel, with 4 out of the top 5 airports experiencing their lowest traffic levels during this period.ts in our top reaching its minimum during this period. 
- Notably, Atlanta's airport reaches an impressive capacity, handling up to 55,000 flights in a single month at its peak.
- This analysis suggests that airports may require a larger workforce during the summer to manage the increased traffic, while in the winter, a reduced number of staff might suffice.

## 2. What are the biggest operation and marketing airlines and how this value changes based on the flight being short of long?

To grasp the following analysis, it’s essential to understand that each flight involves two types of carriers: the marketing carrier, which is the airline selling the flight, and the operating carrier, which is the airline actually operating the flight. These are represented by the MKT_AIRLINE and OP_AIRLINE columns, respectively. While they can be the same, they often differ.

I aim to highlight the largest airlines in each group separately. To do this, we split the data into two distinct datasets, which we then visualize using subplots. Additionally, the data is categorized based on flight duration, dividing it into short and long flights. The criteria for this classification are:
1) Short flights: 120 minutes or less.
2) Long flights: More than 120 minutes.

Two separate dataframes are created, grouping the data by airline name and splitting the counts based on flight duration. We then calculate the percentage of each type relative to the total for each airline. This visualization and method were not initially planned but evolved throughout the analysis. The full process is documented in the Project_Airline file.

Note: We will perform a join (merge) on some foreign keys in our main table, specifically the MKT_UNIQUE_CARRIER and OP_UNIQUE_CARRIER columns.

```python
# Reading new table to get full names of airlines
df_carriers = pd.read_csv('Carriers.csv')

# Joining tables to get both MKT and OP airlines
df_merged = pd.merge(df_2022, df_carriers, 
                     left_on='MKT_UNIQUE_CARRIER', right_on='CODE', how='inner')

# Drop the 'CODE' column and rename 'DESCRIPTION' to 'MKT_AIRLINE'
df_merged = df_merged.drop(columns=['CODE']).rename(columns={'DESCRIPTION': 'MKT_AIRLINE'})

df_merged = pd.merge(df_merged, df_carriers, 
                     left_on='OP_UNIQUE_CARRIER', right_on='CODE', how='inner')

# Drop the 'CODE' column and rename 'DESCRIPTION' to 'OP_AIRLINE'
df_merged = df_merged.drop(columns=['CODE']).rename(columns={'DESCRIPTION': 'OP_AIRLINE'})

# Adding our custom metrics
df_merged['SHORT_OR_LONG'] = df_merged['AIR_TIME'].apply(lambda x: 'Long Flight' if x > 120 else 'Short Flight')

# Clean airline columns
df_merged['MKT_AIRLINE'] = df_merged['MKT_AIRLINE'].str.replace(r'Inc.|Co.', '', regex=True).str.strip()
df_merged['OP_AIRLINE'] = df_merged['OP_AIRLINE'].str.replace(r'Inc.|Co.', '', regex=True).str.strip()

# Finding top 5 MKT and OP airlines
top_5_MKT_airlines = df_merged.groupby('MKT_AIRLINE').size().reset_index(name='COUNT').sort_values(by='COUNT', ascending= False).head(5)
top_5_OP_airlines = df_merged.groupby('OP_AIRLINE').size().reset_index(name='COUNT').sort_values(by='COUNT', ascending= False).head(5)

# Now we need to filter for top 5 airlines since we can't use just a .head()
MKT_airlines = df_merged.groupby(['SHORT_OR_LONG', 'MKT_AIRLINE']).size().reset_index(name = 'COUNT').sort_values(by='COUNT', ascending = False)
top_5_MKT_list = list(top_5_MKT_airlines['MKT_AIRLINE'])
MKT_airlines = MKT_airlines[MKT_airlines['MKT_AIRLINE'].isin(top_5_MKT_list)]

# Let's prepare same dataset for operated airlines
OP_airlines = df_merged.groupby(['SHORT_OR_LONG', 'OP_AIRLINE']).size().reset_index(name = 'COUNT').sort_values(by='COUNT', ascending = False)
top_5_OP_list = list(top_5_OP_airlines['OP_AIRLINE'])
OP_airlines = OP_airlines[OP_airlines['OP_AIRLINE'].isin(top_5_OP_list)]

# Calculate the total number of flights per airline
total_flights = MKT_airlines.groupby('MKT_AIRLINE')['COUNT'].sum().reset_index(name='TOTAL_FLIGHTS')
# Merge total flights back into the original DataFrame
MKT_airlines = MKT_airlines.merge(total_flights, on='MKT_AIRLINE')
# Calculate the percentage
MKT_airlines['PERCENTAGE'] = round((MKT_airlines['COUNT'] / MKT_airlines['TOTAL_FLIGHTS'])* 100, 2)

# Calculate the total number of flights per airline
total_flights = OP_airlines.groupby('OP_AIRLINE')['COUNT'].sum().reset_index(name='TOTAL_FLIGHTS')
# Merge total flights back into the original DataFrame
OP_airlines = OP_airlines.merge(total_flights, on='OP_AIRLINE')
# Calculate the percentage
OP_airlines['PERCENTAGE'] = round((OP_airlines['COUNT'] / OP_airlines['TOTAL_FLIGHTS']) * 100, 2)
```

### Visualize Data

```python
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
sns.barplot(data=MKT_airlines,  x = 'COUNT', y = 'MKT_AIRLINE', ax=ax[0], hue = 'SHORT_OR_LONG', palette='deep')
sns.despine()
ax[0].legend().remove()
ax[0].set_xlim([0, 1_300_000])
ax[0].set_xlabel('')
ax[0].set_ylabel('Marketing Carrier', fontsize=12, fontweight='bold')
ax[0].set_title('Top 5 Airlines by The Number of Flights in the US (2022)',fontsize=16)
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
ax[0].tick_params(axis='y', labelsize=11)
ax[0].tick_params(axis='x', labelsize=11)
# Mapping airlines to indices for MKT_airlines. Once again, we are doing it because we can't manipulate the position of airlines as strings. We need indexes. 
airline_to_index = {airline: i for i, airline in enumerate(MKT_airlines['MKT_AIRLINE'].unique())}

# Annotate Marketing Carrier bars
for index, row in MKT_airlines[MKT_airlines['SHORT_OR_LONG'] == 'Short Flight'].iterrows():
    count = row['COUNT']
    airline = row['MKT_AIRLINE']
    percentage = row['PERCENTAGE']
    y_value = airline_to_index[airline] - 0.175  # Adjust the position slightly
    ax[0].text(count + 5000, y_value, f'{percentage:.1f}%', color='black', va='center', fontsize=11)

for index, row in MKT_airlines[MKT_airlines['SHORT_OR_LONG'] == 'Long Flight'].iterrows():
    count = row['COUNT']
    airline = row['MKT_AIRLINE']
    percentage = row['PERCENTAGE']
    y_value = airline_to_index[airline] + 0.2  # Adjust the position slightly
    ax[0].text(5000, y_value, f'{percentage:.1f}%', color='black', va='center', fontsize=11)


sns.barplot(data=OP_airlines,  x = 'COUNT', y = 'OP_AIRLINE', ax=ax[1], hue = 'SHORT_OR_LONG', palette='deep')
ax[1].set_xlabel('')
ax[1].set_xlim([0, 1_300_000])
ax[1].set_ylabel('Operating Carrier', fontsize=12, fontweight='bold')
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
handles, labels = ax[1].get_legend_handles_labels()
labels = ['Short Flight (≤120 min.)', 'Long Flight (>120 min.)']  # New labels
ax[1].legend(handles=handles, labels=labels, title='Type of Flight', fontsize=12, title_fontsize=13, loc='lower right')
ax[1].tick_params(axis='y', labelsize=11)
ax[1].tick_params(axis='x', labelsize=11)

# Mapping airlines to indices for OP_airlines
airline_to_index_op = {airline: i for i, airline in enumerate(OP_airlines['OP_AIRLINE'].unique())}

# Annotate Operating Carrier bars
for index, row in OP_airlines[OP_airlines['SHORT_OR_LONG'] == 'Short Flight'].iterrows():
    count = row['COUNT']
    airline = row['OP_AIRLINE']
    percentage = row['PERCENTAGE']
    y_value = airline_to_index_op[airline] - 0.175  # Adjust the position slightly
    ax[1].text(count + 5000, y_value, f'{percentage:.1f}%', color='black', va='center', fontsize=11)

for index, row in OP_airlines[OP_airlines['SHORT_OR_LONG'] == 'Long Flight'].iterrows():
    count = row['COUNT']
    airline = row['OP_AIRLINE']
    percentage = row['PERCENTAGE']
    y_value = airline_to_index_op[airline] + 0.2  # Adjust the position slightly
    ax[1].text(5000, y_value, f'{percentage:.1f}%', color='black', va='center', fontsize=11)

plt.show()
```
### Results
![Top 5 Airlines Based on Count of Flights Grouped by Type of Carrier and Type of Flight](Images\top_5_airlines.png)

*Bar Graph visualizing the top 5 most popular airlines based on the type of carrier and type of flight. Sorted by the short flights in descending order.*

### Insights:

- American Airlines, Delta Air Lines, and Southwest Airlines are the top performers, each having a significant percentage of their flights as short flights (≤120 min). For instance, American Airlines and Delta Air Lines have approximately 70% of their flights categorized as short flights, indicating a focus on shorter domestic routes.
- United Air Lines shows a relatively balanced split between short and long flights. This suggests that United has a more diverse range of flight durations, catering to both shorter and longer routes across the US.
- Alaska Airlines has a unique profile compared to other airlines, with a significant proportion (around 43%) of its flights being long flights (>120 min). This likely reflects Alaska’s strategic focus on longer domestic routes, possibly due to its geographic base and the locations it primarily serves.
- SkyWest Airlines, predominantly operating as a regional airline, has an overwhelming majority of its flights categorized as short flights (approximately 88%). This highlights its role in operating shorter, regional routes as a partner to major airlines like United and Delta. In addition, SkyWest Airlines appears prominently as an operating carrier but is absent from the top marketing carriers. This highlights SkyWest’s role as a regional operator, primarily operating flights on behalf of other major airlines. It’s a clear example of how some airlines focus more on operational roles rather than marketing their own brand.
- Marketing vs. Operating Dynamics: The differences between the two graphs illustrate the varied roles airlines play in the market. Some airlines like United and Delta not only market flights but also operate a significant number of them, while others like SkyWest are more focused on operations, often under contract with the larger airlines.
- Note: In case you want to see the biggest airline based on number of trips, there is a version without type of flights in the main project.

## 3. Which aircraft are the most commonly used, and what is the average age of these machines?

First Graph: We utilize sns.displot, which takes a column from our dataset and automatically generates a distribution plot. Afterward, we enhance the visualization by adding lines based on our calculated statistics.

Second Graph: To prepare this, we first need to remove any duplicates based on the TAIL_NUM column, which represents the FAA N-Number/Registration — a unique identifier for each aircraft. Once the data is cleaned, we can group and count the relevant values to generate our insight

```python
# This will be used for the first graph
n_bins = int((df_2022['YEAR OF MANUFACTURE'].max() - df_2022['YEAR OF MANUFACTURE'].min()))

# This will be used for the second graph
df_unqiue_planes = df_merged.drop_duplicates(subset='TAIL_NUM')
# Grouping
df_planes = df_unqiue_planes.groupby(['MANUFACTURER','ICAO TYPE']).size().reset_index(name='COUNT').sort_values(by='COUNT', ascending=False)
# Creating a full plane model name
df_planes['FULL MODEL NAME'] = df_planes['ICAO TYPE'] + ' ' + '(' + df_planes['MANUFACTURER'] + ')'
```

### Visualize Data

```python
# First Graph
sns.displot(data = df_2022['YEAR OF MANUFACTURE'], binwidth=1, bins = n_bins, edgecolor='black',height=5, aspect=1.5,  color = palette[0])
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{int(y/1000)}K'))
plt.xlim(1990, 2022)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
mean_year = df_2022['YEAR OF MANUFACTURE'].mean()
median_year = df_2022['YEAR OF MANUFACTURE'].median()
# Adding lines based on calculations
plt.axvline(x=mean_year, color=palette[1], linestyle='--', label=f'Mean Year: {int(mean_year)}')
plt.axvline(x=median_year, color=palette[2], linestyle='--', label=f'Median Year: {int(median_year)}')

plt.legend()
plt.title('Distribution of Aircraft Manufacture Years in the US (2022)', fontsize=12)
plt.xlabel('')
plt.ylabel('')
plt.show()

# Second Graph
sns.barplot(data=df_planes.head(10),x='COUNT',y = 'FULL MODEL NAME', hue = 'MANUFACTURER', palette='deep')
sns.despine()
plt.ylabel('')
plt.xlabel('')
plt.title('Most Commonly Used Commercial Aircraft in the US (2022)',fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.show()
```
### Results Part I

![Distribution of Aircraft Manufacture Years in the US (2022)](Images\hist_planes.png)

*Distribution of aircraft manufacture years along with median and mean.*

### Insights Part I:

- Average Age of Fleet: The mean and median year of manufacture for aircraft in the US fleet is 2009, indicating that the typical aircraft in service is around 13 years old.
- Bimodal Peaks: There are two significant periods of aircraft manufacturing, with peaks around the early 2000s and mid-2010s, reflecting waves of fleet expansion or renewal during these times.
- Recent Increase in Manufacturing: A rise in the number of aircraft manufactured post-2015 suggests a trend toward fleet modernization in recent years.

### Results Part II
![Most Commonly Used Commercial Aircraft in the US (2022)](Images\bar_planes.png)

*The Bar Graph visualizes the top 10 most popular commercial aircraft along with the manufacturer.*

### Insights Part II:

- Boeing Dominance: Boeing is the leading manufacturer, with three of its models—B738, B737, and B739—occupying the top spots. The B738 is the most used aircraft, with around 750 planes.
- Airbus Presence: Airbus follows closely behind Boeing, with four of its models—A320, A321, A319, and A321—ranked among the top 10, showing Airbus's significant role in the US market.
- Regional Aircraft: Bombardier, a key player in the regional aircraft market, has three models (CRJ9, CRJ2, CRJ7) in the top 10, emphasizing the importance of smaller regional jets in the US.

## 4. What are the primary reasons for flight cancellations?

This section takes a brief look at the reasons behind flight cancellations. To calculate the percentage for each type of cancellation, we will use a new table that can be joined (merged) with our main dataset using the CANCELLED column as a foreign key. After merging the tables, we will group and filter the data to extract meaningful insights.

```python
# Reading new df
cancellation = pd.read_csv('Cancellation.csv')\

# Merging
pd_merged = pd.merge(df_merged,cancellation, left_on = 'CANCELLED', right_on = 'STATUS')
pd_merged.drop('STATUS', inplace = True, axis = 1)

# Grouping
cancelled_reasons_df = pd_merged.groupby('CANCELLATION_REASON').size().reset_index(name='COUNT').sort_values(by='COUNT', ascending = False)

# Filtering
cancelled_reasons_df = cancelled_reasons_df[cancelled_reasons_df['CANCELLATION_REASON'] != 'Not Cancelled']
```

### Visualize Data

```python
colors = sns.color_palette('deep')
fig, ax = plt.subplots() 

slices, labels, percent_labels = ax.pie(
    cancelled_reasons_df['COUNT'], 
    labels=None, # No labels because we use legend instead
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=(0, 0, 0,0.1)
)
ax.legend(
    slices, 
    cancelled_reasons_df['CANCELLATION_REASON'], 
    title="Cancellation Causes",
    loc="center left", 
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=12,
    title_fontsize='13'
)
# Used to avoid change color nad fontsize of percent labels
for percent in percent_labels:
    percent.set_color('black')
    percent.set_fontsize(12)

plt.title("Distribution of Flight Cancellations by Cause in the US (2022)", loc='center', fontsize = 15, x=1)
plt.tight_layout()
plt.show()
```

### Results

![Distribution of Flight Cancellations by Cause in the US (2022))](Images\cancel_reason.png)

*Pie Chart visualizing the distribution of flight cancellation causes in the US.*

### Insights:

- Weather-related Cancellations: Weather is the leading cause of flight cancellations, accounting for 52.8% of all cancellations in 2022. This indicates that external factors beyond the control of airlines are the most significant contributor to disruptions.
- Carrier-related Cancellations: Carrier-related issues, such as mechanical failures or crew shortages, make up 37.3% of the cancellations. This suggests a substantial number of cancellations are due to airline-specific operational challenges.
- National Air System and Security: National air system disruptions, like air traffic control delays, account for 9.8%, while security-related cancellations are extremely rare, comprising only 0.1% of the total. This highlights the relative rarity of cancellations due to security concerns compared to other factors.

## 5. Which airline has the highest rate of successful flights?

Now we are shifting our focus back to airlines, aiming to determine which company has the highest percentage of successful flights. By "successful," I mean the ability of airlines to avoid cancellations, even when weather conditions are factored in. Initially, I considered excluding weather-related cancellations from the analysis, but on second thought, the ability to predict and manage weather-related disruptions is a crucial aspect of the airline industry.

We will also include significant delays in our analysis, specifically those that exceed 30 minutes. To achieve this, I created a function called success_category. This function categorizes each flight as successful, significantly delayed, or canceled, based on the data in other columns. After defining these categories, we will group the data by airline and filter the results to focus on the top four airlines identified in previous visualizations.

```python
# Defining function
def success_category(row):
    if row['CANCELLED'] != 0:
        return 'Cancelled'
    elif row['DEP_DELAY'] > 30:
        return 'More Than 30m Delay'
    else:
        return 'Minor or No Delay'

# Applying self-defined function
pd_merged['SUCCESS_CATEGORY'] = pd_merged.apply(success_category, axis=1)

# Grouping, creating a List, and fitlering
pd_to_plot_success =pd_merged.groupby(['MKT_AIRLINE','SUCCESS_CATEGORY']).size().reset_index(name = 'COUNT').sort_values(by='COUNT',ascending=False)
top_4_airlines_list = list(pd_merged.groupby('MKT_AIRLINE').size().sort_values(ascending=False).head(4).index)
pd_to_plot_success = pd_to_plot_success[pd_to_plot_success['MKT_AIRLINE'].isin(top_4_airlines_list)]
```

### Visualize Data

```python
colors = sns.color_palette('deep')
custom_colors = [colors[0], colors[1], colors[3]]
airline_order = ['Delta Air Lines', 'United Air Lines', 'American Airlines', 'Southwest Airlines'] # used for loop

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes = axes.flatten()

for i, airline in enumerate(airline_order):
    airline_data = pd_to_plot_success[pd_to_plot_success['MKT_AIRLINE'] == airline]
    slices, labels, percent_labels = axes[i].pie(
        airline_data['COUNT'], 
        labels=None, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors = custom_colors,
        explode=(0, 0.05, 0.05)
        )
    axes[i].set_title(airline, fontsize = 13)
    for autotext in percent_labels:
        autotext.set_fontsize(12)  # Increase font size of pie chart values

fig.suptitle('Flight Status Distribution Across Major US Airlines (2022)', fontsize=15, y=1)

fig.legend(
    labels=airline_data['SUCCESS_CATEGORY'].unique(),
    title="Flight Status",
    loc="center", 
    fontsize=12,
    title_fontsize = 14,
    bbox_to_anchor=(0.25, 0, 0.5, 1)
)
    
plt.tight_layout()
plt.show()
```
### Results

![Flight Status Distribution Across Major US Airlines (2022)](Images\airline_flight_status.png)

*Pie Charts visualizing flight status distribution across major US airlines.*

### Insights:

- High Reliability Across Airlines: The majority of flights for all four airlines, Delta Air Lines, United Air Lines, American Airlines, and Southwest Airlines, experienced minor or no delays. Delta Air Lines leads with 88.2% of its flights in this category, indicating strong operational reliability.
- Moderate Delays Vary: The percentage of flights delayed by more than 30 minutes shows some variation among airlines. Southwest Airlines has the highest percentage in this category at 15.5%, suggesting potential challenges in on-time performance.
- High On-Time Performance: After additional research, the data reveals that Hawaiian Airlines leads with 90.47% of its flights experiencing minor or no delay, followed closely by Alaska Airlines and Delta Air Lines at 88.45% and 88.15%, respectively. This indicates strong on-time performance and reliability for these airlines in 2022. Therefore, if you want to avoid being late at all cost choose these airlines when possible.

## 6. What time of day and year offers the lowest risk of flight cancellations or delays?

In this final question, we aim to determine the best and worst times of day and year for your flight. Using the success_category column we created in the previous question, we will analyze the percentages of canceled and significantly delayed flights based on the time of day and the month. To achieve this, we will perform several groupings, total calculations, and data merging.

A crucial step in this process is creating a new column to show the estimated hour of departure since we can't rely on the actual departure time. The real departure time isn't viable because canceled flights typically default to a departure hour of 0, skewing the results.

In summary, this question involves transforming our dataset by adding a new column, grouping and merging the data, calculating totals to derive percentages, and finally visualizing the findings. The complete process, including why the current departure hour wouldn't work, is documented in detail in my full project.

```python
# Creating new column with estimated departure hour
pd_merged['CRS_DEP_TIME'] = pd.to_datetime(pd_merged['CRS_DEP_TIME'])
pd_merged['CRS_DEP_HOUR'] = pd_merged['CRS_DEP_TIME'].dt.hour

# Calculating size of each category
df_split = pd_merged.groupby(['CRS_DEP_HOUR','SUCCESS_CATEGORY']).size().reset_index(name = 'SPLIT')

# Calculating total
df_total = pd_merged.groupby('CRS_DEP_HOUR').size().reset_index(name = 'TOTAL')

# Merging
df_merged_cancellation = pd.merge(df_split,df_total)

# Finding Percent Based on Total
df_merged_cancellation['PERCENT'] = round(df_merged_cancellation['SPLIT'] / df_merged_cancellation['TOTAL'] * 100, 2)

# Avoiding Minor or No Delay flights
df_merged_cancellation = df_merged_cancellation[df_merged_cancellation['SUCCESS_CATEGORY'] != 'Minor or No Delay']

# Repeating same groupings for months (Second graph)
df_split_M = pd_merged.groupby(['FL_MONTH_SHORT','SUCCESS_CATEGORY']).size().reset_index(name = 'SPLIT')
df_total_M = pd_merged.groupby('FL_MONTH_SHORT').size().reset_index(name = 'TOTAL')
df_merged_cancellation_month = pd.merge(df_split_M,df_total_M)
df_merged_cancellation_month['PERCENT'] = round(df_merged_cancellation_month['SPLIT'] / df_merged_cancellation_month['TOTAL'] * 100, 2)
df_merged_cancellation_month = df_merged_cancellation_month[df_merged_cancellation_month['SUCCESS_CATEGORY'] != 'Minor or No Delay']
```

### Visualize Data

```python
# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Plot 1: Cancellation and Delay Percentage by Hour
sns.lineplot(data=df_merged_cancellation, x='CRS_DEP_HOUR', y='PERCENT', hue='SUCCESS_CATEGORY', palette=custom_colors, ax=ax1, legend=False)
ax1.set_xlim(0, 23)
ax1.set_ylim(0, None)
ax1.set_xticks(range(0, 24))
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Fill the area between 'More Than 30m Delay' and 'Cancelled'
ax1.fill_between(df_merged_cancellation[df_merged_cancellation['SUCCESS_CATEGORY'] == 'More Than 30m Delay']['CRS_DEP_HOUR'],
                 df_merged_cancellation[df_merged_cancellation['SUCCESS_CATEGORY'] == 'Cancelled']['PERCENT'],
                 df_merged_cancellation[df_merged_cancellation['SUCCESS_CATEGORY'] == 'More Than 30m Delay']['PERCENT'],
                 color=colors[1], alpha=0.2)

# Fill the area between 'Cancelled' and x-axis
ax1.fill_between(df_merged_cancellation[df_merged_cancellation['SUCCESS_CATEGORY'] == 'Cancelled']['CRS_DEP_HOUR'],
                 df_merged_cancellation[df_merged_cancellation['SUCCESS_CATEGORY'] == 'Cancelled']['PERCENT'],
                 0, color=colors[3], alpha=0.2)

ax1.set_title('Patterns in US Flight Cancellations and Delays (2022)', fontsize = 15)
ax1.set_xlabel('Scheduled Departure Hour', fontsize = 12)
ax1.set_ylabel('')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y}%'))
ax1.tick_params(axis='y', labelsize=11)
ax1.tick_params(axis='x', labelsize=11)

# Plot 2: Cancellation and Delay Percentage by Month
sns.lineplot(data=df_merged_cancellation_month, x='FL_MONTH_SHORT', y='PERCENT', hue='SUCCESS_CATEGORY', palette=custom_colors, ax=ax2)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Fill the area between 'More Than 30m Delay' and 'Cancelled'
ax2.fill_between(df_merged_cancellation_month[df_merged_cancellation_month['SUCCESS_CATEGORY'] == 'More Than 30m Delay']['FL_MONTH_SHORT'],
                 df_merged_cancellation_month[df_merged_cancellation_month['SUCCESS_CATEGORY'] == 'Cancelled']['PERCENT'],
                 df_merged_cancellation_month[df_merged_cancellation_month['SUCCESS_CATEGORY'] == 'More Than 30m Delay']['PERCENT'],
                 color=colors[1], alpha=0.2)

# Fill the area between 'Cancelled' and x-axis
ax2.fill_between(df_merged_cancellation_month[df_merged_cancellation_month['SUCCESS_CATEGORY'] == 'Cancelled']['FL_MONTH_SHORT'],
                 df_merged_cancellation_month[df_merged_cancellation_month['SUCCESS_CATEGORY'] == 'Cancelled']['PERCENT'],
                 0, color=colors[3], alpha=0.2)

ax2.set_xlim([df_merged_cancellation_month['FL_MONTH_SHORT'].min(), df_merged_cancellation_month['FL_MONTH_SHORT'].max()])
ax2.set_xlabel('Scheduled Departure Month', fontsize = 12)
ax2.set_ylabel('')
ax2.set_ylim(0, 22)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y}%'))

# Adjust the layout
plt.tight_layout()

# Add a single legend for both plots
handles, labels = plt.gca().get_legend_handles_labels()
ax2.legend(
    title="Flight Status",
    fontsize=12,
    title_fontsize=13
)
ax2.tick_params(axis='y', labelsize=11)
ax2.tick_params(axis='x', labelsize=11)
plt.show()
```
### Results

![Patterns in US Flight Cancellations and Delays (2022)](Images\cancellation_and_delay.png)

*Line charts visualizing patterns for cancelled and delayed flights based on hour of the day and month of the year.*

### Insights:

- Flights scheduled between midnight and 4 AM have relatively low cancellation and delay rates. This makes early morning hours one of the best times to fly if you want to avoid disruptions.
- The percentage of flights with more than 30 minutes of delay sharply increases from the early afternoon and peaks around 8 PM. This suggests that flying during late afternoon and evening increases the likelihood of encountering delays.
- The visualization indicates that during the winter months, particularly from December to February, flights are more prone to cancellations and delays. This is likely due to adverse weather conditions, which are more frequent during these months.
- Although July shows a slight peak, the overall trend during the summer months (June to August) is lower in terms of cancellations and delays compared to the winter. This makes summer a more reliable period for travel.

# What I Learned

Throughout this project, I developed a deeper understanding of the US airline industry, particularly in analyzing flight data. This experience also helped me sharpen my Python skills, especially in data analysis and visualization. Here are some specific insights:

- **Enhanced Python Proficiency**: Utilizing libraries like Pandas for data manipulation and Seaborn and Matplotlib for visualizations enabled me to efficiently tackle complex data analysis challenges specific to the airline industry.
- **Significance of Data Preparation**: I learned the critical importance of thorough data cleaning and preparation, which is essential for extracting accurate and actionable insights from large datasets like the US airline departure data.
- **Data-Driven Decision Making**: This project underscored the value of understanding how different factors such as flight delays, cancellations, and aircraft types influence airline performance, guiding more informed decision-making within the industry.

# Insights

- **Flight Time Impact on Performance**: The analysis revealed that shorter flights (under 120 minutes) are associated with a higher volume of flights across major airlines, particularly for carriers like Southwest and Delta. This suggests that shorter routes are crucial to their operational strategy and are likely prioritized in terms of scheduling and resource allocation.
- **Marketing vs. Operating Carrier Dynamics**: There is a clear distinction between the performance of marketing carriers and operating carriers. For instance, marketing carriers like American Airlines show a strong presence in long-haul flights, whereas operating carriers like SkyWest specialize in shorter, regional routes. This difference highlights the strategic partnerships and roles within the industry, where certain carriers focus on branding while others handle the logistics of flight operations.
- **Aircraft Age Trends**: The analysis of aircraft manufacturing years indicated that many planes in the US fleet are relatively modern, with a notable increase in planes manufactured post-2000. This suggests that the industry has undergone significant fleet upgrades in recent decades, likely driven by advances in fuel efficiency and technology.
- **Cancellation Reasons**: The majority of flight cancellations are attributed to weather conditions, making up over 50% of all cancellations. This emphasizes the ongoing challenge that airlines face in mitigating weather-related disruptions, which remains a critical factor in operational reliability.
- **Best Times to Fly**: Based on the data, early morning flights (around 6-8 AM) and winter months (particularly January and February) are associated with higher cancellation rates and delays. Conversely, flights scheduled during midday and the summer months tend to have fewer disruptions, making them the optimal times for travelers seeking to avoid delays and cancellations. In addition, to minimize delays and cancellations you can also stick with Hawaiian Airlines, Alaska Airlines, or Delta Air Lines.

# Challenges I Faced

This project came with several challenges that provided valuable learning experiences:

- **Handling Large Datasets**: Managing and processing a substantial dataset with millions of flight records posed difficulties, particularly in terms of performance and memory usage. This required optimizing the code and using efficient data manipulation techniques to ensure smooth analysis.

- **Visualizing Multidimensional Data**: Creating clear and informative visualizations from complex, multidimensional data sets was a significant challenge. It involved experimenting with various graph types and customization options to present the insights effectively.

- **Integrating Various Data Sources**: Merging different datasets, each with its own structure and inconsistencies, was a challenge that required careful data cleaning and alignment to ensure that the analysis was accurate and meaningful. Balancing the integration while maintaining data integrity was crucial for the project's success.

# Conclusion

This exploration into the US airline industry has been highly insightful, revealing the key patterns and trends that influence this dynamic sector. The analysis has deepened my understanding of the factors that affect flight performance, such as delays and cancellations, and offers valuable insights for improving operational efficiency. As the aviation industry continues to evolve, continuous analysis will be crucial for staying informed and competitive. This project lays a strong foundation for future research, emphasizing the need for ongoing learning and adaptation in the rapidly changing landscape of air travel. I already have so many ideas how to imrove this project and visualizations that I use. You can learn about some of them inside my main file - [Project_Airline](Project_Airline.ipynb).

## P.S. Step Back

Before diving into the main analysis, it’s important to note that my project initially started with a different dataset that appeared to contain 20 years' worth of data. However, I quickly realized that this data was likely AI-generated and incorrect. After a brief analysis, it became evident that the trends and patterns were unrealistic and inconsistent with genuine historical data.

Here’s a quick analysis and a graph that illustrate why this dataset was problematic. The patterns did not align with known industry trends, and some data points were implausible, such as the same flight numbers appearing repeatedly across different years without any variation in frequency or patterns.

This discovery led me to switch to a more reliable dataset for the project. Ensuring the accuracy of the data is crucial for drawing meaningful conclusions, and this experience reinforced the importance of data validation in any analysis.

### Analysis on The "Fake" Data

I am not going to go in details here since I have already spend a lot of time analysing this "fake" dataset when I started working on this project >:|

```python
# Load Data
df = pd.read_csv('US Airline Flight Routes and Fares.csv')
# Decided to remove 90s to get a more precise scope, 2024 was deleted since the data for that year was incomplete
years_to_remove = [1993, 1994, 1995, 1996, 1997, 1998, 1999, 2024]
df_excluding_years = df[~df['Year'].isin(years_to_remove)].copy()
# Addding New Column
df_pivot.loc['Total'] = df_pivot.sum()
# Sorting
df_pivot = df_pivot[df_pivot.loc['Total'].sort_values(ascending=False).index]
# Filtering for Top 5
top_N = 5
df_to_plot = df_pivot.iloc[:,:top_N]
# Drop Row
df_to_plot = df_to_plot.drop('Total')
# Reseting and Melting
df_to_plot.reset_index(inplace = True)
df_melted = df_to_plot.melt(id_vars=['Year'], var_name='Airport', value_name='Count')
```
### Visualisation of The "Fake" Data

```python
# Plotting
ax = sns.lineplot(data=df_to_plot, x='Year', y='Count', hue='airport_2')

# Custom Offsets
offsets = {
    'DCA': 0,
    'BWI': -3,
    'IAD': -6,
    'TPA': 0,
    'SFO': 0
}
# Custom Colors
colors = {
    'DCA': 'blue',
    'BWI': 'orange',
    'IAD': 'green',
    'TPA': 'red',
    'SFO': 'purple'
}
# Loop for Labels
for airport in df_to_plot['airport_2'].unique():
    latest_year = df_to_plot[df_to_plot['Year'] == 2023]
    y_value = latest_year[latest_year['airport_2'] == airport]['Count'].values[0]
    plt.text(2023.3, y_value + offsets[airport], airport, color=colors[airport])

# Adding a Custom Line
plt.axvline(x=2019, color='red', linestyle='--')
plt.text(2019.5, 360, 'Start of COVID', color = 'black')

plt.show()
```
### Results of The "Fake" Data

![Visualization of Top Airports Based on The Fake Data](Images\fake_data.png)

As you can see, the five airports identified in this analysis do not correspond with the top five most popular airports in the US, as confirmed by a quick Google search. This discrepancy made it clear that the data was flawed. While I chose to leave this graph somewhat incomplete, I still decided to share it because I believe it's important to present all aspects of my project, even the less successful ones.

## P.S.S.
Thank you and Luke Barousse for the course "Python for Data Analytics" that inspired me to finish this project. Glory to Ukraine!