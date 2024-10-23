import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

# Load the data
rates = pd.read_csv('RATES.csv')
df = pd.DataFrame(rates)

# Ensure 'DATES' is in datetime format and extract the year
df['DATES'] = pd.to_datetime(df['DATES'], format='%d-%m-%Y', errors='coerce')  # Parse dates, errors='coerce' handles invalid formats
df['year'] = df['DATES'].dt.year  # Extract the year from the 'DATES' column

# Drop any rows where the date conversion failed (optional)
df = df.dropna(subset=['DATES'])

# Initialize the app
app = Dash(__name__)

# Create a dropdown for selecting the fuel type (Petrol/Diesel)
fuel_dropdown = dcc.Dropdown(
    options=[
        {'label': 'Petrol (Daily)', 'value': 'PETROL'},
        {'label': 'Diesel (Daily)', 'value': 'DIESEL'}
    ],
    value='PETROL',  # Default value
    id='fuel-dropdown'
)

# Create a dropdown for selecting the year
year_dropdown = dcc.Dropdown(
    options=[{'label': str(year), 'value': year} for year in sorted(df['year'].unique())],
    value=df['year'].min(),  # Default value is the earliest year in the dataset
    id='year-dropdown'
)

# Define the layout
app.layout = html.Div(children=[
    html.H1(children='Fuel Prices Dashboard'),
    html.Label('Select Fuel Type:'),
    fuel_dropdown,
    html.Label('Select Year:'),
    year_dropdown,
    dcc.Graph(id='price-graph')
])

# Define the callback to update the graph based on dropdown selections
@app.callback(
    Output('price-graph', 'figure'),
    [Input('fuel-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_graph(selected_fuel, selected_year):
    # Filter the dataframe based on the selected year
    filtered_df = df[df['year'] == selected_year]
    
    # Create the figure using the filtered data
    fig = px.line(
        filtered_df, x='DATES', y=selected_fuel, 
        title=f'{selected_fuel.capitalize()} Rates in {selected_year}',
        labels={'DATES': 'Date', selected_fuel: 'Price (INR)'}
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

#http://127.0.0.1:8050/..... url for dashboard. 
