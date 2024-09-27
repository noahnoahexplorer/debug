import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from wordcloud import WordCloud
import ast
import seaborn as sns

st.set_page_config(
    page_title='Game Analysis App - Dashboard',
    page_icon="	:bar_chart:",
)


st.title("Game Analysis Dashboard :bar_chart:")


############################################Visualization Functions################################################
# Function to display KPI cards
def display_kpi_card(title, value, delta=None, delta_color="normal", icon=None, color="lightblue"):
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 5px;">
            <div style="font-size: 20px; color: black;">{title}</div>
            <div style="font-size: 28px; font-weight: bold; color: black;">{value}</div>
            {'<div style="font-size: 16px; color: green;">' + icon + '</div>' if icon else ''}
            {'<div style="font-size: 16px; color:' + delta_color + ';">' + delta + '</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to generate scorecard for unique usernames
def generate_unique_username_scorecard(df):
    unique_usernames = df['username'].nunique()
    display_kpi_card("Active Players", unique_usernames, icon="👤", color="#6fa8dc")


# Function to generate scorecard for total cost
def generate_total_cost_scorecard(df):
    total_cost = df['total_cost'].sum()
    display_kpi_card("Total Cost", f"${total_cost:,.2f}", icon="💰", color="#ff9999")
    return total_cost


# Function to generate scorecard for total reward
def generate_total_reward_scorecard(df):
    total_reward = df['rewards'].sum()
    display_kpi_card("Total Reward", f"${total_reward:,.2f}", icon="🏆", color="#b6d7a8")
    return total_reward


# Function to generate scorecard for profit margin
def generate_profit_margin_scorecard(total_cost, total_reward):
    if total_cost != 0:
        profit_margin = ((total_reward - total_cost) / total_cost) * 100
        # Set color based on profit margin
        color = "#b6d7a8" if profit_margin >= 0 else "#ff9999"
        display_kpi_card("Profit Margin", f"{profit_margin:.2f}%", icon="📈", color=color)
    else:
        display_kpi_card("Profit Margin", "N/A", icon="📈", color="#c9daf8")

# Function to generate top 10 winners by net gain/loss
def generate_top_winners_bar_chart(df):
    # Ensure the 'username' column is treated as a string
    df["username"] = df["username"].astype(str)

    # Group by username and calculate total rewards, total cost, and net gain/loss
    user_summary = df.groupby("username").agg(
        total_rewards=pd.NamedAgg(column="rewards", aggfunc="sum"),
        total_cost=pd.NamedAgg(column="total_cost", aggfunc="sum"),
    )

    # Calculate Net Gain/Loss
    user_summary["net_gain_loss"] = (
        user_summary["total_rewards"] - user_summary["total_cost"]
    )

    # Sort by Net Gain/Loss and select top 10
    top_winners = (
        user_summary.sort_values(by="net_gain_loss", ascending=False)
        .head(10)
        .reset_index()
    )

    # Convert usernames to string (to ensure it's treated as discrete data)
    top_winners["username"] = top_winners["username"].astype(str)

    # Create a horizontal bar chart using Plotly and explicitly make the y-axis discrete
    fig = px.bar(
        top_winners,
        x="net_gain_loss",
        y="username",  # Y-axis for usernames
        orientation="h",
        title="Top 10 Winners by Net Gain/Loss",
        labels={"net_gain_loss": "Net Gain/Loss", "username": "Username"},
        color="net_gain_loss",
        color_continuous_scale="Viridis",
        category_orders={
            "username": top_winners["username"].tolist()
        },  # Ensure the order of usernames is maintained
    )

    # Set the y-axis as discrete to avoid continuous formatting
    fig.update_yaxes(type="category")  # This ensures the y-axis is treated as discrete

    # Display the chart
    st.plotly_chart(fig)


# Function to generate profit margin by provider & Product
def generate_profit_margin_bar_chart(df):
    # Create a new column 'Provider & Product' to combine 'ref_provider' and 'product_name_en'
    df["Provider & Product"] = df["ref_provider"] + " - " + df["product_name_en"]

    # Group by 'ref_provider' and 'product_name_en', and calculate total_reward and total_cost
    provider_product_summary = (
        df.groupby(["ref_provider", "product_name_en"])
        .agg(
            total_reward=pd.NamedAgg(column="rewards", aggfunc="sum"),
            total_cost=pd.NamedAgg(column="total_cost", aggfunc="sum"),
        )
        .head(15)
        .reset_index()
    )

    # Calculate the profit margin as total_reward / total_cost
    provider_product_summary["profit_margin"] = (
        -(provider_product_summary["total_reward"] - provider_product_summary["total_cost"])
        / provider_product_summary["total_cost"]
    )

    # Create a new column 'Provider & Product' for the combined label
    provider_product_summary["Provider & Product"] = (
        provider_product_summary["ref_provider"]
        + " - "
        + provider_product_summary["product_name_en"]
    )

    # Sort the data by profit margin in descending order
    provider_product_summary = provider_product_summary.sort_values(
        by="profit_margin", ascending=True
    )

    # Define a custom color scale that goes from red to blue
    custom_color_scale = [
        [0, "red"],      # Most negative values will be red
        [0.5, "yellow"], # Neutral values will be yellow
        [1, "green"],    # Positive values will be green
    ]

    # Create a horizontal bar chart using Plotly
    fig = px.bar(
        provider_product_summary,
        x="profit_margin",
        y="Provider & Product",
        orientation="h",
        title="Profit Margin by Provider & Product",
        labels={
            "profit_margin": "Profit Margin",
            "Provider & Product": "Provider & Product",
        },
        color="profit_margin",
        color_continuous_scale=custom_color_scale,  # Use the custom color scale
        category_orders={
            "Provider & Product": provider_product_summary[
                "Provider & Product"
            ].tolist()
        },  # Ensure the order of the bars
    )

    # Set the y-axis as discrete to avoid continuous formatting
    fig.update_yaxes(type="category")

    # Display the chart
    st.plotly_chart(fig)        
        
def extract_numbers_from_dict(df):
    # Extract the numbers (keys) from the number_cost column and create a new 'number' column
    df['number'] = df['number_cost'].apply(lambda x: list(eval(x).keys()))
    
    # Explode the 'number' column so each number gets its own row
    df_exploded = df.explode('number').reset_index(drop=True)
    
    # Convert 'number' to an integer type for heatmap
    df_exploded['number'] = df_exploded['number'].astype(str)
    
    return df_exploded

# Function to generate word cloud from unique numbers and their frequencies
def generate_unique_number_wordcloud(df_exploded):
    # Count frequency of each unique number
    number_counts = df_exploded['number'].value_counts().to_dict()

    # Create a word cloud with numbers
    wordcloud = WordCloud(
        width=400, 
        height=200, 
        background_color='white', 
        colormap='viridis',
        max_words=100,
        prefer_horizontal=1.0 # Force all numbers to be horizontal
    ).generate_from_frequencies(number_counts)

    # Plot the word cloud (keep high resolution, but adjust display size later)
    fig, ax = plt.subplots(figsize=(4, 2))  # Keeping figure size at a reasonable display size
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    
#Function to display heatmaps based on the betting data
def display_heatmaps(df):
    try:
        # Create a DataFrame for betting data
        number_covered = list(range(1, 101))  # Numbers from 1 to 100
        heatmap_matrix = pd.DataFrame(0, index=number_covered, columns=["Betting Coverage"])

        # Fill the heatmap matrix based on betting data
        for betting_dict in df['number_cost']:
            for number, amount in betting_dict.items():
                heatmap_matrix.loc[int(number)] += amount

        # Visualize the number betting heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_matrix, cmap='YlGnBu', annot=False, cbar=True, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in number betting heatmap: {e}")

# def display_heatmaps(df):
#     try:
#         # Create a DataFrame for betting data with numbers 1 to 100
#         number_covered = list(range(1, 101))  # Numbers from 1 to 100
#         heatmap_matrix = pd.DataFrame(0, index=number_covered, columns=["Betting Coverage"])

#         # Fill the heatmap matrix based on betting data
#         for betting_dict in df['number_cost']:
#             if isinstance(betting_dict, dict):  # Check if the row is a dictionary
#                 for number, amount in betting_dict.items():
#                     # Safeguard against non-integer keys or missing values
#                     if isinstance(number, (int, float)) and number in heatmap_matrix.index:
#                         heatmap_matrix.loc[int(number)] += amount

#         # Visualize the number betting heatmap
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(heatmap_matrix, cmap='YlGnBu', annot=False, cbar=True, ax=ax)
#         ax.set_title('Betting Coverage Heatmap')
#         st.pyplot(fig)
#     except Exception as e:
#         st.error(f"Error in number betting heatmap: {e}")

############################################Visualization Functions################################################

if 'raw_df' in st.session_state:
#     # Access the stored DataFrame
#     df = st.session_state['raw_df']
    
#     # Create a filter with a multiselect for providers
#     provider_filter = st.multiselect('Select Provider(s):', df['ref_provider'].unique())

#     # Create a filter with a multiselect for product names
#     product_filter = st.multiselect('Select Product Name(s):', df['product_name_en'].unique())

#     # Filter data based on the selected providers and product names
#     filtered_df = df.copy()  # Start with the full data
    
#     if provider_filter:
#         filtered_df = filtered_df[filtered_df['ref_provider'].isin(provider_filter)]
    
#     if product_filter:
#         filtered_df = filtered_df[filtered_df['product_name_en'].isin(product_filter)]

#     filtered_df = filtered_df.reset_index(drop=True)
#     st.write("Filtered Data:")
#     st.write(filtered_df)
    # Access the stored DataFrame
    df = st.session_state['raw_df']

    # Create a sidebar for filters
    st.sidebar.title("Filters")

    # Create a filter with a multiselect for providers in the sidebar
    provider_filter = st.sidebar.multiselect('Select Provider(s):', df['ref_provider'].unique())

    # Create a filter with a multiselect for product names in the sidebar
    product_filter = st.sidebar.multiselect('Select Product Name(s):', df['product_name_en'].unique())

    # Filter data based on the selected providers and product names
    filtered_df = df.copy()  # Start with the full data

    if provider_filter:
        filtered_df = filtered_df[filtered_df['ref_provider'].isin(provider_filter)]

    if product_filter:
        filtered_df = filtered_df[filtered_df['product_name_en'].isin(product_filter)]

    filtered_df = filtered_df.reset_index(drop=True)
    st.write("Filtered Data:")
    st.write(filtered_df)
    
    # Create four columns for KPI cards
    col1, col2, col3, col4 = st.columns(4)

    # Place the Total Rewards Amount scorecard in the first column
    with col1:
        total_reward = generate_total_reward_scorecard(filtered_df)

    # Place the Total Costs Amount scorecard in the second column
    with col2:
        total_cost = generate_total_cost_scorecard(filtered_df)

    # Place the unique username scorecard in the third column
    with col3:
        generate_unique_username_scorecard(filtered_df)

    # Place the profit margin scorecard in the fourth column
    with col4:
        generate_profit_margin_scorecard(total_cost, total_reward)
    
    # Create three columns for the scorecards
    col5, col6 = st.columns(2)

    # Place the Profit Margin by Provider & Product in the first column
    with col5:
        # st.subheader("Profit Margin by Provider & Product")
        generate_profit_margin_bar_chart(filtered_df)

    # Place the Top Winners by Net Gain/Loss in the second column
    with col6:
        # st.subheader("Top 10 Winners by Net Gain/Loss")
        generate_top_winners_bar_chart(filtered_df)
    
    # Extract numbers and create exploded dataframe
    df_exploded = extract_numbers_from_dict(filtered_df)
    # Create 1 columns for the charts
    col7, col8 = st.columns(2)

    # Place the Unique Number Word Cloud in the first column
    with col7:
        st.subheader("Hot Numbers")
        generate_unique_number_wordcloud(df_exploded)
        
    with col8:  
        # Number Betting Heatmap
        st.subheader("Number Betting Heatmap")
        # Convert 'number_cost_dict' column to dictionary
        if 'number_cost' in filtered_df.columns:
            filtered_df['number_cost'] = filtered_df['number_cost'].apply(ast.literal_eval)

            # Display heatmaps for betting data
            display_heatmaps(filtered_df)
        else:
            st.warning("The column 'number_cost' is missing in the uploaded CSV.")

#     st.divider()

############################################Geo Map##############################################################
    # Ensure session state keys are initialized
    if "heatmap_generated" not in st.session_state:
        st.session_state.heatmap_generated = False
    if "heatmap_data" not in st.session_state:
        st.session_state.heatmap_data = None
    if "map_state" not in st.session_state:
        st.session_state.map_state = None
    if "providers_state" not in st.session_state:
        st.session_state.providers_state = None
    if "games_state" not in st.session_state:
        st.session_state.games_state = None

    # Streamlit button to trigger the execution of the code
    if st.button("Generate Heatmap"):
        # Update session state to indicate heatmap generation
        st.session_state.heatmap_generated = True

        ip_list = [{"query": ip} for ip in filtered_df['ip'].tolist()]

        # Send the batch request
        response = requests.post("http://ip-api.com/batch", json=ip_list).json()

        # Extract latitude and longitude and add them to the DataFrame
        filtered_df['Latitude'] = [ip_info['lat'] for ip_info in response]
        filtered_df['Longitude'] = [ip_info['lon'] for ip_info in response]
        filtered_df

        # Drop rows without location data
        filtered_locations = filtered_df.dropna(subset=['Latitude', 'Longitude'])

        # Store filtered locations and heatmap data in session state
        st.session_state.heatmap_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_locations.iterrows()]
        st.session_state.map_state = filtered_locations
        st.session_state.providers_state = filtered_locations['ref_provider'].unique()  # Store filtered data for display
        st.session_state.games_state = filtered_locations['product_name_en'].unique()

    # Check if heatmap data exists in session state and display the map
    if st.session_state.heatmap_generated and st.session_state.heatmap_data:
        # Initialize the map centered around an average location
        avg_lat = st.session_state.map_state['Latitude'].mean()
        avg_lon = st.session_state.map_state['Longitude'].mean()
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2)

        # Add the heatmap layer to the map
        HeatMap(st.session_state.heatmap_data).add_to(m)

        # Display the map in Streamlit
        st_data = st_folium(m, width=700, height=500)

       # Create two columns
        col1, col2 = st.columns(2)

        # Display data in the first column
        with col1:
            st.write("Analyzed Providers:")
            st.write(st.session_state.providers_state)

        # Display data in the second column
        with col2:
            st.write("Analyzed Games:")
            st.write(st.session_state.games_state)
############################################Geo Map##############################################################

else:
    st.write("No data loaded. Please load data on HomePage first.")



