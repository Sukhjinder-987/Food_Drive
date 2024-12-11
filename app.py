import sklearn
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib



# Load the dataset with a specified encoding
data = pd.read_csv('Cleaned_food_drive_data.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.image('Untitled design.jpg', width=500, use_column_width='auto')

    st.subheader("💡 Abstract:")

    inspiration = '''
    The Edmonton Food Drive Project \n
    Project Overview: \n
    The Edmonton City Food Drive Project aims to make the best use of machine learning techniques to enhance the efficiency and impact of the food drive in Edmonton, Alberta which will support the local food bank of Edmonton to address the food insecurity in the community. This could be achieved by optimizing the volunteer route assignment and maximizing collection effectively.
    Problem Statement: \n
    The challenges faced by the Edmonton Food Drive are the route planning, pick-up processes, and drop-off locations. Therefore, there is a need to optimize the Edmonton Food Drive by providing better solutions for collecting donated food by enhancing the route planning. There are a limited number of volunteers, so it is necessary to employ them efficiently in areas of greater requirement.

    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    Our project aims to predict the Comment Sentiments from the volunteer's comments, to investigate if the volunteers are facing any problem and the most as well as the least number of donation bags collected based on the Positive, Negative and Netural Comments. The focus of the project is getting training from the 2023 Food Drive Dataset and then making predictions on the 2024 Food Drive Dataset.
    '''

    st.write(what_it_does)


# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis and Visualization")
    st.markdown(f"**Disclaimer:** The data used in this app consists ONLY of data collected in collaboration with NorQuest College during the 2023 Food Drive Project and does not represent the entire Food Drive.")
    # Rename columns for clarity
    data_cleaned = data.copy()

    # Visualize the distribution of numerical features using Plotly
    fig = px.histogram(data_cleaned, x='# of Adult Volunteers for year 2024', nbins=20, labels={'# of Adult Volunteers for year 2024': 'Adult Volunteers'}, title="Distribution of adult volunteers")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data_cleaned, x='# of Youth Volunteers for year 2024', nbins=20, labels={'# of Youth Volunteers for year 2024': 'Youth Volunteers'}, title="Distribution of youth volunteers" )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data_cleaned, x='Donation Bags Collected for the year 2024', nbins=20, labels={'Donation Bags Collected for the year 2024': 'Donation Bags Collected'}, title="Distribution of donation Bags collected")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data_cleaned, x='Time Spent Collecting Donations', nbins=20, labels={'Time Spent Collecting Donations': 'Time to Complete'}, title="Distribution of time to complete")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data_cleaned, x='Comment Sentiments for year 2024', nbins=20, labels={'Comment Sentiments for year 2024': 'Comment Sentiments'}, title="Distribution of Comment Sentiments")
    st.plotly_chart(fig, use_container_width=True)

     ## Add Stake Filter Option ##
    stake_selection = st.multiselect(label='Select Stake(s) to filter the Wards displayed below:',options=data_cleaned['Stake for year 2024'].unique(),
                                    placeholder='Please select 1 or more values.'
                                    )
    filtered_data = data_cleaned.loc[data_cleaned['Stake for year 2024'].isin(stake_selection)]
    if len(stake_selection) == 0:
      filtered_data = data_cleaned
    ward_chart_height = 750
    if len(filtered_data.groupby(by='COMBINED STAKES')) <= 14: ward_chart_height = 450

    # Create the pie chart
    #fig = px.pie(data_cleaned['Donation Bags Collected for the year 2024'], labels=data_cleaned['Stake for year 2024'], autopct='%1.1f%%', startangle=90, title='Percentage Distribution of Bags Collected by Region')
    #px.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the chart in Streamlit
    #st.pyplot(fig)


# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    stake_encoding = {'Edmonton North Stake' : 0,'Gateway Stake' : 1,'Riverbend Stake' : 2,'Bonnie Doon Stake' : 3,'YSA Stake' : 4}
    Comment_Sentiment_encoding = {'Positive' : 1,'Negative' : 0,'Netural' : 2}
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict Comment Sentiments:")

    # Input fields for user to enter data
    stake = st.selectbox("Select a Stake",data['Stake for year 2024'].unique())
    ward_branch = st.selectbox("Select a Ward/Branch",data.loc[data['Stake for year 2024'] == stake,'COMBINED STAKES'].unique())
    completed_routes = st.slider("Completed More Than One Route", 0, 1, 0)
    routes_completed = st.slider("Routes Completed", 1, 10, 5)
    donation_bags_collected = st.slider("Donation Bags Collected", 0, 2000, 1000)
    time_spent = st.slider("Time Spent (minutes)", 10, 300, 60)
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 50, 10)
    doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100)
    youth_volunteers = st.slider("Number of Youth Volunteers", 1, 50, 10)

    # Cols to calc
    donation_bags_collected = data.loc[data['COMBINED STAKES'] == ward_branch,'Donation Bags Collected for the year 2024'].mean()
    bags_per_door = int(donation_bags_collected)/int(doors_in_route)
    bags_per_route = int(donation_bags_collected)/int(routes_completed)
    total_volunteers = int(adult_volunteers) + int(youth_volunteers)

    stake_num = int(stake_encoding[stake])

    # Predict button
    if st.button("Predict"):
        # Load the trained model
        model = joblib.load('random_forest_classifier_model.pkl')

        # Prepare input data for prediction
        input_data = [[completed_routes, routes_completed, time_spent, adult_volunteers, doors_in_route, youth_volunteers, comment_sentiments]]

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"Predicted Comment Sentiments: {prediction[0]}")

        # You can add additional information or actions based on the prediction if needed

# Page 4: Neighbourhood Mapping
# Read geospatial data
#geodata = pd.read_csv("Location_data_updated.csv")

#def neighbourhood_mapping():
#    st.title("Neighbourhood Mapping")

    # Get user input for neighborhood
#    user_neighbourhood = st.text_input("Enter the neighborhood:")

    # Check if user provided input
#    if user_neighbourhood:
        # Filter the dataset based on the user input
#        filtered_data = geodata[geodata['Neighbourhood'] == user_neighbourhood]

        # Check if the filtered data is empty, if so, return a message indicating no data found
#        if filtered_data.empty:
#            st.write("No data found for the specified neighborhood.")
#        else:
            # Create the map using the filtered data
#            fig = px.scatter_mapbox(filtered_data,
#                                    lat='Latitude',
#                                    lon='Longitude',
#                                    hover_name='Neighbourhood',
#                                    zoom=12)

            # Update map layout to use OpenStreetMap style
#            fig.update_layout(mapbox_style='open-street-map')

            # Show the map
#            st.plotly_chart(fig)
#    else:
#        st.write("Please enter a neighborhood to generate the map.")




# Page 5: Data Collection
def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    google_form_url = "https://forms.gle/Sif2hH3zV5fG2Q7P8"#YOUR_GOOGLE_FORM_URL_HERE
    st.markdown(f"[Fill out the form]({google_form_url})")

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA and Visualization", "ML Modeling", "Data Collection"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA and Visualization":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    #elif app_page == "Neighbourhood Mapping":
     #   neighbourhood_mapping()
    elif app_page == "Data Collection":
        data_collection()

if __name__ == "__main__":
    main()
