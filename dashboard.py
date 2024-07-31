import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page config
st.set_page_config(page_title="Soccer Data Analysis", layout="wide")

# Function to load data
@st.cache_data
def load_data(file, skip_first_row=False):
    if skip_first_row:
        df = pd.read_excel(file, skiprows=1)
    else:
        df = pd.read_excel(file)
    df = df.fillna(0)
    df = df.drop_duplicates()
    return df

# Sidebar for page selection
page = st.sidebar.selectbox("Choose a page", ["Pocession Analysis", "League Table Analysis"])

if page == "Pocession Analysis":
    st.title("Chelsea Soccer Pocession Analysis Dashboard")

    # File uploaders
    uploaded_pocession_file = st.file_uploader("Upload Pocession.xlsx file", type="xlsx")
    uploaded_player_file = st.file_uploader("Upload Player basic data.xlsx file", type="xlsx")

    if uploaded_pocession_file is not None and uploaded_player_file is not None:
        pocession_df = load_data(uploaded_pocession_file, skip_first_row=True)
        player_df = load_data(uploaded_player_file, skip_first_row=True)

        # Filter Chelsea players
        chelsea_players = player_df[player_df['Squad'] == 'Chelsea']['Player'].tolist()
        chelsea_pocession_df = pocession_df[pocession_df['Player'].isin(chelsea_players)]

        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Choose analysis type",
            ["Overview", "Third Analysis", "Correlation Analysis", "Top Performers", "Pocession Flow"]
        )

        if analysis_type == "Overview":
            st.header("Chelsea Data Overview")
            st.write(chelsea_pocession_df.head())
            
            st.subheader("Dataset Information")
            st.write(f"Total number of Chelsea players: {len(chelsea_pocession_df)}")
            st.write(f"Number of features: {chelsea_pocession_df.shape[1]}")

        elif analysis_type == "Third Analysis":
            st.header("Analysis by Thirds for Chelsea Players")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Defensive Third")
                st.write(chelsea_pocession_df['Def 3rd'].describe())
                fig, ax = plt.subplots()
                chelsea_pocession_df['Def 3rd'].hist(bins=20, ax=ax)
                ax.set_title('Distribution of Defensive Third Touches (Chelsea)')
                ax.set_xlabel('Def 3rd Touches')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Middle Third")
                st.write(chelsea_pocession_df['Mid 3rd'].describe())
                fig, ax = plt.subplots()
                chelsea_pocession_df['Mid 3rd'].hist(bins=20, ax=ax)
                ax.set_title('Distribution of Middle Third Touches (Chelsea)')
                ax.set_xlabel('Mid 3rd Touches')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            
            with col3:
                st.subheader("Attacking Third")
                st.write(chelsea_pocession_df['Att 3rd'].describe())
                fig, ax = plt.subplots()
                chelsea_pocession_df['Att 3rd'].hist(bins=20, ax=ax)
                ax.set_title('Distribution of Attacking Third Touches (Chelsea)')
                ax.set_xlabel('Att 3rd Touches')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

        elif analysis_type == "Correlation Analysis":
            st.header("Correlation Analysis for Chelsea Players")
            
            correlation_cols = ['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att', 'Live', 'Succ', 'Att', 'Succ', 'Mis', 'Dis', 'Rec', 'PrgR']
            correlation_matrix = chelsea_pocession_df[correlation_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix of Pocession Metrics (Chelsea)')
            st.pyplot(fig)
            
            st.write("This heatmap shows the correlations between various pocession metrics for Chelsea players. Stronger correlations (closer to 1 or -1) are shown in darker colors.")

        elif analysis_type == "Top Performers":
            st.header("Top Chelsea Performers")
            
            metric = st.selectbox("Choose a metric", ['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Carries', 'PrgC'])
            top_n = st.slider("Number of top performers", 5, 20, 10)
            
            top_players = chelsea_pocession_df.nlargest(top_n, metric)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Player', y=metric, data=top_players, ax=ax)
            ax.set_title(f'Top {top_n} Chelsea Players by {metric}')
            ax.set_xlabel('Player')
            ax.set_ylabel(metric)
            plt.xticks(rotation=90)
            st.pyplot(fig)
            
            st.write(top_players[['Player', metric]])

        elif analysis_type == "Pocession Flow":
            st.header("Pocession Flow Analysis for Chelsea Players")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=chelsea_pocession_df, x='Touches', y='Carries', size='PrgC', hue='Att 3rd', ax=ax)
            ax.set_title('Pocession Flow: Touches vs Carries (Chelsea)')
            ax.set_xlabel('Total Touches')
            ax.set_ylabel('Total Carries')
            st.pyplot(fig)
            
            st.write("This scatter plot visualizes the relationship between total touches and carries for Chelsea players. The size of each point represents the number of progressive carries (PrgC), while the color indicates the number of touches in the attacking third.")
            
            st.subheader("Chelsea Player Pocession Profile")
            selected_player = st.selectbox("Select a Chelsea player", chelsea_pocession_df['Player'].tolist())
            player_data = chelsea_pocession_df[chelsea_pocession_df['Player'] == selected_player].iloc[0]
            
            # Radar chart for player pocession profile
            metrics = ['Def 3rd', 'Mid 3rd', 'Att 3rd', 'Carries', 'PrgC', 'PrgR']
            values = player_data[metrics].values
            values = np.concatenate((values, [values[0]]))  # close the loop
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # close the loop

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color='red', alpha=0.25)  # fill area under the plot
            ax.plot(angles, values, color='red', linewidth=2)  # plot outline
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_title(f"Pocession Profile: {selected_player}")
            st.pyplot(fig)
            
            st.write(f"This radar chart shows the pocession profile of {selected_player}, highlighting their involvement in different thirds of the pitch and their progression with the ball.")

    else:
        st.write("Please upload both the Pocession.xlsx and Player basic data.xlsx files to begin the analysis.")
elif page == "League Table Analysis":
    st.title("League Table Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload League table.xlsx file", type="xlsx")

    if uploaded_file is not None:
        # Load the data from the 'DATA' sheet
        data_df = pd.read_excel(uploaded_file, sheet_name='Data')

        # Filter data for Chelsea
        chelsea_data = data_df[data_df['Squad'].str.contains('Chelsea', case=False, na=False)]

        # Bar chart for Points
        st.subheader("Points by Team")
        fig, ax = plt.subplots(figsize=(12, 6))
        points_chart = sns.barplot(x='Pts', y='Squad', data=data_df.sort_values('Pts', ascending=False), palette='viridis', ax=ax)
        ax.set_title('Points by Team', fontsize=16)
        ax.set_xlabel('Points', fontsize=14)
        ax.set_ylabel('Team', fontsize=14)
        plt.axvline(x=chelsea_data['Pts'].values[0], color='red', linestyle='--', label='Chelsea')
        plt.legend()
        st.pyplot(fig)

        # Scatter plot for Goal Difference vs. Points
        st.subheader("Goal Difference vs. Points")
        fig, ax = plt.subplots(figsize=(12, 6))
        gd_vs_pts_chart = sns.scatterplot(x='GD', y='Pts', data=data_df, hue='Squad', palette='viridis', s=100, ax=ax)
        ax.set_title('Goal Difference vs. Points', fontsize=16)
        ax.set_xlabel('Goal Difference', fontsize=14)
        ax.set_ylabel('Points', fontsize=14)
        plt.scatter(chelsea_data['GD'], chelsea_data['Pts'], color='red', s=200, edgecolor='black', label='Chelsea')
        plt.legend()
        st.pyplot(fig)

        # Additional analysis options
        st.subheader("Additional Analysis")
        analysis_option = st.selectbox("Choose an analysis", ["Top Scorers", "Team Performance"])

        if analysis_option == "Top Scorers":
            top_n = st.slider("Number of top scorers", 5, 20, 10)
            top_scorers = data_df.nlargest(top_n, 'GF')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='GF', y='Squad', data=top_scorers, palette='viridis', ax=ax)
            ax.set_title(f'Top {top_n} Scoring Teams', fontsize=16)
            ax.set_xlabel('Goals For', fontsize=14)
            ax.set_ylabel('Team', fontsize=14)
            st.pyplot(fig)

        elif analysis_option == "Team Performance":
            selected_team = st.selectbox("Select a team", data_df['Squad'].tolist())
            team_data = data_df[data_df['Squad'] == selected_team].iloc[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Points", team_data['Pts'])
            col2.metric("Goal Difference", team_data['GD'])
            col3.metric("Position", team_data['Rk'])

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='polar')
            metrics = ['W', 'D', 'L', 'GF', 'GA', 'GD']
            values = team_data[metrics].values
            angles = [n / float(len(metrics)) * 2 * np.pi for n in range(len(metrics))]
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            ax.plot(angles, values)
            ax.fill(angles, values, alpha=0.3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_title(f"Performance Profile: {selected_team}")
            st.pyplot(fig)

    else:
        st.write("Please upload the League table.xlsx file to begin the analysis.")
