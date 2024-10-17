import streamlit as st
import plotly as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dalex as dx
from PIL import Image
import streamlit_authenticator as stauth

import pandas as pd
import csv
import sklearn

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from operator import itemgetter
from itertools import groupby

import warnings
import numpy as np
warnings.filterwarnings("ignore")

logo = Image.open("pacificwide_logo.jpg")

st.set_page_config(
    page_title="Pacificwide KPI System",
    page_icon= logo,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.pacificwide.com/help',
        'Report a bug': "https://www.pacificwide.com/bug",
        'About': "# Pacificwide product. This application is for Internal Use only!"
    }
)

# Login Database: Usernames and Passwords
# Create new users here: 
names = ['Leon Le','Dat Mai', 'Gina Nguyen']
usernames = ['leonle@pacificwide.com','datmai@pacificwide.com', 'gina@pacificwide.com']
passwords = ['Passion2023','Datmai@2205', 'Gina@Pacificwide2023']

# Passwords Security Proctection
hashed_passwords = stauth.Hasher(passwords).generate()

# Add authenticator to the system
# Set cooloe cache at 10 days (save login session for 10 days)
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,'cookie_name', 'signature_key',cookie_expiry_days=10)

# Add Pacificwide logo at sidebar
st.sidebar.image('https://cloud.pacificwide.com/dd/36QRQ21Qmt/?thumbNail=1&w=1200&h=1200&type=proportional&preview=true', use_column_width='auto')
name, authentication_status, username = authenticator.login('Login','sidebar')

# This is where the software begins
if authentication_status:
    def set_work_kpi(file_path, month, sale):
        df = pd.read_csv(file_path)
        
        # Month map dictionary
        month_map = {
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9,
            'October': 10,
            'November': 11,
            'December': 12
        }

        df.Month = df.Month.map(month_map)
        
        # Percentage of responsible devided by job level
        pt_ju = 0.1
        pt_sn = 0.2
        ft_ju = 0.3
        ft_sn = 0.4
        
        # Set threshold
        threshold = 2
        
        # Begin to build predictive ML models: 
        x = df[['Month', 'Loan Amount', 'Missed Milestone',
                'Number of Loan Denied', 'Number of Loan Withdrawn']].dropna()
        y = df['Number of Loan Completed'].dropna()

        # Input Month and Sales Goal: 
        input_a = month
        input_b = sale

        # input_a = int(input_a)
        # input_b = int(input_b)
        # input_c = input('Please input Missed Milestone:')
        # input_d = input('Please input Number of Loan Denied:')
        # input_e = input('Please input Number of Loan Withdrawn:')

        # kpi_para = input('Please input KPI Bonus:')

        # Estimated Missed Milestone
        df1 = df[['Month', 'Loan Amount']].dropna()
        df1_test = df['Missed Milestone'].dropna()
        regr = linear_model.LinearRegression()
        regr.fit(df1, df1_test)
        input_c = regr.predict([[input_a, input_b]])

        # Estimated Number of Loan Denied
        df2 = df[['Month', 'Loan Amount', 'Missed Milestone']].dropna()
        df2_test = df['Number of Loan Denied'].dropna()
        regr.fit(df2, df2_test)
        input_d = regr.predict([[input_a, input_b, input_c]])

        # Estimated Number of Loan Withdrawn
        df3 = df[['Month', 'Loan Amount', 'Missed Milestone',
                  'Number of Loan Denied']].dropna()
        df3_test = df['Number of Loan Withdrawn'].dropna()
        regr.fit(df3, df3_test)
        input_e = regr.predict([[input_a, input_b, input_c, input_d]])

        # kpi_para = int(kpi_para)
        
        # Fit model
        regr.fit(x, y)
        predicted = regr.predict([[input_a, input_b, input_c, input_d, input_e]])

        print('-----------')
        print('Your Sales Target is: ', '${:,.2f}'.format(input_b))
        print('Current Month is: ', input_a)

        # print('Current threshold is: ', threshold, 'file(s)')
    #     print('-----------')
    #     print("\n")
        # print('Number of Missed Milestone allow:', round(int(input_c)) - 1)
        # print('Number of Loan Denied allow:', round(int(input_d)) - 1)
        # print('Number of Loan Withdrawn allow', round(int(input_e))- 1)


    #     if input_c == 1: # <- Adjust # to your preference
    #         input_c = round(int(input_c))
    #         print('Number of Missed Milestone allowed:', input_c + 1)
    #     elif input_c < 0:
    #         print('Number of Missed Milestone allowed:', 0)
    #     else:
    #         print('Number of Missed Milestone allowed:', round(int(input_c)))


    #     if input_d == 0: # <- Adjustable
    #         input_d = round(int(input_d))
    #         print('Number of Loan Denied allowed:', input_d + 1)
    #     elif input_d < 0:
    #         print('Number of Loan Denied allowed:', 0)
    #     else:
    #         print('Number of Loan Denied allowed:', round(int(input_d)))


    #     if input_e == 0: # <- Adjustable
    #         input_e = round(int(input_c))
    #         print('Number of Loan Withdrawn allowed:', input_e + 1)
    #     elif input_e < 0:
    #         print('Number of Loan Withdrawn allowed:', 0)
    #     else:
    #         print('Number of Loan Withdrawn allowed:', round(int(input_e)))


    #     print("\n")
        print('-----------')
        
        # Summary messages:
        if predicted > 0:
            print('Mortgage Team need to complete at least:',
                  int(predicted.round()), 'file(s)')
        else:
            print('Mortgage Team need to complete at least:',
                  int(predicted.round() + threshold), 'file(s)')

        # Create KPI table
        kpi_df = pd.DataFrame(columns=['Month', 'Sale Goal', 'Title', 'Baseline', 'KPI'])

        l1 = round(pt_ju * int(predicted))
        l2 = round(pt_sn * int(predicted))
        l3 = round(ft_ju * int(predicted))
        l4 = round(ft_sn * int(predicted))

        # Add Baseline to KPI table
        kpi_df['Title'] = ['Part-time Junior Loan Processor', 'Part-time Senior Loan Processor',
                           'Full-time Junior Loan Processor', 'Full-time Senior Loan Processor']
        baseline = [l1, l2, l3, l4]
        kpi_df['Baseline'] = baseline
        kpi_df['Month'] = month
        kpi_df['Sale Goal'] = sale

        print('-----------')
        print('WORK BASELINE:')
        print('Part-time Junior Loan Processor baseline:', round(l1))
        print('Part-time Senior Loan Processor baseline:', round(l2))
        print('Full-time Junior Loan Processor baseline:', round(l3))
        print('Full-time Senior Loan Processor baseline:', round(l4))
        #         if l1 < 1:
        #             print('Part-time Junior Loan Processor has to complete: at least',
        #               round(l1) + 1)
        #         else:
        #             print('Part-time Junior Loan Processor has to complete: at least',
        #               round(l1))

        #         if l2 < 1:
        #             print('Part-time Senior Loan Processor has to complete: at least',
        #               round(l2) + 1)
        #         else:
        #             print('Part-time Junior Loan Processor has to complete: at least',
        #               round(l2))

        #         if l3 < 1:
        #             print('Full-time Junior Loan Processor has to complete: at least',
        #               round(l3) + 1)
        #         else:
        #             print('Part-time Junior Loan Processor has to complete: at least',
        #               round(l3))

        #         if l4 < 1:
        #             print('Full-time Senior Loan Processor has to complete: at least',
        #               round(l4) + 1)
        #         else:
        #             print('Part-time Junior Loan Processor has to complete: at least',
        #               round(l4))

        print('-----------')
        print('WORK KPI:')
        if l1 < 2:  # <- Adjustable
            l1_kpi = l1 + threshold
            print('Part-time Junior Loan Processor KPI:',
                  round(l1_kpi))
        else:
            l1_kpi = l1
            print('Part-time Junior Loan Processor KPI:',
                  round(l1_kpi))

        if l2 < 3:  # <- Adjustable
            l2_kpi = l2 + threshold
            print('Part-time Senior Loan Processor KPI:',
                  round(l2_kpi))
        else:
            l2_kpi = l2
            print('Part-time Senior Loan Processor KPI:',
                  round(l2_kpi))

        if l3 < 4:  # <- Adjustable
            l3_kpi = l3 + threshold
            print('Full-time Junior Loan Processor KPI:',
                  round(l3_kpi))
        else:
            l3_kpi = l3
            print('Full-time Junior Loan Processor KPI:',
                  round(l3_kpi))

        if l4 < 5:  # <- Adjustable
            l4_kpi = l4 + threshold
            print('Full-time Senior Loan Processor KPI:',
                  round(l4_kpi))
        else:
            l4_kpi = l4
            print('Full-time Senior Loan Processor KPI:',
                  round(l4_kpi))

        kpi = [l1_kpi, l2_kpi, l3_kpi, l4_kpi]
        kpi_df['KPI'] = kpi

        print('-----------')

        if predicted > 1:
            print('Estimated Loan Amount:', '${:,.2f}'.format(
                round(input_b / int(predicted))))
        else:
            print('Estimated Loan Amount:', '${:,.2f}'.format(round(input_b / 1)))

        return kpi_df


    # Function to set KPI for report:
    def set_kpi(file_path, month, sale):

        df = pd.read_csv(file_path)
        
        # Month Map
        month_map = {
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9,
            'October': 10,
            'November': 11,
            'December': 12
        }

        df.Month = df.Month.map(month_map)

        # Portion of Work by percentage: 
        pt_ju = 0.1 # <- Adujstable
        pt_sn = 0.2 # <- Adujstable
        ft_ju = 0.3 # <- Adujstable
        ft_sn = 0.4 # <- Adujstable

        threshold = 2

        x = df[['Month', 'Loan Amount', 'Missed Milestone',
                'Number of Loan Denied', 'Number of Loan Withdrawn']].dropna()
        y = df['Number of Loan Completed'].dropna()

        # Variables for month and sale input:
        input_a = month
        input_b = sale

        # Estimated Missed Milestone
        df1 = df[['Month', 'Loan Amount']].dropna()
        df1_test = df['Missed Milestone'].dropna()
        regr = linear_model.LinearRegression()
        regr.fit(df1, df1_test)
        input_c = regr.predict([[input_a, input_b]])

        # Estimated Number of Loan Denied
        df2 = df[['Month', 'Loan Amount', 'Missed Milestone']].dropna()
        df2_test = df['Number of Loan Denied'].dropna()
        regr.fit(df2, df2_test)
        input_d = regr.predict([[input_a, input_b, input_c]])

        # Estimated Number of Loan Withdrawn
        df3 = df[['Month', 'Loan Amount', 'Missed Milestone',
                  'Number of Loan Denied']].dropna()
        df3_test = df['Number of Loan Withdrawn'].dropna()
        regr.fit(df3, df3_test)
        input_e = regr.predict([[input_a, input_b, input_c, input_d]])

        # kpi_para = int(kpi_para)

        regr.fit(x, y)
        predicted = regr.predict([[input_a, input_b, input_c, input_d, input_e]])

        # Create KPI table
        kpi_df = pd.DataFrame(columns=['Title', 'Baseline', 'KPI'])

        l1 = round(pt_ju * int(predicted))
        l2 = round(pt_sn * int(predicted))
        l3 = round(ft_ju * int(predicted))
        l4 = round(ft_sn * int(predicted))

        # Add Baseline to KPI table
        kpi_df['Title'] = ['Part-time Junior Loan Processor', 'Part-time Senior Loan Processor',
                           'Full-time Junior Loan Processor baseline', 'Full-time Senior Loan Processor']
        baseline = [l1, l2, l3, l4]
        kpi_df['Baseline'] = baseline

        if l1 < 2:  # <- Adjustable
            l1_kpi = l1 + threshold
        else:
            l1_kpi = l1

        if l2 < 3:  # <- Adjustable
            l2_kpi = l2 + threshold
        else:
            l2_kpi = l2

        if l3 < 4:  # <- Adjustable
            l3_kpi = l3 + threshold
        else:
            l3_kpi = l3

        if l4 < 5:  # <- Adjustable
            l4_kpi = l4 + threshold
        else:
            l4_kpi = l4

        kpi = [l1_kpi, l2_kpi, l3_kpi, l4_kpi]
        kpi_df['KPI'] = kpi

        return l1, l2, l3, l4, l1_kpi, l2_kpi, l3_kpi, l4_kpi



    def create_report(file_path_report, file_path_loan_status, month, sale):

        draft = pd.read_csv(file_path_report)

        month_map = {
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9,
            'October': 10,
            'November': 11,
            'December': 12
        }

        # Sub-dataset after month input
        draft.Month = draft.Month.map(month_map)

        input_month = month

        df_name2 = draft[draft['Month'] == int(input_month)]
        df_name2 = df_name2.reset_index(drop=True)

        kpi_target = set_kpi(file_path=file_path_loan_status,
                             month=month,
                             sale=sale)
        l1 = kpi_target[0]
        l2 = kpi_target[1]
        l3 = kpi_target[2]
        l4 = kpi_target[3]

        l1_kpi = kpi_target[4]
        l2_kpi = kpi_target[5]
        l3_kpi = kpi_target[6]
        l4_kpi = kpi_target[7]

        # Set number of milesotne per transaction:
        milestone_per_transaction = 5 # <- Adujstable

        # New Columns for All Loan (Difficult Loan & Completed 3 Weeks count as 1.5 Normal Loan)
        df_name2['Number of Normal Loan Completed'] = df_name2['Number of Loan Completed'] - \
            df_name2['Number of Difficult Loan'] - df_name2['Number of Loan Completed Within 3 Weeks']

        df_name2['All Loan Completed'] = (df_name2['Number of Normal Loan Completed'] + \
            (df_name2['Number of Difficult Loan'] * 1.5) + (df_name2['Number of Loan Completed Within 3 Weeks'] * 1.5) + df_name2['Number of Loan Takeover']) * df_name2['Check-list Forms & Procedure (%)']

        df_name2['All Loan Completed'] = df_name2['All Loan Completed'] - (df_name2['Number of Missed Milestone'] / (df_name2['All Loan Completed'] * milestone_per_transaction))

        list_title = df_name2['Title'].values.tolist()

        # Work Score
        # # Part-time Junior Loan Processor - Calculation
        for z in range(0, len(df_name2['Title'])):
            if df_name2['Title'][z] == 'Part-time Junior Loan Processor':
                df_name2['KPI'] = l1_kpi / 1
                #                 list_title.count('Part-time Junior Loan Processor')

                df_name2['Work Score'] = df_name2['All Loan Completed'] / df_name2['KPI']

                # df_name2['Work_vs_KPI (%)'] = (df_name2['Work Score'] /
                #                                df_name2['KPI']) * 100
                df_name2['Work Score'] = df_name2['Work Score'] * 10

        # Part-time Senior Loan Processor - Calculation
        for z in range(0, len(df_name2['Title'])):
            if df_name2['Title'][z] == 'Part-time Senior Loan Processor':
                df_name2['KPI'][z] = l2_kpi / 1
                #                 list_title.count('Part-time Senior Loan Processor')

                df_name2['Work Score'][z] = (df_name2['All Loan Completed'][z] / df_name2['KPI'][z])
                # df_name2['Work_vs_KPI (%)'][z] = (df_name2['Work Score'][z] /
                #                                   df_name2['KPI'][z]) * 100
                df_name2['Work Score'][z] = df_name2['Work Score'][z] * 10

        # Full-time Junior Loan Processor - Calculation
        for z in range(0, len(df_name2['Title'])):
            if df_name2['Title'][z] == 'Full-time Junior Loan Processor':
                df_name2['KPI'][z] = l3_kpi / 1
                #                 list_title.count('Full-time Junior Loan Processor')

                df_name2['Work Score'][z] = (df_name2['All Loan Completed'][z] / df_name2['KPI'][z])
                # df_name2['Work_vs_KPI (%)'][z] = (df_name2['Work Score'][z] /
                #                                   df_name2['KPI'][z]) * 100
                df_name2['Work Score'][z] = df_name2['Work Score'][z] * 10

        # Full-time Senior Loan Processor - Calculation
        for z in range(0, len(df_name2['Title'])):
            if df_name2['Title'][z] == 'Full-time Senior Loan Processor':
                df_name2['KPI'][z] = l4_kpi / 1
                #                 list_title.count('Full-time Senior Loan Processor')

                df_name2['Work Score'][z] = (df_name2['All Loan Completed'][z] / df_name2['KPI'][z]) 
                # df_name2['Work_vs_KPI (%)'][z] = (df_name2['Work Score'][z] /
                #                                   df_name2['KPI'][z]) * 100
                df_name2['Work Score'][z] = df_name2['Work Score'][z] * 10

        # Attitude Score
        df_name2['Attitude Score'] = df_name2[[
            'Attendance', 'Teamwork', 'Collaboration', 'Work Behavior', 'Learning',
            'Responsibility'
        ]].mean(axis=1)

        # Total Loan Assigned
        df_name2['Total Loan Assigned'] = df_name2['Number of Loan Completed'] + \
            df_name2['Number of Loan Denied'] + df_name2['Number of Loan Withdrawn']

        # Loan Withdrawn Percentage
        df_name2['Withdrawn Rate (%)'] = round(
            (df_name2['Number of Loan Withdrawn'] /
             df_name2['Total Loan Assigned']) * 100)
        #
        df_name2['Denial Rate (%)'] = round(
            (df_name2['Number of Loan Denied'] / df_name2['Total Loan Assigned']) *
            100)

        # Work Completion
        df_name2['Work Completion Score'] = (df_name2['All Loan Completed'] / df_name2['Total Loan Assigned']) * 10

        # Final Score Calculation:
        work_weight_percentage = 0.3
        attitude_weight_percentage = 0.4
        completion_weight_percentage = 0.2
        customer_weight_percentage = 0.1

        df_name2['Final Score'] = (
            df_name2['Work Score'] * work_weight_percentage
        ) + (df_name2['Attitude Score'] * attitude_weight_percentage) + (
            df_name2['Work Completion Score'] * completion_weight_percentage) + (
                (df_name2['Customer: Good Feedback'] -
                 df_name2['Customer: Bad Feedback']) * 10 *
                customer_weight_percentage)

        # Round values and turn into percentage for easier view
        df_name2['Work Score (%)'] = round(df_name2['Work Score'] * 10)
        df_name2['Work Completion Score (%)'] = round(
            df_name2['Work Completion Score'] * 10)
        df_name2['Attitude Score (%)'] = round(df_name2['Attitude Score'] * 10)

        df_name2['Final Score (%)'] = round(df_name2['Final Score'] * 10)

        # Alert column
        df_name2['Alert'] = np.where(df_name2['Final Score'] <= 5, 'WARNING!', '')

        df_name2['Alert'] = np.where(
            (df_name2['Final Score'] > 5) & (df_name2['Final Score'] < 7), 'CHECK',
            df_name2['Alert'])

        df_name2['Alert'] = np.where(
            (df_name2['Final Score'] >= 7) & (df_name2['Final Score'] < 8),
            'NORMAL', df_name2['Alert'])

        df_name2['Alert'] = np.where(df_name2['Final Score'] >= 8, 'GOOD JOB',
                                     df_name2['Alert'])

        df_name2['Alert'] = np.where(df_name2['Final Score'] >= 10, 'AMAZING!',
                                     df_name2['Alert'])


        # df_name2 = df_name2.drop(['Number of Normal Loan Completed'], axis=1)

        #     selected_cols1 = ['Month', 'Name', 'Title', 'Total Loan Assigned','Number of Loan Completed', 'Number of Loan Takeover', 'KPI', 'Number of Missed Milestone', 'Number of Loan Denied', 'Number of Loan Withdrawn', 'Customer: Good Feedback', 'Customer: Bad Feedback',
        #                      'Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)', 'Final Score (%)', 'Alert']

        #     selected_cols2 = ['Month', 'Name', 'Title', 'Withdrawn Rate (%)', 'Denial Rate (%)', ]

        df_1 = df_name2

        return df_1

    # Color Dataframe Function
    def color_recommend(value):
        if value == 'WARNING!':
            color = 'red'
        elif value == 'CHECK':
            color = 'pink'
        else:
            return
        return f'background-color: {color}'

    def color_negative_red(value):
        if value < 50:
            color = 'pink'
        elif value > 80:
            color = 'lightgreen'
        else:
            color = 'lightgrey'
        return 'background-color: %s' % color

    def final_color_negative_red(value):
        if value < 50:
            color = '#FF8886'
        elif value > 80:
            color = '#00c04b'
        else:
            color = 'lightgrey'
        return 'background-color: %s' % color

    def create_report_fullyear(evaluation_file_path, loan_status_path, kpi_summary_path):    #to create 12 month report of all employees
        df_report = pd.read_csv(evaluation_file)
        employee_list = list(df_report['Name'].unique())
        
        # Month Map
        month_dict = {
        0:'January',
        1:'February',
        2:'March',
        3:'April',
        4:'May',
        5:'June',
        6:'July',
        7:'August',
        8:'September',
        9:'October',
        10:'November',
        11:'December'
        }
        
        # Month Map Converter
        month_dict_convert = {
        1:'January',
        2:'February',
        3:'March',
        4:'April',
        5:'May',
        6:'June',
        7:'July',
        8:'August',
        9:'September',
        10:'October',
        11:'November',
        12:'December'
        }
        
        # Read data from 'Set KPI' link and clean 
        kpi_summary = pd.read_csv(kpi_summary_path)
        kpi_summary = kpi_summary[['Month', 'Sale Goal']]
        kpi_summary = kpi_summary.drop_duplicates()
        kpi_summary = kpi_summary.reset_index()
        
        # Transfer Sales Goal to variable
        month_sale_input = kpi_summary['Sale Goal']
        time_range = 12

        #Generate full year reports and concat
        df_input_month = {}
        
        # Create inputs for 12 months    
        for i in range(time_range):
            df_input_month[i] = create_report(file_path_report = evaluation_file,
                                                    file_path_loan_status = loan_status, month = i+1, sale = month_sale_input[i])

        df_fullyear = pd.concat([df_input_month[i] for i in df_input_month.keys()])
        df_fullyear['Month Text'] = df_fullyear['Month'].map(month_dict_convert)  

        return df_fullyear

    # Query Function
    def query_employee(df,employee):    #Function to query individual
        return df[df['Name']==employee]
    
    # Set colors alerts for dataframe 
    class color_palette:
        def color_recommend(value):
            if value == 'WARNING!':
                color = 'red'
            elif value == 'CHECK':
                color = 'pink'
            else:
                return
            return f'background-color: {color}'

        def color_negative_red(value):
            if value < 50:
                color = 'pink'
            elif value > 80:
                color = 'lightgreen'
            else:
                color = 'lightgrey'
            return 'background-color: %s' % color

        def final_color_negative_red(value):
            if value < 50:
                color = '#FF8886'
            elif value > 80:
                color = '#00c04b'
            else:
                color = 'lightgrey'
            return 'background-color: %s' % color

    # Display key metrics table
    def display_key_metrics(df_full,df_employee):
        ws_mean,att_mean,wc_mean,final_mean = df_full[['Work Score (%)','Attitude Score (%)','Work Completion Score (%)','Final Score (%)']].mean(axis=0)
        withdrawn_mean, denial_mean = df_full[['Withdrawn Rate (%)','Denial Rate (%)']].mean(axis=0)

        col0, col1, col2, col3, col4, col5 = st.columns(6)

        col0.subheader(df_employee['Name'][0] + '\n key metrics')

        col1.metric(label='Work Score',value=df_employee['Work Score (%)'],delta=(round(int(df_employee['Work Score (%)'])-ws_mean)))

        col2.metric(label='Attitude Score',value=df_employee['Attitude Score (%)'],delta=(round(int(df_employee['Attitude Score (%)'])-att_mean)))

        col2.metric(label='Withdrawn Rate',value=df_employee['Withdrawn Rate (%)'],delta=(round(int(df_employee['Withdrawn Rate (%)']-withdrawn_mean))),delta_color='inverse')

        col3.metric(label='Work Completion Score',value=df_employee['Work Completion Score (%)'],delta=(round(int(df_employee['Work Completion Score (%)'])-wc_mean)))

        col3.metric(label='Denial Rate',value=df_employee['Denial Rate (%)'],delta=(round(int(df_employee['Denial Rate (%)']-denial_mean))),delta_color='inverse')

        col4.metric(label='Customer Feedback Bonus',value=(df_employee['Customer: Good Feedback']-df_employee['Customer: Bad Feedback']))

        col5.metric(label='Final Score',value=df_employee['Final Score (%)'],delta=(round(int(df_employee['Final Score (%)'])-final_mean)))

    # Display key metrics with previous month
    def display_key_metrics_previous(df_fullyear,employee_name,current_month):
        df_employee_current_month = query_employee(df_fullyear,employee_name)
        df_employee_previous_month = query_employee(df_fullyear,employee_name)
        df_employee_current_month = df_employee_current_month.loc[df_employee_current_month['Month'] == current_month]

        if current_month == 1:
            df_employee_previous_month = df_employee_current_month.copy()
        else:
            df_employee_previous_month = df_employee_previous_month.loc[df_employee_previous_month['Month'] == current_month-1]

        col0, col1, col2, col3, col4, col5 = st.columns(6)

        col0.subheader(employee_name + '\n key metrics')

        col1.metric(label='Work Score',value=df_employee_current_month['Work Score (%)'],
                    delta=(round(int(df_employee_current_month['Work Score (%)'])-int(df_employee_previous_month['Work Score (%)']))))

        col2.metric(label='Attitude Score',value=df_employee_current_month['Attitude Score (%)'],
                    delta=(round(int(df_employee_current_month['Attitude Score (%)'])-int(df_employee_previous_month['Attitude Score (%)']))))

        col2.metric(label='Withdrawn Rate',value=df_employee_current_month['Withdrawn Rate (%)'],
                    delta=(round(int(df_employee_current_month['Withdrawn Rate (%)'])-int(df_employee_previous_month['Withdrawn Rate (%)']))),delta_color='inverse')

        col3.metric(label='Work Completion Score',value=df_employee_current_month['Work Completion Score (%)'],
                    delta=(round(int(df_employee_current_month['Work Completion Score (%)'])-int(df_employee_previous_month['Work Completion Score (%)']))))

        col3.metric(label='Denial Rate',value=df_employee_current_month['Denial Rate (%)'],
                    delta=(round(int(df_employee_current_month['Denial Rate (%)'])-int(df_employee_previous_month['Denial Rate (%)']))),delta_color='inverse')

        col4.metric(label='Customer Feedback Bonus',
                    value=(df_employee_current_month['Customer: Good Feedback']-df_employee_current_month['Customer: Bad Feedback']))

        col5.metric(label='Final Score',value=df_employee_current_month['Final Score (%)'],
                    delta=(round(int(df_employee_current_month['Final Score (%)'])-int(df_employee_previous_month['Final Score (%)']))))

    ############################# BEGIN MAIN MODULE #####################
    # Link to Pacificwide Cloud datasets:
    evaluation_file = 'https://cloud.pacificwide.com/dd/10zNRyog88/Evaluation_Report.csv_'
    loan_status = 'https://cloud.pacificwide.com/dd/2t4HA95NDp/Loan_Status.csv_'
    kpi_summary = 'https://cloud.pacificwide.com/dd/jRIpFXZrDz/KPI_Summary.csv_'
    
    # Set option for function:
    if name in ['Leon Le', 'Dat Mai']:
        options = ['Read Me','KPI Setting','Employee Dashboard','Yearly Analysis']  
    else:
        options = ['Read Me', 'Monthly KPI Table','Employee Dashboard','Yearly Analysis']
        
    page = st.sidebar.selectbox('Choose a function', options)
    st.sidebar.success("Login Successful, please select a function above.")
    st.sidebar.write("---")
    
    # User welcome and name display:
    st.sidebar.write('Welcome *%s*' % (name))
    
    # Option to log out
    authenticator.logout('Logout', 'sidebar')
    
    # Set KPI for single month
    try:
        if (page == 'Monthly KPI Table'):
            st.header('Monthly KPI Table')
            st.write('---')  
            df_report = pd.read_csv(evaluation_file)
            employee_list = list(df_report['Name'].unique())

            #Make field for number input of 12 months in a year
            month_dict = {
            0:'January',
            1:'February',
            2:'March',
            3:'April',
            4:'May',
            5:'June',
            6:'July',
            7:'August',
            8:'September',
            9:'October',
            10:'November',
            11:'December'
            }

            month_dict_convert = {
            1:'January',
            2:'February',
            3:'March',
            4:'April',
            5:'May',
            6:'June',
            7:'July',
            8:'August',
            9:'September',
            10:'October',
            11:'November',
            12:'December'
            }
            
            time_range = 12
            
            kpi_summary = pd.read_csv(kpi_summary)
            kpi_summary = kpi_summary[['Month', 'Sale Goal']]
            kpi_summary = kpi_summary.drop_duplicates()
            kpi_summary = kpi_summary.reset_index()
            
            sale_input = kpi_summary['Sale Goal']   
            
            #Generate full year reports and concat
            df_input_month_kpi = {}


            for i in range(time_range):    
                df_input_month_kpi[i] = set_work_kpi(loan_status,
                                                        month = i+1, sale = sale_input[i])

            df_fullyear_kpi = pd.concat([df_input_month_kpi[i] for i in df_input_month_kpi.keys()])
            df_fullyear_kpi = df_fullyear_kpi[['Month', 'Title', 'KPI', 'Sale Goal']]
            df_fullyear_kpi['Month'] = df_fullyear_kpi['Month'].map(month_dict_convert)

            ### Display data frame with [Name + Employee Sales Target]
            df_employee_title = pd.read_csv(evaluation_file)
            df_employee_title = df_employee_title[['Name','Title','Month']].drop_duplicates()

            df_fullyear_kpi_per_employee = pd.merge(df_employee_title, df_fullyear_kpi, on=['Month','Title'],how = 'inner')

            #This is to find duplicated title share the same KPI -> so we can divide by the number of the duplicated title
            #For example given default param, in January : title Full-time junior has kpi of 5 but there are 
            #2 employees with the same title -> so we need to share the KPI among them
            duplicated_title_index = list(df_fullyear_kpi_per_employee.loc[
                                        df_fullyear_kpi_per_employee[['Title','Month']].duplicated(keep=False)].index)


            #Build function helps find the divisor
            def find_divisor(list_index,number):    
                ranges =[]
                for k,g in groupby(enumerate(list_index),lambda x:x[0]-x[1]):
                    group = (map(itemgetter(1),g))
                    group = list(map(int,group))
                    ranges.append((group[0],group[-1]))

                for range in ranges:
                    if number >= min(range) and number <= max(range):
                        return max(range) - min(range) +1

            for index in duplicated_title_index:
                df_fullyear_kpi_per_employee.at[index,'KPI'] = df_fullyear_kpi_per_employee.at[index,'KPI'] / find_divisor(duplicated_title_index,index)

            df_month_total_kpi = df_fullyear_kpi.copy()   #Create a dataframe holds Total KPI of each month
            df_month_total_kpi = df_month_total_kpi.groupby('Month')[['Month','KPI']].sum()  

            df_fullyear_kpi_per_employee = pd.merge(df_fullyear_kpi_per_employee,df_month_total_kpi, on='Month').rename(
                                                                columns={'KPI_x': 'KPI', 'KPI_y':'Total_KPI'})

            #Create column Employee Sale Target
            df_fullyear_kpi_per_employee['Est. Sale Target per Employee'] = ((df_fullyear_kpi_per_employee['Sale Goal']) / (
                                                                    df_fullyear_kpi_per_employee['Total_KPI'])) * df_fullyear_kpi_per_employee['KPI']

            cols_to_show = ['Month','Name','Title','KPI','Est. Sale Target per Employee']
            df_fullyear_kpi_per_employee = df_fullyear_kpi_per_employee[cols_to_show]

            #Color the dataframe by their title
            #Color choose from plotly pallete: https://plotly.com/python/discrete-color/
            #Junior Part-time : pastel2[4] = rgb(230,245,201)
            #Junior Full-time : pastel2[4] = rgb(179,226,205)
            #Senior Part-time : pastel2[4] = rgb(203,213,232)       
            #Senior Part-time : pastel2[4] = rgb(244,208,228) 

            def color_row_condition(row):
                if row['Month'] == 'January':
                    color = ['background-color: rgb(240, 255, 255)']*len(row)
                elif row['Month'] == 'Febuary':
                    color = ['background-color: rgb(159, 226, 191)']*len(row)
                elif row['Month'] == 'March':
                    color = ['background-color: rgb(234, 221, 202)']*len(row)
                elif row['Month'] == 'April':
                    color = ['background-color: rgb(178, 190, 181)']*len(row)
                elif row['Month'] == 'May':
                    color = ['background-color: rgb(236, 255, 220)']*len(row)
                elif row['Month'] == 'June':
                    color = ['background-color: rgb(248, 131, 121)']*len(row)
                elif row['Month'] == 'July':
                    color = ['background-color: rgb(255, 245, 238)']*len(row)
                elif row['Month'] == 'August':
                    color = ['background-color: rgb(248, 200, 220)']*len(row)
                elif row['Month'] == 'September':
                    color = ['background-color: rgb(150,213,232)']*len(row)
                elif row['Month'] == 'October':
                    color = ['background-color: rgb(230, 230, 250)']*len(row)
                elif row['Month'] == 'November':
                    color = ['background-color: rgb(245, 245, 220)']*len(row)
                else:
                    color = ['background-color: rgb(147, 197, 114)']*len(row)
                return color

            # df_fullyear_kpi_per_employee = df_fullyear_kpi_per_employee.set_index('Month')
            df_fullyear_kpi_per_employee = df_fullyear_kpi_per_employee.style.apply(color_row_condition, axis=1)
            df_fullyear_kpi_per_employee = df_fullyear_kpi_per_employee.format({'KPI': '{:.0f} loan(s)', 'Est. Sale Target per Employee': '${0:,.0f}'})


    #         df_fullyear_kpi_per_employee.KPI = round(df_fullyear_kpi_per_employee.KPI, 0)
    #         df_fullyear_kpi_per_employee["Est. Sale Target per Employee"] = round(df_fullyear_kpi_per_employee["Est. Sale Target per Employee"], 0)

            hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            st.table(df_fullyear_kpi_per_employee) #Final Chart

            
#         if (page == '1. KPI Report'):
#             st.header('KPI Calculator System')
#     #         st.write(
#     #             'Upload the necessary files for the KPI program to run')

#     #         evaluation_file = st.file_uploader('Upload Evaluation CSV File Below', type=[
#     #                                 'csv'], accept_multiple_files=False, key = "1")


#     #         loan_status = st.file_uploader('Upload Loan Status CSV File Below', type=[
#     #                                 'csv'], accept_multiple_files=False, key = "2")


#             # month_input = st.number_input('Insert month to set KPI', key = "1")
#             # st.write('The current month is ', month_input)
#     #         note = 'Please make sure the uploaded csv files are in exactly the same format.(i.e.Same number of columns and same columns names).'
#     #         st.markdown(
#     #             f'<h1 style="color:#f20f12;font-size:16px;">{note}</h1>', unsafe_allow_html=True)

#             st.write('-------------------------------------')
#             st.subheader("Month Selection:")
#             month_input = st.selectbox(
#             'Month to set KPI',
#             (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), key = 1)
#             st.write('The Current Month is:', month_input)


#             st.subheader("Set Sale Goal:")
#             sale_input = st.number_input('Your Sale Goal', 1000000, key = 2)
#             st.write('The Current Sale Goal is ', '${:,.2f}'.format(sale_input))

#             # sale_input = st.slider('Your Sale Goal', min_value = 100000, max_value = 10000000, step = 100000, key = "1")
#             # st.write('The Current Sale Goal is ', sale_input)


#             df = create_report(file_path_report = evaluation_file,
#                     file_path_loan_status = loan_status, month = month_input, sale = sale_input)

#             selected_cols4 = ['Name', 'Title', 'Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)', 'Final Score (%)', 'Alert']

#             kpi_df = df[selected_cols4]
#             kpi_df = kpi_df.set_index('Name')

#             kpi_df = (
#                 kpi_df.style.applymap(color_negative_red, subset=['Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)'])
#                     .applymap(color_recommend, subset=['Alert'])
#                     .applymap(final_color_negative_red, subset=['Final Score (%)'])

#             )


#             if evaluation_file:
#                 # if st.checkbox("Show File"):
#                 st.write('-------------------------------------')
#                 st.write('This is how the report looks like: ')
#                 st.dataframe(kpi_df, use_container_width=True)


#                 # fname = st.text_input('Output file name(No suffixes):')
#                 # fname = fname+'.csv'

#                 csv = df.to_csv(index=False)
#                 st.download_button('Download detail KPI Report file', data=csv)

#             st.write('-------------------------------------')


#             # selected_cols4 = ['Name', 'Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)']
#             # df2 = df[selected_cols4]
#             # df2 = df2.set_index('Name')
#             # st.bar_chart(df2)

#     ##### Breake down section start in 1.
#             # list = df['Name'].tolist()

#             # processor_name = st.selectbox(
#             # 'Select Processor to View Breakdown',
#             # list, key = 3)

#             # if st.checkbox("Breakdown for WORK SCORE"):
#             #     df_processor = df[(df['Name'] == processor_name)]
#             #     df_processor = df_processor.reset_index()

#             #     X = df[['Number of Normal Loan Completed', 'Number of Loan Takeover', 'Number of Loan Completed Within 3 Weeks', 'Number of Difficult Loan', 'Number of Missed Milestone', 'Check-list Forms & Procedure (%)']]
#             #     X2 = df_processor[['Number of Normal Loan Completed', 'Number of Loan Takeover', 'Number of Loan Completed Within 3 Weeks', 'Number of Difficult Loan', 'Number of Missed Milestone', 'Check-list Forms & Procedure (%)']]

#             #     y = df['Work Score (%)']

#             #     numerical_features = ['Number of Normal Loan Completed', 'Number of Loan Takeover', 'Number of Loan Completed Within 3 Weeks', 'Number of Difficult Loan', 'Number of Missed Milestone', 'Check-list Forms & Procedure (%)']
#             #     numerical_transformer = Pipeline(
#             #         steps=[
#             #             ('imputer', SimpleImputer(strategy='median')),
#             #             ('scaler', StandardScaler())
#             #         ]
#             #     )

#             #     preprocessor = ColumnTransformer(
#             #         transformers=[
#             #             ('num', numerical_transformer, numerical_features),
#             #             # ('cat', categorical_transformer, categorical_features)
#             #         ]
#             #     )

#             #     lr = LinearRegression()

#             #     model = Pipeline(steps=[('preprocessor', preprocessor),
#             #               ('lr', lr)])

#             #     model.fit(X, y)

#             #     for i in range(0, len(df_processor)):
#             #         model_exp = dx.Explainer(model, X, y, 
#             #                     label = "Features effect WORK SCORE")
#             #         example = X2.iloc[i]

#             #         bd = model_exp.predict_parts(example, 
#             #                 type = 'break_down')

#             #         f = bd.plot(show=False)

#             #         st.plotly_chart(f, use_container_width=True)


#             # if st.checkbox("Breakdown for ATTITUDE SCORE"):
#             #     df_processor = df[(df['Name'] == processor_name)]
#             #     df_processor = df_processor.reset_index()

#             #     X = df[['Attendee', 'Teamwork', 'Colab', 'Work behavior', 'Learning', 'Responsibility']]
#             #     X2 = df_processor[['Attendee', 'Teamwork', 'Colab', 'Work behavior', 'Learning', 'Responsibility']]

#             #     y = df['Attitude Score (%)']

#             #     numerical_features = ['Attendee', 'Teamwork', 'Colab', 'Work behavior', 'Learning', 'Responsibility']
#             #     numerical_transformer = Pipeline(
#             #         steps=[
#             #             ('imputer', SimpleImputer(strategy='median')),
#             #             ('scaler', StandardScaler())
#             #         ]
#             #     )

#             #     preprocessor = ColumnTransformer(
#             #         transformers=[
#             #             ('num', numerical_transformer, numerical_features),
#             #         ]
#             #     )

#             #     lr = LinearRegression()

#             #     model = Pipeline(steps=[('preprocessor', preprocessor),
#             #               ('lr', lr)])

#             #     model.fit(X, y)

#             #     for i in range(0, len(df_processor)):
#             #         model_exp = dx.Explainer(model, X, y, 
#             #                     label = df_processor['Name'][i] + "'S ATTITUDE SCORE - " + df_processor['Alert'][i])
#             #         example = X2.iloc[i]

#             #         bd = model_exp.predict_parts(example, 
#             #                 type = 'break_down')

#             #         f = bd.plot(show=False)

#             #         st.plotly_chart(f, use_container_width=True)

#             # if st.checkbox("Breakdown for WORK COMPLETION SCORE"):
#             #     df_processor = df[(df['Name'] == processor_name)]
#             #     df_processor = df_processor.reset_index()

#             #     X = df[['All Loan Completed', 'Number of Loan Denied', 'Number of Loan Withdrawn']]
#             #     X2 = df_processor[['All Loan Completed', 'Number of Loan Denied', 'Number of Loan Withdrawn']]

#             #     y = df['Work Completion Score (%)']

#             #     numerical_features = ['All Loan Completed', 'Number of Loan Denied', 'Number of Loan Withdrawn']
#             #     numerical_transformer = Pipeline(
#             #         steps=[
#             #             ('imputer', SimpleImputer(strategy='median')),
#             #             ('scaler', StandardScaler())
#             #         ]
#             #     )

#             #     preprocessor = ColumnTransformer(
#             #         transformers=[
#             #             ('num', numerical_transformer, numerical_features),
#             #         ]
#             #     )

#             #     lr = LinearRegression()

#             #     model = Pipeline(steps=[('preprocessor', preprocessor),
#             #               ('lr', lr)])

#             #     model.fit(X, y)

#             #     for i in range(0, len(df_processor)):
#             #         model_exp = dx.Explainer(model, X, y, 
#             #                     label = df_processor['Name'][i] + "'S WORK COMPLETION SCORE - " + df_processor['Alert'][i])
#             #         example = X2.iloc[i]

#             #         bd = model_exp.predict_parts(example, 
#             #                 type = 'break_down')

#             #         f = bd.plot(show=False)

#             #         st.plotly_chart(f, use_container_width=True)


    ########################################    KPI SETTING FOR WHOLE YEAR  ####################################################
    
        elif (page == 'KPI Setting'):
            st.header('Set Sale Goal and KPI for Whole Year')

            st.write('-------------------------------------')  
            df_report = pd.read_csv(evaluation_file)
            employee_list = list(df_report['Name'].unique())

            #Make field for number input of 12 months in a year
            month_dict = {
            0:'January',
            1:'February',
            2:'March',
            3:'April',
            4:'May',
            5:'June',
            6:'July',
            7:'August',
            8:'September',
            9:'October',
            10:'November',
            11:'December'
            }

            month_dict_convert = {
            1:'January',
            2:'February',
            3:'March',
            4:'April',
            5:'May',
            6:'June',
            7:'July',
            8:'August',
            9:'September',
            10:'October',
            11:'November',
            12:'December'
            }
            month_sale_input = []
            time_range = 12

            for i in range(time_range):
                month_sale_input.append(st.number_input('Enter sale goal (of all employees) for {}'.format(month_dict[i]), value= 33000000,step=200000))    


            #Generate full year reports and concat
            df_input_month_kpi = {}


            for i in range(time_range):    
                df_input_month_kpi[i] = set_work_kpi(loan_status,
                                                        month = i+1, sale = month_sale_input[i])

            df_fullyear_kpi = pd.concat([df_input_month_kpi[i] for i in df_input_month_kpi.keys()])
            df_fullyear_kpi = df_fullyear_kpi[['Month', 'Title', 'KPI', 'Sale Goal']]
            df_fullyear_kpi['Month'] = df_fullyear_kpi['Month'].map(month_dict_convert)
            ####
            st.write('This is how the KPI Summary looks like: ')

            ### Display data frame with [Name + Employee Sales Target]
            df_employee_title = pd.read_csv(evaluation_file)
            df_employee_title = df_employee_title[['Name','Title','Month']].drop_duplicates()

            df_fullyear_kpi_per_employee = pd.merge(df_employee_title, df_fullyear_kpi, on=['Month','Title'],how = 'inner')

            #This is to find duplicated title share the same KPI -> so we can divide by the number of the duplicated title
            #For example given default param, in January : title Full-time junior has kpi of 5 but there are 
            #2 employees with the same title -> so we need to share the KPI among them
            duplicated_title_index = list(df_fullyear_kpi_per_employee.loc[
                                        df_fullyear_kpi_per_employee[['Title','Month']].duplicated(keep=False)].index)


            #Build function helps find the divisor
            def find_divisor(list_index,number):    
                ranges =[]
                for k,g in groupby(enumerate(list_index),lambda x:x[0]-x[1]):
                    group = (map(itemgetter(1),g))
                    group = list(map(int,group))
                    ranges.append((group[0],group[-1]))

                for range in ranges:
                    if number >= min(range) and number <= max(range):
                        return max(range) - min(range) +1

            for index in duplicated_title_index:
                df_fullyear_kpi_per_employee.at[index,'KPI'] = df_fullyear_kpi_per_employee.at[index,'KPI'] / find_divisor(duplicated_title_index,index)

            df_month_total_kpi = df_fullyear_kpi.copy()   #Create a dataframe holds Total KPI of each month
            df_month_total_kpi = df_month_total_kpi.groupby('Month')[['Month','KPI']].sum()  

            df_fullyear_kpi_per_employee = pd.merge(df_fullyear_kpi_per_employee,df_month_total_kpi, on='Month').rename(
                                                                columns={'KPI_x': 'KPI', 'KPI_y':'Total_KPI'})

            #Create column Employee Sale Target
            df_fullyear_kpi_per_employee['Est. Sale Target per Employee'] = ((df_fullyear_kpi_per_employee['Sale Goal']) / (
                                                                    df_fullyear_kpi_per_employee['Total_KPI'])) * df_fullyear_kpi_per_employee['KPI']

            cols_to_show = ['Month','Name','Title','KPI','Est. Sale Target per Employee']
            df_fullyear_kpi_per_employee = df_fullyear_kpi_per_employee[cols_to_show]

            #Color the dataframe by their title
            #Color choose from plotly pallete: https://plotly.com/python/discrete-color/
            #Junior Part-time : pastel2[4] = rgb(230,245,201)
            #Junior Full-time : pastel2[4] = rgb(179,226,205)
            #Senior Part-time : pastel2[4] = rgb(203,213,232)       
            #Senior Part-time : pastel2[4] = rgb(244,208,228) 

            def color_row_condition(row):
                if row['Month'] == 'January':
                    color = ['background-color: rgb(240, 255, 255)']*len(row)
                elif row['Month'] == 'Febuary':
                    color = ['background-color: rgb(159, 226, 191)']*len(row)
                elif row['Month'] == 'March':
                    color = ['background-color: rgb(234, 221, 202)']*len(row)
                elif row['Month'] == 'April':
                    color = ['background-color: rgb(178, 190, 181)']*len(row)
                elif row['Month'] == 'May':
                    color = ['background-color: rgb(236, 255, 220)']*len(row)
                elif row['Month'] == 'June':
                    color = ['background-color: rgb(248, 131, 121)']*len(row)
                elif row['Month'] == 'July':
                    color = ['background-color: rgb(255, 245, 238)']*len(row)
                elif row['Month'] == 'August':
                    color = ['background-color: rgb(248, 200, 220)']*len(row)
                elif row['Month'] == 'September':
                    color = ['background-color: rgb(150,213,232)']*len(row)
                elif row['Month'] == 'October':
                    color = ['background-color: rgb(230, 230, 250)']*len(row)
                elif row['Month'] == 'November':
                    color = ['background-color: rgb(245, 245, 220)']*len(row)
                else:
                    color = ['background-color: rgb(147, 197, 114)']*len(row)
                return color

            # df_fullyear_kpi_per_employee = df_fullyear_kpi_per_employee.set_index('Month')
            df_fullyear_kpi_per_employee = df_fullyear_kpi_per_employee.style.apply(color_row_condition, axis=1)
            df_fullyear_kpi_per_employee = df_fullyear_kpi_per_employee.format({'KPI': '{:.0f} loan(s)', 'Est. Sale Target per Employee': '${0:,.0f}'})


    #         df_fullyear_kpi_per_employee.KPI = round(df_fullyear_kpi_per_employee.KPI, 0)
    #         df_fullyear_kpi_per_employee["Est. Sale Target per Employee"] = round(df_fullyear_kpi_per_employee["Est. Sale Target per Employee"], 0)

            hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            st.table(df_fullyear_kpi_per_employee) #Final Chart

            # st.dataframe(df_fullyear_kpi_per_employee.style.format({'KPI': '{:.0f} loan(s)', 'Est. Sale Target per Employee': '${0:,.0f}'}), use_container_width=True) #final result
            csv1 = df_fullyear_kpi.to_csv(index = False) 
            st.download_button('Download KPI Summary file for Pacificwide Cloud', data=csv1, file_name='KPI Summary.csv', mime='text/csv', key='download1')
            st.markdown('For system to run. Please download and save this file to Pacificwide Cloud at Shared / OFFICE / 9. DATA SCIENCE / KPIs for Mortgage / (subject to change).')
            
#             csv2 = df_fullyear_kpi_per_employee.to_csv(index = False) 
#             st.download_button('Download Monthly KPI Summary file', data=csv2, file_name='Monthly KPI Summary.csv', mime='text/csv', key='download2')
#             st.markdown('This file is for users to view.')

    ########################################    READ ME  ####################################################
    
        elif (page == 'Read Me'):
            st.header("READ ME - User's Manual")
            st.write('-------------------------------------')
            
            st.subheader('How-To-Use')
            st.write('For the KPI System to work, it will need KPI Summary, Loan Status, and Evaluation Report. All the files are located in "Shared / OFFICE / 9. DATA SCIENCE / KPIs for Mortgage / Mortgage Data".')
            st.write('Just replace them with new files (remember to put them in CSV format and ensure the names are the same). Make sure the current version on Pacificwide Cloud of the file is set to latest upload.')
            note3 = 'Step-by-step to operate the KPI System:'
            st.markdown(
                    f'<h1 style="color:grey;font-size:16px;">{note3}</h1>', unsafe_allow_html=True)
            st.write('•	Step 1: Use the “Set KPI” function to set the Sales Goal for every month in the designated year > Click the “Download” button and save it on Pacificwide Cloud Drive at Shared / OFFICE / 9. DATA SCIENCE / KPIs for Mortgage / Mortgage Data (subject to change in the future).')
            st.warning('*%s*' % ("*Only need to be done once a year."))
            
            st.write('•	Step 2: Update the Loan Status file with the latest monthly version. All information can be extracted from Encompass.')
            st.warning('*%s*' % ("*Work best if it can be updated every month."))
            
            st.write("•	Step 3: Update the Evaluation Report with each team member's final evaluation (on Work and Attitude) at the end of every month.")
            st.warning('*%s*' % ("*Need to update at the end of every month."))
            
            
            st.write('-------------------------------------')
            note2 = 'Mortgage KPI - Explanation'
            st.markdown(
                    f'<h1 style="color:grey;font-size:16px;">{note2}</h1>', unsafe_allow_html=True)

            st.write('How Does The KPI System Works? See how we calculate the Mortgage KPI')
            st.write('Link: https://cloud.pacificwide.com/dl/DxMV9wu4SF')


            st.write('-------------------------------------')
            note4 = 'Break down - Explanation'
            st.markdown(
                    f'<h1 style="color:grey;font-size:16px;">{note4}</h1>', unsafe_allow_html=True)

            st.write('See we calculate variable importance using breakdown with the link down below')
            st.write('Link: https://cloud.pacificwide.com/dl/WqAWLSmWAT')
            st.write('')

            st.write('-------------------------------------')
            note5 = 'Github Repository'
            st.markdown(
                    f'<h1 style="color:grey;font-size:16px;">{note5}</h1>', unsafe_allow_html=True)

            st.write('Source code for the program')
            st.write('Link: https://github.com/dmai287/kpi_pacificwide')
            st.write('')
            
            # Future Updates
            st.write('-------------------------------------')
            st.subheader("[ Future Updates ] Coming Soon!")
            st.markdown("1. Mortgage Loan Volume Forecast: Using Predictive Machine Learning Model to forecast future Mortgage Loan Volume based the past mortgage department's performance + business development strategies and market trends.")
            st.write("Mortgage Loan Volume forecasting allows our company to efficiently allocate resources for future growth and manage its cash flow. It also helps us to estimate their costs and revenue accurately based on which they are able to predict their short-term and long-term performance.")

            st.markdown("2. Average Cycle Time: (Sum of Days from Application to Funding for All Loans) / (# of Loans Funded in Same Period)")
            st.write("Poor cycle time has been shown to correlate directly to pull-through rates and loan profitability metrics. Referral partners and borrowers have expectations that can quickly sour relationships when loans do not close on time.")

            st.markdown("3. Pull-Through Rate: (# of Funded Loans) / (# of Applications Submitted in Same Period)")
            st.write("The pull-through rate provides a high-level perspective on the overall health of your mortgage operation. Pull-through rate is not used to identify any single portion of your process that is failing, but instead to understand if there are problematic inefficiencies at all, or, conversely, if your process is ready to scale to take on more loan applications.")

            st.markdown("4. Average Mortgage Loan Value: (Total Loan Volume Originated) / (# of Loans Funded in Same Period)")
            st.write("The closer your average mortgage loan volume is to the conforming limit, the more likely you are to generate strong profit from those revenues.")

            st.markdown("5. Cost Per Unit Originated: (Total Business Expenses) / (# of Loans Funded in Same Period)")
            st.write("Keeping costs in line with expected performance is critical to maintaining profitability at scale.")

            st.markdown("6. Cycle Stage Length: (Sum of Days in Stage for All Loans) / (# of Loans Funded in Same Period)")
            st.write("When a Loan Processors average cycle time is high or is rising, being able to quickly identify negative changes across various segments makes it much easier to diagnose and resolve problematic processes.")

            st.markdown("7. Profit Per Loan: ((Total Business Revenue) – (Total Business Expense)) / (# of Loans Funded in Same Period)")
            st.write("If there is one KPI that should always be considered during evaluation and decision making, it’s this one.")

    ########################################    INDIVIDUAL DETAIL  ####################################################

        elif (page == 'Employee Dashboard'):
            df_fullyear = create_report_fullyear(evaluation_file_path=evaluation_file, 
                                                    loan_status_path=loan_status, 
                                                    kpi_summary_path=kpi_summary)


            st.header('Individual Detail')
            st.write('-------------------------------------')   #choose employee for detail
            df_report = pd.read_csv(evaluation_file)
            employee_full_list = df_fullyear['Name'].unique().tolist()
            month_input = st.selectbox(
            'Month',(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), key='month_input')

            month_map = {
                'January': 1,
                'February': 2,
                'March': 3,
                'April': 4,
                'May': 5,
                'June': 6,
                'July': 7,
                'August': 8,
                'September': 9,
                'October': 10,
                'November': 11,
                'December': 12
            }

            df_report.Month = df_report.Month.map(month_map)

            df_report_month = df_report[df_report['Month'] == month_input]
            employee_list = list(df_report_month['Name'].unique())
            employee_1 = st.selectbox('Select employee for the details',(employee_list), key='radio1')       
            employee_2 = st.selectbox('Select another employee for comparison',(['None']+employee_list), key='radio2') 

            #Plot line chart whole year

            #Query one employee full year
            df_average_fullyear = df_fullyear.groupby(by='Month').mean().reset_index()
            month_dict_convert = {
            1:'January',
            2:'February',
            3:'March',
            4:'April',
            5:'May',
            6:'June',
            7:'July',
            8:'August',
            9:'September',
            10:'October',
            11:'November',
            12:'December'
            }
            df_average_fullyear['Month Text'] = df_average_fullyear['Month'].map(month_dict_convert)

            # df_employee_fullyear = query_employee(df_fullyear,employee_1)  #outcome is DF full year by name stats, month by month, skip nan if data missing value

            # df_month = pd.DataFrame({'Month Text': ['January','February','March','April','May','June',
            #                          'July','August','September','October','November','December']})

            # df_employee_fullyear_final = pd.merge(df_employee_fullyear, df_month, on='Month Text', how='right')   #make the employee df month text skip for Nan data
            # df_employee_fullyear_concat_average = pd.merge(df_employee_fullyear_final, df_average_fullyear, on='Month Text',how='right')

            # #for plotly express chart - merge axis = 0
            # df_employee_fullyear_final['Label'] = employee_1
            df_average_fullyear['Label'] = 'Average'

            # df_employee_fullyear_concat_average_0 = pd.concat([df_employee_fullyear_final,df_average_fullyear],axis=0)

            month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
             'August', 'September', 'October', 'November', 'December']

       
            st.write('-------------------------------------')

            kpi_summary = pd.read_csv(kpi_summary)
            kpi_summary = kpi_summary[['Month', 'Sale Goal']]
            kpi_summary.Month = kpi_summary.Month.map(month_map)
            kpi_summary = kpi_summary.drop_duplicates()
            kpi_summary = kpi_summary.reset_index()


            sale_input = kpi_summary[kpi_summary['Month'] == month_input]
            sale_input = sale_input['Sale Goal']

            df_input = create_report(file_path_report = evaluation_file,
                    file_path_loan_status = loan_status, month = month_input, sale = sale_input)

            selected_cols = ['Name', 'Title', 'Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)', 'Final Score (%)', 'Alert']

            kpi_df_show = df_input[selected_cols]
            kpi_df_show = kpi_df_show.set_index('Name')

            kpi_df_show = (
                kpi_df_show.style.applymap(color_negative_red, subset=['Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)'])
                    .applymap(color_recommend, subset=['Alert'])
                    .applymap(final_color_negative_red, subset=['Final Score (%)'])

            )
            
            kpi_df_show = kpi_df_show.format({'Work Score (%)': '{:.0f}%', 'Attitude Score (%)': '{:,.0f}%', 
                                        'Work Completion Score (%)': '{:,.0f}%',  'Final Score (%)': '{:,.0f}%'})

            st.subheader('Employees in the Selected Month')
            st.dataframe(kpi_df_show, use_container_width=True)

            st.caption('See how your scores calculated: https://cloud.pacificwide.com/dl/DxMV9wu4SF')    

            st.write('-------------------------------------')

            st.subheader('Dashboard: Employee Detail')

            if st.session_state.radio2 == 'None':
                df_employee1 = query_employee(df_input,employee_1).reset_index()
                df_employee_concat = df_employee1

                if df_employee1['Name'].empty:
                    st.warning('Employee {} is not in the evaluation file or no records in the select month'.format(employee_1))


                else:
                    cols_to_show = ['Month','Title','Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)', 'Final Score (%)','Alert']
                    df_employee1_core = df_employee1[cols_to_show]
#                     df_employee1_core.rename(columns={0: employee_1},inplace=True)
                    
#                     df_employee1_color = df_employee1_core.style.applymap(
#                         color_palette.color_negative_red, subset=pd.IndexSlice[
#                             ['Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)'],:]).applymap(
#                         color_palette.color_recommend, subset=pd.IndexSlice[['Alert'],:]).applymap(
#                         color_palette.final_color_negative_red, subset=pd.IndexSlice[['Final Score (%)'],:])

                    df_employee1_color = (
                        df_employee1_core.style.applymap(color_negative_red, subset=['Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)'])
                            .applymap(color_recommend, subset=['Alert'])
                            .applymap(final_color_negative_red, subset=['Final Score (%)'])
                    )
    
                    df_employee1_color = df_employee1_color.format({'Work Score (%)': '{:.0f}%', 'Attitude Score (%)': '{:,.0f}%', 'Work Completion Score (%)': '{:,.0f}%',  'Final Score (%)': '{:,.0f}%'})

                    hide_dataframe_row_index = """
                                            <style>
                                            .row_heading.level0 {display:none}
                                            .blank {display:none}
                                            </style>
                                            """
                    st.markdown(hide_dataframe_row_index,unsafe_allow_html=True)
                    st.table(df_employee1_color)
#                   st.dataframe(df_employee1_color, use_container_width=True) #Show dataframe employee 1


                    #KEY METRICS
                    #call function
                    # display_key_metrics(df_input, df_employee1)
                    # st.caption('Changes based on average performance of all employees in selected month and sales goal.')

                    display_key_metrics_previous(df_fullyear, employee_1, month_input)
                    st.caption('Changes based on previous month performance.')



                    df_employee1_ws = df_employee1[['KPI','All Loan Completed']].T.reset_index().rename(columns={'index':'Name',0:'Value'})
                    st.write('---')
                    col0,col1,col2,col3,col4,col5 = st.columns(6)
                    col0.write('Work Completed vs KPI')
                    col2.metric(label='KPI',value=df_employee1_ws['Value'][0])
                    col3.metric(label='Completed', value=round(df_employee1_ws['Value'][1],1))
                    col4.metric(label='Difference', value=round(df_employee1_ws['Value'][1]-df_employee1_ws['Value'][0],1))

                    #Bar Chart - Work Score
                    st.write('---')


                    #Plot metric
                    # df_difference = pd.DataFrame({'Name':['Difference'], 'Value':[df_employee1_ws['Value'][1] - df_employee1_ws['Value'][0]]})
                    # df_employee1_ws_diff = df_employee1_ws.append(df_difference,ignore_index=True)




                    #Bar Chart - with difference
                    # change = 100*(df_employee1_ws['Value'][1] - df_employee1_ws['Value'][0]) / df_employee1_ws['Value'][0]

                    # ws2 = go.Figure(data=[
                    #         go.Bar(x=df_employee1_ws['Name'], y=df_employee1_ws['Value'],
                    #                 text=f"+{change:.0f}%" if change > 0 else f"{change:.0f}%",
                    #                 textposition='outside',
                    #                 textfont_size=18,
                    #                 textfont_color='red')])

                    # st.plotly_chart(ws2)


                    #RADAR CHART - Attitude

                    attitude_features = ['Attendance','Teamwork','Collaboration','Work Behavior','Learning','Responsibility']
    #                 st.subheader('{} performance'.format(employee_1))
                    df_employee1_attitude = df_employee1[attitude_features]  #mask to get DF of employee 1 with only attitude
                    df_plot_employee1 = df_employee1_attitude.T.reset_index().rename(columns={'index':'theta',0:'r'}) #Format dataframe for plotly input

                    #Begin to plot spider chart
                    fig1 = px.line_polar(df_plot_employee1,r='r',theta = 'theta',line_close=True,height=430,width=430)
                    fig1.update_traces(fill='toself')
                    fig1.update_polars(radialaxis=dict(visible=True,range=[0, 10]))


                    #PIE CHART - Work Completion


                    workCompletion_features = ['Number of Loan Completed','Number of Loan Withdrawn','Number of Loan Denied']

                    df_employee1_wc = df_employee1[workCompletion_features].T.reset_index().rename(columns={'index':'Name',0:"Value"})

                    #Begin to plot
                    wc1 = px.pie(df_employee1_wc, names='Name',values='Value',hole=0.4,
                                    color_discrete_sequence=['rgb(27,158,119)','rgb(117, 112, 179)','rgb(102, 102, 102)'])

                    col1, col2 = st.columns(2)
                    with col1:
                        st.header("Work Completion Chart")
                        st.plotly_chart(wc1, use_container_width=True)  

                    with col2:
                        st.header("Attitude Chart")
                        st.plotly_chart(fig1, use_container_width=True)

    ##Begin to plot line chart for individual 1 whole year







    ######### add breakdown chart to section 4 - Employee2 == NONE
    #                 list = df_input['Name'].tolist()

    #                 processor_name = st.selectbox(
    #                 'Select Processor to View Breakdown',
    #                 list, key = 3)
                    st.write('---')
                    st.subheader('Breakdown Explaination')
    #                 st.write("WORK SCORE - Breakdown")
                    df_processor = df_input[(df_input['Name'] == employee_1)]
                    df_processor = df_processor.reset_index()

                    X = df_input[['Number of Normal Loan Completed', 'Number of Loan Takeover', 'Number of Loan Completed Within 3 Weeks', 'Number of Difficult Loan', 'Number of Missed Milestone', 'Check-list Forms & Procedure (%)']]
                    X2 = df_processor[['Number of Normal Loan Completed', 'Number of Loan Takeover', 'Number of Loan Completed Within 3 Weeks', 'Number of Difficult Loan', 'Number of Missed Milestone', 'Check-list Forms & Procedure (%)']]

                    y = df_input['Work Score (%)']

                    numerical_features = ['Number of Normal Loan Completed', 'Number of Loan Takeover', 'Number of Loan Completed Within 3 Weeks', 'Number of Difficult Loan', 'Number of Missed Milestone', 'Check-list Forms & Procedure (%)']
                    numerical_transformer = Pipeline(
                        steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())
                        ]
                    )

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numerical_transformer, numerical_features),
                            # ('cat', categorical_transformer, categorical_features)
                        ]
                    )

                    lr = LinearRegression()

                    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('lr', lr)])

                    model.fit(X, y)

                    for i in range(0, len(df_processor)):
                        model_exp = dx.Explainer(model, X, y, 
                                    label = "Feature affect " + df_processor['Name'][i] + "'S WORK SCORE")
                        example = X2.iloc[i]

                        bd = model_exp.predict_parts(example, 
                                type = 'break_down')

                        f = bd.plot(show=False)

                        st.plotly_chart(f, use_container_width=True)


    #                 if st.checkbox("Breakdown for ATTITUDE SCORE"):
    #                 st.write("ATTITUDE SCORE - Breakdown")
                    df_processor = df_input[(df_input['Name'] == employee_1)]
                    df_processor = df_processor.reset_index()

                    X = df_input[['Attendance', 'Teamwork', 'Collaboration', 'Work Behavior', 'Learning', 'Responsibility']]
                    X2 = df_processor[['Attendance', 'Teamwork', 'Collaboration', 'Work Behavior', 'Learning', 'Responsibility']]

                    y = df_input['Attitude Score (%)']

                    numerical_features = ['Attendance', 'Teamwork', 'Collaboration', 'Work Behavior', 'Learning', 'Responsibility']
                    numerical_transformer = Pipeline(
                        steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())
                        ]
                    )

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numerical_transformer, numerical_features),
                        ]
                    )

                    lr = LinearRegression()

                    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('lr', lr)])

                    model.fit(X, y)

                    for i in range(0, len(df_processor)):
                        model_exp = dx.Explainer(model, X, y, 
                                    label = "Feature affect " + df_processor['Name'][i] + "'S ATTITUDE SCORE")
                        example = X2.iloc[i]

                        bd = model_exp.predict_parts(example, 
                                type = 'break_down')

                        f = bd.plot(show=False)

                        st.plotly_chart(f, use_container_width=True)


    #                 st.write("WORK COMPLETION SCORE - Breakdown")
                    df_processor = df_input[(df_input['Name'] == employee_1)]
                    df_processor = df_processor.reset_index()

                    X = df_input[['All Loan Completed', 'Number of Loan Denied', 'Number of Loan Withdrawn']]
                    X2 = df_processor[['All Loan Completed', 'Number of Loan Denied', 'Number of Loan Withdrawn']]

                    y = df_input['Work Completion Score (%)']

                    numerical_features = ['All Loan Completed', 'Number of Loan Denied', 'Number of Loan Withdrawn']
                    numerical_transformer = Pipeline(
                        steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())
                        ]
                    )

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numerical_transformer, numerical_features),
                        ]
                    )

                    lr = LinearRegression()

                    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('lr', lr)])

                    model.fit(X, y)

                    for i in range(0, len(df_processor)):
                        model_exp = dx.Explainer(model, X, y, 
                                    label = "Features affect " + df_processor['Name'][i] + "'S WORK COMPLETION SCORE")
                        example = X2.iloc[i]

                        bd = model_exp.predict_parts(example, 
                                type = 'break_down')

                        f = bd.plot(show=False)

                        st.plotly_chart(f, use_container_width=True)
                st.caption('1. Intercept: Understand it as average score of the whole team in selected month.')
                st.caption('2. Contribution: How the each variables contribute to the score.')
                st.caption('See how it works: https://cloud.pacificwide.com/dl/WqAWLSmWAT')

            # Plot Average Line Chart vs Employee
                st.write('---')
                st.subheader('Performance Trend')
                select_list = st.multiselect('Choose more employees to plot',employee_full_list)
                st.write('You are now viewing current performance of ', employee_1)
                st.write('')


            #create dataframe for combine name plot##############
                df_employee_fullyear_more = df_fullyear[(df_fullyear['Name'].isin(select_list)) | (df_fullyear['Name'] == employee_1)]
                df_employee_fullyear_more = df_employee_fullyear_more[['Name','Month Text','Final Score (%)']]

                #make a new dataframe as combination of Name - Month prior to merge
            
                months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
                            'August', 'September', 'October', 'November', 'December']
                all_combinations_employee_select = pd.DataFrame([(employee, month) for employee in select_list for month in months],
                                    columns=['Name', 'Month Text'])
                all_combinations_employee_1 = pd.DataFrame({'Name':[employee_1]*12, 'Month Text': months})
                all_combinations = pd.concat([all_combinations_employee_select,all_combinations_employee_1],axis=0)

                df_employee_fullyear_final_more = pd.merge(df_employee_fullyear_more, all_combinations,
                                                            on=['Name','Month Text'], how='right')

                #make average dataframe
                df_average_fullyear['Name'] = df_average_fullyear['Label']
                df_average_fullyear = df_average_fullyear[['Name','Month Text','Final Score (%)']]

                #Concat 
                df_employee_fullyear_concat_average_more = pd.concat([df_employee_fullyear_final_more, df_average_fullyear], axis=0)
        
                #Plot Chart more employee
                fig_score_year_more = px.line(df_employee_fullyear_concat_average_more, x='Month Text', y='Final Score (%)', color='Name', markers=True)
                fig_score_year_more.update_traces(line_color='black', line_width=2.5, line_dash='dot', selector={'name': 'Average'})
                st.plotly_chart(fig_score_year_more)

            else: #Employee2 not "None"
                df_employee1 = query_employee(df_input,employee_1).reset_index()
                df_employee2 = query_employee(df_input,employee_2).reset_index()
                df_employee_concat = pd.concat([df_employee1,df_employee2],ignore_index=True)

                df_employee_concat_copy = df_employee_concat.copy()
                df_employee_concat_copy = df_employee_concat_copy.apply(pd.to_numeric,errors='coerce')
                df_employee_concat_diff = df_employee_concat_copy.diff(periods=-1)
                df_employee_concat_diff['Month'] = np.nan
                df_employee_concat_diff = df_employee_concat_diff.replace(np.nan,'-')
                df_employee_concat_diff = df_employee_concat_diff.iloc[0,:]
                df_employee_concat = df_employee_concat.append(df_employee_concat_diff).reset_index()

                cols_to_show = ['Month','Title','Work Score (%)','Work Completion Score (%)','Attitude Score (%)', 'Final Score (%)']
                df_employee_concat = df_employee_concat[cols_to_show]     

                df_employee_concat_T = df_employee_concat.copy().T.rename(columns={0:employee_1,1:employee_2,2:'DIFFERENCE'})
                st.table(df_employee_concat_T)   
#                 st.dataframe(df_employee_concat_T, use_container_width=True)    #Show dataframe


                ###########KEY METRICS DISPLAY 
                ###########KEY METRICS DISPLAY 
                # display_key_metrics(df_input,df_employee1)
                # st.write('---')
                # display_key_metrics(df_input,df_employee2)
                # st.caption('Changes based on average performance of all employees in selected month and sales goal')

                #### DISPLOAY METRICS COMPARE PREVIOUS
                display_key_metrics_previous(df_fullyear,employee_1, month_input)
                st.write('---')
                display_key_metrics_previous(df_fullyear,employee_2, month_input)
                st.caption('Changes based on previous month perfomance')

                #KPI vs Completed
                st.write('---')
                df_employee1_ws = df_employee1[['KPI','All Loan Completed']].T.reset_index().rename(columns={'index':'Name',0:'Value'})
                df_employee2_ws = df_employee2[['KPI','All Loan Completed']].T.reset_index().rename(columns={'index':'Name',0:'Value'})

                col0,col1,col2,col3,col4,col5 = st.columns(6)
                col0.write('Work Completed vs KPI - {}'.format(employee_2))
                col2.metric(label='KPI',value=df_employee1_ws['Value'][0])
                delta1= round(df_employee1_ws['Value'][1]-df_employee1_ws['Value'][0],1)
                col3.metric(label='Completed', value=round(df_employee1_ws['Value'][1],1),delta=delta1)


                st.write('---')

                col0,col1,col2,col3,col4,col5 = st.columns(6)
                col0.write('Work Completed vs KPI - {}'.format(employee_2))
                col2.metric(label='KPI',value=df_employee2_ws['Value'][0])
                delta2= round(df_employee2_ws['Value'][1]-df_employee2_ws['Value'][0],1)
                col3.metric(label='Completed', value=round(df_employee2_ws['Value'][1],1),delta=delta2)



                #Bar chart Work Score
                st.write('---')
                # st.subheader('Work Score')

                # col1,col2 = st.columns(2)

                # #Begin to plot
                # with col1:
                #     df_employee1_ws = df_employee1[['KPI','All Loan Completed']].T.reset_index().rename(columns={'index':'Name',0:'Value'})
                #     ws1 = px.bar(df_employee1_ws, x='Name',y='Value',text_auto=True,color='Name')
                #     ws1.update_layout(showlegend=True, legend=dict(font=dict(size=9)))
                #     st.plotly_chart(ws1, use_container_width=True)

                # with col2:
                #     df_employee2_ws = df_employee2[['KPI','All Loan Completed']].T.reset_index().rename(columns={'index':'Name',0:'Value'})
                #     ws2 = px.bar(df_employee2_ws, x='Name',y='Value',text_auto=True,color='Name')
                #     ws2.update_layout(showlegend=True, legend=dict(font=dict(size=9)))
                #     st.plotly_chart(ws2, use_container_width=True)


                #RADAR CHART - Attitude

                st.subheader('Attitude Score')
                attitude_features = ['Attendance', 'Teamwork', 'Collaboration', 'Work Behavior', 'Learning', 'Responsibility']

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader('{} performance'.format(employee_1))
                    df_employee1_attitude = df_employee1[attitude_features]  #mask to get DF of employee 1 with only attitude
                    df_plot_employee1 = df_employee1_attitude.T.reset_index().rename(columns={'index':'theta',0:'r'}) #Format dataframe for plotly input

                    #Begin to plot spider chart
                    fig1 = px.line_polar(df_plot_employee1,r='r',theta='theta',line_close=True,height=400,width=400)
                    fig1.update_traces(fill='toself')
                    fig1.update_polars(radialaxis=dict(visible=True,range=[0, 10]))
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.subheader('{} performance'.format(employee_2))
                    df_employee2_attitude = df_employee2[attitude_features]  #mask to get DF of employee 1 with only attitude
                    df_plot_employee2 = df_employee2_attitude.T.reset_index().rename(columns={'index':'theta',0:'r'}) #Format dataframe for plotly input

                    #Begin to plot spider chart
                    fig2 = px.line_polar(df_plot_employee2,r='r',theta = 'theta',line_close=True,height=400,width=400)
                    fig2.update_traces(fill='toself')
                    fig2.update_polars(radialaxis=dict(visible=True,range=[0, 10]))
                    st.plotly_chart(fig2, use_container_width=True)

                ##### Button to show stacked radar chart
                if st.checkbox('Press for stacked Radar Chart'):
                    df_plot_employee1['employee'] = employee_1
                    df_plot_employee2['employee'] = employee_2
                    df_plot_employee_concat = pd.concat([df_plot_employee1,df_plot_employee2], axis=0)

                    fig_radar = px.line_polar(df_plot_employee_concat,r='r',theta='theta',color='employee',labels = {'employee': 'Employee Name'} , line_close=True)
                    fig_radar.update_traces(fill='toself')
                    fig_radar.update_polars(radialaxis=dict(visible=True,range=[0, 10]))
                    st.plotly_chart(fig_radar, use_container_width=True)            

                ###########

    #                     fig_radar = go.Figure()
    #                     fig_radar.add_trace(go.Scatterpolar(r=list(df_plot_employee1['r']),
    #                                                         theta=attitude_features,name=employee_1,fill='toself',
    #                                                         fillcolor='blue', opacity=0.5))
    #                     fig_radar.add_trace(go.Scatterpolar(r=list(df_plot_employee2['r']),
    #                                                          theta=attitude_features,name=employee_2,fill='toself',
    #                                                          fillcolor='red',opacity =0.5))
    #                     fig_radar.update_layout(
    #                                             polar = dict(
    #                                             radialaxis = dict(tickangle = 0),
    #                                             angularaxis = dict(
    #                                                     dtick = 45,
    #                                                     rotation=90,
    #                                                     direction = "clockwise"                                               
    #                                                          )
    #                                             ))
    #                     fig_radar.update_polars(radialaxis=dict(range=[0, 10]))

    #                     st.plotly_chart(fig_radar)


                #PIE CHART - Work Completion
                st.subheader('Work Completion Score')
                workCompletion_features = ['Number of Loan Completed','Number of Loan Withdrawn','Number of Loan Denied']

                col1, col2 = st.columns(2)

                with col1:
                    df_employee1_wc = df_employee1[workCompletion_features].T.reset_index().rename(columns={'index':'Name',0:"Value"})
                    #Begin to plot
                    wc1 = px.pie(df_employee1_wc, names='Name',values='Value',hole=0.4,
                                    color_discrete_sequence=['rgb(27,158,119)','rgb(117, 112, 179)','rgb(102, 102, 102)'])

                    wc1.update_layout(legend=dict(orientation="h"))
                    st.plotly_chart(wc1, use_container_width=True)

                with col2:
                    df_employee2_wc = df_employee2[workCompletion_features].T.reset_index().rename(columns={'index':'Name',0:"Value"})
                    #Begin to plot
                    wc2 = px.pie(df_employee2_wc, names='Name',values='Value',hole=0.4,
                                    color_discrete_sequence=['rgb(27,158,119)','rgb(117, 112, 179)','rgb(102, 102, 102)'])
                    wc2.update_layout(legend=dict(orientation="h"))
                    st.plotly_chart(wc2, use_container_width=True)


            ##################### WHOLE YEAR Analysis  ################
        elif (page == 'Yearly Analysis'):

    #        st.write(
    #             'Upload the necessary files for the KPI program to run')

    #         evaluation_file = st.file_uploader('Upload Evaluation CSV File Below', type=[
    #                                 'csv'], accept_multiple_files=False, key = "1")

    #         loan_status = st.file_uploader('Upload Loan Status CSV File Below', type=[
    #                                 'csv'], accept_multiple_files=False, key = "2")

            # month_input = st.number_input('Insert month to set KPI', key = "1")
            # st.write('The current month is ', month_input)

    #         note = 'Please make sure the uploaded csv files are in exactly the same format.(i.e.Same number of columns and same columns names).'
    #         st.markdown(
    #             f'<h1 style="color:#f20f12;font-size:16px;">{note}</h1>', unsafe_allow_html=True)    
    #       
            st.header('Yearly Analysis')
            st.write('---')   #choose employee for detail
            
            # Read the link and extract data from Evaluation File
            df_report = pd.read_csv(evaluation_file)
            employee_list = list(df_report['Name'].unique())

            #Make field for number input of 12 months in a year
            month_dict = {
            0:'January',
            1:'February',
            2:'March',
            3:'April',
            4:'May',
            5:'June',
            6:'July',
            7:'August',
            8:'September',
            9:'October',
            10:'November',
            11:'December'
            }

            month_dict_convert = {
            1:'January',
            2:'February',
            3:'March',
            4:'April',
            5:'May',
            6:'June',
            7:'July',
            8:'August',
            9:'September',
            10:'October',
            11:'November',
            12:'December'
            }
            
            # Read data from KPI Summary link
            kpi_summary = pd.read_csv(kpi_summary)
            kpi_summary = kpi_summary[['Month', 'Sale Goal']]
            kpi_summary = kpi_summary.drop_duplicates()
            kpi_summary = kpi_summary.reset_index()

            month_sale_input = kpi_summary['Sale Goal']
            time_range = 12

            #Generate full year reports and concat
            df_input_month = {}

            
            for i in range(time_range):
                df_input_month[i] = create_report(file_path_report = evaluation_file,
                                                        file_path_loan_status = loan_status, month = i+1, sale = month_sale_input[i])

            df_fullyear = pd.concat([df_input_month[i] for i in df_input_month.keys()])
            df_fullyear['Month Text'] = df_fullyear['Month'].map(month_dict_convert)        


#             ####
#             st.subheader("Month Selection:")
#             month_input = st.selectbox(
#             'Month to set KPI',
#             ('January','February','March','April','May','June',
#                                      'July','August','September','October','November','December'), key = 'month_select')

#             st.write('The Current Month is:', month_input)

#             df_month_input = df_fullyear[df_fullyear['Month Text'] == month_input]
#             selected_cols = ['Month', 'Name', 'Title', 'Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)', 'Final Score (%)', 'Alert']

#             kpi_df = df_month_input[selected_cols]
#             kpi_df = kpi_df.set_index('Name')


#             kpi_df = (
#                 kpi_df.style.applymap(color_negative_red, subset=['Work Score (%)', 'Attitude Score (%)', 'Work Completion Score (%)'])
#                     .applymap(color_recommend, subset=['Alert'])
#                     .applymap(final_color_negative_red, subset=['Final Score (%)'])
#                     )
#             st.dataframe(kpi_df, use_container_width=True)



            #Click to see comparison final score:
            df_fullyear_grouped = df_fullyear.groupby(['Title','Name']).mean().reset_index()
            fig_fullyear = px.bar(df_fullyear_grouped, 
                                    x='Final Score (%)', y='Title', color='Name', orientation='h',barmode='group',text_auto=True)
            fig_fullyear.update_traces(width=0.15)
            fig_fullyear.update_layout(bargap=0.1 ,bargroupgap=0.1)
            fig_fullyear.update_layout(yaxis={'categoryorder':'category ascending'})      
            st.plotly_chart(fig_fullyear, use_container_width=True)

            # #PLOT BY GO TO MAKE GROUP BY CHART EASILY READ
            # st.dataframe(df_fullyear_grouped)
            # title = list(df_fullyear_grouped['Title'].unique())
            # fig_fullyear_go = go.Figure()
            # fig_fullyear_go.add_trace(go.Bar(x=list(df_fullyear_grouped.loc[df_fullyear_grouped['Title']==title[0]]['Name']),
            #                                 y=list(df_fullyear_grouped.loc[df_fullyear_grouped['Title']==title[0]]['Final Score (%)']),
            #                                 name=title[0])
            # )
            # fig_fullyear_go.add_trace(go.Bar(x=list(df_fullyear_grouped.loc[df_fullyear_grouped['Title']==title[1]]['Name']),
            #                                 y=list(df_fullyear_grouped.loc[df_fullyear_grouped['Title']==title[1]]['Final Score (%)']),
            #                                 name=title[1])
            # )
            # fig_fullyear_go.add_trace(go.Bar(x=list(df_fullyear_grouped.loc[df_fullyear_grouped['Title']==title[2]]['Name']),
            #                                 y=list(df_fullyear_grouped.loc[df_fullyear_grouped['Title']==title[2]]['Final Score (%)']),
            #                                 name=title[2])
            # )
            # fig_fullyear_go.add_trace(go.Bar(x=list(df_fullyear_grouped.loc[df_fullyear_grouped['Title']==title[3]]['Name']),
            #                                 y=list(df_fullyear_grouped.loc[df_fullyear_grouped['Title']==title[3]]['Final Score (%)']),
            #                                 name=title[3])
            # )
            # fig_fullyear_go.update_layout(xaxis={'categoryorder':'category ascending'})                          

            # st.plotly_chart(fig_fullyear_go)

            # Plot All employees to see count of reach KPI in 1 year
            df_fullyear['Pass'] = df_fullyear['Number of Loan Completed'] >= df_fullyear['KPI']
            df_fullyear['Pass'] = df_fullyear['Pass'].astype(int)
            df_fullyear['Count'] = 1
            df_fullyear_groupby = df_fullyear.groupby('Name').sum()
            df_fullyear_groupby['Pass (%)'] = 100*df_fullyear_groupby['Pass'] / df_fullyear_groupby['Count']

            fig_kpi = px.bar(df_fullyear_groupby.reset_index(),
                            x='Name',y='Pass (%)',color='Name',text_auto=True)

            fig_kpi.update_layout(yaxis_title='Pass KPI (%)')
            st.plotly_chart(fig_kpi, use_container_width=True)


            st.write('---')

#             #select employee:
#             employee_pick = st.selectbox('select employee to see his performance', (employee_list), key='employee_pick')


            #PLOT LINE CHART FOR INDIVIDUAL
            # df_average_fullyear = df_fullyear.groupby(by='Month').mean().reset_index()
            # df_average_fullyear['Month Text'] = df_average_fullyear['Month'].map(month_dict_convert)

            # df_employee_fullyear = query_employee(df_fullyear,employee_pick)  #outcome is DF full year by name stats, month by month, skip nan if data missing value

            # df_month = pd.DataFrame({'Month Text': ['January','February','March','April','May','June',
            #                          'July','August','September','October','November','December']})

            # df_employee_fullyear_final = pd.merge(df_employee_fullyear, df_month, on='Month Text', how='right')   #make the employee df month text skip for Nan data
            # df_employee_fullyear_concat_average = pd.merge(df_employee_fullyear_final, df_average_fullyear, on='Month Text',how='right')

            # #for plotly express chart - merge axis = 0
            # df_employee_fullyear_final['Label'] = employee_pick
            # df_average_fullyear['Label'] = 'Average'

            # df_employee_fullyear_concat_average_0 = pd.concat([df_employee_fullyear_final,df_average_fullyear],axis=0)

            # month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
            #  'August', 'September', 'October', 'November', 'December']

            # fig_score_year = px.line(df_employee_fullyear_concat_average_0, x='Month Text', y='Final Score (%)', color='Label', labels={"Month Text": ""}, markers=True)
            # st.plotly_chart(fig_score_year, use_container_width=True)





    except:
        pass


# Login Interfere with details on the project
elif authentication_status == False:
    # Welcome Message
    st.header('Welcome to Pacificwide KPI Evaluation System')
    st.write('---')
    
    # About the system
    st.subheader('About this system:')
    st.write('A KPI dashboard displays key performance indicators in interactive charts and graphs, allowing for quick, organized review and analysis. Key performance indicators are quantifiable measures of performance over time for specific strategic objectives.')
    st.write('This KPI System allows managers to easily explore the data behind the KPIs and uncover actionable insights. In this way, a KPI dashboard transforms massive data sets from across an organization into data-driven decisions that can improve our business.')
    
    # System features
    st.subheader('What do KPI dashboards include?')
    st.write('Within it, you can:')
    st.write("- Generative target KPI: Set a doable KPI for each Loan Processor based on companys's historical data and target sales goal.")
    st.write("- Provide a performance check: KPIs give you a realistic look at the performance of your employees and organization over the time, from risk factors to financial indicators.")
    st.write('- Make adjustments: KPIs help you clearly see your successes and failures so you can do more of what’s working, and less of what’s not.')
    st.write('- Hold your teams accountable: Make sure everyone provides value with key performance indicators that help employees track their progress and help managers move things along.')
    
    # Future Updates
    st.subheader("[ Future Updates ] Coming Soon!")
    st.markdown("1. Mortgage Loan Volume Forecast: Using Predictive Machine Learning Model to forecast future Mortgage Loan Volume based the past mortgage department's performance + business development strategies and market trends.")
    st.write("Mortgage Loan Volume forecasting allows our company to efficiently allocate resources for future growth and manage its cash flow. It also helps us to estimate their costs and revenue accurately based on which they are able to predict their short-term and long-term performance.")
    
    st.markdown("2. Average Cycle Time: (Sum of Days from Application to Funding for All Loans) / (# of Loans Funded in Same Period)")
    st.write("Poor cycle time has been shown to correlate directly to pull-through rates and loan profitability metrics. Referral partners and borrowers have expectations that can quickly sour relationships when loans do not close on time.")
    
    st.markdown("3. Pull-Through Rate: (# of Funded Loans) / (# of Applications Submitted in Same Period)")
    st.write("The pull-through rate provides a high-level perspective on the overall health of your mortgage operation. Pull-through rate is not used to identify any single portion of your process that is failing, but instead to understand if there are problematic inefficiencies at all, or, conversely, if your process is ready to scale to take on more loan applications.")
    
    st.markdown("4. Average Mortgage Loan Value: (Total Loan Volume Originated) / (# of Loans Funded in Same Period)")
    st.write("The closer your average mortgage loan volume is to the conforming limit, the more likely you are to generate strong profit from those revenues.")
    
    st.markdown("5. Cost Per Unit Originated: (Total Business Expenses) / (# of Loans Funded in Same Period)")
    st.write("Keeping costs in line with expected performance is critical to maintaining profitability at scale.")
    
    st.markdown("6. Cycle Stage Length: (Sum of Days in Stage for All Loans) / (# of Loans Funded in Same Period)")
    st.write("When a Loan Processors average cycle time is high or is rising, being able to quickly identify negative changes across various segments makes it much easier to diagnose and resolve problematic processes.")
    
    st.markdown("7. Profit Per Loan: ((Total Business Revenue) – (Total Business Expense)) / (# of Loans Funded in Same Period)")
    st.write("If there is one KPI that should always be considered during evaluation and decision making, it’s this one.")
    
    # Wrong username/password input
    st.sidebar.error('Username/password is incorrect. Please Try Again.')
    
elif authentication_status == None:
    # Welcome message
    st.header('Welcome to Pacificwide KPI Evaluation System')
    st.write('---')
    
    # About the system
    st.subheader('About this system:')
    st.write('A KPI dashboard displays key performance indicators in interactive charts and graphs, allowing for quick, organized review and analysis. Key performance indicators are quantifiable measures of performance over time for specific strategic objectives.')
    st.write('This KPI System allows managers to easily explore the data behind the KPIs and uncover actionable insights. In this way, a KPI dashboard transforms massive data sets from across an organization into data-driven decisions that can improve our business.')
    
    # System Features
    st.subheader('What do KPI dashboards include?')
    st.write('Within it, you can:')
    st.write("- Generative target KPI: Set a doable KPI for each Loan Processor based on companys's historical data and target sales goal.")
    st.write("- Provide a performance check: KPIs give you a realistic look at the performance of your employees and organization over the time, from risk factors to financial indicators.")
    st.write('- Make adjustments: KPIs help you clearly see your successes and failures so you can do more of what’s working, and less of what’s not.')
    st.write('- Hold your teams accountable: Make sure everyone provides value with key performance indicators that help employees track their progress and help managers move things along.')
    
    # Future Updates
    st.subheader("[ Future Updates ] Coming Soon!")
    st.markdown("1. Mortgage Loan Volume Forecast: Using Predictive Machine Learning Model to forecast future Mortgage Loan Volume based the past mortgage department's performance + business development strategies and market trends.")
    st.write("Mortgage Loan Volume forecasting allows our company to efficiently allocate resources for future growth and manage its cash flow. It also helps us to estimate their costs and revenue accurately based on which they are able to predict their short-term and long-term performance.")
    
    st.markdown("2. Average Cycle Time: (Sum of Days from Application to Funding for All Loans) / (# of Loans Funded in Same Period)")
    st.write("Poor cycle time has been shown to correlate directly to pull-through rates and loan profitability metrics. Referral partners and borrowers have expectations that can quickly sour relationships when loans do not close on time.")
    
    st.markdown("3. Pull-Through Rate: (# of Funded Loans) / (# of Applications Submitted in Same Period)")
    st.write("The pull-through rate provides a high-level perspective on the overall health of your mortgage operation. Pull-through rate is not used to identify any single portion of your process that is failing, but instead to understand if there are problematic inefficiencies at all, or, conversely, if your process is ready to scale to take on more loan applications.")
    
    st.markdown("4. Average Mortgage Loan Value: (Total Loan Volume Originated) / (# of Loans Funded in Same Period)")
    st.write("The closer your average mortgage loan volume is to the conforming limit, the more likely you are to generate strong profit from those revenues.")
    
    st.markdown("5. Cost Per Unit Originated: (Total Business Expenses) / (# of Loans Funded in Same Period)")
    st.write("Keeping costs in line with expected performance is critical to maintaining profitability at scale.")
    
    st.markdown("6. Cycle Stage Length: (Sum of Days in Stage for All Loans) / (# of Loans Funded in Same Period)")
    st.write("When a Loan Processors average cycle time is high or is rising, being able to quickly identify negative changes across various segments makes it much easier to diagnose and resolve problematic processes.")
    
    st.markdown("7. Profit Per Loan: ((Total Business Revenue) – (Total Business Expense)) / (# of Loans Funded in Same Period)")
    st.write("If there is one KPI that should always be considered during evaluation and decision making, it’s this one.")
    
    # Ask to enter the username and password
    st.sidebar.warning('Please enter your username and password.')