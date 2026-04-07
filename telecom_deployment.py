import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout= 'wide', page_title= 'Telecom Analysis Dashboard')

page = st.sidebar.radio('Pages', ["Home / OverView", "Service Subscribtions", "Contracts & Billing"])
df = pd.read_csv('cleaned_data.csv', index_col= 0)
if page == 'Home / OverView':
    st.image('Data-Analytics-in-The-Telecom-Industry_-Use-Cases_big-new-min-1024x512.webp',width= 1500)
    html_title = """<h1 style="color:white;text-align:center;"> Telecom Exploratory Data Analysis </h1>"""
    st.markdown(html_title, unsafe_allow_html=True)
    st.subheader('Dataset Overview')
    column_descriptions = {
        "gender": "Customer's gender, e.g. 'Male' or 'Female'.",

        "seniorcitizen": "Indicates whether the customer is a senior citizen (e.g. senior vs non-senior).",

        "partner": "Whether the customer has a spouse/partner (Yes/No).",

        "dependents": "Whether the customer has dependents (children, family members relying on them).",

        "multiplelines": "Whether the customer has multiple phone lines with the company (e.g. Yes/No).",

        "internetservice": "Type of internet service the customer has: DSL, Fiber optic, or No internet service.",

        "onlinesecurity": "Whether the customer subscribes to an online security add-on service (Yes/No).",

        "onlinebackup": "Whether the customer subscribes to an online backup service (Yes/No).",

        "deviceprotection": "Whether the customer subscribes to a device protection plan (Yes/No).",

        "techsupport": "Whether the customer subscribes to additional technical support (Yes/No).",

        "streamingtv": "Whether the customer subscribes to streaming TV service from the company (Yes/No).",

        "streamingmovies": "Whether the customer subscribes to streaming movies service from the company (Yes/No).",

        "tenure": "Number of months the customer has stayed with the company.",

        "phoneservice": "Whether the customer has phone service with the company (Yes/No).",

        "contract": "Type of contract the customer has: Month-to-month, One year, or Two year.",

        "paperlessbilling": "Whether the customer receives bills electronically (Yes) or on paper (No).",

        "paymentmethod": "Customer's payment method (e.g. Electronic check, Mailed check, Bank transfer, Credit card).",

        "monthlycharges": "The amount the customer is currently charged per month.",

        "totalcharges": "Total amount the customer has been charged over their entire tenure.",

        "churn": "Target label indicating whether the customer has churned (left the service) or not (Yes/No).",

        "cust-loyality": "Customer loyalty category derived from tenure or similar (e.g. New, Somewhat Loyal, Loyal, Very Loyal).",

        "family_member": "Family status segment combining partner and dependents (e.g. Single, Married, Single with dependents, Married with dependents).",

        "subscription_count": "Number of value-added internet services the customer is subscribed to among: onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingtv, streamingmovies."
    }
    desc_df = pd.DataFrame(list(column_descriptions.items()), columns=["Column Name", "Description"])

    st.subheader("📝 Column Descriptions")
    st.table(desc_df)
    st.subheader("View by Column Type")
    view_option = st.radio("Select view:",("Show all", "Numerical only", "Categorical only"))
    if view_option == "Show all":
        st.write("Showing **all columns**:")
        st.dataframe(df)

    elif view_option == "Numerical only":
        st.write("Showing **numerical columns only**:")
        num_df = df.select_dtypes(include=["int64", "float64"])
        st.dataframe(num_df)

    elif view_option == "Categorical only":
        st.write("Showing **categorical columns only**:")
        cat_df = df.select_dtypes(include=["object", "category", "bool"])
        st.dataframe(cat_df)


    col1, col2= st.columns(2)
    col1.plotly_chart(px.pie(data_frame= df , names= 'churn',hole=0.5,title='The percentage of Churn'))
    categ_col = df.select_dtypes(include='object').drop('churn',axis = 1).columns
    option = col2.selectbox('select the column you want to see the count of the churn', categ_col)
    col2.plotly_chart(px.histogram(data_frame= df, x = option , color='churn',barmode='group').update_xaxes(categoryorder = 'max descending')) 

    col1, col2= st.columns(2)
    contract_df = df.groupby('contract')['totalcharges'].mean().round(2).reset_index().sort_index(ascending=False)
    col1.plotly_chart(px.bar(data_frame=contract_df,x='contract',y = 'totalcharges',title = 'the total charges for every contract type'))
    payment_df = df.groupby('paymentmethod')['totalcharges'].mean().round(2).reset_index().sort_index(ascending=False)
    col2.plotly_chart(px.bar(data_frame=payment_df,x='paymentmethod',y = 'totalcharges',title = 'the total charges for every payment method').update_xaxes(categoryorder = 'max descending'))

    
    num_col1 = df.select_dtypes(include=["int64", "float64"]).columns
    st.plotly_chart(px.scatter_matrix(data_frame=df[num_col1],height=600,title = 'the correlation between all numerical columns  for every contract type'))

    col1, col2= st.columns(2)
    col1.plotly_chart(px.scatter(data_frame=df,x='tenure',y = 'subscription_count',color='churn',title = 'the count of months with the count of subscribtions showing churn'))
    billing_df = df.groupby('paperlessbilling')['subscription_count'].count().round(2).reset_index().sort_index(ascending=False)
    col2.plotly_chart(px.bar(data_frame=billing_df,x='paperlessbilling',y = 'subscription_count',title = 'the count of subscribtions for paperlessbilling'))

    ##col3.metric("👥 Total Customers", f"{total_customers:,}")

    
elif page == 'Service Subscribtions':
    html_title = """<h1 style="color:white;text-align:center;"> The Churn For Every Subsciption Type  </h1>"""
    st.markdown(html_title, unsafe_allow_html=True)
    sub_col = df[['onlinesecurity', 'onlinebackup','deviceprotection','techsupport','streamingtv','streamingmovies']].columns
    s_op = st.selectbox('select the service you want to see the count of churn',sub_col)
    service_churn = (df.groupby([s_op, 'churn']).size().reset_index(name='count'))
    st.plotly_chart(px.bar(data_frame = service_churn ,  x= s_op ,y='count',color='churn',barmode='group',text_auto=True,title=f'Churn count by {s_op}'))


elif page == 'Contracts & Billing':
    html_title = """<h1 style="color:white;text-align:center;"> Contracts & Billing Insights </h1>"""
    st.markdown(html_title, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ---- churn by contract ----
    contract_churn = (
        df.groupby(['contract', 'churn'])
          .size()
          .reset_index(name='count')
    )
    contract_churn['total'] = contract_churn.groupby('contract')['count'].transform('sum')
    contract_churn['churn_rate'] = contract_churn['count'] / contract_churn['total']

    fig_contract = px.bar(
        contract_churn,
        x='contract',
        y='churn_rate',
        color='churn',
        barmode='group',
        text_auto=".1%",
        title='Churn rate by contract type'
    )
    fig_contract.update_yaxes(tickformat=".0%")
    fig_contract.update_xaxes(categoryorder='category ascending')
    col1.plotly_chart(fig_contract, use_container_width=True)

    payment_churn = (
        df.groupby(['paymentmethod', 'churn'])
          .size()
          .reset_index(name='count')
    )
    payment_churn['total'] = payment_churn.groupby('paymentmethod')['count'].transform('sum')
    payment_churn['churn_rate'] = payment_churn['count'] / payment_churn['total']

    fig_payment = px.bar(
        payment_churn,
        x='paymentmethod',
        y='churn_rate',
        color='churn',
        barmode='group',
        text_auto=".1%",
        title='Churn rate by payment method'
    )
    fig_payment.update_yaxes(tickformat=".0%")
    fig_payment.update_xaxes(categoryorder='category ascending')
    col2.plotly_chart(fig_payment, use_container_width=True)   







