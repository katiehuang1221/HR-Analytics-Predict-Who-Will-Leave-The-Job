"""
script for streamlit web app

"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score, log_loss


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()
takeaway = st.beta_container()


with header:
    st.title('Welcome to Smart HR!')
    st.write('Tell me about the candidates and I will let you know if they are actually looking for a new job!\
    This will save you considerable amount of time reaching out or interviewing the candidates!')

    df_temp = pd.read_csv('data/df_test_altair_viz.csv')
    c = alt.Chart(df_temp.iloc[:200]).mark_circle().encode(
    x=alt.X('Candidate',axis=alt.Axis(labels=False)), y='Training Hours', size='Probability', color='Current company',
    tooltip=['Probability', 'Candidate ID', 'Gender', 'Major', 'Education',
             'Training Hours', 'Experience (yr)']).properties(width=700, height=250)
    st.write(c) 


st.sidebar.header('Info of Candidates')
st.sidebar.markdown('### Select the feature(s) you want to use for filtering!')

specify = st.sidebar.selectbox('Features',
('All','Specify'))

if specify == 'Specify':

    # Set default values
    gender_filter=['Female','Male','Other','unknown']
    major_filter=['STEM','Humanities','Business Degree','Arts','No major','Other','unknown']
    education_filter=['Primary School','High School','Graduate','Masters','Phd','unknown']
    city_filter=[0.5,1]
    company_type_filter=['Pvt Ltd','unknown','Funded Startup','Public Sector','Early Stage Startup','NGO','Other']
    company_size_filter=['50-99', '<10', '10000+', '5000-9999', '1000-4999', '10/49','100-500', '500-999']
    enrolled_university_filter=['no_enrollment', 'Full time course', 'unknown', 'Part time course']
    training_hours_filter=[0,500]
    relevant_experience_filter=['Has relevent experience', 'No relevent experience']
    experience_filter =[0,25]


    st.sidebar.markdown('### Current status')

    if st.sidebar.checkbox('Gender',key='gender'):
        option = st.sidebar.selectbox('Select gender',
        ('All','Specify'))

        if option == 'Specify':
            Female = st.sidebar.checkbox('Female',key='female')
            Male = st.sidebar.checkbox('Male',key='male')
            Other = st.sidebar.checkbox('Other',key='gender_other')
            secret = st.sidebar.checkbox('Secret',value=True,key='gender_secret')

            gender_filter=[]
            if Female:
                gender_filter.append('Female')
            if Male:
                gender_filter.append('Male')
            if Other:
                gender_filter.append('Other')
            if secret:
                gender_filter.append('unknown')

    

    st.sidebar.header('')    
    if st.sidebar.checkbox('Major',key='major'):
        option = st.sidebar.selectbox('Select major',
        ('All','Specify'))

        if option == 'Specify':
            STEM = st.sidebar.checkbox('STEM',key='STEM')
            Humanities = st.sidebar.checkbox('Humanities',key='Humanities')
            Business = st.sidebar.checkbox('Business',key='Business')
            Arts = st.sidebar.checkbox('Arts',key='Arts')
            Other = st.sidebar.checkbox('Other',key='major_other')
            No = st.sidebar.checkbox('No major',key='major_none')
            secret = st.sidebar.checkbox('Secret',value=True,key='major_secret')

            major_filter=[]
            if STEM:
                major_filter.append('STEM')
            if Humanities:
                major_filter.append('Humanities')
            if Business:
                major_filter.append('Business Degree')
            if Arts:
                major_filter.append('Arts')
            if No:
                major_filter.append('No major')
            if Other:
                major_filter.append('Other')
            if secret:
                major_filter.append('unknown')


    st.sidebar.header('')
    if st.sidebar.checkbox('Education Level',key='education_level'):
        option = st.sidebar.selectbox('Select education level',
        ('All','Specify'))

        if option == 'Specify':
            PM = st.sidebar.checkbox('Primary School',key='PM')
            HS = st.sidebar.checkbox('High School',key='HS')
            Graduate = st.sidebar.checkbox('Graduate',key='Graduate')
            Master = st.sidebar.checkbox('Master',key='Master')
            PhD = st.sidebar.checkbox('PhD',key='PhD')
            secret = st.sidebar.checkbox('Secret',value=True,key='education_secret')

            education_filter=[]
            if PM:
                education_filter.append('Primary School')
            if HS:
                education_filter.append('High School')
            if Graduate:
                education_filter.append('Graduate')
            if Master:
                education_filter.append('Masters')
            if PhD:
                education_filter.append('Phd')
            if secret:
                education_filter.append('unknown')

    st.sidebar.header('')
    if st.sidebar.checkbox('Current City',key='city_development_index'):
        x = st.sidebar.slider('Select a range of city development index',
        0.5,1.0,(0.6, 0.8))  # 👈 this is a widget
        city_filter[0],city_filter[1] = x
        
    st.sidebar.header('')
    if st.sidebar.checkbox('Current company type',key='company_type'):
        option = st.sidebar.selectbox('Select current company type',
        ('All','Specify'))

        if option == 'Specify':
            ESS = st.sidebar.checkbox('Early Stage Startup',key='ESS')
            FS = st.sidebar.checkbox('Funded Startup',key='FS')
            NGO = st.sidebar.checkbox('NGO',key='NGO')
            PS = st.sidebar.checkbox('Public Sector',key='PS')
            Pvt = st.sidebar.checkbox('Pvt Ltd',key='Pvt')
            Other = st.sidebar.checkbox('Other',key='company_type_Other')
            secret = st.sidebar.checkbox('Secret',value=True,key='company_type_secret')

            company_type_filter=[]
            if ESS:
                company_type_filter.append('Early Stage Startup')
            if FS:
                company_type_filter.append('Funded Startup')
            if NGO:
                company_type_filter.append('NGO')
            if PS:
                company_type_filter.append('Public Sector')
            if Pvt:
                company_type_filter.append('Pvt Ltd')
            if Other:
                company_type_filter.append('Other')    
            if secret:
                company_type_filter.append('unknown')



    st.sidebar.header('')
    if st.sidebar.checkbox('Current company size',key='company_size'):
        option = st.sidebar.selectbox('Select current company size',
        ('All','Specify'))

        if option == 'Specify':
            a = st.sidebar.checkbox('<10',key='a')
            b = st.sidebar.checkbox('10-49',key='b')
            c = st.sidebar.checkbox('50-99',key='c')
            d = st.sidebar.checkbox('100-499',key='d')
            e = st.sidebar.checkbox('500-999',key='e')
            f = st.sidebar.checkbox('1000-4999',key='f')
            g = st.sidebar.checkbox('5000-9999',key='g')
            h = st.sidebar.checkbox('10000+',value=True,key='h')
            secret = st.sidebar.checkbox('Secret',value=True,key='company_size_secret')

            company_size_filter=[]
            if a:
                company_size_filter.append('<10')
            if b:
                company_size_filter.append('10/49')
            if c:
                company_size_filter.append('50-99')
            if d:
                company_size_filter.append('100-500')
            if e:
                company_size_filter.append('500-999')
            if f:
                company_size_filter.append('1000-4999')
            if g:
                company_size_filter.append('5000-9999')
            if h:
                company_size_filter.append('10000+')      
            if secret:
                company_size_filter.append('unknown')



    st.sidebar.markdown('### Training')

    if st.sidebar.checkbox('Enrolled Course',key='enrolled_university'):
        option = st.sidebar.selectbox('Select enrolled course',
        ('All','Specify'))

        if option == 'Specify':
            full_time = st.sidebar.checkbox('Full time',key='full_time')
            part_time = st.sidebar.checkbox('Part time',key='part_time')
            no_enrollment = st.sidebar.checkbox('No enrollment',key='no_enrollment')
            unknown = st.sidebar.checkbox('Unknown',value=True,key='unknown_enrollment')

            enrolled_university_filter=[]
            if full_time:
                enrolled_university_filter.append('Full time course')
            if part_time:
                enrolled_university_filter.append('Part time course')
            if no_enrollment:
                enrolled_university_filter.append('no_enrollment')
            if unknown:
                enrolled_university_filter.append('unknown')



    st.sidebar.header('')
    if st.sidebar.checkbox('Training Hours',key='training_hours'):
        x = st.sidebar.slider('Select a range of training hours',
        0,500,(24, 300))  # 👈 this is a widget
        training_hours_filter[0],training_hours_filter[1] = x
        # st.write(training_hours_filter_min)

    st.sidebar.header('')
    if st.sidebar.checkbox('Relevant Experience',key='relevant_experience'):
        option = st.sidebar.selectbox('Select relevant experience',
        ('Yes','No','All'))

        relevant_experience_filter=[]
        if option == 'All':
            relevant_experience_filter=['Has relevent experience', 'No relevent experience']
        if option == 'Yes':
            relevant_experience_filter=['Has relevent experience']
        if option == 'No':
            relevant_experience_filter=['No relevent experience']



    st.sidebar.header('')
    if st.sidebar.checkbox('Experience (years)',key='experience'):
        x = st.sidebar.slider('Select a range of experience (in years)',
        0,25,(5, 10))  # 👈 this is a widget
        experience_filter[0], experience_filter[1] = x
        # st.write(training_hours_filter_min)



with dataset:
    st.header('Candidate Statistics')
   

    df_train = pd.read_pickle('data/df_train.csv')
    df_test = pd.read_pickle('data/df_test.csv')

    X_train = pd.read_pickle('data/X_adasyn')
    y_train = pd.read_pickle('data/y_adasyn')
    X_test = pd.read_pickle('data/X_test_processed')
    y_test = df_test['target']


    # st.write('Total number of candidates',df_train_viz.shape[0])

    # if st.checkbox('Show dataframe'):
    #     st.write(df_train_viz.head())

    if specify == 'All':
        df_train_filtered = df_train

    else:
        df_train_filtered = df_train[
            (df_train.gender.isin(gender_filter)) & 
            (df_train.major_discipline.isin(major_filter)) &
            (df_train.education_level.isin(education_filter)) &
            (df_train.city_development_index > city_filter[0]) & (df_train.city_development_index < city_filter[1]) &
            (df_train.company_type.isin(company_type_filter)) &
            (df_train.company_size.isin(company_size_filter)) &
            (df_train.enrolled_university.isin(enrolled_university_filter)) &
            (df_train.training_hours > training_hours_filter[0]) & (df_train.training_hours < training_hours_filter[1]) &
            (df_train.relevent_experience.isin(relevant_experience_filter)) &
            (df_train.experience > experience_filter[0]) & (df_train.experience < experience_filter[1])
        ]

    # Save the df
    # df_train_filtered.to_csv('../dump/df_train_customized')

    df_train_filtered_display=df_train_filtered[['enrollee_id','gender', 'major_discipline',  'education_level',
        'city_development_index', 'company_type', 'company_size',
        'enrolled_university', 'training_hours', 'relevent_experience', 'experience','target']]


    
    df_train_viz = df_train_filtered_display.copy()
    df_train_viz['target'] = df_train_viz['target'].apply(lambda x: 'Yes' if x==1 else 'No')
    df_train_viz['enrolled_university'] = \
        df_train_viz['enrolled_university'].map({'Full time course':'Full time',\
                                                 'Part time course':'Part time',\
                                                 'no_enrollment':'No enrollment',\
                                                 'unknown':'unknown'
                                                })
    df_train_viz['major_discipline'] = df_train_viz['major_discipline'].\
                                        replace('Business Degree','Business')
    df_train_viz['relevent_experience'] = df_train_viz['relevent_experience'].\
                                        map({'Has relevent experience':'Yes',\
                                                                              'No relevent experience':'No'})
    target_count = pd.DataFrame(df_train_viz['target'].value_counts())
    yes_count = target_count.target[1]
    no_count = target_count.target[0]

    df_train_viz.columns=['Enrollee ID', 'Gender', 'Major',  'Education Level',
        'Current City', 'Current Company Type', 'Current Company Size',
        'Enrolled Course', 'Training Hours', 'Relevant Experience', 'Experience (years)','Target']


    # st.markdown('### Here are the filtered candidates:')
    st.write('Qualified candidate count:',df_train_viz.shape[0])
    if st.checkbox('Show dataframe',key='filtered'):
            st.write(df_train_viz)        

    

    source = pd.DataFrame({
    'Looking for a new job?': ['Yes', 'No'],
    'Count': [yes_count, no_count]
    })
    c = alt.Chart(source).mark_bar().encode(
    x='Looking for a new job?',
    y='Count', tooltip=['Count']
    ).configure_axisX(labelAngle=0).configure_axis(labelFontSize=20,titleFontSize=20)\
    .properties(width=600, height=400)
    st.altair_chart(c)




with model_training:
    st.header('Model')

    # Training data
    df_test = pd.read_pickle('data/df_test.csv')
    X_train = pd.read_pickle('data/X_adasyn')
    y_train = pd.read_pickle('data/y_adasyn')
    X_test = pd.read_pickle('data/X_test_processed')
    y_test = df_test['target']

    lm = LogisticRegression(solver='newton-cg',  # For comparison, use the same solver as statsmodels default
                          C=100000)  # No regularization
    lm.fit(X_train, y_train)
    y_predict = lm.predict(X_test)
    accuracy = str(round(accuracy_score(y_test,y_predict),3)*100) + '%'
    AUC = str(round(roc_auc_score(y_test,lm.predict_proba(X_test)[:, 1]),3)*100) + '%'
    recall = str(round(recall_score(y_test,y_predict),3)*100) + '%'
    precision = str(round(precision_score(y_test,y_predict),3)*100) + '%'
    
    st.subheader('Overall Performance')
    # st.write('Accuracy:', accuracy)
    col1,col2 = st.beta_columns([1,2])
    col1.markdown('### ROC AUC \n (on unseen test data)')
    col2.title(AUC)
    
    
    

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


    # ROC plot with threshold annotate
    y_scores = lm.predict_proba(X_test)
    fpr, tpr, threshold_array = roc_curve(y_test, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)

    threshold_value = st.slider('Select threshold value (%):', min_value=0, max_value=100, value=50, step=1)
    threshold = threshold_value/100
    i = find_nearest(threshold_array, threshold)

    fig, ax = plt.subplots()
    plt.stackplot(fpr, tpr,alpha=0.3)
    plt.plot(fpr, tpr, 'cadetblue', linewidth=3.5, label = 'Logistic Regression')
    plt.plot([0, 1], [0, 1],'k--',linewidth=1, label = 'Random Guess')
    plt.scatter(fpr[i], tpr[i],linewidth=3,color='orange',zorder=10)
    ax.annotate('Threshold= %.2f' % (threshold_array[i]),[fpr[i]+0.03,tpr[i]-0.02])

    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.title('Model Performance (ROC)',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12);

    st.pyplot(fig)


    # Candidate probability vs head count
    df = pd.read_csv('data/candidate_proba.csv')

    st.markdown('### Customize Metric')
    option = st.selectbox('Select the metric you wish to use:',
            ('Recall','Threshold','Number of candidates'),index=1)

    if option == 'Threshold':

        probability = threshold*100
        probability = st.slider('Select probability threshold (%):',min_value=0, max_value=100, value=50, step=1)

        i = find_nearest(df['Probability (%)'], probability)

        y_predict = (lm.predict_proba(X_test)[:, 1] >= probability/100)
        recall = str(round(recall_score(y_test,y_predict)*100,2)) + '%'
        precision = str(round(precision_score(y_test,y_predict)*100,2)) + '%'

        col1, col2, col3, col4 = st.beta_columns(4)
        col1.subheader('Recall')
        col1.write(recall)
        col2.subheader('Precision')
        col2.write(precision)
        col3.subheader('Threshold')
        col3.write(str(probability) + '%')
        col4.subheader('Number of Candidates')
        col4.write(i)
        
        
        fig, ax = plt.subplots()
        plt.scatter(x=df.index, y=(df['Probability (%)']))

        plt.axvline(x=i,linestyle='--',color='g')
        plt.axhline(y=probability,color='g')

        plt.xlim([-20,2200])
        plt.ylim([0,100])
        plt.ylabel('Probability of Yes (%)',fontsize=14)
        plt.xlabel('Candidate Head Count',fontsize=14)
        plt.title('Probability vs Number of Candidates',fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12);

        st.pyplot(fig)

    if option == 'Number of candidates':
    # Specify number of candidates
        st.markdown('### Specify number of candidates you plan to interview:')
        input_number_of_candidate = st.number_input('How many candidates?', value=200, step=10)
        prob = df.iloc[input_number_of_candidate]['Probability (%)']
        threshold = prob/100

        y_predict = (lm.predict_proba(X_test)[:, 1] >= threshold)
        recall = str(round(recall_score(y_test,y_predict)*100,2)) + '%'
        precision = str(round(precision_score(y_test,y_predict)*100,2)) + '%'

        col1, col2, col3, col4 = st.beta_columns(4)
        col1.subheader('Recall')
        col1.write(recall)
        col2.subheader('Precision')
        col2.write(precision)
        col3.subheader('Threshold')
        col3.write(str(prob) + '%')
        col4.subheader('Number of Candidates')
        col4.write(input_number_of_candidate)

        fig2, ax2 = plt.subplots()
        plt.scatter(x=df.index, y=(df['Probability (%)']))

        plt.axvline(x=input_number_of_candidate,color='r')
        plt.axhline(y=prob,linestyle='--',color='r')

        plt.xlim([-20,2200])
        plt.ylim([0,100])
        plt.ylabel('Probability of Yes (%)',fontsize=14)
        plt.xlabel('Candidate Head Count',fontsize=14)
        plt.title('Probability vs Number of Candidate',fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12);

        
        st.pyplot(fig2)


    if option == 'Recall':

        
        recall = st.slider('Select desired recall (%):',min_value=0, max_value=100, value=50, step=1)

        threshold_list = [x/100 for x in range(0,100)]
        recall_list=[]
        for value in threshold_list:
            threshold = value
            y_predict = (lm.predict_proba(X_test)[:, 1] >= threshold)
            recall_list.append(recall_score(y_test,y_predict))

        i = find_nearest(recall_list, recall/100)
        probability = threshold_list[i]*100
        y_predict = (lm.predict_proba(X_test)[:, 1] >= threshold_list[i])
        recall = str(round(recall_score(y_test,y_predict)*100,2)) + '%'
        precision = str(round(precision_score(y_test,y_predict)*100,2)) + '%'

        i = find_nearest(df['Probability (%)'], probability)

        col1, col2, col3, col4 = st.beta_columns(4)
        col1.subheader('Recall')
        col1.write(recall)
        col2.subheader('Precision')
        col2.write(precision)
        col3.subheader('Threshold')
        col3.write(str(round(probability,2)) + '%')
        col4.subheader('Number of Candidates')
        col4.write(i)
        
        
        fig, ax = plt.subplots()
        plt.scatter(x=df.index, y=(df['Probability (%)']))

        plt.axvline(x=i,linestyle='--',color='orange')
        plt.axhline(y=probability,linestyle='--',color='orange')

        plt.xlim([-20,2200])
        plt.ylim([0,100])
        plt.ylabel('Probability of Yes (%)',fontsize=14)
        plt.xlabel('Candidate Head Count',fontsize=14)
        plt.title('Probability vs Number of Candidates',fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12);

        st.pyplot(fig)
    



with takeaway:
    st.header('Final List of Candidates')
    

    df['Probability (%)'] =  df['Probability (%)'].apply(lambda x: round(x,2))
    df.rename(columns={'Enrollee_id':'Enrollee ID'},inplace=True)
    # st.write(df[['Enrollee ID','Probability (%)']])
    # st.write(df_test)


# Create filter df_train
    if specify == 'All':
        df_test_filtered = df_test

    else:
        df_test_filtered = df_test[
            (df_test.gender.isin(gender_filter)) & 
            (df_test.major_discipline.isin(major_filter)) &
            (df_test.education_level.isin(education_filter)) &
            (df_test.city_development_index > city_filter[0]) & (df_test.city_development_index < city_filter[1]) &
            (df_test.company_type.isin(company_type_filter)) &
            (df_test.company_size.isin(company_size_filter)) &
            (df_test.enrolled_university.isin(enrolled_university_filter)) &
            (df_test.training_hours > training_hours_filter[0]) & (df_test.training_hours < training_hours_filter[1]) &
            (df_test.relevent_experience.isin(relevant_experience_filter)) &
            (df_test.experience > experience_filter[0]) & (df_test.experience < experience_filter[1])
        ]

    # Save the df
    # df_train_filtered.to_csv('../dump/df_train_customized')

    df_test_filtered_display=df_test_filtered[['enrollee_id','gender', 'major_discipline',  'education_level',
        'city_development_index', 'company_type', 'company_size',
        'enrolled_university', 'training_hours', 'relevent_experience', 'experience']]


    df_test_filtered_display.columns=['Enrollee ID', 'Gender', 'Major',  'Education Level',
        'Current City', 'Current Company Type', 'Current Company Size',
        'Enrolled Course', 'Training Hours', 'Relevant Experience', 'Experience (years)']



    final = df_test_filtered_display.merge(df,on='Enrollee ID',how='inner')

    # Last clean up for display
    final['Relevant Experience'] = final['Relevant Experience'].replace('Has relevent experience','Yes').replace('No relevent experience','No')


    st.write('Filtered number of candidates:',final.shape[0])
    st.write(final[['Enrollee ID', 'Probability (%)','Gender', 'Major',  'Education Level',
        'Current City', 'Current Company Type', 'Current Company Size',
        'Enrolled Course', 'Training Hours', 'Relevant Experience', 'Experience (years)']].\
            sort_values('Probability (%)',ascending=False).reset_index(drop=True))

