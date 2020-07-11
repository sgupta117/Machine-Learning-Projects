import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

decisionTreeClassifier = pickle.load(open('DecisionTreeClassifier.pkl', 'rb'))
kNeighborsClassifier = pickle.load(open('KNeighborsClassifier.pkl', 'rb'))
randomForestClassifier = pickle.load(open('RandomForestClassifier.pkl', 'rb'))
adaBoostClassifier = pickle.load(open('AdaBoostClassifier.pkl', 'rb'))

married_status = {'Married' : 1, 'UnMarried' : 0}
property_area = {'Rural' : 0 , 'SemiUrban' : 1 , 'Urban' : 2}

def app():

	html_temp = """
	    <div style="background-color:tomato;padding:10px;font-family:calibri">
	    <h2 style="color:white;text-align:center;">Loan Eligibility Prediction🙂 </h2>
	    </div>
	    """

	st.markdown(html_temp, unsafe_allow_html=True)

	activities=["Predict the loan status","About"]
	choice = st.sidebar.selectbox("Select Your Activity",activities)

	if choice=="Predict the loan status":
		st.subheader("Find out the loan status of a person based upon the below inputs.:")
		st.subheader("Please Enter the details below :")

		credit_History = st.selectbox("Select the credit History 1 or 0 :",[1,0])
		dependents = st.text_input("Enter the number of dependents :")
		married = st.radio("Select the marital Status :",('Married', 'UnMarried'))
		applicantIncome = st.text_input("Enter the applicant income like 2500, 6000,3500... :")
		coapplicantIncome = st.text_input("Enter the coaplicant income like 1500,2000,1800... :")
		property_Area = st.selectbox("Select the property Area :",['Urban','Rural','SemiUrban'])
		loanAmount = st.text_input("Enter the loan amount like 100,250,300... :")

		#haCk
		credit_History = 1 if credit_History == 0 else 0
		married = married_status[married]
		property_Area = property_area[property_Area]

		Analyzer_choice = st.selectbox("Select the Model",  ["Decision Tree" , "KNN model" , "Random Forest" , "Adaboost"])


		if st.button("Analyze"):
			if Analyzer_choice == "Decision Tree":
				st.success("Checking the status with Decision Tree Classifier...")
				def predict_decision_tree(credit_History,dependents,married,applicantIncome,property_Area,coapplicantIncome,loanAmount):
					return decisionTreeClassifier.predict([[credit_History,dependents,married,applicantIncome,property_Area,coapplicantIncome,loanAmount]])

				result = predict_decision_tree(credit_History,dependents,married,applicantIncome,property_Area,coapplicantIncome,loanAmount)

				if result == 1:
					st.markdown('Customer is **_Eligible_ for personal loan!!!**')
				else:
					st.markdown('Customer is **_Not Eligible_ for personal loan**.')

			elif Analyzer_choice == "KNN model":

				st.success("Checking the status with KNN model Classifier...")

				def predict_knn_classifier(credit_History, dependents, married, applicantIncome, property_Area, coapplicantIncome, loanAmount):
					return kNeighborsClassifier.predict([[credit_History, dependents, married, applicantIncome,property_Area, coapplicantIncome, loanAmount]])
				result = predict_knn_classifier(credit_History, dependents, married, applicantIncome, property_Area,coapplicantIncome, loanAmount)

				if result == 1:
					st.markdown('Customer is **_Eligible_ for personal loan!!!**')
				else:
					st.markdown('Customer is **_Not Eligible_ for personal loan**.')


			elif Analyzer_choice == "Random Forest":

				st.success("Checking the status with Random Forest Classifier...")

				def predict_random_forest(credit_History, dependents, married, applicantIncome, property_Area, coapplicantIncome, loanAmount):
					return randomForestClassifier.predict([[credit_History, dependents, married, applicantIncome,property_Area, coapplicantIncome, loanAmount]])
				result = predict_random_forest(credit_History, dependents, married, applicantIncome, property_Area,coapplicantIncome, loanAmount)

				if result == 1:
					st.markdown('Customer is **_Eligible_ for personal loan!!!**')
				else:
					st.markdown('Customer is **_Not Eligible_ for personal loan**.')

			elif Analyzer_choice == "Adaboost":

				st.success("Checking the status with Adaboost Classifier...")

				def predict_adaboost_classifier(credit_History, dependents, married, applicantIncome, property_Area, coapplicantIncome, loanAmount):
					return randomForestClassifier.predict([[credit_History, dependents, married, applicantIncome,property_Area, coapplicantIncome, loanAmount]])
				result = predict_adaboost_classifier(credit_History, dependents, married, applicantIncome, property_Area,coapplicantIncome, loanAmount)

				if result == 1:
					st.markdown('Customer is **_Eligible_ for personal loan!!!**')
				else:
					st.markdown('Customer is **_Not Eligible_ for personal loan**.')


			else:
				pass
	else:
		if choice == 'About':
			st.subheader("Introduction:")
			st.write("In this web app , I will try to show you how different models can improve just by doing simple process on the data.")

			st.write("We are going to work on binary classification problem, where we got some information about sample of peoples , and we need to predict whether we should give some one a loan or not depending on his information . we actually have a few sample size (614 rows), so we will go with machine learning techniques to solve our problem .")

			st.subheader("What you will learn in this kernel ?")

			st.write("1) Basics of visualizing the data.")
			st.write("2) How to compare between feature importance (at less in this data).")
			st.write("3) Feature selection.")
			st.write("4) Feature engineer.")
			st.write("5) Handling missing data.")
			st.write("6) How to deal with categorical and numerical data.")

			st.subheader("What we will use ?")

			st.write("1) Some important libraries like sklearn, matplotlib, numpy, pandas, seaborn, scipy.")
			st.write("2) Fill the values using backward 'bfill' method for numerical columns , and most frequent value for categorical columns (simple techniques)")
			st.write("3) 3 different models to train your data, so we can compare between them : ")

			st.subheader("a) KNeighborsClassifier")
			st.subheader("b) Adaboost")
			st.subheader("c) DecisionTreeClassifier")
			st.subheader("d) RandomForestClassifier")

			st.write("")
			st.write("")
			st.markdown("**Thank You...** 👍")
			st.markdown("Streamlit reference : https://docs.streamlit.io/en/latest/index.html")
			st.markdown("My LinkedIn : https://www.linkedin.com/in/shubham-gupta-941457a4/")
	footer = """
			<style>
		.footer {
		  position: fixed;
		  left: 0;
		  bottom: 0;
		  width: 100%;
		  background-color: tomato;
		  color: white;
		  text-align: center;
		}
		</style>

		<div class="footer">
		  <p>Created By : Shubham Gupta! 😎...</p>
		</div>

			"""

	#st.subheader('Created By : Shubham Gupta! 😉...')
	st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
	app()