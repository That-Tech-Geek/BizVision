import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.linear_model import LinearRegression
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Set the title of the app
st.title("Financial and Optimization Toolkit")

# Sidebar options
st.sidebar.title("Choose a Function")
option = st.sidebar.selectbox(
    "Select Functionality",
    ("Regression", "NPV Calculation", "Linear Programming", "Integer Programming", "Nonlinear Programming")
)

# Regression Function
if option == "Regression":
    st.header("Regression Analysis")
    st.write("Upload your dataset for regression analysis (CSV format).")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        X_cols = st.multiselect("Select independent variable(s):", data.columns)
        Y_col = st.selectbox("Select dependent variable:", data.columns)

        if X_cols and Y_col:
            X = data[X_cols].values.reshape(-1, len(X_cols))
            y = data[Y_col].values.reshape(-1, 1)

            reg = LinearRegression()
            reg.fit(X, y)

            st.write("Intercept:", reg.intercept_)
            st.write("Coefficients:", reg.coef_)

            y_pred = reg.predict(X)
            st.write("Predicted values:", y_pred)

# NPV Calculation Function
elif option == "NPV Calculation":
    st.header("Net Present Value (NPV) Calculation")
    
    cash_flows = st.text_input("Enter cash flows separated by commas (e.g., 1000, 1500, -500, ...):")
    rate = st.number_input("Enter discount rate (as a percentage, e.g., 5 for 5%):", value=0.0, step=0.1)
    
    if cash_flows:
        cash_flows = [float(x) for x in cash_flows.split(",")]
        rate = rate / 100  # Convert percentage to decimal
        npv = np.sum([cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows)])
        st.write("Net Present Value (NPV):", npv)

# Linear Programming Function
elif option == "Linear Programming":
    st.header("Linear Programming Problem")
    
    st.write("Enter coefficients for the objective function (e.g., 1, -2, 3):")
    obj_func = st.text_input("Objective function coefficients (comma-separated):")
    
    st.write("Enter coefficients and bounds for inequalities (e.g., '1, -1, 0 <= 10'):") 
    num_constraints = st.number_input("Number of constraints:", min_value=1, value=1)
    constraints = []

    for i in range(num_constraints):
        constraint = st.text_input(f"Constraint {i+1}:", key=f"constraint_{i}")
        if constraint:
            constraints.append(constraint)
    
    if obj_func and constraints:
        obj_coeffs = np.array([float(x) for x in obj_func.split(",")])
        lhs_ineq = []
        rhs_ineq = []

        for constraint in constraints:
            parts = constraint.split("<=")
            lhs_ineq.append([float(x) for x in parts[0].split(",")])
            rhs_ineq.append(float(parts[1]))

        bounds = [(0, float('inf'))] * len(obj_coeffs)
        result = linprog(c=obj_coeffs, A_ub=np.array(lhs_ineq), b_ub=np.array(rhs_ineq), bounds=bounds, method='highs')
        
        st.write("Optimal solution:", result.x)
        st.write("Optimal value:", result.fun)

# Integer Programming Function
elif option == "Integer Programming":
    st.header("Integer Programming Problem")
    
    st.write("Enter coefficients for the objective function (e.g., 1, -2, 3):")
    obj_func = st.text_input("Objective function coefficients (comma-separated):")

    st.write("Enter coefficients and bounds for inequalities (e.g., '1, -1, 0 <= 10'):") 
    num_constraints = st.number_input("Number of constraints:", min_value=1, value=1)
    constraints = []

    for i in range(num_constraints):
        constraint = st.text_input(f"Constraint {i+1}:", key=f"constraint_{i}")
        if constraint:
            constraints.append(constraint)
    
    if obj_func and constraints:
        obj_coeffs = [float(x) for x in obj_func.split(",")]
        prob = LpProblem("Integer Programming Problem", LpMaximize)
        
        vars = [LpVariable(f"x{i}", lowBound=0, cat="Integer") for i in range(len(obj_coeffs))]
        prob += lpSum([obj_coeffs[i] * vars[i] for i in range(len(obj_coeffs))])
        
        for constraint in constraints:
            parts = constraint.split("<=")
            lhs = [float(x) for x in parts[0].split(",")]
            rhs = float(parts[1])
            prob += lpSum([lhs[i] * vars[i] for i in range(len(lhs))]) <= rhs
        
        prob.solve()

        solution = [var.value() for var in vars]
        st.write("Optimal solution:", solution)
        st.write("Optimal value:", prob.objective.value())

# Nonlinear Programming Function
elif option == "Nonlinear Programming":
    st.header("Nonlinear Programming Problem")

    st.write("Enter your objective function as a Python lambda function of x (e.g., 'lambda x: x[0]**2 + x[1]**2'):")
    obj_func = st.text_input("Objective function:")
    
    st.write("Enter your initial guess for the solution (comma-separated):")
    x0 = st.text_input("Initial guess (comma-separated):")
    
    if obj_func and x0:
        objective = eval(obj_func)
        x0 = np.array([float(x) for x in x0.split(",")])
        result = minimize(objective, x0)
        st.write("Optimal solution:", result.x)
        st.write("Optimal value:", result.fun)
