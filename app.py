import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Function to compute the exact solution
def f(x, initial_condition):
    return initial_condition*np.exp(x)

# Function to compute the approximate solution with a custom initial condition
def _f(k, h, initial_condition):
    if k == 0:
        return initial_condition
    else:
        return (1+h)*_f(k-1, h, initial_condition)
    
# Function to compute the central difference approximation
def central_diff_f(k, h, initial_condition):
    if k == 0:
        return initial_condition
    elif k == 1 or k == steps-1:
        return initial_condition*np.exp(k*h)  # use exact values at the boundary
    else:
        x = k*h
        return f(x-h, initial_condition) + h * (f(x+h, initial_condition) - f(x-h, initial_condition)) / (2*h)

# Function to compute the MSE
def MSE(_y, y):
    return np.mean((_y-y)**2)

# Function to compute the pointwise squared error
def pointwise_error(_y, y):
    return (_y - y)**2

# Streamlit sidebar widgets
st.sidebar.title("Input Parameters")
initial_condition = st.sidebar.number_input("Initial Condition:", min_value=0.1, value=1.0, step=0.1)
h = st.sidebar.slider("Step size (h)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
steps = st.sidebar.slider("Number of steps", min_value=5, max_value=30, value=10, step=1)
st.sidebar.title("A BytePotion App")
st.sidebar.image("bytepotion.png", width=200)  # Replace with your image URL or file path
st.sidebar.write("This app provides insight into finite differences method to solve ODE's.")
st.sidebar.write("https://bytepotion.com")
st.sidebar.title("Author")
st.sidebar.image("roman2.png", width=200)  # Replace with your image URL or file path
st.sidebar.write("Roman Paolucci")
st.sidebar.write("MSOR Graduate Student @ Columbia University")
st.sidebar.write("roman.paolucci@columbia.edu")

# Main title
st.title("Finite Differences Method Visualization")

st.subheader("First-Order Forward Difference")

with st.expander("Explanation"):
    st.write("The ordinary differential equation is:")
    st.latex(r"f' = f")
    st.write("With the initial condition:")
    st.latex(r"f(0) = 1")
    st.write("This leads to the exact solution:")
    st.latex(r"f(x) = e^x")

    st.write("However, using the first-order forward difference approximation:")
    st.latex(r"f'(x) \approx \frac{f(x+h) - f(x)}{h}")

    st.write("For small values of \( h \), this implies:")
    st.latex(r"\frac{f(x+h) - f(x)}{h} = f(x)")
    st.latex(r"\implies f(x+h) = (1+h) f(x)")

    st.write("Starting from the initial condition \( f(0) = 1 \), we can solve this equation incrementally:")
    st.latex(r"f(x+h) = (1+h) f(x)")
    st.latex(r"f(0) = 1")
    st.latex(r"f(h) = (1+h) f(0)")
    st.latex(r"f(2h) = (1+h) f(h)")

    st.write("We denote")
    st.latex(r"f(x_k) \approx f_k")
    st.write("where")
    st.latex(r"x_k = kh ")
    st.write("and therefore:")
    st.latex(r"f_{k+1} = (1+h) f_k")

    st.write("The error for this method is O(h), which means the error decreases linearly with h.")


# Calculate the exact and approximate solutions
x = np.linspace(0, h*(steps-1), 100)
y = f(x, initial_condition)

_x = [i*h for i in range(0, steps)]
_y = [_f(i, h, initial_condition) for i in range(0, steps)]

# Calculate and display the MSE and pointwise error
mse = MSE(np.array(_y), f(np.array(_x), initial_condition))
pointwise_err = pointwise_error(np.array(_y), f(np.array(_x), initial_condition))

# Create the plot using Plotly
fig = go.Figure()

# Add traces for exact and approximate solutions
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Exact Solution'))
fig.add_trace(go.Scatter(x=_x, y=_y, mode='markers', name='Approximate Solution'))

# Create a twin axis and add the pointwise error
fig.add_trace(go.Scatter(x=_x, y=pointwise_err, mode='lines+markers', name='Pointwise Error', yaxis='y2'))

# Update layout for the twin axis
fig.update_layout(
    title="Solutions and Pointwise Error",
    xaxis_title="X",
    yaxis_title="Solution",
    yaxis2=dict(
        title="Pointwise Error",
        overlaying='y',
        side='right'
    )
)

st.plotly_chart(fig)

st.subheader(f"Mean Squared Error Value: {mse}")

st.subheader("Central Difference")

with st.expander("Explanation"):
    st.write("The central difference scheme is another way to approximate the same ordinary differential equation:")
    st.latex(r"f' = f")

    st.write("Using the central difference approximation, we have:")
    st.latex(r"f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}")

    st.write("For \( x \geq h \):")
    st.latex(r"f(x) \approx f(x-h) + h \left( \frac{f(x+h) - f(x-h)}{2h} \right)")

    st.write("This equation allows us to approximate \( f(x) \) incrementally, just like the first-order method. The initial condition \( f(0) = 1 \) is also used to start the approximation.")
    st.latex(r"f_{k} \approx f(x_{k-1}) + h \left( \frac{f(x_{k+1}) - f(x_{k-1})}{2h} \right)")

    st.write("The error for this method is O(h^2), which means the error decreases quadratically with h.")


# Calculate the central difference approximation
central_diff_y = [central_diff_f(i, h, initial_condition) for i in range(0, steps)]

# Calculate and display the MSE and pointwise error for central difference
central_diff_mse = MSE(np.array(central_diff_y), f(np.array(_x), initial_condition))
central_diff_pointwise_err = pointwise_error(np.array(central_diff_y), f(np.array(_x), initial_condition))

# Create the plot using Plotly for central difference
central_diff_fig = go.Figure()

# Add traces for exact and central difference solutions
central_diff_fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Exact Solution'))
central_diff_fig.add_trace(go.Scatter(x=_x, y=central_diff_y, mode='markers', name='Central Difference'))

# Create a twin axis and add the pointwise error
central_diff_fig.add_trace(go.Scatter(x=_x, y=central_diff_pointwise_err, mode='lines+markers', name='Pointwise Error', yaxis='y2'))

# Update layout for the twin axis
central_diff_fig.update_layout(
    title="Central Difference: Solutions and Pointwise Error",
    xaxis_title="X",
    yaxis_title="Solution",
    yaxis2=dict(
        title="Pointwise Error",
        overlaying='y',
        side='right'
    )
)

st.plotly_chart(central_diff_fig)

st.subheader(f"Central Difference - Mean Squared Error Value: {central_diff_mse}")
