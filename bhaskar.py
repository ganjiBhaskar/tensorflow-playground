import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import io

st.header('A Neural Network Playground ')

# Function to load datasets
@st.cache_data
def load_data(file):
    return pd.read_csv(file, header=None)

def create_layer(units, activation):
    return Dense(units=units, activation=activation, use_bias=True)

def app():
    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    st.sidebar.title("Configuration")
    num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 1)
    epochs = st.sidebar.slider("Select number of epochs", 100, 1000, 100)
    lr = st.sidebar.number_input('Enter Learning Rate: ')
    hidden_layers = []

    for i in range(num_hidden_layers):
        st.sidebar.markdown(f"### Hidden Layer {i+1}")
        units = st.sidebar.slider(f"Number of units for hidden layer {i+1}", 1, 10, 1)
        activation = st.sidebar.selectbox(f"Activation function for hidden layer {i+1}", ['tanh', 'sigmoid', 'relu', 'linear'], key=f"activation_{i}")
        hidden_layers.append((units, activation))


    if uploaded_file is not None:
        # Load dataset
        dataset = load_data(uploaded_file)

        # Split dataset into training and testing sets
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values.astype(np.int_)  # Convert target variable to integers
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        if st.sidebar.button("Submit"):
            def build_model():
                input_layer = Input(shape=(X_train.shape[1],))
                x = input_layer
                for units, activation in hidden_layers:
                    x = create_layer(units, activation)(x)
                output_layer = Dense(units=1, activation='sigmoid')(x)  # Output layer for binary classification
                model = Model(inputs=input_layer, outputs=output_layer)
                model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
                return model

            model = build_model()

            history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=0)

            # Plot training and testing loss
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Testing Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            loss_plot = fig

            # Plot the output layer decision region
            fig, ax = plt.subplots()
            plot_decision_regions(X, Y, clf=model, ax=ax)
            decision_region_plot = fig

            # Display the plots side by side
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Decision Region for the Output Layer")
                    st.pyplot(decision_region_plot)
                with col2:
                    st.write("Training and Testing Loss")
                    st.pyplot(loss_plot)

            # Get all hidden layers
            hidden_layers_output = [layer.output for layer in model.layers if isinstance(layer, Dense)]

            # Extract the output of each neuron from all hidden layers
            for layer_num, layer_output in enumerate(hidden_layers_output[:-1]):  # Exclude the output layer
                num_neurons = layer_output.shape[1]
                cols = st.columns(3)  # Create 3 columns for the plots
                for neuron_num in range(num_neurons):
                    neuron_model = Model(inputs=model.input, outputs=layer_output[:, neuron_num])
                    col = cols[neuron_num % 3]  # Cycle through the columns
                    with col:
                        st.write(f"Plotting decision region for neuron {neuron_num+1} in hidden layer {layer_num+1}")
                        fig, ax = plt.subplots()
                        plot_decision_regions(X, Y, clf=neuron_model, ax=ax)
                        st.pyplot(fig)

if __name__ == "__main__":
    app()
