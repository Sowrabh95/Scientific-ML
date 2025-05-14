# Scientific-ML
SIR MODEL USING NEURAL ODEs in JULIA
Problem Background
The SIR (Susceptible-Infected-Recovered) model is a classical epidemio-logical model that simulates how a contagious disease spreads through a population in the region. It consists of three ordinary differential equations (ODEs) describing the rates of change of the susceptible/ not infected S(t), infected I(t), and recovered R(t) populations over time (days).

The dynamics are governed by:
𝑑𝑆(𝑡)/𝑑𝑡 = −𝛽× 𝑆(𝑡)× 𝐼(𝑡)𝑁 

𝑑𝐼(𝑡)/𝑑𝑡 =𝛽 × 𝑆(𝑡)× 𝐼(𝑡)𝑁− 𝛾 × 𝐼(𝑡)

𝑑𝑅(𝑡) /𝑑𝑡 = 𝛾 × 𝐼(𝑡)

Where:

•N is the total population (N=1000)

•β (beta) is the transmission rate, calculated as: β= Contact Rate × Transmission Probability =0.3

•γ (gamma) is the recovery rate, calculated as: γ=1/D = 1/10 where D is the average duration of infection

![image](https://github.com/user-attachments/assets/244e0d8a-4548-4375-8103-89dd8a11fb26)
Image credits: 3Blue1Brown
In this scenario, we use both a traditional ODE which is formula based and a Neural ODE to model the SIR system, then compare the results.

1) Generate Synthetic Data
a. Load necessary packages

• ComponentArrays: For handling and optimizing neural network parameters cleanly.

•Lux: A lightweight and fast neural network library in Julia.

•DiffEqFlux: Bridges differential equations with machine learning (Neural ODEs).

•OrdinaryDiffEq: For solving ordinary differential equations (ODEs).

•Optimization & related: For optimization-based training (like gradient descent).

•Random: Controls random number generation for reproducibility.

•Plots: For visualizing results.

b. Set random number generator (RNG)

•	Xoshiro is a fast pseudo-random number generator.

•	Ensures that any randomness (like neural net initialization) is reproducible.

c. Define the SIR model parameters
# SIR model parameters
N = 1000.0f0

β = 0.3f0

γ = 0.1f0

•	N: Total population = 1000

•	β (beta): Transmission rate is how likely an infected person infects a susceptible person.

•	γ (gamma): Recovery rate is the rate at which infected people recover.

•	All are stored as Float32 values (f0) for compatibility with neural network code.

d. Set initial conditions

# Initial conditions

I0, R0 = 1.0f0, 0.0f0

S0 = N - I0 - R0

u0 = Float32[S0, I0, R0]

•	Initially:

I0 = 1 infected

R0 = 0 recovered

S0 = 999 susceptible (since total is 1000)

•	u0 is the initial state vector: [S0, I0, R0].

e. Define the time span

# Time domain

tspan = (0.0f0, 160.0f0)

datasize = 161

tsteps = range(tspan[1], tspan[2]; length=datasize)

•	Simulate the model from day 0 to day 160.

•	datasize = 161 gives us daily values (including day 0).

•	tsteps is a vector of time points from 0 to 160 (used for saving output at each day).

f. Define the true SIR model

# True SIR ODE function

function true_SIR!(du, u, p, t)

    S, I, R = u
    
    du[1] = -β * S * I / N
    
    du[2] = β * S * I / N - γ * I
    
    du[3] = γ * I
end

This is the SIR differential equation model written in-place for performance (! indicates mutation).

•	u = current state [S, I, R]

•	du = change of state

•	du[1]: Change in susceptibles

‡decreases as they get infected

•	du[2]: Change in infected

‡ increases with new infections, decreases with recoveries

•	du[3]: Change in recovered

‡ increases as infected people recover

g. Solve the ODE (generate synthetic data)

# Solve true ODE for synthetic data

prob_true = ODEProblem(true_SIR!, u0, tspan)

ode_data = Array(solve(prob_true, Tsit5(); saveat=tsteps))

•	ODEProblem: Defines the initial value problem.

	true_SIR!: the function

	u0: initial condition

	tspan: time range

•	solve(..., Tsit5()): Solves the system using Tsit5, a 5th-order Runge-Kutta method.

•	saveat = tsteps: Tells the solver to record values at each time step.

•	ode_data shall contain simulated data for [S(t), I(t), R(t)] at all 161 time points.

#Save the data generated for S, I, R in csv format

using CSV

using DataFrames

df = DataFrame(S=ode_data[1, :], I=ode_data[2, :], R=ode_data[3, :])

CSV.write("ode_data.csv", df)

The Purpose is to generate "ground truth" data using known parameters for later comparison. We can also save them using CSV and the DataFrames libraries.

2) Implement a Neural ODE to Understand the Dynamics
   
# Define the neural network model

nn_model = Chain(Dense(3, 64, tanh), Dense(64, 3))

p, st = Lux.setup(rng, nn_model)

# Create the neural ODE problem

neural_ode = NeuralODE(nn_model, tspan, Tsit5(); saveat=tsteps)

The chain of dense layers is used in this neural network approximates the SIR derivative function. The NeuralODE command Integrates the Neural Network (NN) over time just like the true ODE.

3) Train the Neural ODE on Generated Data

Prediction Function

# Prediction function

function predict_neuralode(p)

Array(neural_ode(u0, p, st)[1])

end

•	This runs the Neural ODE solver with parameter p.

•	Returns the predicted SIR values over time (as an array).

Loss Function

# Loss function

function loss_neuralode(p)

pred = predict_neuralode(p)

return sum(abs2, pred .- ode_data)

end

•	Calculates the difference (error) between predictions and ground truth (ode_data).

•	Uses sum of squared differences as the loss.



Callback Function (Optional Plotting)

# Optional plotting callback

function callback(state, l; doplot=true)

    println("Loss = ", l)
    
    if doplot
    
        pred = predict_neuralode(state.u)
        
        plt = plot(tsteps, ode_data', lw=2, linestyle=:dash, label=["S true"         "I true" "R true"])
        
        plot(plt, tsteps, pred', lw=2, label=["S pred" "I pred" "R pred"], title="Neural ODE vs True SIR", xlabel="Time (days)", ylabel="Population")
        
        display(plt)
    end

    
    
    return false
end


•	Prints the current loss after each training step.

•	If doplot=true, plots both the true and predicted curves for S, I, and R.

•	This helps visually monitor training progress.

Initialization

# Initialize parameter array

p_init = ComponentArray(p)


# First run callback to visualize initial prediction

callback((; u=p_init), loss_neuralode(p_init); doplot=true)


•	Prepares the neural network parameters for training.
•	Runs a first plot to visualize the model before training.

Optimization Setup

# Set up optimization

adtype = Optimization.AutoZygote()

optf = OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

optprob = OptimizationProblem(optf, p_init)

•	Uses automatic differentiation (Zygote) for computing gradients.

•	Wraps the loss function into an OptimizationProblem.


Training: ADAM Optimizer

# Train using ADAM
res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); maxiters=300, callback=callback)

•	Trains the neural network using the Adam optimizer, which is good for deep learning.

•	Runs for 300 iterations while tracking progress using the callback.

Fine-tuning: BFGS Optimizer

# Fine-tune using BFGS

optprob2 = remake(optprob; u0=res1.u)

res2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm=0.01); callback=callback)

# Final plot

callback((; u=res2.u), loss_neuralode(res2.u); doplot=true)

•	Uses the result from ADAM (res1.u) to start fine-tuning with BFGS (a second-order optimizer).

•	BFGS can help find a more precise local minimum after Adam's rough optimization.

Optimization loop adjusts NN parameters to match the neural model output with the true SIR data. ADAM and BFGS performs two-stage optimization strategy for stable convergence. BFGS is used to fine-tune the NN.

d) Compare Neural ODE Predictions to True Data

# Final plot

callback((; u=res2.u), loss_neuralode(res2.u); doplot=true)

This final plot overlays predictions from the trained neural model and the original SIR solution.

Additionally, to save the prediction results in a csv file

#save the predicted data in csv format

df_pred = DataFrame(S=predict_neuralode(res2.u)[1, :], I=predict_neuralode(res2.u)[2, :], R=predict_neuralode(res2.u)[3, :])

CSV.write("predicted_data.csv", df_pred)

Visual plots help confirm how well the model learned the dynamics.
