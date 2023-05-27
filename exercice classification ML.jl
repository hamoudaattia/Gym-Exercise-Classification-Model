using CSV
using DataFrames
using MLJ

# Read the CSV file into a DataFrame
df = CSV.read("megaGymDataset.csv", DataFrame)

# Display summary statistics of the DataFrame
describe(df)

# Get the number of rows and columns in the DataFrame
nrow(df), ncol(df)

# Select the two columns from the original DataFrame
selected_columns = df[:, [:Title,  :BodyPart]]
# Create a new DataFrame with the selected columns
new_df = DataFrame(selected_columns)
nrow(new_df), ncol(new_df)

schema(new_df)

using Flux

X = new_df.Title  # Exercise names
y = new_df.BodyPart   # Target body parts

# Step 2: Feature Extraction
# In this simple example, we'll use one-hot encoding for the exercise names
X_encoded = Flux.onehotbatch(X, sort(unique(X))) |> Matrix{Float32}
X_unique = sort(unique(X))

using Statistics
# Standardize the input features using z-score normalization
X_standardized = (X_encoded .- mean(X_encoded, dims=1)) ./ std(X_encoded, dims=1)


# Encode target labels as integers
label_mapping = Dict(unique(y) .=> 1:length(unique(y)))
y_encoded = [label_mapping[label] for label in y]

# Step 3: Model Training
# Split the data into training and testing sets
data = [(x, y) for (x, y) in zip(eachrow(X_encoded), y_encoded)]
train_size = Int(round(length(data) * 0.8))
train_data = data[1:train_size]
test_data = data[train_size+1:end]

# Define the model architecture
input_size = size(X_encoded, 2)
model = Chain(
    Dense(input_size, 64, relu),
    Dense(64, 32, relu),
    Dense(32, length(label_mapping))
)

# Define the loss function
loss(x, y) = Flux.crossentropy(softmax(model(x)), Flux.onehotbatch(y, 1:length(label_mapping)))
# Define the optimizer
optimizer = Flux.ADAM()

# Train the model
for epoch in 1:10
    Flux.train!(loss, Flux.params(model), train_data, optimizer)
end

# Step 4: Model Evaluation
# Make predictions on the test set
X_test = [x for (x, _) in test_data]
y_test = [y for (_, y) in test_data]
y_pred = Flux.argmax(model.(X_test), dims=2)

# Calculate accuracy
accuracy = sum(y_pred .== reshape(y_test, :)') / length(y_test)

println("Accuracy: $accuracy")

println("Dimension of X_encoded: ", size(X_encoded))
println("Dimension of X_standardized: ", size(X_standardized))


# Step 5: Model Deployment
# You can now use the trained model to make predictions on new exercise names

exercise_name = "Wrist Roller" # Example exercise name for prediction
exercise_encoded = Flux.onehotbatch([exercise_name], sort(unique(X))) |> Matrix{Float32}
predicted_body_part = Flux.argmax(model(exercise_encoded), dims=2)[1]

inverse_label_mapping = invert(label_mapping)
predicted_body_part_label = inverse_label_mapping[predicted_body_part]

println("Predicted body part: $predicted_body_part_label")
