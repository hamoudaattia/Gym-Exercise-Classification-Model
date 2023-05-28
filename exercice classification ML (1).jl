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

last_df = unique(new_df; keep = :first)


using DataFrames
# Reset the row indices of the DataFrame
last_df[!, :index] = 1:size(last_df, 1)

using Flux

X = last_df.Title  # Exercise names

y = last_df.BodyPart   # Target body parts

using Flux

X = last_df.Title  # Exercise names 
y = last_df.BodyPart   # Target body parts
# Step 2: Feature Extraction
X_encoded = Flux.onehotbatch(X, sort(unique(X))) |> Matrix{Float32}

# Encode target labels as integers
label_mapping = Dict(unique(y) .=> 1:length(unique(y)))
y_encoded = [label_mapping[label] for label in y] # Step 3: Model Training
# Split the data into training and testing sets
data = [(x, y) for (x, y) in zip(eachrow(X_encoded), y_encoded)] 

using Random

# Shuffle the data randomly
shuffled_data = shuffle(data)

# Define the desired train-test split ratio
train_ratio = 0.8

# Calculate the sizes of the training and test sets
train_size = Int(round(length(shuffled_data) * train_ratio))
test_size = length(shuffled_data) - train_size

# Split the data into training and test sets
train_data = shuffled_data[1:train_size]
test_data = shuffled_data[train_size+1:end]

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

exercise_name = "Band low-to-high twist" # Example exercise name for prediction
exercise_encoded = Flux.onehotbatch([exercise_name], sort(unique(X_encoded))) |> Matrix{Float32}
predicted_body_part = Flux.argmax(model(exercise_encoded), dims=2)[1]

inverse_label_mapping = Dict(value => key for (key, value) in label_mapping)
predicted_body_part_label = inverse_label_mapping[predicted_body_part]

println("Predicted body part: $predicted_body_part_label")








