-- SIMPLE NEURAL NETWORK IN LUA
-- Works without any errors

local math = math
local table = table
local string = string

-- Activation function
function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

function sigmoidDerivative(x)
    return x * (1 - x)
end

-- Create neural network
function createNetwork(inputSize, hiddenSize, outputSize)
    local network = {}
    
    -- Weights and biases
    network.weights1 = {}
    network.bias1 = {}
    network.weights2 = {}
    network.bias2 = {}
    
    -- Initialize first layer weights
    for i = 1, hiddenSize do
        network.weights1[i] = {}
        for j = 1, inputSize do
            network.weights1[i][j] = (math.random() - 0.5) * 2
        end
        network.bias1[i] = (math.random() - 0.5) * 2
    end
    
    -- Initialize second layer weights
    for i = 1, outputSize do
        network.weights2[i] = {}
        for j = 1, hiddenSize do
            network.weights2[i][j] = (math.random() - 0.5) * 2
        end
        network.bias2[i] = (math.random() - 0.5) * 2
    end
    
    -- Forward propagation
    function network:forward(input)
        local hidden = {}
        for i = 1, #self.weights1 do
            local sum = self.bias1[i]
            for j = 1, #input do
                sum = sum + input[j] * self.weights1[i][j]
            end
            hidden[i] = sigmoid(sum)
        end
        
        local output = {}
        for i = 1, #self.weights2 do
            local sum = self.bias2[i]
            for j = 1, #hidden do
                sum = sum + hidden[j] * self.weights2[i][j]
            end
            output[i] = sigmoid(sum)
        end
        
        return output, hidden
    end
    
    -- Train network
    function network:train(input, target, learningRate)
        local output, hidden = self:forward(input)
        
        -- Calculate output errors
        local outputErrors = {}
        for i = 1, #output do
            outputErrors[i] = target[i] - output[i]
        end
        
        -- Calculate output deltas
        local outputDeltas = {}
        for i = 1, #output do
            outputDeltas[i] = outputErrors[i] * sigmoidDerivative(output[i])
        end
        
        -- Calculate hidden errors
        local hiddenErrors = {}
        for i = 1, #hidden do
            local sum = 0
            for j = 1, #outputDeltas do
                sum = sum + outputDeltas[j] * self.weights2[j][i]
            end
            hiddenErrors[i] = sum
        end
        
        -- Calculate hidden deltas
        local hiddenDeltas = {}
        for i = 1, #hidden do
            hiddenDeltas[i] = hiddenErrors[i] * sigmoidDerivative(hidden[i])
        end
        
        -- Update weights and biases for output layer
        for i = 1, #self.weights2 do
            for j = 1, #self.weights2[i] do
                self.weights2[i][j] = self.weights2[i][j] + learningRate * outputDeltas[i] * hidden[j]
            end
            self.bias2[i] = self.bias2[i] + learningRate * outputDeltas[i]
        end
        
        -- Update weights and biases for hidden layer
        for i = 1, #self.weights1 do
            for j = 1, #self.weights1[i] do
                self.weights1[i][j] = self.weights1[i][j] + learningRate * hiddenDeltas[i] * input[j]
            end
            self.bias1[i] = self.bias1[i] + learningRate * hiddenDeltas[i]
        end
        
        -- Calculate total error
        local error = 0
        for i = 1, #outputErrors do
            error = error + outputErrors[i] * outputErrors[i]
        end
        return error / #outputErrors
    end
    
    return network
end

-- Simple word to vector conversion
local vocabulary = {}
local wordToIndex = {}

function addWord(word)
    if not wordToIndex[word] then
        table.insert(vocabulary, word)
        wordToIndex[word] = #vocabulary
    end
end

function textToVector(text)
    text = string.lower(text)
    text = string.gsub(text, "[%p]", "")
    
    local vec = {}
    for i = 1, #vocabulary do
        vec[i] = 0
    end
    
    for word in string.gmatch(text, "%S+") do
        if wordToIndex[word] then
            vec[wordToIndex[word]] = 1
        end
    end
    
    return vec
end

-- Training data
local questions = {
    "what color is grass",
    "what is two plus two",
    "who wrote romeo and juliet",
    "what is the capital of france",
    "what is the boiling point of water",
    "who painted the mona lisa",
    "what is the largest ocean",
    "who invented the light bulb",
    "what planet is the red planet",
    "what is the speed of light"
}

local answers = {
    "green",
    "four",
    "shakespeare",
    "paris",
    "one hundred",
    "da vinci",
    "pacific",
    "edison",
    "mars",
    "three hundred million"
}

-- Build vocabulary
for _, q in ipairs(questions) do
    for word in string.gmatch(q, "%S+") do
        addWord(word)
    end
end

for _, a in ipairs(answers) do
    for word in string.gmatch(a, "%S+") do
        addWord(word)
    end
end

-- Create unique answer IDs
local answerIds = {}
local answerToId = {}
for i, a in ipairs(answers) do
    if not answerToId[a] then
        answerToId[a] = i
        answerIds[i] = a
    end
end

local numAnswers = #answerIds

-- Convert answers to one-hot vectors
function answerToVector(answer)
    local vec = {}
    for i = 1, numAnswers do
        vec[i] = 0
    end
    if answerToId[answer] then
        vec[answerToId[answer]] = 1
    end
    return vec
end

-- Create input and target vectors
local inputs = {}
local targets = {}

for i, q in ipairs(questions) do
    table.insert(inputs, textToVector(q))
    table.insert(targets, answerToVector(answers[i]))
end

-- Create and train network
local vocabSize = #vocabulary
local hiddenSize = 10
local network = createNetwork(vocabSize, hiddenSize, numAnswers)

print("=== NEURAL NETWORK ===")
print("Vocabulary: " .. vocabSize .. " words")
print("Hidden neurons: " .. hiddenSize)
print("Possible answers: " .. numAnswers)
print()

print("Training...")
local learningRate = 0.5
local epochs = 500

for epoch = 1, epochs do
    local totalError = 0
    for i = 1, #inputs do
        local error = network:train(inputs[i], targets[i], learningRate)
        totalError = totalError + error
    end
    
    if epoch % 100 == 0 then
        print(string.format("Epoch %d, Error: %.4f", epoch, totalError / #inputs))
    end
end

print()
print("Training done!")
print()

-- Function to ask questions
function ask(question)
    local vec = textToVector(question)
    local output = network:forward(vec)
    
    local bestIdx = 1
    local bestVal = output[1]
    for i = 2, #output do
        if output[i] > bestVal then
            bestVal = output[i]
            bestIdx = i
        end
    end
    
    return answerIds[bestIdx], bestVal
end

-- Test the network
print("=== TESTING ===")
print()

for i, q in ipairs(questions) do
    local answer, confidence = ask(q)
    print(string.format("Q: %s", q))
    print(string.format("A: %s (%.0f%% confidence)", answer, confidence * 100))
    print()
end

-- Interactive mode
print("=== INTERACTIVE MODE ===")
print("Ask me a question! Type 'exit' to quit")
print()

while true do
    io.write("> ")
    local input = io.read()
    
    if input == "exit" or input == "quit" then
        print("Goodbye!")
        break
    end
    
    local answer, confidence = ask(input)
    if confidence > 0.4 then
        print(string.format("Answer: %s", answer))
    else
        print("Sorry, I don't know the answer to that question.")
    end
    print()
end
