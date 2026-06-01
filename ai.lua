-- ============================================================
-- NEURAL NETWORK WITH MULTIPLE ANSWER VARIATIONS
-- Each question can be answered in different ways
-- ============================================================

local math = math
local table = table
local string = string

function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

function sigmoidDerivative(x)
    return x * (1 - x)
end

function randomWeight()
    return (math.random() - 0.5) * 2
end

-- ========== NEURAL NETWORK CLASS ==========

local NeuralNetwork = {}
NeuralNetwork.__index = NeuralNetwork

function NeuralNetwork.new(layers)
    local self = setmetatable({}, NeuralNetwork)
    self.layers = layers
    self.numLayers = #layers
    
    self.weights = {}
    self.biases = {}
    
    for i = 2, self.numLayers do
        local layerWeights = {}
        local inputSize = layers[i-1]
        local outputSize = layers[i]
        
        for j = 1, outputSize do
            local neuronWeights = {}
            for k = 1, inputSize do
                table.insert(neuronWeights, randomWeight())
            end
            table.insert(layerWeights, neuronWeights)
        end
        self.weights[i-1] = layerWeights
        
        local layerBiases = {}
        for j = 1, outputSize do
            table.insert(layerBiases, randomWeight())
        end
        self.biases[i-1] = layerBiases
    end
    
    self.layerInputs = {}
    self.layerOutputs = {}
    
    return self
end

function NeuralNetwork:forward(input)
    self.layerInputs = {}
    self.layerOutputs = {}
    
    local current = input
    table.insert(self.layerOutputs, current)
    
    for layer = 1, self.numLayers - 1 do
        local weights = self.weights[layer]
        local biases = self.biases[layer]
        local nextOutputs = {}
        local nextInputs = {}
        
        for neuron = 1, #weights do
            local sum = biases[neuron]
            for i = 1, #current do
                sum = sum + current[i] * weights[neuron][i]
            end
            table.insert(nextInputs, sum)
            table.insert(nextOutputs, sigmoid(sum))
        end
        
        table.insert(self.layerInputs, nextInputs)
        table.insert(self.layerOutputs, nextOutputs)
        current = nextOutputs
    end
    
    return current
end

function NeuralNetwork:train(input, target, learningRate)
    local output = self:forward(input)
    
    local errors = {}
    local gradients = {}
    
    local outputErrors = {}
    for i = 1, #output do
        outputErrors[i] = target[i] - output[i]
    end
    table.insert(errors, outputErrors)
    
    for layer = self.numLayers - 1, 2, -1 do
        local layerErrors = {}
        local nextErrors = errors[#errors]
        local weights = self.weights[layer]
        
        for neuron = 1, #self.layerOutputs[layer-1] do
            local sum = 0
            for nextNeuron = 1, #nextErrors do
                sum = sum + nextErrors[nextNeuron] * weights[nextNeuron][neuron]
            end
            table.insert(layerErrors, sum)
        end
        table.insert(errors, layerErrors)
    end
    
    local reversedErrors = {}
    for i = #errors, 1, -1 do
        table.insert(reversedErrors, errors[i])
    end
    
    for layer = 1, self.numLayers - 1 do
        local layerGradients = {}
        local layerOutput = self.layerOutputs[layer]
        local layerInput = self.layerOutputs[layer-1]
        local layerErrors = reversedErrors[layer]
        
        for neuron = 1, #self.weights[layer] do
            local gradient = layerErrors[neuron] * sigmoidDerivative(self.layerInputs[layer][neuron])
            table.insert(layerGradients, gradient)
            
            for w = 1, #self.weights[layer][neuron] do
                self.weights[layer][neuron][w] = self.weights[layer][neuron][w] + learningRate * gradient * layerInput[w]
            end
            
            self.biases[layer][neuron] = self.biases[layer][neuron] + learningRate * gradient
        end
        table.insert(gradients, layerGradients)
    end
    
    local totalError = 0
    for i = 1, #outputErrors do
        totalError = totalError + outputErrors[i] ^ 2
    end
    
    return totalError / #outputErrors
end

-- ========== TEXT PROCESSING ==========

local vocabulary = {}
local wordToIndex = {}

function addToVocabulary(word)
    if not wordToIndex[word] then
        table.insert(vocabulary, word)
        wordToIndex[word] = #vocabulary
    end
end

function textToVector(text)
    text = string.lower(text)
    text = string.gsub(text, "[%p]", "")
    
    local vector = {}
    for i = 1, #vocabulary do
        vector[i] = 0
    end
    
    for word in string.gmatch(text, "%S+") do
        if wordToIndex[word] then
            vector[wordToIndex[word]] = vector[wordToIndex[word]] + 1
        end
    end
    
    local sum = 0
    for i = 1, #vector do
        sum = sum + vector[i]
    end
    if sum > 0 then
        for i = 1, #vector do
            vector[i] = vector[i] / sum
        end
    end
    
    return vector
end

-- ========== QUESTIONS WITH MULTIPLE ANSWER VARIATIONS ==========

local trainingData = {
    -- Question 1: Grass color
    {
        question = "what color is grass",
        answers = {"green", "it is green", "grass is green", "green color"}
    },
    {
        question = "grass color",
        answers = {"green", "the grass is green", "green"}
    },
    
    -- Question 2: 2+2
    {
        question = "what is 2 plus 2",
        answers = {"four", "4", "the answer is four", "equals four"}
    },
    {
        question = "2+2",
        answers = {"four", "4", "four"}
    },
    {
        question = "how much is two plus two",
        answers = {"four", "4", "it is four"}
    },
    
    -- Question 3: Shakespeare
    {
        question = "who wrote romeo and juliet",
        answers = {"shakespeare", "william shakespeare", "shakespeare wrote it", "william"}
    },
    {
        question = "romeo author",
        answers = {"shakespeare", "william shakespeare", "shakespeare"}
    },
    
    -- Question 4: Paris
    {
        question = "what is the capital of france",
        answers = {"paris", "the capital is paris", "paris france", "paris"}
    },
    {
        question = "france capital",
        answers = {"paris", "paris", "paris"}
    },
    
    -- Question 5: Boiling point
    {
        question = "what is the boiling point of water",
        answers = {"one hundred", "100 degrees", "100 celsius", "one hundred celsius", "100"}
    },
    {
        question = "water boiling point",
        answers = {"one hundred", "100 c", "100 degrees celsius", "one hundred"}
    },
    
    -- Question 6: Mona Lisa
    {
        question = "who painted the mona lisa",
        answers = {"da vinci", "leonardo da vinci", "da vinci painted it", "leonardo"}
    },
    {
        question = "mona lisa painter",
        answers = {"da vinci", "leonardo da vinci", "da vinci"}
    },
    
    -- Question 7: Pacific Ocean
    {
        question = "what is the largest ocean on earth",
        answers = {"pacific", "pacific ocean", "the pacific", "pacific"}
    },
    {
        question = "largest ocean",
        answers = {"pacific", "pacific ocean", "pacific"}
    },
    
    -- Question 8: Light bulb
    {
        question = "who invented the light bulb",
        answers = {"edison", "thomas edison", "edison invented it", "thomas"}
    },
    {
        question = "light bulb inventor",
        answers = {"edison", "thomas edison", "edison"}
    },
    
    -- Question 9: Mars
    {
        question = "what planet is known as the red planet",
        answers = {"mars", "mars planet", "the red planet is mars", "mars"}
    },
    {
        question = "red planet",
        answers = {"mars", "mars", "mars"}
    },
    
    -- Question 10: Speed of light
    {
        question = "what is the speed of light",
        answers = {"three hundred million", "300 million", "300000000", "three hundred million m/s", "300 million meters per second"}
    },
    {
        question = "light speed",
        answers = {"three hundred million", "300 million", "three hundred million"}
    },
}

-- Build vocabulary from all questions and answers
for _, item in ipairs(trainingData) do
    local text = string.lower(item.question)
    text = string.gsub(text, "[%p]", "")
    for word in string.gmatch(text, "%S+") do
        addToVocabulary(word)
    end
    
    for _, answer in ipairs(item.answers) do
        local ansText = string.lower(answer)
        ansText = string.gsub(ansText, "[%p]", "")
        for word in string.gmatch(ansText, "%S+") do
            addToVocabulary(word)
        end
    end
end

-- Create unique answer categories
local answerCategories = {}
local answerToIndex = {}

for _, item in ipairs(trainingData) do
    for _, answer in ipairs(item.answers) do
        local normalized = string.lower(answer)
        if not answerToIndex[normalized] then
            answerToIndex[normalized] = #answerCategories + 1
            table.insert(answerCategories, normalized)
        end
    end
end

print("=== NEURAL NETWORK WITH VARIABLE ANSWERS ===")
print("Vocabulary size: " .. #vocabulary .. " words")
print("Unique answer variations: " .. #answerCategories)
print("Training examples: " .. #trainingData)
print()

-- Create training pairs (multiple pairs per question)
local trainInputs = {}
local trainTargets = {}

for _, item in ipairs(trainingData) do
    local inputVec = textToVector(item.question)
    
    for _, answer in ipairs(item.answers) do
        local targetVec = {}
        for i = 1, #answerCategories do
            targetVec[i] = 0
        end
        local ansKey = string.lower(answer)
        targetVec[answerToIndex[ansKey]] = 1
        
        table.insert(trainInputs, inputVec)
        table.insert(trainTargets, targetVec)
    end
end

print("Total training pairs: " .. #trainInputs)
print()

-- Create and train network
local vocabSize = #vocabulary
local numAnswers = #answerCategories
local nn = NeuralNetwork.new({vocabSize, 25, 20, numAnswers})

print("Network: " .. vocabSize .. " -> 25 -> 20 -> " .. numAnswers)
print("Training...")
print()

local learningRate = 0.7
local epochs = 800

for epoch = 1, epochs do
    local totalError = 0
    
    for i = 1, #trainInputs do
        local error = nn:train(trainInputs[i], trainTargets[i], learningRate)
        totalError = totalError + error
    end
    
    if epoch == 300 then
        learningRate = 0.4
    elseif epoch == 600 then
        learningRate = 0.2
    end
    
    if epoch % 100 == 0 then
        local avgError = totalError / #trainInputs
        print(string.format("Epoch %d, Error: %.6f, LR: %.2f", epoch, avgError, learningRate))
    end
end

print()
print("Training complete!")
print("==========================================")
print()

-- ========== ANSWERING FUNCTION ==========

function askQuestion(question)
    local vector = textToVector(question)
    local output = nn:forward(vector)
    
    -- Find best answer with confidence
    local bestIndex = 1
    local bestConfidence = output[1]
    
    for i = 2, #output do
        if output[i] > bestConfidence then
            bestConfidence = output[i]
            bestIndex = i
        end
    end
    
    local answer = answerCategories[bestIndex]
    return answer, bestConfidence
end

-- ========== DEMONSTRATION ==========

print("=== TESTING DIFFERENT QUESTION VARIATIONS ===")
print()

local testQuestions = {
    -- Question 1 variations
    "what color is grass",
    "grass color",
    "color of grass",
    "what colour is the grass",
    
    -- Question 2 variations
    "2+2",
    "what is two plus two",
    "how much is 2 plus 2",
    
    -- Question 3 variations
    "who wrote romeo",
    "romeo author",
    "shakespeare play",
    
    -- Question 4 variations
    "france capital",
    "capital of france",
    "paris",
    
    -- Question 5 variations
    "boiling point of water",
    "water boiling temperature",
    "what temp does water boil",
    
    -- Question 6 variations
    "mona lisa painter",
    "who painted mona lisa",
    "leonardo",
    
    -- Question 7 variations
    "largest ocean",
    "biggest ocean on earth",
    "pacific",
    
    -- Question 8 variations
    "light bulb inventor",
    "who invented light bulb",
    "edison",
    
    -- Question 9 variations
    "red planet",
    "which planet is red",
    "mars",
    
    -- Question 10 variations
    "speed of light",
    "how fast is light",
    "light speed"
}

for _, q in ipairs(testQuestions) do
    local answer, confidence = askQuestion(q)
    print(string.format("Q: %s", q))
    print(string.format("A: %s (confidence: %.2f%%)", answer, confidence * 100))
    print()
end

-- ========== INTERACTIVE MODE ==========

print("==========================================")
print("=== INTERACTIVE MODE ===")
print("Ask me anything! Each answer may vary.")
print("Type 'exit' to quit")
print()

while true do
    io.write("\nYou: ")
    local input = io.read()
    
    if not input or input == "exit" or input == "quit" then
        print("Goodbye!")
        break
    end
    
    if input == "help" then
        print("Try asking about: grass, 2+2, shakespeare, paris, boiling point, mona lisa, ocean, light bulb, mars, light speed")
    else
        local answer, confidence = askQuestion(input)
        if confidence > 0.3 then
            print(string.format("Bot: %s", answer))
            print(string.format("(confidence: %.0f%%)", confidence * 100))
        else
            print("Bot: I'm not sure about that. Try rephrasing your question.")
        end
    end
end

print("\n=== EXAMPLE OF VARIABLE ANSWERS ===")
print("Same question can get different answers:")
print()

-- Show that same question can yield different answers
local sameQuestion = "what color is grass"
for i = 1, 5 do
    local answer, conf = askQuestion(sameQuestion)
    print(string.format("Try %d: %s", i, answer))
end
