------------------------------- for a single object ---------------------------
local DerenderSingleCriterion, parent = torch.class('nn.DerenderSingleCriterion', 'nn.Criterion')

function DerenderSingleCriterion:__init(weights, lens, withDet, sizeAverage)

  parent.__init(self)
  self.sizeAverage = sizeAverage or true

  self.criterion = {}
  --[[self.criterion[1] = nn.MSECriterion(self.sizeAverage)
  self.criterion[2] = nn.MSECriterion(self.sizeAverage)
  self.criterion[3] = nn.MSECriterion(self.sizeAverage)
  self.criterion[4] = nn.MSECriterion(self.sizeAverage)
  self.criterion[5] = nn.MSECriterion(self.sizeAverage)
  self.criterion[6] = nn.MSECriterion(self.sizeAverage)
--]]
  self.criterion[1] = nn.CrossEntropyCriterion()
  self.criterion[2] = nn.CrossEntropyCriterion()
  self.criterion[3] = nn.AbsCriterion(self.sizeAverage)
  self.criterion[4] = nn.AbsCriterion(self.sizeAverage)
  self.criterion[5] = nn.CrossEntropyCriterion()
  self.criterion[6] = nn.AbsCriterion(self.sizeAverage)
    
  self.weights = weights or torch.ones(6)
  assert((#self.weights)[1] == 6, 'incorrect number of weights in criterion')
  
  self.lens = lens
  self.withDet = withDet or false

end

function DerenderSingleCriterion:updateOutput(input, target)

  assert( input:nElement() == target:nElement(),
    "input and target size mismatch")
  assert( input:size(2) == self.lens[-1], "input dimension mismatch" )

  local nBatch = (#input)[1]
 
  self.output = 0
  for i = 1, 6 do
  if self.criterion[i].__typename == 'nn.CrossEntropyCriterion' then
      local _, instTarget = torch.max(target[{{}, {self.lens[i] + 1, self.lens[i + 1]}}], 2) 
      self.output = self.output + self.weights[i] * self.criterion[i]:forward(
        input[{{}, {self.lens[i] + 1, self.lens[i + 1]}}], instTarget)
    else
      self.output = self.output + self.weights[i] * self.criterion[i]:forward(
        input[{{}, {self.lens[i] + 1, self.lens[i + 1]}}],
        target[{{}, {self.lens[i] + 1, self.lens[i + 1]}}]) 
    end
  end

  return self.output

end

-- must have called updateOutput with the same input/target pair right before
function DerenderSingleCriterion:updateGradInput(input, target)

  assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

  if self.gradInput == nil then
    self.gradInput = input.new()
  end
  self.gradInput:resizeAs(input):zero()
  
  local nBatch = (#input)[1]
 
  for i = 1, 6 do
    if self.criterion[i].__typename == 'nn.CrossEntropyCriterion' then
      local _, instTarget = torch.max(target[{{}, {self.lens[i] + 1, self.lens[i + 1]}}], 2) 
      self.gradInput[{{}, {self.lens[i] + 1, self.lens[i + 1]}}] =
        self.weights[i] * self.criterion[i]:backward(
        input[{{}, {self.lens[i] + 1, self.lens[i + 1]}}], instTarget)
    else
      self.gradInput[{{}, {self.lens[i] + 1, self.lens[i + 1]}}] =
        self.weights[i] * self.criterion[i]:backward(
        input[{{}, {self.lens[i] + 1, self.lens[i + 1]}}],
        target[{{}, {self.lens[i] + 1, self.lens[i + 1]}}])
    end
  end

  -- for boxes with no objects, only classifying obj/non-obj matters
  if self.withDet then
    local mask = self.gradInput.new():resizeAs(self.gradInput):zero()
    mask[{{}, {self.lens[2] + 1, self.lens[7]}}] = 
      target[{{}, 1}]:eq(1):view(input:size(1), 1):repeatTensor(1, self.lens[7] - self.lens[2])
    self.gradInput:maskedFill(mask, 0)
  end
  return self.gradInput

end

------------------------------- for multi objects ---------------------------
local DerenderCriterion, parent = torch.class('nn.DerenderCriterion', 'nn.Criterion')

function DerenderCriterion:__init(weights, lens, sizeAverage)

  parent.__init(self)
  self.sizeAverage = sizeAverage or true

  self.criterion = {}
  self.criterion[1] = nn.MSECriterion(self.sizeAverage)
  self.criterion[2] = nn.MSECriterion(self.sizeAverage)
  self.criterion[3] = nn.MSECriterion(self.sizeAverage)
  self.criterion[4] = nn.MSECriterion(self.sizeAverage)
  self.criterion[5] = nn.MSECriterion(self.sizeAverage)
  self.criterion[6] = nn.MSECriterion(self.sizeAverage)
    
  self.weights = weights or torch.ones(6)
  self.lens = lens
  assert((#self.weights)[1] == 6, 'incorrect number of weights in criterion')

end

function DerenderCriterion:updateOutput(input, target)

  assert( input:nElement() == target:nElement(),
    "input and target size mismatch")
  assert( input:size(3) == self.lens[-1], "input dimension mismatch" )

  local nBatch = (#input)[1]
  local nObj = (#input)[2]
 
  self.output = 0
  for i = 1, 6 do
    self.output = self.output + self.weights[i] * self.criterion[i]:forward(
      input[{{}, {}, {self.lens[i] + 1, self.lens[i + 1]}}],
      target[{{}, {}, {self.lens[i] + 1, self.lens[i + 1]}}]) 
  end

  return self.output

end

-- must have called updateOutput with the same input/target pair right before
function DerenderCriterion:updateGradInput(input, target)

  assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

  if self.gradInput == nil then
    self.gradInput = input.new()
  end
  self.gradInput:resizeAs(input):zero()
  
  local nBatch = (#input)[1]
  local nObj = (#input)[2]
 
  for i = 1, 6 do
    self.gradInput[{{}, {}, {self.lens[i] + 1, self.lens[i + 1]}}] =
      self.weights[i] * 
      self.criterion[i]:backward(
        input[{{}, {}, {self.lens[i] + 1, self.lens[i + 1]}}],
        target[{{}, {}, {self.lens[i] + 1, self.lens[i + 1]}}])
  end

  return self.gradInput

end
