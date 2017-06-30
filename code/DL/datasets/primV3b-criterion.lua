------------------------------- for a single object ---------------------------
local CustomCriterion, parent = torch.class('nn.CustomCriterion', 'nn.Criterion')

function CustomCriterion:__init(types, lens, weights, sizeAverage)

  parent.__init(self)
  self.sizeAverage = sizeAverage or true

  self.num = (#types)[1]
  self.criterion = {}
  for i = 1, self.num do
    if types[i] == 1 then
      self.criterion[i] = nn.CrossEntropyCriterion()
    elseif types[i] == 2 then
      self.criterion[i] = nn.AbsCriterion()
    elseif types[i] == 3 then
      self.criterion[i] = nn.BCECriterion()
    end
  end

  self.lens = lens
  assert((#self.lens)[1] == self.num + 1, "number of output lengths mismatch")

  self.weights = weights or torch.ones(#self.lens)
  assert((#self.weights)[1] == self.num, "number of output weights mismatch")
end

function CustomCriterion:updateOutput(input, target)

  assert( input:nElement() == target:nElement(),
  "input and target size mismatch")
  assert( input:size(2) == self.lens[-1], "input dimension mismatch" )

  local nBatch = (#input)[1]

  self.output = 0
  for i = 1, self.num do
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
function CustomCriterion:updateGradInput(input, target)

  assert( input:nElement() == target:nElement(),
  "input and target size mismatch")

  if self.gradInput == nil then
    self.gradInput = input.new()
  end
  self.gradInput:resizeAs(input):zero()

  local nBatch = (#input)[1]

  for i = 1, self.num do
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

  return self.gradInput
end

