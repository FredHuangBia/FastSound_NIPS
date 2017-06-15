--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('cnnI.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
  self.bestAvg = 0
  self.model = model
  self.criterion = criterion
  self.optimState = optimState or {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    nesterov = true,
    dampening = 0.0,
    weightDecay = opt.weightDecay,
  }
  self.opt = opt
  self.params, self.gradParams = model:getParameters()

  self.logger = {
    train = io.open(paths.concat(opt.resume, 'train.log'), 'a+'),
    val = io.open(paths.concat(opt.resume, 'test.log'), 'a+')
  }
end

function Trainer:train(epoch, dataloader)
  -- Trains the model for a single epoch
  self.optimState.learningRate = self:learningRate(epoch)

  local timer = torch.Timer()
  local dataTimer = torch.Timer()

  local function feval()
    return self.criterion.output, self.gradParams
  end
  
  local trainSize = dataloader:size()
  local lossSum = 0.0
  local N = 0
  
  local visIms = {}
  local visCaptions = {}

  print('=> Training epoch # ' .. epoch .. ', LR ' .. self.optimState.learningRate)
  -- set the batch norm to training mode
  self.model:training()
  for n, sample in dataloader:run() do
    if self.opt.debug and n >= 10 then 
      break
    end
    
    local dataTime = dataTimer:time().real

    -- Copy input and target to the GPU
    self:copyInputs(sample)

    self.output = self.model:forward(self.input)

    local loss = self.criterion:forward(self.model.output, self.target)

    self.model:zeroGradParameters()
    self.criterion:backward(self.model.output, self.target)
    self.model:backward(self.input, self.criterion.gradInput)

    optim.sgd(feval, self.params, self.optimState)

    lossSum = lossSum + loss
    N = N + 1

    local log = ('Epoch: [%d][%d/%d] Time %.3f Data %.3f Err %1.4f')
      :format(epoch, n, trainSize, timer:time().real, dataTime, loss)
    self.logger.train:write(log .. '\n')
    ut.progress(n, trainSize, log)

    if N <= self.opt.visTrain then
      self:visualize(visIms, visCaptions, dataloader)
    end

    -- check that the storage didn't get changed do to an unfortunate getParameters call
    assert(self.params:storage() == self.model:parameters()[1]:storage())

    timer:reset()
    dataTimer:reset()
  end
  
  local log = (' * Finished epoch # %d, Err %1.4f\n'):format(epoch, lossSum / N)
  self.logger.train:write(log .. '\n')
  print(log)
  
  if self.opt.visTrain > 0 then
    local visDir = paths.concat(self.opt.www, 'train_' .. epoch)
    pl.dir.makepath(visDir)
    wut.renderHtml(visDir, visIms, visCaptions, self.opt.visWidth)
  end

  return lossSum / N
end

function Trainer:test(epoch, dataloader)
  -- Computes the top-1 and top-5 err on the validation set

  local timer = torch.Timer()
  local dataTimer = torch.Timer()
  local size = dataloader:size()

  local lossSum = 0.0
  local inferSum = 0.0
  local reconSum = 0.0
  local N = 0

  local visIms = {}
  local visCaptions = {}

  self.model:evaluate()
  for n, sample in dataloader:run() do
    if self.opt.debug and n >= 10 then 
      break
    end
    
    local dataTime = dataTimer:time().real

    -- Copy input and target to the GPU
    self:copyInputs(sample)

    self.output = self.model:forward(self.input)

    local loss = self.criterion:forward(self.model.output, self.target)

    lossSum = lossSum + loss
    N = N + 1
    
    local log = ('Test: [%d][%d/%d] Time %.3f Data %.3f Err %1.4f')
      :format(epoch, n, size, timer:time().real, dataTime, loss)
    self.logger.val:write(log .. '\n')
    ut.progress(n, size, log)

    if N <= self.opt.visTest then
      self:visualize(visIms, visCaptions, dataloader)
    end

    timer:reset()
    dataTimer:reset()
  end
  self.model:training()

  local log = (' * Finished epoch # %d, Err %1.4f\n'):format(epoch, lossSum / N)
  self.logger.val:write(log .. '\n')
  print(log)

  if self.opt.visTest > 0 then
    local visDir = paths.concat(self.opt.www, epoch)
    pl.dir.makepath(visDir)
    wut.renderHtml(visDir, visIms, visCaptions, self.opt.visWidth)
  end

  return lossSum / N
end

function Trainer:sanitize()
  nnut.sanitize(self.model)
  nnut.sanitizeCriterion(self.criterion)
end

function Trainer:copyInputs(sample)
  self.input = self.input or torch.CudaTensor()
  self.input:resize(sample.img:size()):copy(sample.img)
  
  self.target = self.target or torch.CudaTensor()
  self.target:resize(sample.mtl:size()):copy(sample.mtl)
end

function Trainer:learningRate(epoch)
  -- Training schedule
  local decay = 0
  if self.opt.dataset == 'imagenet' then
    decay = math.floor((epoch - 1) / 30)
  elseif self.opt.dataset == 'cifar10' then
    decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
  end
  return self.opt.LR * math.pow(0.1, decay)
end

function Trainer:visualize(visCaptions, dataloader)
  for i = 1, (#self.input)[1] do
    table.insert(visIms, dataloader.dataset:postprocess()(self.input[i]:float()))

    local targetXml = dataloader.dataset:postprocessMtl()(self.target[i]:float())  
    local outputXml = dataloader.dataset:postprocessMtl()(self.output[i]:float())  
    local targetStr = 'target: '
    local outputStr = 'output: '
    for i = 1, (#targetXml)[1] do
      targetStr = targetStr .. targetXml[i] .. ' '
      outputStr = outputStr .. outputXml[i] .. ' '
    end
    table.insert(visCaptions, targetStr .. '<br/>' .. outputStr) 
  end
end

return M.Trainer
