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

require 'rnn'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('inst.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
  self.model = model
  self.criterion = criterion
  self.optimState = optimState or {
    learningRate = opt.LR,
    learningRateDecay = opt.LRDecay == 'anneal' and opt.LRDParam or 0.0,
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

  local function eval()
    return self.criterion.output, self.gradParams
  end
  
  local trainSize = dataloader:size()
  local lossSum = 0.0
  local N = 0
  
  local visIms = {}

  print('=> Training epoch # ' .. epoch .. ', LR: ' .. self.optimState.learningRate)
  -- set the batch norm to training mode
  self.model:training()
  for n, sample in dataloader:run() do
    if self.opt.debug and n >= 10 then 
      break
    end
    
    local dataTime = dataTimer:time().real

    -- Copy input and target to the GPU
    self:copyInputs(sample)

    local loss = 0
    if self.input:dim() ~= 0 then
      self.output = self.model:forward(self.input)
      loss = self.criterion:forward(self.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(eval, self.params, self.optimState)
    end

    lossSum = lossSum + loss
    N = N + 1

    local log = ('Epoch: [%d][%d/%d] Time %.3f Data %.3f Err %1.4f')
      :format(epoch, n, trainSize, timer:time().real, dataTime, loss)
    self.logger.train:write(log .. '\n')
    ut.progress(n, trainSize, log)

    if N <= self.opt.visTrain and self.input:dim() ~= 0 then
      self:visualize(visIms, dataloader)
    end

    -- check that the storage didn't get changed do to an unfortunate getParameters call
    assert(self.params:storage() == self.model:parameters()[1]:storage())

    timer:reset()
    dataTimer:reset()
  end
  
  local log = (' * Finished epoch # %d     Err %1.4f\n'):format(epoch, lossSum / N)
  self.logger.train:write(log .. '\n')
  print(log)
  
  if self.opt.visTrain > 0 then
    local visDir = paths.concat(self.opt.www, 'train_' .. epoch)
    pl.dir.makepath(visDir)
    wut.renderHtml(visDir, visIms, {}, self.opt.visWidth)
  end

  return lossSum / N
end

function Trainer:test(epoch, dataloader)
  -- Computes the top-1 and top-5 err on the validation set

  local timer = torch.Timer()
  local dataTimer = torch.Timer()
  local size = dataloader:size()

  local lossSum = 0.0
  local evalSum = 0.0
  local N = 0

  local visIms = {}

  self.model:evaluate()
  for n, sample in dataloader:run() do
    if self.opt.debug and n >= 10 then 
      break
    end
    
    local dataTime = dataTimer:time().real

    -- Copy input and target to the GPU
    self:copyInputs(sample)

    local loss = 0
    local eval = 0
    if self.input:dim() ~= 0 then
      self.output = self.model:forward(self.input)
      loss = self.criterion:forward(self.output, self.target)
    end

    lossSum = lossSum + loss
    N = N + 1

    local log = ('Test: [%d][%d/%d] Time %.3f Data %.3f Err %1.4f')
      :format(epoch, n, size, timer:time().real, dataTime, loss)
    self.logger.val:write(log .. '\n')
    ut.progress(n, size, log)

    if N <= self.opt.visTest and self.input:dim() ~= 0 then
      self:visualize(visIms, dataloader)
    end

    timer:reset()
    dataTimer:reset()
  end
  self.model:training()

  local log = (' * Finished epoch # %d     Err %1.4f\n'):format(epoch, lossSum / N)
  self.logger.val:write(log .. '\n')
  print(log)

  if self.opt.visTest > 0 then
    local visDir = paths.concat(self.opt.www, epoch)
    pl.dir.makepath(visDir)
    wut.renderHtml(visDir, visIms, {}, self.opt.visWidth)
  end

  return lossSum / N
end

function Trainer:sanitize()
  nnut.sanitize(self.model)
  nnut.sanitizeCriterion(self.criterion)
end

function Trainer:copyInputs(sample)
  -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
  -- if using DataParallelTable. The target is always copied to a CUDA tensor
  assert(self.opt.batchSize == 1, 'only batchsize 1 supported')
  self.input = self.input or (self.opt.nGPU == 1
    and torch.CudaTensor()
    or cutorch.createCudaHostTensor())
  local nMasks = sample.mask:max()
  self.input:resize(nMasks, table.unpack(sample.img[1]:size():totable())):fill(0)

  if nMasks == 0 then
    return
  end

  self.target = self.target or {}
  self.target[1] = self.target[1] or torch.CudaTensor()
  self.target[1]:resize(nMasks, sample.xml[1]:size(2)):copy(sample.xml[1][{{1, nMasks}}])
  self.target[2] = self.target[2] or {}
  self.target[2][1] = self.target[2][1] or torch.CudaTensor()
  self.target[2][1]:resize(nMasks, table.unpack(sample.ori[1]:size():totable())):fill(0)
  self.target[2][2] = self.target[2][2] or torch.CudaTensor()
  self.target[2][2]:resize(nMasks, table.unpack(sample.oriMask[1]:size():totable())):fill(0)
  
  for i = 1, sample.mask:max() do
    local mask = sample.mask:eq(i)
    self.input[i]:maskedCopy(mask:cuda(), sample.img:maskedSelect(mask):cuda()) 
    self.target[2][1][i]:resize(sample.ori:size()):copy(sample.ori)
    self.target[2][2][i] = sample.oriMask:eq(0) + sample.oriMask:eq(i)
  end
end

function Trainer:learningRate(epoch)
  -- Training schedule
  local decay = 0
  if self.opt.LRDecay == 'stepwise' then
    decay = math.floor((epoch - 1) / self.opt.LRDParam)
  elseif self.opt.LRDecay == 'pow' then
    decay = (epoch - 1) * self.opt.LRDParam
  end
  return self.opt.LR * math.pow(0.1, decay)
end

function Trainer:visualize(visIms, dataloader)
  table.insert(visIms, dataloader.dataset:postprocess()(sample.img[1]))
  table.insert(visIms, dataloader.dataset:renderAll(
  dataloader.dataset:postprocessXml()(self.output[1]:float())))
  for i = 1, self.opt.visWidth - 2 do
    table.insert(visIms, torch.ones(3, 200, 200))
  end

  for i = 1, (#self.input)[1] do
    table.insert(visIms, dataloader.dataset:postprocess()(self.input[i]:float()))
    if self.opt.debug then
      table.insert(visIms, dataloader.dataset:renderSingle(
      dataloader.dataset:postprocessXml()(self.target[1][{{i}}]:float())))
      table.insert(visIms, self.target[2][1][i]:float())
      local vis = self.target[2][1][i]:clone()
      local mask = 1 - self.target[2][2][i]
      vis:maskedFill(mask, 0)
      table.insert(visIms, vis:float())
    end
    if self.opt.deLoss then
      table.insert(visIms, dataloader.dataset:renderSingle(
      dataloader.dataset:postprocessXml()(self.output[1][{{i}}]:float())))
    end
    if self.opt.reLoss then
      table.insert(visIms, self.output[2][1][i]:float()) --already rendered now
      local vis = self.output[2][1][i]:clone()
      local mask = 1 - self.target[2][2][i]
      vis:maskedFill(mask, 0)
      table.insert(visIms, vis:float())
    end
  end

  for i = 1, self.opt.visWidth - ((#self.input)[1] * self.opt.visPerInst) % self.opt.visWidth do
    table.insert(visIms, torch.ones(3, 200, 200))
  end
end 

return M.Trainer
