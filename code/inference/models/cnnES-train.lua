--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent right0
--
--  The training loop and learning rate schedule
--

local wwwut = require '../util/www_utils'

local optim = require 'optim'

local visepoch = 10
local vissound = 200

local M = {}
local Trainer = torch.class('cnnES.Trainer', M)

local total_test=0
local top5_correct=0
local top1_correct=0

function Trainer:__init(model, criterion, opt, optimState)
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

  local visAudio1s = {}
  local visAudio2s = {}
  local visAudio3s = {}
  local visCaptions = {}

  print('=> Training epoch # ' .. epoch .. ', LR ' .. self.optimState.learningRate)
  -- set the batch norm to training mode
  self.model:training()
  for n, sample in dataloader:run() do  -- a sample is a batch
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
      self:visualize(visAudio1s, visAudio2s, visAudio3s, visCaptions, dataloader)
      -- self:visualize(visAudios, visCaptions, dataloader)
    end

    -- check that the storage didn't get changed do to an unfortunate getParameters call
    assert(self.params:storage() == self.model:parameters()[1]:storage())

    timer:reset()
    dataTimer:reset()
  end

  local log = (' * Finished epoch # %d, Err %1.4f\n'):format(epoch, lossSum / N)
  self.logger.train:write(log .. '\n')
  print(log)

  if self.opt.visTrain > 0 and epoch%visepoch==0 then
    local visDir = paths.concat(self.opt.www, 'train_' .. epoch)
    pl.dir.makepath(visDir)

    local sound = (epoch%vissound == 0)
    wwwut.renderABSHtml( sound, self.opt.dataset, visDir, visAudio1s, visAudio2s, visAudio3s, visCaptions, self.opt.visWidth)
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
  local visAudio1s = {}
  local visAudio2s = {}
  local visAudio3s = {}
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
      self:visualize(visAudio1s, visAudio2s, visAudio3s, visCaptions, dataloader)
    end

    timer:reset()
    dataTimer:reset()
  end
  self.model:training()

  local log = (' * Finished epoch # %d, Err %1.4f\n'):format(epoch, lossSum / N)
  self.logger.val:write(log .. '\n')
  print(log)

  if self.opt.visTest > 0 and epoch%visepoch==0 then
    local visDir = paths.concat(self.opt.www, epoch)
    pl.dir.makepath(visDir)

    local sound = (epoch%vissound == 0)

    wwwut.renderABSHtml(sound, self.opt.dataset, visDir, visAudio1s, visAudio2s, visAudio3s, visCaptions, self.opt.visWidth)
  end

  return lossSum / N
end

function Trainer:sanitize()
  nnut.sanitize(self.model)
  nnut.sanitizeCriterion(self.criterion)
end

function Trainer:copyInputs(sample)
  self.input = self.input or torch.CudaTensor()
  self.input:resize(sample.audio:size()):copy(sample.audio)

  self.target = self.target or torch.CudaTensor()
  self.target:resize(sample.xml:size()):copy(sample.xml)

  self.ID = self.ID or torch.CudaTensor()
  self.ID:resize(sample.ID:size()):copy(sample.ID)
  -- self.img:resize(sample.img:size()):copy(sample.img)

end

function Trainer:learningRate(epoch)
  -- Training schedule
  local decay = 0.001
  if self.opt.dataset == 'imagenet' then
    decay = math.floor((epoch - 1) / 30)
  elseif self.opt.dataset == 'cifar10' then
    decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
  end
  return self.opt.LR * math.pow(0.1, decay)
end

function Trainer:visualize(visAudio1s, visAudio2s, visAudio3s, visCaptions, dataloader)
  for i = 1, (#self.input)[1] do
    -- table.insert(visIms, dataloader.dataset:postprocess()(self.img[i]:float()))
     
    local targetID = self.ID[i]:int()
    table.insert(visAudio1s, ('%.06d'):format(targetID[1])) -- corresponding ID of audio
    table.insert(visAudio2s, 1) -- each 1 means there should be a audio
    table.insert(visAudio3s, 1)
    -- print(self.output[i])

    local targetXml = dataloader.dataset:postprocessXml()(self.target[i]:float())
    local outputXml = dataloader.dataset:postprocessXml()(self.output[i]:float())
    
    total_test = total_test+1
    if targetXml[1]==outputXml[1] then
        top1_correct = top1_correct +1
    end
    
    local output = self.output[i]:float()
    for j = 1,5 do
        local rot = argmax_1D(output)
        if rot ==targetXml[1] then
	    top5_correct = top5_correct+1
        end
        output[rot]=-1
    end

    -- print(self.output[i])
    -- print(outputXml)
    local targetStr = 'target:  '
    local outputStr = 'output: '
    local synthStr =  'synth:  '

    targetStr = targetStr .. ('%.03d'):format(targetXml[1]) .. ' '
    outputStr = outputStr .. ('%.03d'):format(outputXml[1]) .. ' '


    table.insert(visCaptions, targetStr .. '<br/>' .. outputStr ..'<br/>' .. synthStr)
  end
  print('top1: ')
  print(top1_correct/total_test)
  print('top5: ')
  print(top5_correct/total_test)
end

function argmax_1D(v)
   local length = v:size(1)
   assert(length > 0)
   -- examine on average half the entries
   local maxValue = torch.max(v)
   for i = 1, v:size(1) do
      if v[i] == maxValue then
         return i
      end
   end
end



return M.Trainer
