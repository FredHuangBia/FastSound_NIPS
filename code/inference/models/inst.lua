--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'
require 'rnn'

local Linear = nn.Linear
local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local LSTM = cudnn.LSTM

local function ConvInit(model, name)
  for k, v in pairs(model:findModules(name)) do
    local n = v.kW * v.kH * v.nOutputPlane
    v.weight:normal(0, math.sqrt(2 / n))
    if cudnn.version >= 4000 then
      v.bias = nil
      v.gradBias = nil
    else
      v.bias:zero()
    end
  end
end

local function BNInit(model, name)
  for k, v in pairs(model:findModules(name)) do
    v.weight:fill(1)
    v.bias:zero()
  end
end

local function init(model, opt)
  ConvInit(model, 'cudnn.SpatialConvolution')
  ConvInit(model, 'nn.SpatialConvolution')
  BNInit(model, 'fbnn.SpatialBatchNormalization')
  BNInit(model, 'cudnn.SpatialBatchNormalization')
  BNInit(model, 'nn.SpatialBatchNormalization')
  for k, v in pairs(model:findModules('nn.Linear')) do
    v.bias:zero()
  end
end

local function cudnnize(model, opt)
  model:cuda()
  cudnn.convert(model, cudnn)

  if opt.cudnn == 'deterministic' then
    model:apply(function(m)
      if m.setMode then m:setMode(1,1,1) end
    end)
  end
end

local function createModel(opt, dataLoader)

  local batchSize = opt.batchSize
  local imgDim = opt.imgDim

  local nc = opt.nc or 3
  local ndf = opt.ndf or 64
  local ngf = opt.ngf or 64

  -- input is (nc) x 256 x 256
  local model 
  
  if opt.pretrain == 'none' then
    model = nn.Sequential()
    model:add(Convolution(nc, ndf, 5, 5, 4, 4, 2, 2))
    model:add(SBatchNorm(ndf)):add(ReLU(true))
    -- state size: (ndf) x 64 x 64
    model:add(Convolution(ndf, ndf * 2, 5, 5, 2, 2, 2, 2))
    model:add(SBatchNorm(ndf * 2)):add(ReLU(true))
    -- state size: (ndf*2) x 32 x 32
    model:add(Convolution(ndf * 2, ndf * 4, 5, 5, 2, 2, 2, 2))
    model:add(SBatchNorm(ndf * 4)):add(ReLU(true))
    -- state size: (ndf*4) x 16 x 16
    model:add(Convolution(ndf * 4, ndf * 8, 5, 5, 2, 2, 2, 2))
    model:add(SBatchNorm(ndf * 8)):add(ReLU(true))
    -- state size: (ndf*8) x 8 x 8
    model:add(Convolution(ndf * 8, ndf * 8, 5, 5, 2, 2, 2, 2))
    model:add(SBatchNorm(ndf * 8)):add(ReLU(true))
    -- state size: (ndf*8) x 4 x 4
    model:add(Convolution(ndf * 8, ndf * 8, 3, 3, 1, 1))
    model:add(SBatchNorm(ndf * 8)):add(ReLU(true))
    -- state size: (ndf*8) x 2 x 2
    model:add(Convolution(ndf * 8, ndf * 4, 1, 1, 1, 1))
    model:add(SBatchNorm(ndf * 4)):add(ReLU(true))
    -- state size: (ndf*4) x 2 x 2
    model:add(Convolution(ndf * 4, ndf, 1, 1, 1, 1))
    model:add(SBatchNorm(ndf))
    -- state size: (ndf) x 2 x 2
    model:add(nn.View(1, -1):setNumInputDims(3))

    init(model, opt)
    cudnnize(model, opt)
  elseif opt.pretrain:startswith('res') then
    if opt.pretrain == 'res18' then
      model = nnut.loadResNet18()
      model:remove()
      model:add(Linear(512, opt.outputSize))
    elseif opt.pretrain == 'res34' then
      model = nnut.loadResNet34()
      model:remove()
      model:add(Linear(512, opt.outputSize))
    elseif opt.pretrain == 'res50' then
      model = nnut.loadResNet50()
      model:remove()
      model:add(Linear(2048, opt.outputSize))
    elseif opt.pretrain == 'res101' then
      model = nnut.loadResNet101()
      model:remove()
      model:add(Linear(2048, opt.outputSize))
    elseif opt.pretrain == 'res200' then
      model = nnut.loadResNet200()
      model:remove()
      model:add(Linear(2048, opt.outputSize))
    end 
  end

  local renderer = nn.Sequential()
  
  local decompose = nn.ConcatTable()
  for i = 1, (#opt.outputSplitSize)[1] do
    decompose:add(nn.Narrow(2, opt.outputSplitPSum[i] + 1, opt.outputSplitSize[i]))
  end
  renderer:add(decompose)

  local reinforce = nn.ParallelTable()
  for i = 1, (#opt.outputSplitType)[1] do
    if opt.outputSplitType[i] == 1 then
      reinforce:add(nn.Sequential():add(nn.SoftMax()):add(nn.ReinforceCategorical()))
    elseif opt.outputSplitType[i] == 2 then
      reinforce:add(nn.Sequential():add(nn.Clamp(0, 1)):add(nn.ReinforceNormal(opt.locationStd)))
    elseif opt.outputSplitType[i] == 3 then
      reinforce:add(nn.Sequential():add(nn.Clamp(0, 1)):add(nn.ReinforceBernoulli()))
    end
  end
  renderer:add(reinforce)

  renderer:add(nn.JoinTable(2))
  local dataset = dataLoader.dataset -- for saving
  renderer:add(nn.Call(function (x) return dataset:renderEach(x):cuda() end))

  local baseline = nn.Sequential()
  baseline:add(nn.Constant(1, 1))
  baseline:add(nn.Add(1))
  
  local concat = nn.ConcatTable():add(renderer):add(baseline)
  
  local concat2 = nn.ConcatTable()
  concat2:add(nn.Identity())
  if opt.reLoss then
    concat2:add(concat)
  else
    concat2:add(nn.Identity())
  end
  model:add(concat2)

  init(model, opt) 
  cudnnize(model, opt) 
  if opt.debug then
    nnut.dumpNetwork(model, torch.zeros(batchSize, 3, imgDim, imgDim):cuda())
  end

  return model 
end

return createModel
