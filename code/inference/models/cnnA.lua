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

local Linear = nn.Linear
local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

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
      if m.setMode then m:setMode(1, 1, 1) end
    end)
  end
end

local function createModel(opt)

  local model = nn.Sequential()
  local batchSize = opt.batchSize
  local imgDim = opt.imgDim

  local nc = opt.nc or 3
  local ndf = opt.ndf or 64
  local ngf = opt.ngf or 64

  if opt.pretrain ~= 'none' and opt.pretrain ~= 'default' then
    model = torch.load(path.join(opt.gen, '..', opt.pretrain, 
                       opt.netType .. '_' .. opt.netSpec .. '.t7'))
  else
    if opt.netSpec == 'custom' then
    elseif opt.netSpec == 'soundnet5' then
    elseif opt.netSpec == 'soundnet8' then
      model = nnut.loadSoundNet8()
      while #model.modules > 24 do
        model:remove()
      end
      model:add(nn.SpatialAdaptiveMaxPooling(1, 1))
      model:add(nn.View(-1):setNumInputDims(3))
      model:add(Linear(1024, opt.outputSize))
      -- print(model:get(1).weight)
    end

    if opt.pretrain == 'none' then
      init(model, opt) 
      -- print(model:get(1).weight)
    end
  end

  cudnnize(model, opt) 
  if opt.debug then
    nnut.dumpNetwork(model, torch.zeros(opt.batchSize, 1, opt.audioDim, 1):cuda())
  end

  return model
end

return createModel
