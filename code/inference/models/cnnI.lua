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
require 'cudnn'

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
      -- input is (nc) x 256 x 256
      model:add(Convolution(nc, ndf, 5, 5, 4, 4, 2, 2))
      model:add(SBatchNorm(ndf)):add(nn.LeakyReLU(0.2, true))
      -- state size: (ndf) x 64 x 64
      model:add(Convolution(ndf, ndf * 2, 5, 5, 2, 2, 2, 2))
      model:add(SBatchNorm(ndf * 2)):add(nn.LeakyReLU(0.2, true))
      -- state size: (ndf*2) x 32 x 32
      model:add(Convolution(ndf * 2, ndf * 4, 5, 5, 2, 2, 1, 1))
      model:add(SBatchNorm(ndf * 4)):add(nn.LeakyReLU(0.2, true))
      -- state size: (ndf*4) x 14 x 14
      model:add(Convolution(ndf * 4, ndf * 8, 5, 5, 2, 2, 1, 1))
      model:add(SBatchNorm(ndf * 8)):add(nn.LeakyReLU(0.2, true))
      -- state size: (ndf*8) x 5 x 5
      model:add(Convolution(ndf * 8, ndf * 8, 3, 3, 1, 1))
      model:add(SBatchNorm(ndf * 8)):add(nn.LeakyReLU(0.2, true))
      -- state size: (ndf*8) x 5 x 5
      model:add(Convolution(ndf * 8, ndf * 8, 1, 1, 1, 1))
      model:add(SBatchNorm(ndf * 8)):add(nn.LeakyReLU(0.2, true))
      -- state size: (ndf*8) x 5 x 5
      model:add(Convolution(ndf * 8, ndf * 2, 1, 1, 1, 1))
      model:add(SBatchNorm(ndf * 2)):add(nn.LeakyReLU(0.2, true))
      -- state size: (ndf*2) x 5 x 5
      model:add(Linear(ndf * 50, opt.outputSize))
      -- state size: (18) x 50
    elseif opt.netSpec:startswith('res') then
      if opt.netSpec == 'res18' then
        model = nnut.loadResNet18()
        model:remove()
        model:add(Linear(512, opt.outputSize))
      elseif opt.netSpec == 'res34' then
        model = nnut.loadResNet34()
        model:remove()
        model:add(Linear(512, opt.outputSize))
      elseif opt.netSpec == 'res50' then
        model = nnut.loadResNet50()
        model:remove()
        model:add(Linear(2048, opt.outputSize))
      elseif opt.netSpec == 'res101' then
        model = nnut.loadResNet101()
        model:remove()
        model:add(Linear(2048, opt.outputSize))
      elseif opt.netSpec == 'res200' then
        model = nnut.loadResNet200()
        model:remove()
        model:add(Linear(2048, opt.outputSize))
      end 
    end

    if opt.pretrain == 'none' then
      init(model, opt) 
    end
  end

  cudnnize(model, opt) 
  if opt.debug then
    nnut.dumpNetwork(model, torch.zeros(batchSize, 3, imgDim, imgDim):cuda())
  end

  return model
end

return createModel
