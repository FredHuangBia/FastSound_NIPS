--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Sound V1a dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

-- require locally for threads
package.path = './util/lua/?.lua;' .. package.path
local ut = require 'utils'
local mat = require 'fb.mattorch'

local M = {}
local HitsaDataset = torch.class('HitsaDataset', M)

function HitsaDataset:__init(dataInfo, opt, split)
  self.dataInfo = dataInfo[split]
  self.opt = opt
  self.split = split
  self.dir = dataInfo.basedir
  self.shapeAttr = dataInfo.shapeAttr
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function HitsaDataset:get(i, loadImg, loadAudio, loadSil, loadMfcc)
  local path = ffi.string(self.dataInfo.dataPath[i]:data())
  
  -- load image
  local img = nil
  if loadImg then
    img = ut.loadImage(paths.concat(self.dir, path .. '.jpg'))
  end

  -- load audio
  local audio = nil
  if loadAudio then
    audio = torch.load(paths.concat(self.dir, path .. '.t7'))
  end

  -- load mfcc
  local mfcc = nil
  if loadMfcc then
    mfcc = mat.load(paths.concat(self.dir, path .. '.mat'))
    mfcc = mfcc.c:view(-1)
  end

  -- load scene
  local xml = self.dataInfo.xml[i]
  local xmlLen = self.dataInfo.xmlLen[i]

  return {
    img = img,
    audio = audio,
    xml = xml,
    xmlLen = xmlLen,
    mfcc = mfcc,
  }
end

function HitsaDataset:size()
  if self.dataInfo then
    return self.dataInfo.dataPath:size(1)
  else
    return 0
  end
end

-- Computed from random subset of Imagenet training images
local meanstd = {
  mean = { 0.485, 0.456, 0.406 },
  std = { 0.229, 0.224, 0.225 },
}
local pca = {
  eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
  eigvec = torch.Tensor{
    { -0.5675,  0.7192,  0.4009 },
    { -0.5808, -0.0045, -0.8140 },
    { -0.5836, -0.6948,  0.4203 },
  },
}

function HitsaDataset:preprocess()
  if self.split == 'train' then
    return t.Compose{
      t.RandomSizedCrop(224),
      t.Scale(self.opt.imgDim, 'max'),
      t.Pad(self.opt.imgDim),
      t.CenterCrop(self.opt.imgDim),
      t.ColorJitter({
        brightness = 0.4,
        contrast = 0.4,
        saturation = 0.4,
      }),
      t.Lighting(0.1, pca.eigval, pca.eigvec),
      t.ColorNormalize(meanstd),
      t.HorizontalFlip(0.5),
    }
  elseif self.split == 'val' then
    return t.Compose{
      t.Scale(self.opt.imgDim, 'max'),
      t.Pad(self.opt.imgDim),
      t.CenterCrop(self.opt.imgDim),
      t.ColorNormalize(meanstd),
    }
  else
    error('invalid split: ' .. self.split)
  end
end

function HitsaDataset:postprocess()
  return t.Compose{
    t.ColorUnnormalize(meanstd),
  }
end


function HitsaDataset:preprocessAudio()
  if self.split == 'train' then
    return t.Compose{
      t.SetChannel(),
      t.AudioScale(0.5, 2),
      t.AudioTranslate(-0.5, 0.5, self.opt.audioRate),
      t.AudioNormalize(),
      t.AmpNormalize(2),
      t.AudioJitter(0.01),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      HitsaDataset.soundnetView(),
    }
  elseif self.split == 'val' then
    return t.Compose{
      t.SetChannel(),
      t.AudioNormalize(),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      HitsaDataset.soundnetView(),
    }
  else
    error('invalid split: ' .. self.split)
  end
end

function HitsaDataset:postprocessAudio()
  return t.Compose{
    HitsaDataset.wavView(),
    t.AudioUnnormalize(),
  }
end

function HitsaDataset:soundnetView()
  return function(input)
    return input:view(1, -1, 1)
  end
end

function HitsaDataset:wavView()
  return function(input)
    return input:view(-1, 1)
  end
end

function HitsaDataset:preprocessMtl()
  return function(input)
    local processed = torch.zeros(self.opt.outputSize)
    for i = 1, (#self.opt.outputSplitSize)[1] do 
      processed[self.opt.outputSplitPSum[i] + input[i * 2 + 1] + 1] = 1
    end

    return processed
  end
end

function HitsaDataset:postprocessMtl()
  return function(input)
    local num = (#self.opt.outputSplitSize)[1]
    local processed = torch.zeros(num)
    for i = 1, num do
      _, processed[i] = torch.max(input[{{self.opt.outputSplitPSum[i] + 1, 
        self.opt.outputSplitPSum[i + 1]}}], 1)
      processed[i] = processed[i] - 1
    end

    return processed
  end
end

function HitsaDataset:preprocessXml()
  return function(input)
    local processed = torch.zeros(self.opt.outputSize)
    processed[input[1] - 999] = 1    -- get scene id: 1000 -> 1, one-hot encoding
    for i = 2, (#input)[1] do
      if input[i] < 100 then
        processed[self.opt.outputSplitPSum[i] + input[i] + 1] = 1
      else
        processed[self.opt.outputSplitPSum[i] + 101] = 1
      end
    end

    return processed
  end
end

function HitsaDataset:postprocessXml()
  return function(input)
    local num = (#self.opt.outputSplitSize)[1]
    local processed = torch.zeros(num)
    for i = 1, num do
      _, processed[i] = torch.max(input[{{self.opt.outputSplitPSum[i] + 1, 
        self.opt.outputSplitPSum[i + 1]}}], 1)
      if i == 1 then
        processed[i] = processed[i] + 999
      else
        processed[i] = processed[i] - 1
      end
    end

    return processed
  end
end

function HitsaDataset:preprocessAttr()
  return function(xml, attr)
    local processed = torch.zeros(self.opt.outputSize)
    processed[xml[1] - 999] = 1    -- get scene id: 1000 -> 1, one-hot encoding
    for i = 1, (#attr)[1] do
      processed[{{self.opt.outputSplitPSum[i + 1] + 1, self.opt.outputSplitPSum[i + 2]}}] = attr[i]
    end

    return processed
  end
end

function HitsaDataset:postprocessAttr()
  return function(input)
    local num = (#self.opt.outputSplitSize)[1]
    local scene = torch.zeros(1)
    local attr = torch.zeros(num - 1, self.opt.numAttr)

    _, scene = torch.max(input[{{self.opt.outputSplitPSum[1] + 1, 
      self.opt.outputSplitPSum[2]}}], 1)
    scene = scene + 999
    for i = 2, num do
      attr[i - 1]  = torch.round(input[{{self.opt.outputSplitPSum[i] + 1, self.opt.outputSplitPSum[i + 1]}}])
    end

    return scene, attr
  end
end

return M.HitsaDataset
