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
local Phys101aDataset = torch.class('Phys101aDataset', M)

function Phys101aDataset:__init(dataInfo, opt, split)
  self.dataInfo = dataInfo[split]
  self.opt = opt
  self.split = split
  self.dir = dataInfo.basedir
  self.shapeAttr = dataInfo.shapeAttr
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function Phys101aDataset:get(i, loadImg, loadAudio, loadSil, loadMfcc)
  local path = ffi.string(self.dataInfo.dataPath[i]:data())
  
  -- load image
  local img = nil
  if loadImg then
    local imgPath = paths.concat(self.dir, path, 'render', string.format('%03d.jpg', math.random(1, 25)))
    while lfs.attributes(imgPath, 'mode') ~= 'file' do
      imgPath = paths.concat(self.dir, path, 'render', string.format('%03d.jpg', math.random(1, 25)))
    end
    img = ut.loadImage(imgPath)
  end

  -- load audio
  local audio = nil
  if loadAudio then
    audio = torch.load(paths.concat(self.dir, path, 'audio.t7'))
  end

  -- load sil
  local sil = nil
  if loadSil then
    sil = ut.loadImage(paths.concat(self.dir, path, 'matting', string.format('%03d.png', math.random(1, 20))))
    testSil = ut.loadImage(paths.concat(self.dir, path, 'matting', 'Kinect_RGB_1', '020.png'))
    if sil:dim() == 2 then
      sil = sil:view(1, (#sil)[1], (#sil)[2])
      testSil = testSil:view(1, (#testSil)[1], (#testSil)[2])
    end
    if (#sil)[1] == 1 then
      sil = sil:expand(3, (#sil)[2], (#sil)[3])
      testSil = sil:expand(3, (#testSil)[2], (#testSil)[3])
    end
  end

  -- load mfcc
  local mfcc = nil
  if loadMfcc then
    mfcc = mat.load(paths.concat(self.dir, path, 'mfcc.mat'))
    mfcc = mfcc.c:view(-1)
  end

  -- load scene
  local xml = self.dataInfo.xml[i]
  local xmlLen = self.dataInfo.xmlLen[i]

  -- load shapeAttr
  local num = math.floor((#xml)[1] / 2)
  local attr = torch.zeros(num, (#self.shapeAttr)[2])
  for i = 1, num do
    if xml[i * 2] < self.opt.numObjId then
      attr[i] = self.shapeAttr[xml[i * 2]]
    end
  end

  return {
    img = img,
    audio = audio,
    xml = xml,
    xmlLen = xmlLen,
    attr = attr,
    sil = sil,
    testSil = testSil,
    mfcc = mfcc,
  }
end

function Phys101aDataset:size()
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

function Phys101aDataset:preprocess()
  if self.split == 'train' then
    return t.Compose{
      t.Scale(self.opt.imgDim, 'max'),
      t.Pad(self.opt.imgDim),
      t.CenterCrop(self.opt.imgDim),
--      t.ColorJitter({
--        brightness = 0.4,
--        contrast = 0.4,
--        saturation = 0.4,
--      }),
--      t.Lighting(0.1, pca.eigval, pca.eigvec),
      t.ColorNormalize(meanstd),
--      t.HorizontalFlip(0.5),
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

function Phys101aDataset:postprocess()
  return t.Compose{
    t.ColorUnnormalize(meanstd),
  }
end

function Phys101aDataset:preprocessSil()
  if self.split == 'train' then
    return t.Compose{
      t.Scale(self.opt.imgDim, 'max'),
      t.Pad(self.opt.imgDim),
      t.CenterCrop(self.opt.imgDim),
    }
  elseif self.split == 'val' then
    return t.Compose{
      t.Scale(self.opt.imgDim, 'max'),
      t.Pad(self.opt.imgDim),
      t.CenterCrop(self.opt.imgDim),
    }
  else
    error('invalid split: ' .. self.split)
  end
end

function Phys101aDataset:postprocessSil()
  return t.Compose{
  }
end

function Phys101aDataset:preprocessAudio()
  if self.split == 'train' then
    return t.Compose{
      t.SetChannel(),
      t.AudioScale(0.5, 2),
      t.AudioTranslate(0, 0.5, self.opt.audioRate),
      t.AudioNormalize(),
      t.AmpNormalize(2),
      t.AudioJitter(0.01),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      Phys101aDataset.soundnetView(),
    }
  elseif self.split == 'val' then
    return t.Compose{
      t.SetChannel(),
      t.AudioNormalize(),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      Phys101aDataset.soundnetView(),
    }
  else
    error('invalid split: ' .. self.split)
  end
end

function Phys101aDataset:postprocessAudio()
  return t.Compose{
    Phys101aDataset.wavView(),
    t.AudioUnnormalize(),
  }
end

function Phys101aDataset:soundnetView()
  return function(input)
    return input:view(1, -1, 1)
  end
end

function Phys101aDataset:wavView()
  return function(input)
    return input:view(-1, 1)
  end
end

function Phys101aDataset:preprocessMtl()
  return function(input)
    local processed = torch.zeros(self.opt.outputSize)
    for i = 1, (#self.opt.outputSplitSize)[1] do 
      processed[self.opt.outputSplitPSum[i] + input[i * 2 + 1] + 1] = 1
    end

    return processed
  end
end

function Phys101aDataset:postprocessMtl()
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

function Phys101aDataset:preprocessXml()
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

function Phys101aDataset:postprocessXml()
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

function Phys101aDataset:preprocessAttr()
  return function(xml, attr)
    local processed = torch.zeros(self.opt.outputSize)
    processed[xml[1] - 999] = 1    -- get scene id: 1000 -> 1, one-hot encoding
    for i = 1, (#attr)[1] do
      processed[{{self.opt.outputSplitPSum[i + 1] + 1, self.opt.outputSplitPSum[i + 2]}}] = attr[i]
    end

    return processed
  end
end

function Phys101aDataset:postprocessAttr()
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

function Phys101aDataset:preprocessAS()
  return function(attr)
    local processed = torch.zeros(self.opt.outputSize)
    for i = 1, (#attr)[1] do
      processed[{{self.opt.outputSplitPSum[i] + 1, self.opt.outputSplitPSum[i + 1]}}] = attr[i]
    end

    return processed
  end
end

function Phys101aDataset:postprocessAS()
  return function(input)
    local num = (#self.opt.outputSplitSize)[1]
    local attr = torch.zeros(num, self.opt.numAttr)

    for i = 1, num do
      attr[i]  = torch.round(input[{{self.opt.outputSplitPSum[i] + 1, self.opt.outputSplitPSum[i + 1]}}])
    end

    return attr
  end
end

return M.Phys101aDataset
