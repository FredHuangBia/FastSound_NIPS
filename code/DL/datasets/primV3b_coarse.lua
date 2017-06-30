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

-- local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

-- require locally for threads
package.path = './util/lua/?.lua;' .. package.path
local ut = require 'utils'

local M = {}
local PrimV3b_coarseDataset = torch.class('PrimV3b_coarseDataset', M)

function PrimV3b_coarseDataset:__init(dataInfo, opt, split)
  self.dataInfo = dataInfo[split]
  self.opt = opt
  self.split = split
  self.dir = dataInfo.basedir
  -- self.shapeAttr = dataInfo.shapeAttr
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

-- function PrimV3b_coarseDataset:get(i, loadAudio)

function PrimV3b_coarseDataset:get(i, loadAudio) -- return audio xml xmlLen
  local path = ffi.string(self.dataInfo.dataPath[i]:data())  

  -- load audio
  local audio = nil
  if loadAudio then
    audio = torch.load(paths.concat(self.dir, path, 'merged.t7'))
  end

  -- load scene
  local xml = self.dataInfo.xml[i]
  local xmlLen = self.dataInfo.xmlLen[i]

  return {
    -- img = img,
    audio = audio,
    xml = xml,
    xmlLen = xmlLen,
    -- attr = attr,
  }
end

function PrimV3b_coarseDataset:size()
  return self.dataInfo.dataPath:size(1)
end


function PrimV3b_coarseDataset:preprocessAudio()
  if self.split == 'train' then
    return t.Compose{
      t.SetChannel(),
      -- t.AudioScale(0.5, 2),
      -- t.AudioTranslate(-0.5, 0.5, self.opt.audioRate),
      t.AudioNormalize(),
      t.AmpNormalize(2),
      t.AudioJitter(0.0001),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      PrimV3b_coarseDataset.soundnetView(),
    }
  elseif self.split == 'val' then
    return t.Compose{
      t.SetChannel(),
      t.AudioNormalize(),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      PrimV3b_coarseDataset.soundnetView(),
    }
  else
    error('invalid split: ' .. self.split)
  end
end

function PrimV3b_coarseDataset:postprocessAudio()
  return t.Compose{
    PrimV3b_coarseDataset.wavView(),
    t.AudioUnnormalize(),
  }
end

function PrimV3b_coarseDataset:soundnetView()
  return function(input)
    return input:view(1, -1, 1)
  end
end

function PrimV3b_coarseDataset:wavView()
  return function(input)
    return input:view(-1, 1)
  end
end


function PrimV3b_coarseDataset:preprocessXml()
  return function(input)
    local processed = torch.zeros(self.opt.outputSize)
    processed[input[1]] = 1        -- shape
    processed[5+input[2]] = 1    -- height  
    processed[9+input[3]] = 1    -- alpha
    processed[11+input[4]] = 1      -- restitution    
    return processed
  end
end

function PrimV3b_coarseDataset:postprocessXml()
  return function(input)

    local processed = torch.zeros(4)

    local shape = torch.zeros(5)
    local height = torch.zeros(4)
    local alpha = torch.zeros(2)
    local restitution = torch.zeros(2)

    for i = 1,5 do
      shape[i] = input[i]
    end

    for i = 1,4 do
      height[i] = input[5+i]
    end

    for i = 1,2 do
      alpha[i] = input[9+i]
      restitution[i] = input[11+i]
    end
    -- print(shape)
    -- print(argmax_1D(shape))
    processed[1] = argmax_1D(shape)
    processed[2] = argmax_1D(height)
    processed[3] = argmax_1D(alpha)
    processed[4] = argmax_1D(restitution)
    return processed

  end
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

return M.PrimV3b_coarseDataset
