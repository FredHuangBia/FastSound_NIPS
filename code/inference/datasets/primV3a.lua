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
local PrimV3aDataset = torch.class('PrimV3aDataset', M)

function PrimV3aDataset:__init(dataInfo, opt, split)
  self.dataInfo = dataInfo[split]
  self.opt = opt
  self.split = split
  self.dir = dataInfo.basedir
  -- self.shapeAttr = dataInfo.shapeAttr
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

-- function PrimV3aDataset:get(i, loadAudio)

function PrimV3aDataset:get(i, loadAudio) -- return audio xml xmlLen
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

function PrimV3aDataset:size()
  return self.dataInfo.dataPath:size(1)
end


function PrimV3aDataset:preprocessAudio()
  if self.split == 'train' then
    return t.Compose{
      t.SetChannel(),
      -- t.AudioScale(0.5, 2),
      -- t.AudioTranslate(-0.5, 0.5, self.opt.audioRate),
      t.AudioNormalize(),
      t.AmpNormalize(2),
      t.AudioJitter(0.001),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      PrimV3aDataset.soundnetView(),
    }
  elseif self.split == 'val' then
    return t.Compose{
      t.SetChannel(),
      t.AudioNormalize(),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      PrimV3aDataset.soundnetView(),
    }
  else
    error('invalid split: ' .. self.split)
  end
end

function PrimV3aDataset:postprocessAudio()
  return t.Compose{
    PrimV3aDataset.wavView(),
    t.AudioUnnormalize(),
  }
end

function PrimV3aDataset:soundnetView()
  return function(input)
    return input:view(1, -1, 1)
  end
end

function PrimV3aDataset:wavView()
  return function(input)
    return input:view(-1, 1)
  end
end


function PrimV3aDataset:preprocessXml()
  return function(input)
    local processed = torch.zeros(self.opt.outputSize)
    processed[input[1]] = 1   -- rotation
    processed[65] = input[2]  -- height
    processed[66] = input[3]  -- alpha
    processed[67] = input[4]  -- beta
    processed[68] = input[5]  -- restitution
    return processed
  end
end

function PrimV3aDataset:postprocessXml()
  return function(input)
    local processed = torch.zeros(5)
    local rotation=torch.zeros(64)
    for i = 1,64 do
      rotation[i] = input[i]
    end
    processed[1] = argmax_1D(rotation)
    processed[2] = input[65]
    processed[3] = input[66]
    processed[4] = input[67]
    processed[5] = input[68]
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




return M.PrimV3aDataset
