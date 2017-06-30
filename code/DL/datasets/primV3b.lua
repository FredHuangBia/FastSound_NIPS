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
local PrimV3bDataset = torch.class('PrimV3bDataset', M)

function PrimV3bDataset:__init(dataInfo, opt, split)
  self.dataInfo = dataInfo[split]
  self.opt = opt
  self.split = split
  self.dir = dataInfo.basedir
  -- self.shapeAttr = dataInfo.shapeAttr
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

-- function PrimV3bDataset:get(i, loadAudio)

function PrimV3bDataset:get(i, loadAudio) -- return audio xml xmlLen
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

function PrimV3bDataset:size()
  return self.dataInfo.dataPath:size(1)
end


function PrimV3bDataset:preprocessAudio()
  if self.split == 'train' then
    if self.opt.netType=='cnnAE'then
      return t.Compose{
        t.SetChannel(),
        -- t.AudioScale(0.5, 2),
        -- t.AudioTranslate(-0.5, 0.5, self.opt.audioRate),
        t.AudioNormalize(),
        t.AmpNormalize(2),
        t.AudioJitter(0.0001),
        t.AudioHeadCrop(self.opt.audioDim),
        t.AmpNormalize(2),
        t.ScaleAudio(16384),
        PrimV3bDataset.soundnetView(),
      }
    else
      return t.Compose{
        t.SetChannel(),
        -- t.AudioScale(0.5, 2),
        -- t.AudioTranslate(-0.5, 0.5, self.opt.audioRate),
        t.AudioNormalize(),
        t.AmpNormalize(2),
        t.AudioJitter(0.0001),
        t.AudioHeadCrop(self.opt.audioDim),
        t.AmpNormalize(2),
        PrimV3bDataset.soundnetView(),
      }
    end
  elseif self.split == 'val' then
    if self.opt.netType=='cnnAE'then
      return t.Compose{
        t.SetChannel(),
        t.AudioNormalize(),
        t.AudioHeadCrop(self.opt.audioDim),
        t.AmpNormalize(2),
        t.ScaleAudio(4096),
        PrimV3bDataset.soundnetView(),
      }
    else
      return t.Compose{
        t.SetChannel(),
        t.AudioNormalize(),
        t.AudioHeadCrop(self.opt.audioDim),
        t.AmpNormalize(2),
        PrimV3bDataset.soundnetView(),
      }
    end
  else
    error('invalid split: ' .. self.split)
  end
end

function PrimV3bDataset:postprocessAudio()
  return t.Compose{
    PrimV3bDataset.wavView(),
    t.AudioUnnormalize(),
  }
end

function PrimV3bDataset:soundnetView()
  return function(input)
    return input:view(1, -1, 1)
  end
end

function PrimV3bDataset:wavView()
  return function(input)
    return input:view(-1, 1)
  end
end


function PrimV3bDataset:preprocessXml()
  return function(input)
    local processed = torch.zeros(self.opt.outputSize)
    if self.opt.netType == 'cnnF' then
      processed[input[1]] = 1        -- shape
      processed[14+input[2]] = 1    -- specific    
      processed[24+input[3]] = 1    -- rotation
      processed[89] = input[4]      -- height    
      processed[90] = input[5]      -- alpha
      processed[91] = input[6]      -- beta
      processed[92] = input[7]      -- restitution
    elseif self.opt.netType == 'cnnFS' then
      processed[input[1]] = 1        -- shape
      processed[14+input[2]] = 1    -- specific    
    elseif self.opt.netType == 'cnnAE' then
      processed = processed
    end
    return processed
  end
end

function PrimV3bDataset:postprocessXml()
  return function(input)
    if self.opt.netType == 'cnnF' then
      local processed = torch.zeros(7)

      local shape = torch.zeros(14)
      local specific = torch.zeros(10)
      local rotation = torch.zeros(64)

      for i = 1,14 do
        shape[i] = input[i]
      end

      for i = 1,10 do
        specific[i] = input[14+i]
      end

      for i = 1,64 do
        rotation[i] = input[24+i]
      end
      -- print(shape)
      -- print(argmax_1D(shape))
      processed[1] = argmax_1D(shape)
      processed[2] = argmax_1D(specific)
      processed[3] = argmax_1D(rotation)
      processed[4] = input[89]
      processed[5] = input[90]
      processed[6] = input[91]
      processed[7] = input[92]
      return processed
    elseif self.opt.netType == 'cnnAE' then
      processed = processed
      return processed
    end
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




return M.PrimV3bDataset
