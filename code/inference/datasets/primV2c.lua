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
local PrimV2cDataset = torch.class('PrimV2cDataset', M)

function PrimV2cDataset:__init(dataInfo, opt, split)
  self.dataInfo = dataInfo[split]
  self.opt = opt
  self.split = split
  self.dir = dataInfo.basedir
  -- self.shapeAttr = dataInfo.shapeAttr
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

-- function PrimV2cDataset:get(i, loadAudio)
function PrimV2cDataset:get(i, loadAudio)
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

function PrimV2cDataset:size()
  return self.dataInfo.dataPath:size(1)
end


function PrimV2cDataset:preprocessAudio()
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
      PrimV2cDataset.soundnetView(),
    }
  elseif self.split == 'val' then
    return t.Compose{
      t.SetChannel(),
      t.AudioNormalize(),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      PrimV2cDataset.soundnetView(),
    }
  else
    error('invalid split: ' .. self.split)
  end
end

function PrimV2cDataset:postprocessAudio()
  return t.Compose{
    PrimV2cDataset.wavView(),
    t.AudioUnnormalize(),
  }
end

function PrimV2cDataset:soundnetView()
  return function(input)
    return input:view(1, -1, 1)
  end
end

function PrimV2cDataset:wavView()
  return function(input)
    return input:view(-1, 1)
  end
end

-- function PrimV2cDataset:preprocessMtl() -- ??????????????????
--   return function(input)
--     local processed = torch.zeros(self.opt.outputSize)
--     for i = 1, (#self.opt.outputSplitSize)[1] do 
--       processed[self.opt.outputSplitPSum[i] + input[i * 2 + 1] + 1] = 1
--     end

--     return processed
--   end
-- end

-- function PrimV2cDataset:postprocessMtl()
--   return function(input)
--     local num = (#self.opt.outputSplitSize)[1]
--     local processed = torch.zeros(num)
--     for i = 1, num do
--       _, processed[i] = torch.max(input[{{self.opt.outputSplitPSum[i] + 1, 
--         self.opt.outputSplitPSum[i + 1]}}], 1)
--       processed[i] = processed[i] - 1
--     end

--     return processed
--   end
-- end


function PrimV2cDataset:preprocessXml()
  return function(input)
    local processed = torch.zeros(self.opt.outputSize)
    local a = {-1,-0.6,-0.2,0.2,0.6,1}
    processed[1] = a[input[1]+1]
    a = {-1,-2/3,-1/3,0,1/3,2/3,1}
    processed[2] = a[input[2]+1]
    a = {-1,-0.6,-0.2,0.2,0.6,1}
    processed[3] = a[input[3]+1]
    a = {-1,-0.6,-0.2,0.2,0.6,1}
    processed[4] = a[input[4]+1]
    a = {-1,-0.6,-0.2,0.2,0.6,1}
    processed[5] = a[input[5]+1]

    -- processed[input[1] - 999] = 1    -- get scene id: 1000 -> 1, one-hot encoding
    -- for i = 2, (#input)[1] do
    --   if input[i] < 100 then   -- < 100 means it is not scene id
    --     processed[self.opt.outputSplitPSum[i] + input[i] + 1] = 1
    --   else
    --     processed[self.opt.outputSplitPSum[i] + 101] = 1
    --   end
    -- end

    return processed
  end
end

function PrimV2cDataset:postprocessXml()
  return function(input)
    local processed = torch.zeros(self.opt.outputSize)
    -- local a = {-1,-0.6,-0.2,0.2,0.6,1}
    -- processed[1] = a[math.floor(input[1]+0.5)+1]
    -- a = {-1,-2/3,-1/3,0,1/3,2/3,1}
    -- processed[2] = a[math.floor(input[2]+0.5)+1]
    -- a = {-1,-0.6,-0.2,0.2,0.6,1}
    -- processed[3] = a[math.floor(input[3]+0.5)+1]
    -- a = {-1,-0.6,-0.2,0.2,0.6,1}
    -- processed[4] = a[math.floor(input[4]+0.5)+1]
    -- a = {-1,-0.6,-0.2,0.2,0.6,1}
    -- processed[5] = a[math.floor(input[5]+0.5)+1]

    processed[1] = input[1]
    processed[2] = input[2]
    processed[3] = input[3]
    processed[4] = input[4]
    processed[5] = input[5]


    -- local num = (#self.opt.outputSplitSize)[1]
    -- local processed = torch.zeros(num)
    -- for i = 1, num do
    --   _, processed[i] = torch.max(input[{{self.opt.outputSplitPSum[i] + 1, 
    --     self.opt.outputSplitPSum[i + 1]}}], 1)
    --   if i == 1 then
    --     processed[i] = processed[i] + 999
    --   else
    --     processed[i] = processed[i] - 1
    --   end
    -- end

    return processed
  end
end

-- function PrimV2cDataset:preprocessAttr()
--   return function(xml, attr)
--     local processed = torch.zeros(self.opt.outputSize)
--     processed[xml[1] - 999] = 1    -- get scene id: 1000 -> 1, one-hot encoding
--     for i = 1, (#attr)[1] do
--       processed[{{self.opt.outputSplitPSum[i + 1] + 1, self.opt.outputSplitPSum[i + 2]}}] = attr[i]
--     end

--     return processed
--   end
-- end

-- function PrimV2cDataset:postprocessAttr()
--   return function(input)
--     local num = (#self.opt.outputSplitSize)[1]
--     local scene = torch.zeros(1)
--     local attr = torch.zeros(num - 1, self.opt.numAttr)

--     _, scene = torch.max(input[{{self.opt.outputSplitPSum[1] + 1, 
--       self.opt.outputSplitPSum[2]}}], 1)
--     scene = scene + 999
--     for i = 2, num do
--       attr[i - 1]  = torch.round(input[{{self.opt.outputSplitPSum[i] + 1, self.opt.outputSplitPSum[i + 1]}}])
--     end

--     return scene, attr
--   end
-- end

-- function PrimV2cDataset:preprocessAS()
--   return function(attr)
--     local processed = torch.zeros(self.opt.outputSize)
--     for i = 1, (#attr)[1] do
--       processed[{{self.opt.outputSplitPSum[i] + 1, self.opt.outputSplitPSum[i + 1]}}] = attr[i]
--     end

--     return processed
--   end
-- end

-- function PrimV2cDataset:postprocessAS()
--   return function(input)
--     local num = (#self.opt.outputSplitSize)[1]
--     local attr = torch.zeros(num, self.opt.numAttr)

--     for i = 1, num do
--       attr[i]  = torch.round(input[{{self.opt.outputSplitPSum[i] + 1, self.opt.outputSplitPSum[i + 1]}}])
--     end

--     return attr
--   end
-- end


return M.PrimV2cDataset
