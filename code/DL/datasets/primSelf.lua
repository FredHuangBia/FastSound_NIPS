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
local PrimSelfDataset = torch.class('PrimSelfDataset', M)

function PrimSelfDataset:__init(dataInfo, opt, split)
  self.dataInfo = dataInfo[split]
  self.opt = opt
  self.split = split
  self.dir = dataInfo.basedir
  -- self.shapeAttr = dataInfo.shapeAttr
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

-- function PrimSelfDataset:get(i, loadAudio)

function PrimSelfDataset:get(i, loadAudio) -- return audio xml xmlLen
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

function PrimSelfDataset:size()
  return self.dataInfo.dataPath:size(1)
end


function PrimSelfDataset:preprocessAudio()
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
      PrimSelfDataset.soundnetView(),}
  elseif self.split == 'val' then
    return t.Compose{
      t.SetChannel(),
      t.AudioNormalize(),
      t.AudioHeadCrop(self.opt.audioDim),
      t.AmpNormalize(2),
      PrimSelfDataset.soundnetView(),}
  else
    error('invalid split: ' .. self.split)
  end
end

function PrimSelfDataset:postprocessAudio()
  return t.Compose{
    PrimSelfDataset.wavView(),
    t.AudioUnnormalize(),
  }
end

function PrimSelfDataset:soundnetView()
  return function(input)
    return input:view(1, -1, 1)
  end
end

function PrimSelfDataset:wavView()
  return function(input)
    return input:view(-1, 1)
  end
end


function PrimSelfDataset:preprocessXml()
  return function(input)
    local processed = torch.zeros(self.opt.outputSize)
    if self.opt.netType == 'cnnAll' then
      processed[input[1]] = 1        -- shape
      processed[14+input[2]] = 1    -- specific    
      processed[25] = input[3]    -- rotation
      processed[26] = input[4]    -- rotation
      processed[27] = input[5]    -- rotation
      processed[28] = input[6]    -- rotation
      processed[29] = input[7]      -- height    
      processed[30] = input[8]      -- alpha
      processed[31] = input[9]      -- beta
      processed[32] = input[10]      -- restitution
      return processed
      -- print(processed)

    elseif self.opt.netType == 'cnnAll2' then
      processed[input[1]] = 1        -- shape
      processed[14+input[2]] = 1    -- specific    
      processed[25] = input[7]    -- height
      processed[26] = input[8]    -- alpha
      processed[27] = input[10]    -- restitution
      return processed

    elseif self.opt.netType == 'cnnCor' then
      local processed = torch.zeros(self.opt.outputSize)
      local shape_id = math.floor(input[1]+0.5)

      if shape_id == 1 or shape_id == 8 or shape_id == 10 then
        processed[1] = 1     -- is pointy
      end

      if shape_id == 4 or shape_id == 5 or shape_id == 6 or shape_id == 7 or shape_id == 9 or shape_id == 14 or shape_id == 8 then
        processed[2] = 1     -- curved face
      end

      if shape_id ~= 6 and shape_id ~= 14 then
        processed[3] = 1     -- have edge
      end

      if input[2] <= 4.01 then  -- soft
        processed[4] = 1
      elseif input[2] <= 7.01 then  -- mid
        processed[5] = 1
      elseif input[2] <= 10.01 then  -- hard
        processed[6] = 1
      end

      if input[7]>0.0 then  -- high
        processed[7] = 1
      end

      if input[8] <= -0.47 then -- metal
        processed[8] = 1
      elseif input[8] <= 0.0 then  -- ceramic
        processed[9] = 1
      elseif input[8] <= 0.41 then  -- polyst
        processed[10] = 1
      elseif input[8] <= 1.01 then  -- wood
        processed[11] = 1
      end

      if input[10] > 0.0 then  -- bouncy
        processed[12] = 1  
      end
      return processed

    elseif self.opt.netType == 'cnnCor2' or self.opt.netType == 'cnnCor4'  then -- 1,1,1,1
      local processed = torch.zeros(self.opt.outputSize)
      local weight = {1,1,1,1}
      local shape_id = math.floor(input[1]+0.5)
      if shape_id == 1 or shape_id == 8 or shape_id == 10 then
        processed[1] = 1     -- is pointy
      end

      if shape_id == 4 or shape_id == 5 or shape_id == 6 or shape_id == 7 or shape_id == 9 or shape_id == 14 or shape_id == 8 then
        processed[2] = 1     -- curved face
      end

      if shape_id ~= 6 and shape_id ~= 14 then
        processed[3] = 1     -- have edge
      end

      if input[7]>0.0 then  -- high
        processed[4] = 1
      end

      local matVec = label2Val({input[2],input[8],input[9],input[10]})
      local mat = nearest_neighbour(matVec,weight)
      processed[4+mat] = 1
      return processed

    elseif self.opt.netType == 'cnnCor3' then -- 1,1,1,1
      local processed = torch.zeros(self.opt.outputSize)
      local weight = {3,3,1,1}
      local shape_id = math.floor(input[1]+0.5)
      if shape_id == 1 or shape_id == 8 or shape_id == 10 then
        processed[1] = 1     -- is pointy
      end

      if shape_id == 4 or shape_id == 5 or shape_id == 6 or shape_id == 7 or shape_id == 9 or shape_id == 14 or shape_id == 8 then
        processed[2] = 1     -- curved face
      end

      if shape_id ~= 6 and shape_id ~= 14 then
        processed[3] = 1     -- have edge
      end

      if input[7]>0.0 then  -- high
        processed[4] = 1
      end

      local matVec = label2Val({input[2],input[8],input[9],input[10]})
      local mat = nearest_neighbour(matVec,weight)
      processed[4+mat] = 1

      return processed

    end
    
  end
end

function PrimSelfDataset:postprocessXml()
  return function(input)
    if self.opt.netType == 'cnnAll' then
      local processed = torch.zeros(10)
      local shape = torch.zeros(14)
      local specific = torch.zeros(10)
      for i = 1,14 do
        shape[i] = input[i]
      end
      for i = 1,10 do
        specific[i] = input[14+i]
      end
      processed[1] = argmax_1D(shape)
      processed[2] = argmax_1D(specific)
      processed[3] = input[25]
      processed[4] = input[26]
      processed[5] = input[27]
      processed[6] = input[28]
      processed[7] = input[29]
      processed[8] = input[30]
      processed[9] = input[31]
      processed[10] = input[32]
      return processed
      
    elseif self.opt.netType == 'cnnAll2' then
      local processed = torch.zeros(5)
      local shape = torch.zeros(14)
      local specific = torch.zeros(10)
      for i = 1,14 do
        shape[i] = input[i]
      end
      for i = 1,10 do
        specific[i] = input[14+i]
      end
      processed[1] = argmax_1D(shape)
      processed[2] = argmax_1D(specific)
      processed[3] = input[25]
      processed[4] = input[26]
      processed[5] = input[27]
      return processed

    elseif self.opt.netType == 'cnnCor' then
      local processed = torch.zeros(7)

      local specific = torch.zeros(3)
      local alpha = torch.zeros(4)

      for i = 1,3 do
        if input[i] > 0.5 then
          processed[i] = 1
        end
      end

      for i = 1,3 do
        specific[i] = input[3+i]
      end    
      processed[4] = argmax_1D(specific)

      if input[7] > 0.5 then
        processed[5] = 1
      end

      for i = 1,4 do
        alpha[i] = input[7+i]
      end      
      processed[6] = argmax_1D(alpha)
      
      if input[12] > 0.5 then
        processed[7] = 1
      end
      return processed

    elseif self.opt.netType == 'cnnCor2' or self.opt.netType == 'cnnCor3' or self.opt.netType == 'cnnCor4' then -- 1,1,1,1
      local processed = torch.zeros(5)

      local material = torch.zeros(4)

      for i = 1,3 do
        if input[i] > 0.5 then
          processed[i] = 1
        end
      end

      if input[4] > 0.5 then
        processed[4] = 1
      end

      for i = 1,4 do
        material[i] = input[4+i]
      end      

      processed[5] = argmax_1D(material)-1 -- 0 indexing output

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

function argmin_1D(ipt)
   local v = torch.Tensor(#ipt)
   for i = 1, #ipt do
      v[i] = ipt[i]
   end
   local length = v:size(1)
   assert(length > 0)
   -- examine on average half the entries
   local minValue = torch.min(v)
   for i = 1, v:size(1) do
      if v[i] == minValue then
         return i
      end
   end
end

function linear_scaling(value, max_val, min_val)
  local mean = (max_val + min_val)/2.0
  local var = max_val - min_val
  return 2.0*(value - mean)/var
end

function to_nnlabel(matVec)
  local youngs = linear_scaling(matVec[1], 9, 0)
  local alpha = linear_scaling(math.log(matVec[2], 10), -5, -8)
  local beta = linear_scaling(math.log(matVec[3], 2), 5, 0)
  local res = linear_scaling(matVec[4], 0.9, 0.6)
  return {youngs, alpha, beta, res}
end

function get_distance(matVec0, matVec1, weight)
  local label0 = to_nnlabel(matVec0)
  local label1 = to_nnlabel(matVec1)
  local weight = weight or {1,1,1,1}
  local sum = 0
  for i = 1,#weight do
    sum = sum + weight[i]*(label0[i]-label1[i])^2
  end
  return sum
end

function nearest_neighbour(matVec, weight)
  local mat0 = {8, 10^(-7.5), 2^2.3, 0.77} -- steel
  local mat1 = {7, 10^(-6.9), 2^2.3, 0.65} -- ceramic
  local mat2 = {1, 10^(-6.1), 2^4.9, 0.83} -- poly
  local mat3 = {5, 10^(-5.7), 2^5.9, 0.78} -- wood
  local base_mat = {mat0, mat1, mat2, mat3}
  local distance = {0,0,0,0}
  for i = 1,#base_mat do
    distance[i] = get_distance(base_mat[i], matVec, weight)
  end
  return argmin_1D(distance)
end

function label2Val(labels)
  -- pass in params estimated by nn all ranges between -1 and 1
  local modulus = labels[1]   
  local alpha = 10^((-5-8)/2 - labels[2]*(-8+5)/2)  
  local beta = 2^((0+5)/2 + labels[3]*(5-0)/2)
  local restitution = (0.6+0.9)/2 + labels[4]*(0.9-0.6)/2  
  return {modulus, alpha, beta, restitution}
end

return M.PrimSelfDataset
