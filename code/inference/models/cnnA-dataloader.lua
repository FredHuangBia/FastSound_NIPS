--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('cnnA.DataLoader', M)

function DataLoader.create(opt)
  -- The train and val loader
  local loaders = {}

  for i, split in ipairs{'train', 'val'} do
    local dataset = datasets.create(opt, split)
    loaders[i] = M.DataLoader(dataset, opt, split)
  end

  return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
  local manualSeed = opt.manualSeed
  local function init()
    require('datasets/' .. opt.dataset)
  end
  local function main(idx)
    if manualSeed ~= 0 then
      torch.manualSeed(manualSeed + idx)
    end
    torch.setnumthreads(1)
    _G.dataset = dataset
    -- _G.preprocess = dataset:preprocess()
    _G.preprocessAudio = dataset:preprocessAudio()
    _G.preprocessXml = dataset:preprocessXml()
    return dataset:size()
  end

  local threads, sizes = Threads(opt.nThreads, init, main)
  self.threads = threads
  self.__size = sizes[1][1]
  self.batchSize = opt.batchSize
  self.dataset = dataset
  self.split = split
end

function DataLoader:size()
  return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
  local threads = self.threads
  local size, batchSize = self.__size, self.batchSize
  local perm 
  if self.split == 'train' then
    perm = torch.randperm(size)
  elseif self.split == 'val' then
    perm = torch.linspace(1, size, size)
  end

  local idx, sample = 1, nil
  local function enqueue()
    while idx <= size and threads:acceptsjob() do
      local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
      threads:addjob(
      function(indices)
        local sz = indices:size(1)
        -- local batchImg, batchAudio, batchXml, imgSize, audioSize, xmlSize
        local batchAudio, batchXml, audioSize, xmlSize
        for i, idx in ipairs(indices:totable()) do
          local sample = _G.dataset:get(idx, true)
          local audio = _G.preprocessAudio(sample.audio)
          local xml = _G.preprocessXml(sample.xml)

          -- if not batchImg then
          --   imgSize = img:size():totable()
          --   batchImg = torch.FloatTensor(sz, table.unpack(imgSize))
          -- end
          -- batchImg[i]:copy(img)

          if not batchAudio then
            audioSize = audio:size():totable()
            batchAudio = torch.FloatTensor(sz, table.unpack(audioSize))
          end
          batchAudio[i]:copy(audio)
          
          if not batchXml then
            xmlSize = xml:size():totable()
            batchXml = torch.FloatTensor(sz, table.unpack(xmlSize))
          end
          batchXml[i]:copy(xml)
      
        end
        collectgarbage()
        return {
          -- img = batchImg:view(sz, table.unpack(imgSize)),
          audio = batchAudio:view(sz, table.unpack(audioSize)),
          xml = batchXml:view(sz, table.unpack(xmlSize)),

        }
      end,
      function(_sample_)
        sample = _sample_
      end,
      indices
      )
      idx = idx + batchSize
    end
  end

  local n = 0
  local function loop()
    enqueue()
    if not threads:hasjob() then
      return nil
    end
    threads:dojob()
    if threads:haserror() then
      threads:synchronize()
    end
    enqueue()
    n = n + 1
    return n, sample
  end

  return loop
end

return M.DataLoader
