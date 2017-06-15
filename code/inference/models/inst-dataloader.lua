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
local DataLoader = torch.class('inst.DataLoader', M)

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
    _G.preprocess = dataset:preprocess()
    _G.preprocessXml = dataset:preprocessXml()
    _G.preprocessMask = dataset:preprocessMask()
    return dataset:size()
  end

  local threads, sizes = Threads(opt.nThreads, init, main)
  self.threads = threads
  self.__size = sizes[1][1]
  self.batchSize = opt.batchSize
  self.dataset = dataset
end

function DataLoader:size()
  return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
  local threads = self.threads
  local size, batchSize = self.__size, self.batchSize
  local perm = torch.randperm(size)

  local idx, sample = 1, nil
  local function enqueue()
    while idx <= size and threads:acceptsjob() do
      local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
      threads:addjob(
      function(indices)
        local sz = indices:size(1)
        local batchOri, batchImg, batchXml, batchMask, batchOriMask
        local oriSize, imgSize, xmlSize, maskSize, oriMaskSize
        for i, idx in ipairs(indices:totable()) do
          local sample = _G.dataset:get(idx, true, true, false, false)
          if not batchOri then
            oriSize = sample.img:size():totable()
            batchOri = torch.FloatTensor(sz, table.unpack(oriSize))
          end
          batchOri[i]:copy(sample.img)

          local img = _G.preprocess(sample.img)
          if not batchImg then
            imgSize = img:size():totable()
            batchImg = torch.FloatTensor(sz, table.unpack(imgSize))
          end
          batchImg[i]:copy(img)

          local xml = _G.preprocessXml(sample.xml)
          if not batchXml then
            xmlSize = xml:size():totable()
            batchXml = torch.FloatTensor(sz, table.unpack(xmlSize))
          end
          batchXml[i]:copy(xml)

          if not batchOriMask then
            oriMaskSize = sample.mask:size():totable()
            batchOriMask = torch.FloatTensor(sz, table.unpack(oriMaskSize))
          end
          batchOriMask[i]:copy(sample.mask)
          
          local mask = _G.preprocessMask(sample.mask)
          if not batchMask then
            maskSize = mask:size():totable()
            batchMask = torch.FloatTensor(sz, table.unpack(maskSize))
          end
          batchMask[i]:copy(mask)
        end
        collectgarbage()
        return {
          ori = batchOri:view(sz, table.unpack(oriSize)),
          img = batchImg:view(sz, table.unpack(imgSize)),
          xml = batchXml:view(sz, table.unpack(xmlSize)),
          oriMask = batchOriMask:view(sz, table.unpack(oriMaskSize)),
          mask = batchMask:view(sz, table.unpack(maskSize)),
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
