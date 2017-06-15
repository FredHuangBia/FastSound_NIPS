--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local ffi = require 'ffi'

local M = {}

local function findAudios(opt)
  local maxLength = -1
  local dataPaths = {}

  -- Generate a list of data points
  local xmlRaw = io.open(path.join(opt.dataRoot, opt.dataset .. '.txt'))
  local numData = opt.numEntry 
  local xml = torch.zeros(numData, opt.maxXmlLen) -- opt.maxXmlLen for single_shape is 5
  local xmlLen = torch.zeros(numData)
 
  for i = 1, numData do
    local line = xmlRaw:read('*l') -- read the next line
    local pieces = line:split(' ')
    xml[i][1] = pieces[2]   -- label 1 (rotation)
    xml[i][2] = pieces[3]   -- label 2 (height)
    xml[i][3] = pieces[4]
    xml[i][4] = pieces[5]
    xml[i][5] = pieces[6]
    xml[i][6] = pieces[1]   --  sound id
    xmlLen[i] = 6 -- current length of xml

    local str = string.format("%06d", i)

    dataPaths[i] = path.join('./',str)

    maxLength = math.max(maxLength, #dataPaths[i] + 1)  -- ??????????????????????????
  end

  -- Convert the generated list to a tensor for faster loading
  local dataPath = torch.CharTensor(numData, maxLength):zero()
  for i, path in ipairs(dataPaths) do
    ffi.copy(dataPath[i]:data(), path)
  end
  
  return dataPath, xml, xmlLen
end


local function mergeAudios(dataPath, opt)  -- write .raw to .t7
  for i = 1, opt.numEntry do
    local path = ffi.string(dataPath[i]:data())
    local t7filename = paths.concat(opt.data, path, 'merged.t7') 
    
    if lfs.attributes(t7filename, 'mode') ~= 'file' then
      local merged = torch.zeros(opt.audioDim)

      local filename = paths.concat(opt.data, path, 'sound.raw')
      if lfs.attributes(filename, 'mode') == 'file' then
        local audioRaw = io.open(filename)
        for k = 1, opt.audioDim do
          -- print(tonumber(audioRaw:read('*l')))
          merged[k]= merged[k] + tonumber(audioRaw:read('*l'))
        end
      end

      torch.save(t7filename, merged)
    end
    ut.progress(i, opt.numEntry) 
  end
end


function M.exec(opt, cacheFile)
  -- find the image path names
  local dataPath = torch.CharTensor()  -- path to each image in dataset

  assert(paths.dirp(opt.data), 'data directory not found: ' .. opt.data)

  print("=> Generating list of audios")
  local dataPath, xml, xmlLen = findAudios(opt)
  -- local shapeAttr = loadShapeAttr(opt)
  mergeAudios(dataPath, opt)

  local numData = (#dataPath)[1]
  local numTrainData = math.floor(numData * opt.trainPctg)

  print("=> Shuffling")
  local shuffle = torch.randperm(numData):long()
  local trainDataPath = dataPath:index(1, shuffle[{{1, numTrainData}}])
  local valDataPath = dataPath:index(1, shuffle[{{numTrainData + 1, -1}}])
  local trainXml = xml:index(1, shuffle[{{1, numTrainData}}])
  local valXml = xml:index(1, shuffle[{{numTrainData + 1, -1}}])
  local trainXmlLen = xmlLen:index(1, shuffle[{{1, numTrainData}}])
  local valXmlLen = xmlLen:index(1, shuffle[{{numTrainData + 1, -1}}])

  local info = {
    basedir = opt.data,
    train = {
      dataPath = trainDataPath,
      xml = trainXml,
      xmlLen = trainXmlLen,
    },
    val = {
      dataPath = valDataPath,
      xml = valXml,
      xmlLen = valXmlLen,
    },
  }

  print(" | saving list of audios to " .. cacheFile)
  torch.save(cacheFile, info)
  return info
end

return M
