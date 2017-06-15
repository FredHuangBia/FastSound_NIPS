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

local function findImages(opt)
  local maxLength = -1
  local dataPaths = {}

  -- Generate a list of data points
  local xmlRaw = io.open(path.join(opt.dataRoot, opt.dataset .. '.txt'))
  local numData = opt.numEntry 
  local xml = torch.zeros(numData, opt.maxXmlLen)
  local xmlLen = torch.zeros(numData)
 
  for i = 1, numData do
    local line = xmlRaw:read('*l')
    local pieces = line:split(' ')
    xml[i][1] = pieces[1]
    xmlLen[i] = 1

    dataPaths[i] = path.join('scene-' .. xml[i][1], 'obj')
    for j = 2, #pieces, 2 do
      dataPaths[i] = dataPaths[i] .. '-' .. pieces[j]
      if tonumber(pieces[j]) < 100 then
        xml[i][xmlLen[i] + 1] = pieces[j]
        xml[i][xmlLen[i] + 2] = pieces[j + 1]
        xmlLen[i] = xmlLen[i] + 2
      end
    end
    dataPaths[i] = path.join(dataPaths[i], 'mat')
    for j = 3, #pieces, 2 do
      dataPaths[i] = dataPaths[i] .. '-' .. pieces[j]
    end

    maxLength = math.max(maxLength, #dataPaths[i] + 1)
  end

  -- Convert the generated list to a tensor for faster loading
  local dataPath = torch.CharTensor(numData, maxLength):zero()
  for i, path in ipairs(dataPaths) do
    ffi.copy(dataPath[i]:data(), path)
  end
  
  return dataPath, xml, xmlLen
end

local function loadShapeAttr(opt)
  local attrRaw = io.open(path.join(opt.dataRoot, 'shapeAttr.txt'))
  local attr = torch.zeros(opt.numObjId, opt.numAttr)
 
  for i = 1, opt.numObj do
    local line = attrRaw:read('*l')
    local pieces = line:split(' ')
    attr[pieces[1] + 1] = torch.Tensor({table.unpack(pieces, 2)})
  end

  return attr
end

local function mergeAudios(dataPath, xml, opt)
  for i = 1, opt.numEntry do
    local path = ffi.string(dataPath[i]:data())
    local filename = paths.concat(opt.data, path, 'merged.t7')
    
    if lfs.attributes(filename, 'mode') ~= 'file' then
      local merged = torch.zeros(opt.audioDim)

      for j = 2, (#xml)[2], 2 do
        local filename = paths.concat(opt.data, path, 
        string.format('obj-%04d.raw', xml[i][j]))
        if lfs.attributes(filename, 'mode') == 'file' then
          local audioRaw = io.open(filename)
          for k = 1, opt.audioDim do
            merged[k]= merged[k] + tonumber(audioRaw:read('*l'))
          end
        end
      end

      torch.save(filename, merged)
    end
    ut.progress(i, opt.numEntry) 
  end
end

function M.exec(opt, cacheFile)
  -- find the image path names
  local dataPath = torch.CharTensor()  -- path to each image in dataset

  assert(paths.dirp(opt.data), 'data directory not found: ' .. opt.data)

  print("=> Generating list of images")
  local dataPath, xml, xmlLen = findImages(opt)
  local shapeAttr = loadShapeAttr(opt)
  mergeAudios(dataPath, xml, opt)

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
    shapeAttr = shapeAttr,
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

  print(" | saving list of images to " .. cacheFile)
  torch.save(cacheFile, info)
  return info
end

return M
