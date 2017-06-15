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

local function findImages(opt, map, mtl)
  local maxLength = -1
  local dataPaths = {}

  -- Generate a list of data points
  local xmlRaw = io.open(path.join(opt.dataRoot, opt.dataset .. '.txt'))
  local numData = opt.numEntry 
  local xml = torch.zeros(numData, opt.maxXmlLen)
  local xmlLen = torch.zeros(numData)
 
  for i = 1, numData do
    local line = xmlRaw:read('*l')
    local pieces = line:split('/')

    xml[i][1] = 1000
    xml[i][2] = map[pieces[3]]
    xml[i][3] = mtl[xml[i][2]]
    xmlLen[i] = 3
    dataPaths[i] = line 

    maxLength = math.max(maxLength, #dataPaths[i] + 1)
  end

  -- Convert the generated list to a tensor for faster loading
  local dataPath = torch.CharTensor(numData, maxLength):zero()
  for i, path in ipairs(dataPaths) do
    ffi.copy(dataPath[i]:data(), path)
  end
  
  return dataPath, xml, xmlLen
end

local function loadMtl(opt)
  local mtlRaw = io.open(path.join(opt.dataRoot, 'phys101a_mtl.txt'))
  local map = {}
  local mtl = torch.zeros(opt.numObj)
 
  for i = 1, opt.numObj do
    local line = mtlRaw:read('*l')
    local pieces = line:split(' ')
    map[pieces[1]] = i
    mtl[i] = tonumber(pieces[2]) - 1
  end

  return map, mtl
end

local function loadShapeAttr(opt, map)
  local attrRaw = io.open(path.join(opt.dataRoot, 'phys101a_shapeAttr.txt'))
  local attr = torch.zeros(opt.numObj, opt.numAttr)
 
  for i = 1, opt.numObj do
    local line = attrRaw:read('*l')
    local pieces = line:split(' ')
    local id = map[pieces[1]]
    attr[id] = torch.Tensor({table.unpack(pieces, 2)})
  end
  return attr
end

local function parseVideos(dataPath, opt)
  for i = 1, opt.numEntry do
    local path = ffi.string(dataPath[i]:data())
    local filename = paths.concat(opt.data, path, 'Camera_1.mp4')
    
    local imgName = paths.concat(opt.data, path, 'render')
    if lfs.attributes(paths.concat(imgName, '020.jpg'), 'mode') ~= 'file' then
      os.execute('mkdir ' .. imgName)
      os.execute('ffmpeg -i ' .. filename .. ' ' .. imgName .. '/%03d.jpg')
    end
    local audioT7Name = paths.concat(opt.data, path, 'audio.t7')
    if lfs.attributes(audioT7Name, 'mode') ~= 'file' then
      local audioName = paths.concat(opt.data, path, 'audio.wav')
      os.execute('ffmpeg -i ' .. filename .. ' ' .. audioName)
      local audioWav = audio.load(audioName)
      torch.save(audioT7Name, audioWav[{{}, {1}}])
    end

    ut.progress(i, opt.numEntry) 
  end
end

function M.exec(opt, cacheFile)
  -- find the image path names
  local dataPath = torch.CharTensor()  -- path to each image in dataset

  assert(paths.dirp(opt.data), 'data directory not found: ' .. opt.data)

  print("=> Generating list of images")
  local map, mtl = loadMtl(opt)
  local shapeAttr = loadShapeAttr(opt, map)
  local dataPath, xml, xmlLen = findImages(opt, map, mtl)
  parseVideos(dataPath, opt)

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
    map = map,
    mtl = mtl, 
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
