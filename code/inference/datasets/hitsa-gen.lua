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

local function loadMtl(opt)
  local mtlRaw = io.open(path.join(opt.dataRoot, 'hitsa_mtl.txt'))
  local map = {}
  local mtl = torch.zeros(opt.numMtl)
 
  for i = 1, opt.numMtl do
    local line = mtlRaw:read('*l')
    local pieces = line:split(' ')
    map[pieces[1]] = i
    mtl[i] = tonumber(pieces[2]) - 1
  end

  return map, mtl
end


local function findImages(opt, map, mtl, trainVal)
  local maxLength = -1
  local dataPaths = {}

  -- Generate a list of data points
  local xmlRaw = io.open(path.join(opt.dataRoot, opt.dataset .. '_' .. trainVal .. '.txt'))
  local numData 
  if trainVal == 'train' then
    numData = opt.numTrainEntry 
  else
    numData = opt.numTestEntry 
  end
  local xml = torch.zeros(numData, opt.maxXmlLen)
  local xmlLen = torch.zeros(numData)
 
  for i = 1, numData do
    local line = xmlRaw:read('*l')
    local pieces = line:split('/')

    dataPaths[i] = line:sub(1, -5)

    xml[i][1] = 1000
    xml[i][2] = 1
    xml[i][3] = mtl[map[pieces[1]]]
    xmlLen[i] = 3

    maxLength = math.max(maxLength, #dataPaths[i] + 1)
  end

  -- Convert the generated list to a tensor for faster loading
  local dataPath = torch.CharTensor(numData, maxLength):zero()
  for i, path in ipairs(dataPaths) do
    ffi.copy(dataPath[i]:data(), path)
  end
  
  return dataPath, xml, xmlLen
end

local function parseVideos(dataPath, xml, opt, numEntry)
  for i = 1, numEntry do
    local path = ffi.string(dataPath[i]:data())
    local filename = paths.concat(opt.data, path .. '.mp4')
    
    local imgName = paths.concat(opt.data, path .. '.jpg')
    if lfs.attributes(imgName, 'mode') ~= 'file' then
      os.execute('ffmpeg -i ' .. filename .. ' -vframes 1 ' .. imgName)
    end
    local audioT7Name = paths.concat(opt.data, path .. '.t7')
    if lfs.attributes(audioT7Name, 'mode') ~= 'file' then
      local audioName = paths.concat(opt.data, path .. '.wav')
      os.execute('ffmpeg -i ' .. filename .. ' ' .. audioName)
      local audioWav = audio.load(audioName)
      torch.save(audioT7Name, audioWav[{{}, {1}}])
    end

    ut.progress(i, numEntry) 
  end
end

function M.exec(opt, cacheFile)
  -- find the image path names
  local dataPath = torch.CharTensor()  -- path to each image in dataset

  assert(paths.dirp(opt.data), 'data directory not found: ' .. opt.data)

  print("=> Generating list of images")
  local map, mtl = loadMtl(opt)
  local trainDataPath, trainXml, trainXmlLen = findImages(opt, map, mtl, 'train')
  local valDataPath, valXml, valXmlLen = findImages(opt, map, mtl, 'test')
  parseVideos(trainDataPath, trainXml, opt, opt.numTrainEntry)
  parseVideos(valDataPath, valXml, opt, opt.numTestEntry)

  local info = {
    basedir = opt.data,
    map = map,
    mtl = mtl, 
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
