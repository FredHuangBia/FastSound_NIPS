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

function reverse_mapping(label)
  H_MAX = 2
  H_MIN = 1
  ALPHA_MAX = -5
  ALPHA_MIN = -8
  BETA_MAX = 5
  BETA_MIN = 0
  RES_MAX = 0.9
  RES_MIN = 0.6
  new_label = {}
  for i = 1,10 do
    label[i] = tonumber(label[i])
  end
  for i = 1,6 do
    new_label[i] = label[i]
  end
  new_label[7] = 2*(label[7]-(H_MIN+H_MAX)/2)/(H_MAX-H_MIN)
  new_label[8] = 2*(math.log(label[8],10)-(ALPHA_MIN+ALPHA_MAX)/2)/(ALPHA_MAX-ALPHA_MIN)
  new_label[9] = 2*(math.log(label[9],2)-(BETA_MIN+BETA_MAX)/2)/(BETA_MAX-BETA_MIN)
  new_label[10] = 2*(label[10]-(RES_MIN+RES_MAX)/2)/(RES_MAX-RES_MIN)
  return new_label
end

local function findAudios(opt)
  local maxLength = -1
  local dataPaths = {}

  -- Generate a list of data points
  local xmlRaw = io.open(path.join(opt.dataRoot, 'self_sup', 'label.txt'))
  local prefixRaw = io.open(path.join(opt.dataRoot, 'self_sup', 'prefix.txt'))
  local numData = opt.numEntry 
  local xml = torch.zeros(numData, opt.maxXmlLen) -- opt.maxXmlLen for single_shape is 5
  local xmlLen = torch.zeros(numData)
 
  for i = 1, numData do
    local line = xmlRaw:read('*l') -- read the next line
    local prefix_line = prefixRaw:read('*l') -- read the next line
    local pieces = line:split(' ')
    local prefix = prefix_line:split(' ')[1]
    local labels = reverse_mapping(pieces)
    xml[i][1] = labels[1]+1 -- label 1 (shape)
    xml[i][2] = labels[2]+1 -- label 2 (specific)
    xml[i][3] = labels[3]   -- label 3 (rotation)
    xml[i][4] = labels[4]   -- label 4 (rotation)
    xml[i][5] = labels[5]   -- label 5 (rotation)
    xml[i][6] = labels[6]   -- label 6 (rotation)
    xml[i][7] = labels[7]   -- label 7 (height)
    xml[i][8] = labels[8]   -- label 8 (alpha)
    xml[i][9] = labels[9]  -- label 9 (beta)
    xml[i][10] = labels[10]  -- label 10 (rest)
    xml[i][11] = prefix   --  sound id
    -- print(xml[i])
    xmlLen[i] = 11 -- current length of xml

    local outer_folter = ('%.03d'):format( math.floor((tonumber(prefix))/100) +1)
    local str = string.format("%06d", tonumber(prefix))

    dataPaths[i] = path.join('./', outer_folter, str)

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




