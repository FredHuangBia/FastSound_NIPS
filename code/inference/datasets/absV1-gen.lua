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

local function findImages(dir)
  local imagePath = torch.CharTensor()
  local numSets = 1002
  local numInstsPerSet = 10
  local numImgs = numSets * numInstsPerSet

  local maxLength = -1
  local imagePaths = {}

  -- Generate a list of all the images and their class
  for i = 1, numSets do
    for j = 1, numInstsPerSet do
      local line = path.join('RenderedScenes', string.format('Scene%d_%d.png', i - 1, j - 1))

      table.insert(imagePaths, line)
      maxLength = math.max(maxLength, #line + 1)
    end
  end

  -- Convert the generated list to a tensor for faster loading
  local nImages = #imagePaths
  assert(nImages == numImgs, 'Wrong number of images')
  local imagePath = torch.CharTensor(nImages, maxLength):zero()
  for i, path in ipairs(imagePaths) do
    ffi.copy(imagePath[i]:data(), path)
  end

  -- Load all xmls
  local xmls = {}
  local xmlRaw = io.open(path.join(dir, 'Scenes_10020.txt'))
  local xmlLen = torch.Tensor(nImages)
  xmlRaw:read('*l')
  for i = 1, nImages do
    local line = xmlRaw:read('*l')
    xmlLen[i] = line:split('\t')[2] + 1

    local xml = torch.zeros(xmlLen[i], 6)
    for j = 1, xmlLen[i] - 1 do
      line = xmlRaw:read('*l'):split('\t')
      table.remove(line, 1)
      xml[j] = torch.Tensor(line)
    end
    xml[xmlLen[i]][1] = -1

    table.insert(xmls, xml)
  end
  xmlRaw:close()

  -- Convert the xmls to a tensor for faster loading
  local xml = torch.zeros(nImages, xmlLen:max(), 6) 
  for i = 1, nImages do
    xml[i][{{1, xmlLen[i]}}] = xmls[i]
  end

  return imagePath, xml, xmlLen
end

local function collectPieces(dir)
  local bg = image.load(paths.concat(dir, 'Pngs', 'background.png'))
   
  local prefixes = {'s', 'p', 'hb0', 'hb1', 'a', 'c', 'e', 't'}
  local pieces = {}
  for i = 0, 7 do
    pieces[i] = {}
    for j = 0, 50 do
      local objFile = paths.concat(dir, 'Pngs', prefixes[i + 1] .. '_' .. j .. 's.png')
      if lfs.attributes(objFile, 'mode') == 'file' then  
        local piece = image.load(objFile)
        pieces[i][j] = {}

        for k = 0, 2 do
          pieces[i][j][k] = {}

          local scale = (k == 0) and 1 or (k == 1) and 0.7 or (k == 2) and 0.49
          local postPiece = image.scale(piece, piece:size(3) * scale, piece:size(2) * scale)
          pieces[i][j][k][0] = postPiece
          pieces[i][j][k][1] = image.hflip(postPiece)
        end
      end
    end
  end

  return bg, pieces
end

local function genMasks(dir, imagePath, xml, dataset)
  local maxLength = -1
  local maskPaths = {}

  for i = 1, (#imagePath)[1] do
    local line = path.join('Masks', ffi.string(imagePath[i]:data())
      :split('/')[2]:sub(1, -4) .. 't7')
    if not pl.path.exists(path.join(dir, line)) then
      local mask = dataset:render(xml[i], true)
      torch.save(path.join(dir, line), mask)
    end

    table.insert(maskPaths, line)
    maxLength = math.max(maxLength, #line + 1)
    
    ut.progress(i, (#imagePath)[1])
  end

  local maskPath = torch.CharTensor(#maskPaths, maxLength):zero()
  for i, path in ipairs(maskPaths) do
    ffi.copy(maskPath[i]:data(), path)
  end

  return maskPath
end

local function collectBoxes(opt, imagePath, xml, xmlLen, dataset)
  local dir = opt.data
  local maxLength = -1
  local boxPaths = {}

  for i = 1, (#imagePath)[1] do
    local line = path.join('Boxes', ffi.string(imagePath[i]:data())
      :split('/')[2]:sub(1, -4) .. 't7')
    if not pl.path.exists(path.join(dir, line)) then
      local box = {}
      
      local boxLine = path.join('Boxes', ffi.string(imagePath[i]:data())
        :split('/')[2]:sub(1, -4) .. 'npy')
      local rawBox = np.loadnpy(path.join(dir, boxLine))
      box.box = rawBox:clone()
      box.box[{{}, 2}] = dataset.bg:size(2) - rawBox[{{}, 4}] 
      box.box[{{}, 4}] = dataset.bg:size(2) - rawBox[{{}, 2}] 

      local gtBox = torch.zeros(xmlLen[i] - 1, 4)
      for k = 1, xmlLen[i] - 1 do
        gtBox[k] = dataset:getPosition(xml[i][k])
      end

      box.boxGt = torch.zeros((#box.box)[1], (#xml)[3])
      box.boxGt[{{}, 1}] = -1
      for j = 1, (#box.box)[1] do
        local maxIou = 0
        for k = 1, xmlLen[i] - 1 do
          local iou, recall = ut.nms(gtBox[k], box.box[j])
          if iou > maxIou and iou > opt.iouThres and recall > opt.recallThres then
            box.boxGt[j] = xml[i][k]
            maxIou = iou
          end
        end
      end

      torch.save(path.join(dir, line), box)
    end

    table.insert(boxPaths, line)
    maxLength = math.max(maxLength, #line + 1)
    
    ut.progress(i, (#imagePath)[1])
  end

  local boxPath = torch.CharTensor(#boxPaths, maxLength):zero()
  for i, path in ipairs(boxPaths) do
    ffi.copy(boxPath[i]:data(), path)
  end

  return boxPath
end

local function collectSegs(opt, imagePath, xml, xmlLen, dataset)
  local dir = opt.data
  local maxLength = -1
  local segPaths = {}

  for i = 1, (#imagePath)[1] do
    local line = path.join('Segments', ffi.string(imagePath[i]:data())
      :split('/')[2]:sub(1, -4) .. 't7')

    if not pl.path.exists(path.join(dir, line)) then
      local boxLine = path.join('Segments', ffi.string(imagePath[i]:data())
        :split('/')[2]:sub(1, -5) .. '_boxes.npy')
      local rawBox = np.loadnpy(path.join(dir, boxLine)) + 1 -- 0-indexed to 1-indexed
      rawBox[{{}, 1}] = torch.cmin(torch.cmax(rawBox[{{}, 1}], 1), (#dataset.bg)[3])
      rawBox[{{}, 2}] = torch.cmin(torch.cmax(rawBox[{{}, 2}], 1), (#dataset.bg)[2])
      rawBox[{{}, 3}] = torch.cmin(torch.cmax(rawBox[{{}, 3}], 1), (#dataset.bg)[3])
      rawBox[{{}, 4}] = torch.cmin(torch.cmax(rawBox[{{}, 4}], 1), (#dataset.bg)[2])

      local maskLine = path.join('Segments', ffi.string(imagePath[i]:data())
        :split('/')[2]:sub(1, -5) .. '_masks.npy')
      local rawMask = np.loadnpy(path.join(dir, maskLine))

      local seg = {}
      seg.seg = torch.zeros((#rawMask)[1], table.unpack((#dataset.bg):totable()))
      for j = 1, (#seg.seg)[1] do
        seg.seg[j][{{}, {rawBox[j][2], rawBox[j][4]}, {rawBox[j][1], rawBox[j][3]}}] = 
          image.scale(rawMask[{{j}}]:repeatTensor(3, 1, 1), 
                      rawBox[j][3] - rawBox[j][1] + 1, rawBox[j][4] - rawBox[j][2] + 1)
      end
      seg.seg = image.scale(seg.seg:reshape((#rawMask)[1] * 3, 400, 500), 125, 100)
                :reshape((#rawMask)[1], 3, 100, 125):gt(opt.binThres)

      local gtMask = torch.zeros(xmlLen[i] - 1, table.unpack((#dataset.bg):totable())):byte()
      for k = 1, xmlLen[i] - 1 do
        gtMask[k] = dataset:renderSingle(xml[i][k], true, 1)
      end
      if xmlLen[i] > 1 then
        gtMask = image.scale(gtMask:reshape((xmlLen[i] - 1) * 3, 400, 500), 125, 100)
                 :reshape(xmlLen[i] - 1, 3, 100, 125):gt(opt.binThres)
      end
      local gtS = {}
      for k = 1, xmlLen[i] - 1 do
        gtS[k] = gtMask[k]:sum()
      end

      seg.segGt = torch.zeros((#seg.seg)[1], (#xml)[3])
      seg.segGt[{{}, 1}] = -1
      for j = 1, (#seg.seg)[1] do
        local maxIou = 0
        local segS = seg.seg[j]:sum()
        for k = 1, xmlLen[i] - 1 do
          local interS = seg.seg[j][gtMask[k]]:sum()
          local iou = interS / (segS + gtS[k] - interS)
          local recall = interS / gtS[k] 
          if iou > maxIou and iou > opt.iouThres and recall > opt.recallThres then
            seg.segGt[j] = xml[i][k]
            maxIou = iou
          end
        end
      end

      seg.seg = nil
      seg.rawMask = rawMask
      seg.rawBox = rawBox
      torch.save(path.join(dir, line), seg)
    end

    table.insert(segPaths, line)
    maxLength = math.max(maxLength, #line + 1)
    
    ut.progress(i, (#imagePath)[1])
  end

  local segPath = torch.CharTensor(#segPaths, maxLength):zero()
  for i, path in ipairs(segPaths) do
    ffi.copy(segPath[i]:data(), path)
  end

  return segPath
end

function M.exec(opt, cacheFile)
  -- find the image path names
  local imagePath = torch.CharTensor()  -- path to each image in dataset

  assert(paths.dirp(opt.data), 'data directory not found: ' .. opt.data)

  print("=> Generating list of images")
  local imagePath, xml, xmlLen = findImages(opt.data)
  print("=> Collecting pieces") 
  local bg, pieces = collectPieces(opt.data)

  local dummySet = require('datasets/' .. opt.dataset)({
    basedir = opt.data,
    bg = bg,
    pieces = pieces,
  }, opt, nil)  
  print("=> Generating masks") -- create a dummy dataset object for rendering
  local maskPath = genMasks(opt.data, imagePath, xml, dummySet) 
  print("=> Generating gt for boxes") -- create a dummy dataset object for rendering
  local boxPath = collectBoxes(opt, imagePath, xml, xmlLen, dummySet)
  print("=> Generating gt for segments") -- create a dummy dataset object for rendering
  local segPath = collectSegs(opt, imagePath, xml, xmlLen, dummySet)

  local numImages = (#imagePath)[1]
  local numTrainImages = math.floor(numImages * opt.trainPctg)

  print("=> Shuffling")
  local shuffle = torch.randperm(numImages):long()
  local trainImagePath = imagePath:index(1, shuffle[{{1, numTrainImages}}])
  local valImagePath = imagePath:index(1, shuffle[{{numTrainImages + 1, -1}}])
  local trainMaskPath = maskPath:index(1, shuffle[{{1, numTrainImages}}])
  local valMaskPath = maskPath:index(1, shuffle[{{numTrainImages + 1, -1}}])
  local trainBoxPath = boxPath:index(1, shuffle[{{1, numTrainImages}}])
  local valBoxPath = boxPath:index(1, shuffle[{{numTrainImages + 1, -1}}])
  local trainSegPath = segPath:index(1, shuffle[{{1, numTrainImages}}])
  local valSegPath = segPath:index(1, shuffle[{{numTrainImages + 1, -1}}])
  local trainXml = xml:index(1, shuffle[{{1, numTrainImages}}])
  local valXml = xml:index(1, shuffle[{{numTrainImages + 1, -1}}])
  local trainXmlLen = xmlLen:index(1, shuffle[{{1, numTrainImages}}])
  local valXmlLen = xmlLen:index(1, shuffle[{{numTrainImages + 1, -1}}])

  local info = {
    basedir = opt.data,
    bg = bg,
    pieces = pieces,
    train = {
      imagePath = trainImagePath,
      maskPath = trainMaskPath,
      boxPath = trainBoxPath,
      segPath = trainSegPath,
      xml = trainXml,
      xmlLen = trainXmlLen,
    },
    val = {
      imagePath = valImagePath,
      maskPath = valMaskPath,
      boxPath = valBoxPath,
      segPath = valSegPath,
      xml = valXml,
      xmlLen = valXmlLen,
    },
  }

  print(" | saving list of images to " .. cacheFile)
  torch.save(cacheFile, info)
  return info
end

return M
