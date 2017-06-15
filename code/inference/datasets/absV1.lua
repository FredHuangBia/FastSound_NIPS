--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Abs V1 'Mike and Jenny' dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

-- require locally for threads
package.path = './util/lua/?.lua;' .. package.path
local ut = require 'utils'

local M = {}
local AbsV1Dataset = torch.class('AbsV1Dataset', M)

function AbsV1Dataset:__init(imageInfo, opt, split)
  self.imageInfo = imageInfo[split]
  self.opt = opt
  self.split = split
  self.dir = imageInfo.basedir
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  self.bg = imageInfo.bg[{{1, 3}}]
  self.pieces = imageInfo.pieces
  self.dict = imageInfo.dict
end

function AbsV1Dataset:get(i, isXmlRaw, loadMask, loadBox, loadSeg)
  local path = ffi.string(self.imageInfo.imagePath[i]:data())
  local img = ut.loadImage(paths.concat(self.dir, path))

  local xml = self.imageInfo.xml[i]
  if not isXmlRaw then
    xml = xml[{{1, self.imageInfo.xmlLen[i]}}]
  end

  local mask
  if loadMask then
    path = ffi.string(self.imageInfo.maskPath[i]:data())
    mask = torch.load(paths.concat(self.dir, path))
  end

  local box = {}
  if loadBox then
    path = ffi.string(self.imageInfo.boxPath[i]:data())
    box = torch.load(paths.concat(self.dir, path))
  end

  local seg = {}
  if loadSeg then
    path = ffi.string(self.imageInfo.segPath[i]:data())
    seg = torch.load(paths.concat(self.dir, path))
    seg.seg = torch.zeros((#seg.rawMask)[1], table.unpack((#self.bg):totable()))
    for j = 1, (#seg.seg)[1] do
      seg.seg[j][{{}, {seg.rawBox[j][2], seg.rawBox[j][4]}, {seg.rawBox[j][1], seg.rawBox[j][3]}}] = 
        image.scale(seg.rawMask[{{j}}]:repeatTensor(3, 1, 1), 
        seg.rawBox[j][3] - seg.rawBox[j][1] + 1, seg.rawBox[j][4] - seg.rawBox[j][2] + 1)
    end
  end

  local caption = self.imageInfo.caption[i] 

  return {
    img = img,
    xml = xml,
    mask = mask,
    box = box.box,
    boxGt = box.boxGt,
    seg = seg.seg,
    segGt = seg.segGt,
    caption = caption,
  }
end

function AbsV1Dataset:size()
  return self.imageInfo.imagePath:size(1)
end

-- Computed from random subset of Imagenet training images
local meanstd = {
  mean = { 0.485, 0.456, 0.406 },
  std = { 0.229, 0.224, 0.225 },
}
local pca = {
  eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
  eigvec = torch.Tensor{
    { -0.5675,  0.7192,  0.4009 },
    { -0.5808, -0.0045, -0.8140 },
    { -0.5836, -0.6948,  0.4203 },
  },
}

function AbsV1Dataset:preprocess()
  if self.split == 'train' then
    return t.Compose{
      t.Scale(self.opt.imgDim, 'max'),
      t.Pad(self.opt.imgDim),
      t.CenterCrop(self.opt.imgDim),
--      t.ColorJitter({
--        brightness = 0.4,
--        contrast = 0.4,
--        saturation = 0.4,
--      }),
--      t.Lighting(0.1, pca.eigval, pca.eigvec),
      t.ColorNormalize(meanstd),
--      t.HorizontalFlip(0.5),
    }
  elseif self.split == 'val' then
    return t.Compose{
      t.Scale(self.opt.imgDim, 'max'),
      t.Pad(self.opt.imgDim),
      t.CenterCrop(self.opt.imgDim),
      t.ColorNormalize(meanstd),
    }
  else
    error('invalid split: ' .. self.split)
  end
end

function AbsV1Dataset:postprocess()
  return t.Compose{
    t.ColorUnnormalize(meanstd),
  }
end

function AbsV1Dataset:preprocessMask()
  return t.Compose{
    t.Scale(self.opt.imgDim, 'max', 'simple'),
    t.Pad(self.opt.imgDim),
    t.CenterCrop(self.opt.imgDim),
  }
end

function AbsV1Dataset:preprocessBox()
  if self.split == 'train' then
    return t.Compose{
      t.ScaleBox(self.opt.imgDim, 'max'),
      t.PadBox(self.opt.imgDim),
      t.CenterCropBox(self.opt.imgDim),
      t.ClampBox(),
--      t.ImgFromBox(t.ColorJitter({
--        brightness = 0.4,
--        contrast = 0.4,
--        saturation = 0.4,
--      })),
--      t.ImgFromBox(t.Lighting(0.1, pca.eigval, pca.eigvec)),
      t.ImgFromBox(t.ColorNormalize(meanstd)),
    }
  elseif self.split == 'val' then
    return t.Compose{
      t.ScaleBox(self.opt.imgDim, 'max'),
      t.PadBox(self.opt.imgDim),
      t.CenterCropBox(self.opt.imgDim),
      t.ClampBox(),
      t.ImgFromBox(t.ColorNormalize(meanstd)),
    }
  end
end

function AbsV1Dataset:preprocessXml()
  return function(input)
    local maxXY = 500
    local numType, numObject, numXY, numZ, numFlip = 9, 35, 2, 3, 1
    local numTotal = numType + numObject + numXY + numZ + numFlip

    local processed = torch.zeros(input:size(1), numTotal)
    for i = 1, input:size(1) do
      processed[i][input[i][1] + 2] = 1
      processed[i][input[i][2] + numType + 1] = 1
      processed[i][numType + numObject + 1] = input[i][3] / maxXY
      processed[i][numType + numObject + 2] = input[i][4] / maxXY
      processed[i][input[i][5] + numType + numObject + numXY + 1] = 1
      processed[i][numType + numObject + numXY + numZ + 1] = input[i][6]
    end

    return processed
  end
end

function AbsV1Dataset:postprocessXml()
  return function(input)
    local maxXY = 500
    local numType, numObject, numXY, numZ, numFlip = 9, 35, 2, 3, 1
    local numTotal = numType + numObject + numXY + numZ + numFlip

    local processed = torch.zeros(input:size(1), 7)
    for i = 1, input:size(1) do
      _, processed[i][1] = torch.max(input[i][{{1, numType}}], 1) 
      processed[i][1] = processed[i][1] - 2
      _, processed[i][2] = torch.max(input[i][{{numType + 1, numType + numObject}}], 1) 
      processed[i][2] = processed[i][2] - 1
      processed[i][3] = input[i][numType + numObject + 1] * maxXY
      processed[i][4] = input[i][numType + numObject + 2] * maxXY
      _, processed[i][5] = torch.max(input[i]
        [{{numType + numObject + numXY + 1, numType + numObject + numXY + numZ}}], 1) 
      processed[i][5] = processed[i][5] - 1
      processed[i][6] = ut.clamp(torch.round(
        input[i][numType + numObject + numXY + numZ + 1]), 0, 1)
      processed[i][7] = torch.max(nn.SoftMax():forward(
        input[i][{{numType + 1, numType + numObject}}]), 1) 
    end

    return processed
  end
end

function AbsV1Dataset:preprocessCaption()
  return function(input)
    local numTotal = self.opt.dictSize 

    local processed = torch.zeros(input:size(1), numTotal)
    for i = 1, input:size(1) do
      processed[i][input[i]] = 1
    end

    return processed
  end
end

function AbsV1Dataset:postprocessCaption()
  return function(input)
    local processed = torch.zeros(input:size(1))
    for i = 1, input:size(1) do
        _, processed[i] = torch.max(input[i], 1)
    end

    return processed
  end
end

function AbsV1Dataset:translate(input)
  local str = ''

  for i = 1, input:size(1) do
    if input[i] == self.opt.dictSize then
      break
    elseif input[i] <= self.opt.dictSize - 2 then
      str = str .. self.dict[input[i]] .. ' '
    end
  end

  return str
end

function AbsV1Dataset:getPosition(obj)
  local piece = self.pieces[obj[1]][obj[2]][obj[5]][obj[6]]
 
  local h = piece:size(2)
  local w = piece:size(3)
  local hMin = torch.round(obj[4] - h / 2)
  local wMin = torch.round(obj[3] - w / 2)
  local hMax = hMin + h - 1
  local wMax = wMin + w - 1
  
  return torch.Tensor({math.max(wMin, 1), math.max(hMin, 1), 
                       math.min(wMax, self.bg:size(3)), math.min(hMax, self.bg:size(2))})
end

function AbsV1Dataset:renderObject(im, obj, renderMask, val)
  if obj[1] < 0 or not self.pieces[obj[1]][obj[2]] then
    return im
  end
  renderMask = renderMask or false
  val = val or 1

  local piece = self.pieces[obj[1]][obj[2]][obj[5]][obj[6]]
 
  local h = piece:size(2)
  local w = piece:size(3)
  local hMin = torch.round(obj[4] - h / 2)
  local wMin = torch.round(obj[3] - w / 2)
  local hMax = hMin + h - 1
  local wMax = wMin + w - 1
  if hMin > im:size(2) or hMax < 1 or wMin > im:size(3) or wMax < 1 then
    return im
  end
    
  local hMinShift = math.max(hMin, 1) - hMin
  local wMinShift = math.max(wMin, 1) - wMin
  local hMaxShift = hMax - math.min(hMax, im:size(2))
  local wMaxShift = wMax - math.min(wMax, im:size(3))
  
  local subPiece = piece[{{}, {hMinShift + 1, h - hMaxShift}, {wMinShift + 1, w - wMaxShift}}]
  local mask = subPiece[{{4}}]:gt(self.opt.binThres):repeatTensor(3, 1, 1)
  if not renderMask then
    im[{{1, 3}, {hMin + hMinShift, hMax - hMaxShift}, {wMin + wMinShift, wMax - wMaxShift}}]
      :maskedCopy(mask, subPiece[{{1, 3}}]:maskedSelect(mask))
  else 
    im[{{1, 3}, {hMin + hMinShift, hMax - hMaxShift}, {wMin + wMinShift, wMax - wMaxShift}}]
      :maskedFill(mask, val)
  end

  return im
end

function AbsV1Dataset:renderSingle(xml, renderMask, val)
  renderMask = renderMask or false
  local im 
  if not renderMask then 
    im = self.bg:clone()
  else
    im = torch.zeros(#self.bg):byte()
  end

  if (#xml)[1] == 1 then
    return self:renderObject(im, xml[1], renderMask, val)
  else
    return self:renderObject(im, xml, renderMask, val)
  end
end

function AbsV1Dataset:renderEach(rawXml)
  local batchIms
  local xml = self:postprocessXml()(rawXml:float())

  for i = 1, (#xml)[1] do
    local im = self:renderSingle(xml[i])
    batchIms = batchIms or torch.FloatTensor((#xml)[1], table.unpack((#im):totable()))
    batchIms[i]:copy(im)
  end

  return batchIms
end

function AbsV1Dataset:renderAll(xml)
  im = self.bg:clone()
  if xml:dim() == 0 then
    return im
  end

  for depth = 3, 0, -1 do
    for i = 1, (#xml)[1] do
      if (depth == 3 and xml[i][1] == 0) or (depth == xml[i][5] and xml[i][1] > 0) then
        im = self:renderObject(im, xml[i])
      end
    end
  end

  return im
end

function AbsV1Dataset:render(xml, renderMask)
  renderMask = renderMask or false
  local im 
  if not renderMask then 
    im = self.bg:clone()
  else
    im = torch.zeros(#self.bg):byte()
  end

  -- find first negative element and set as length
  local len = xml[{{}, 1}]:lt(0):nonzero()
  if len:dim() == 0 then
    len = (#xml)[1]
  else
    len = len[1][1] - 1
  end

  for depth = 3, 0, -1 do
    for i = 1, len do
      if (depth == 3 and xml[i][1] == 0) or (depth == xml[i][5] and xml[i][1] > 0) then
        im = self:renderObject(im, xml[i], renderMask, i)
      end
    end
  end

  return im
end

function AbsV1Dataset:renderBatch(xmls)
  local im = self:render(self:postprocessXml()(xmls[1]:float()))
  ims = torch.Tensor((#xmls)[1], table.unpack((#im):totable()))
  ims[1] = im

  for i = 2, (#xmls)[1] do 
    ims[i] = self:render(self:postprocessXml()(xmls[i]:float()))
  end

  return ims:cuda()
end

function AbsV1Dataset:sort(xmls)
  local _, id = torch.sort(xmls[{{}, 2}])
  xmls = xmls:index(1, id)
  _, id = torch.sort(xmls[{{}, 1}])
  xmls = xmls:index(1, id)

  return xmls
end

function AbsV1Dataset:nms(xmls)
  local selected = {}
  for i = 1, (#xmls)[1] do
    if xmls[i][1] >= 0 and self.pieces[xmls[i][1]][xmls[i][2]] then
      table.insert(selected, i)
    end
  end
  if #selected == 0 then
    return torch.Tensor()
  end
  xmls = xmls:index(1, torch.LongTensor(selected))
  
  local _, id = torch.sort(xmls[{{}, 7}], 1, true)
  
  selected = {}
  for i = 1, (#id)[1] do
    local isSelected = true
    local pos = self:getPosition(xmls[id[i]])
    for j = 1, i - 1 do
      if xmls[id[j]][1] == xmls[id[i]][1] and
        ut.nms(self:getPosition(xmls[id[j]]), pos) > self.opt.nmsThres then
        isSelected = false
        break
      end
    end

    if isSelected then
      table.insert(selected, id[i])
    end
  end

  xmls = xmls:index(1, torch.LongTensor(selected))
 
  return self:sort(xmls)
end

return M.AbsV1Dataset
