--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'

local C = {}

function C.setup(opt, checkpoint, model)
  local criterion
  local criterionHandler = require('models/' .. opt.netType .. '-criterion')
  
  if checkpoint then
    local criterionPath = paths.concat(opt.resume, checkpoint.criterionFile)
    if not paths.filep(criterionPath) then
      print('=> WARNING: Saved criterion not found: ' .. criterionPath)
    else
      print('=> Resuming criterion from ' .. criterionPath)
      criterion = torch.load(criterionPath)
      criterionHandler.initCriterion(criterion, model)
    end
  end
  if not criterion then
    print('=> Creating criterion from file: models/' .. opt.netType .. '-criterion.lua')
    criterion = criterionHandler.createCriterion(opt, model)
  end

  return criterion
end

return C
