
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local opts = require 'opts'
local opt = opts.parse(arg)

package.path = './util/lua/?.lua;' .. package.path
require 'ginit' (opt)
opts.init(opt)

----------------------------------------------------

local models = require 'models/init'
local criterions = require 'criterions/init'
local DataLoader = require('models/' .. opt.netType .. '-dataloader')
local Trainer = require('models/' .. opt.netType .. '-train')
local checkpoints = require 'checkpoints'

----------------------------------------------------
-- Data loading
print('=> Setting up data loader')
local trainLoader, valLoader = DataLoader.create(opt)

-- Load previous checkpoint, if it exists
print('=> Checking checkpoints')
local checkpoint = checkpoints.load(opt)

-- Create model
print('=> Setting up model and criterion')

local model, optimState = models.setup(opt, checkpoint, valLoader)
local criterion = criterions.setup(opt, checkpoint, model) -- netType-criterion.createCriterion(opt, model)

-- The trainer handles the training loop and evaluation on validation set
print('=> Loading trainer')
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local loss = trainer:test(0, valLoader)
   print(string.format(' * Results Err %1.4f', loss))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or math.max(1, opt.epochNumber)
local bestLoss = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss = trainer:train(epoch, trainLoader)
   -- Run model on validation set
   local testLoss = trainer:test(epoch, valLoader)

   local bestModel = false
   if testLoss < bestLoss then
      bestModel = true
      bestLoss = testLoss
      print(' * Best model ', testLoss)
   end

   trainer:sanitize()
   checkpoints.save(epoch, model, criterion, trainer.optimState, bestModel, opt)
end

print(string.format(' * Finished Err %1.4f', bestLoss))
