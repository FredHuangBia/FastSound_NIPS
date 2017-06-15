--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local checkpoint = {}

function checkpoint.latest(opt)
  print('=> Loading the latest checkpoint')
  local latestPath = paths.concat(opt.resume, 'latest.t7')
  if not paths.filep(latestPath) then
    return nil
  end

  print('=> Loading checkpoint ' .. latestPath)
  return torch.load(latestPath)
end

function checkpoint.best(opt)
  print('=> Loading the best checkpoint')
  local bestPath = paths.concat(opt.resume, 'best.t7')
  if not paths.filep(bestPath) then
    return nil
  end

  print('=> Loading checkpoint ' .. bestPath)
  return torch.load(bestPath)
end

function checkpoint.load(opt)
  local epoch = opt.epochNumber
  if epoch == 0 then
    return nil
  elseif epoch == -1 then
    -- finding the latest epoch, requiring 'latest.t7'
    return checkpoint.latest(opt)
  elseif epoch == -2 then
    -- finding the best epoch, requiring 'best.t7'
    return checkpoint.best(opt)
  end

  local modelFile = 'model_' .. epoch .. '.t7'
  local criterionFile = 'criterion_' .. epoch .. '.t7'
  local optimFile = 'optimState_' .. epoch .. '.t7'

  local loaded = {
    epoch = epoch,
    modelFile = modelFile,
    criterionFile = criterionFile,
    optimFile = optimFile,
  }

  return loaded
end

function checkpoint.save(epoch, model, criterion, optimState, bestModel, opt)
  -- Don't save the DataParallelTable for easier loading on other machines
  if torch.type(model) == 'nn.DataParallelTable' then
    model = model:get(1)
  end

  local modelFile = 'model_' .. epoch .. '.t7'
  local criterionFile = 'criterion_' .. epoch .. '.t7'
  local optimFile = 'optimState_' .. epoch .. '.t7'

  if bestModel or (epoch % opt.saveEpoch == 0) then
    torch.save(paths.concat(opt.resume, modelFile), model)
    torch.save(paths.concat(opt.resume, criterionFile), criterion)
    torch.save(paths.concat(opt.resume, optimFile), optimState)
    torch.save(paths.concat(opt.resume, 'latest.t7'), {
      epoch = epoch,
      modelFile = modelFile,
      criterionFile = criterionFile,
      optimFile = optimFile,
    })
  end

  if bestModel then
    torch.save(paths.concat(opt.resume, 'best.t7'), {
      epoch = epoch,
      modelFile = modelFile,
      criterionFile = criterionFile,
      optimFile = optimFile,
    })
    torch.save(paths.concat(opt.resume, 'model_best.t7'), model)
  end
end

return checkpoint
