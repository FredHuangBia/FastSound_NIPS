local handler = {}

function handler.initCriterion(criterion, model)
  if criterion.__typename == 'nn.MultiCriterion' or 
    criterion.__typename == 'nn.ParallelCriterion' then
    for i = 1, #criterion.criterions do
      handler.initCriterion(criterion.criterions[i], model)
    end
  end
end

function handler.createCriterion(opt, model)
  local criterion = nn.MultiCriterion() 

  if opt.absLoss ~= 0 then
    criterion:add(nn.AbsCriterion(), opt.absLoss)
  end
  if opt.mseLoss ~= 0 then
    criterion:add(nn.MSECriterion(), opt.mseLoss)
  end
  if opt.gdlLoss ~= 0 then
    criterion:add(nn.GDLCriterion(), opt.gdlLoss)
  end
  if opt.customLoss ~= 0 then
    -- criterion:add(nn.CustomCriterion(opt.outputSplitType,
    --   opt.outputSplitPSum, opt.criterionWeights), opt.customLoss)
    criterion:add(nn.CrossEntropyCriterion(), opt.customLoss)
  end

  criterion:cuda()

  return criterion
end

return handler
