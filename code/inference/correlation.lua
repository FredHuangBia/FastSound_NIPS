function get_correlation(signal1,signal2)
	-- body
	assert(signal1:size()[1] == signal2:size()[1], "two signals have different length")
	local length = signal1:size()[1]
	local correlation = torch.zeros(length)
	for i = 1, length do
		correlation[i] = torch.dot(signal1[{{1,length+1-i}}],signal2[{{i,length}}])
	end
	return correlation
end