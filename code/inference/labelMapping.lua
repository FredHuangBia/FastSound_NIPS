function labelMapping(scn0, scn1, mat0, mat1, mat2)
	local tblScn0 = {-1,-0.6,-0.2,0.2,0.6,1}
	local tblScn1 = {-1,-2/3,-1/3,0,1/3,2/3,1}
	local tblMat0 = {-1,-0.6,-0.2,0.2,0.6,1}
	local tblMat1 = {-1,-0.6,-0.2,0.2,0.6,1}
	local tblMat2 = {-1,-0.6,-0.2,0.2,0.6,1}
	return tblScn0[scn0+1], tblScn1[scn1+1], tblMat0[mat0+1], tblMat1[mat1+1], tblMat2[mat2+1]
end
