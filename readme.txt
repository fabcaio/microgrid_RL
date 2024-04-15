to import the dataset:

	cbuy, csell, cprod, power_load, power_res = np.load('price_load_res_profiles.npy', allow_pickle=True)


to show the GUROBI model variables:

	#show all variables and their respective values and index
	i=0
	for v in mdl.getVars():
		print('%s %g %i' % (v.VarName, v.X, i))
		i+=1

to get the value of a GUROBI model varible:
	
	x = np.array([[mdl.getVars()[1].x]])

to call the optimal solution:
	
	x0 = np.random.rand(1,1)*225+25 #minimum battery level is 25
        
	i = np.random.randint(cbuy.shape[0]-N)
    
	cbuy_tmp = cbuy[i:i+N+1]
	csell_tmp = csell[i:i+N+1]
	cprod_tmp = cprod[i:i+N+1]
	power_res_tmp = power_res[i:i+N+1]
	power_load_tmp = power_load[i:i+N+1]
    
	mdl = hybrid_fhocp(x0, N, power_res_tmp, power_load_tmp, cbuy_tmp, csell_tmp, cprod_tmp)

*objective value is given by mdl.ObjVal
*the "delta" input to the gurobi_qp function should have as size (N,5)