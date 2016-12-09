# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 19:35:28 2016

@author: vinic
"""

        solution_expr=sy.lambdify(variables,solution_output,'numpy')
        granul_nums=[np.linspace(0,0.95,granul,dtype='float16')]*num_free_param
        free_mesh=np.meshgrid(*granul_nums,sparse=True,indexing='ij')
        chunk_free_mesh=[]        
        for i in range(0,num_free_param):
            chunked_axis=np.split(free_mesh[i],granul/2,i)
            chunk_free_mesh.append(chunked_axis)
            
            
    if (A_rank==A_b_rank and A_rank<A.shape[1]) or A_b_rank>A_rank or solution=="overdetermined":
        A_prime_rank=np.linalg.matrix_rank(A_prime)
        #num_free_param=int(A_prime.shape[1])-A_prime_rank  
        indices=np.linspace(1,A_prime.shape[1],A_prime.shape[1],dtype='int8')            
        variables=[]        
        for i in indices:
            exec "x%s=sy.symbols('x'+str(i))" % (i)
            exec "variables.append(x%s)" % (i)
        solution_output=list(sy.linsolve(sy.Matrix(A_prime_b),variables))[0]  #initial solution output from sympy, not in lmabdify form
        
        coeffs=[]
        consts=[]        
        for i in range(0,A_prime_rank):
            coeff=[]
            temp_expr=solution_output[i]
            for j in range(A_prime_rank,len(variables)):
                coeff.append(solution_output[i].coeff(variables[j],1))
                temp_expr=temp_expr.coeff(variables[j],0)
            coeffs.append(coeff)
            consts.append([temp_expr])
        #Setting up non-linear problem as transformed linear problem below
        A_prime_eq=np.concatenate(((-1*np.identity(A_prime_rank)),np.array(coeffs)),1)
        A_prime_eq=np.concatenate((A_prime_eq,np.array(consts)),1)
        extra_fractional_constraint=fsugar_functional_denominator_coeff+[0]
        A_prime_eq=np.concatenate((A_prime_eq,np.array(extra_fractional_constraint)),0)
        b_eq=np.zeros([A_prime_eq.shape[0]+1,1]); b_eq[-1]=1                
        A_prime_ub=np.array([[1]*len(variables)+[-1]])
        b_ub=np.array([0])
        minimizing_function=-np.array(fsugar_functional_numerator_coeff+[0])
        
        linprog_result=sp.optimize.linprog(minimizing_function, A_eq=A_prime_eq, b_eq=b_eq, A_ub=A_prime_ub, b_ub=b_ub,bounds=(0.005, None))  #look back at this 0.005 later
        if linprog_result.status==2 or linprog_result.status==1:
            solution="overdetermined"
        else:
            solution=linprog_result.x[:-1]/linprog_result.x
            
            
        #G0_l=A_prime_ub
        #G1_l=extra_fractional_constraint
        #Gl=np.concatenate(([G0_l],[G1_l]),0)
            
        #hl=cvx.matrix([0.0,1.0]+[0]*G1_q.shape[1])
            
            
            
            
#        over_prob = pic.Problem()
#        y_t = over_prob.add_variable('y_t',A_prime.shape[1]+1,lower=0)
#        max_func_coeffs=np.array(fsugar_functional_numerator_coeff+[0])
#        objective_func=sum([max_func_coeffs[i]*y_t[i] for i in range(A_prime.shape[1]+1)])
#        over_prob.set_objective('max',objective_func)
#        mass_conserv_constraint=sum([A_prime_ub[i]*y_t[i] for i in range(A_prime.shape[1]+1)])
#        over_prob.add_constraint(mass_conserv_constraint<0)
#        transform_constraint=sum([extra_fractional_constraint[i]*y_t[i] for i in range(A_prime.shape[1]+1)])
#        over_prob.add_constraint(transform_constraint<1)
#        A_prime_eq_cvx=cvx.matrix(A_prime_eq.astype(np.double))
#        print A_prime_eq
#        print A_prime_eq_cvx        
#        over_prob.add_constraint(abs(A_prime_eq_cvx*y_t)<tol*y_t[int(A_prime.shape[1])])
#        for i in range(A_prime.shape[1]+1):
#              over_prob.add_constraint(y_t[i]>0)      
#        print over_prob        
#        over_prob.solve(solver='cvxopt',verbose=False)
#        if over_prob.status=='infeasible' or over_prob.status=="primal infeasible":
#            solution= 'infeasible'
#        else:
#            solution=np.array(y_t.value)
#            #solution=np.array(y_t.value)[:-1]/np.array(y_t.value)[-1]        