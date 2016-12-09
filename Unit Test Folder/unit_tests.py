# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 22:52:36 2016

@author: Vinicius Thaddeu dos Santos Ferreira: Sugarwise Chief Scientist
"""
from Implementation_of_Patent import *
import unittest

class Test_Patent_code(unittest.TestCase):
   
    def setUp(self):
        self.parsed_negative_number=parser("test single_sol.csv","ingredients.csv","lab_negative_number.csv")
        self.sys_negative_number=build_lin_system(self.parsed_negative_number[0],self.parsed_negative_number[1])
        self.parsed_bigger_one=parser("test single_sol.csv","ingredients.csv","lab_bigger_one.csv")
        self.sys_bigger_one=build_lin_system(self.parsed_bigger_one[0],self.parsed_bigger_one[1])
        self.parsed_single_solution=parser("test single_sol.csv","ingredients.csv","lab_single_solution.csv")
        self.sys_single_solution=build_lin_system(self.parsed_single_solution[0],self.parsed_single_solution[1])
        self.parsed_under_max=parser("test under.csv","ingredients under.csv","lab_under_max.csv")
        self.sys_under_max=build_lin_system(self.parsed_under_max[0],self.parsed_under_max[1])
        self.parsed_under_zero=parser("test under_zero.csv","ingredients under.csv","lab_under_zero.csv")
        self.sys_under_zero=build_lin_system(self.parsed_under_zero[0],self.parsed_under_zero[1])
        self.parsed_under_negative=parser("test under_neg.csv","ingredients under_neg.csv","lab_under_negative.csv")
        self.sys_under_negative=build_lin_system(self.parsed_under_negative[0],self.parsed_under_negative[1])
        self.parsed_under_bigger1=parser("test under.csv","ingredients under.csv","lab_under_bigger1.csv")
        self.sys_under_bigger1=build_lin_system(self.parsed_under_bigger1[0],self.parsed_under_bigger1[1])
        self.parsed_over=parser("test over.csv","ingredients_over.csv","lab_over.csv")
        self.sys_over=build_lin_system(self.parsed_over[0],self.parsed_over[1])
        
        
        
    def test_negative_number(self):
        sys= self.sys_negative_number       
        A_b=np.concatenate((sys[0],sys[1]),1)
        A_rank=np.linalg.matrix_rank(sys[0])
        A_b_rank=np.linalg.matrix_rank(A_b)        
        status=get_recipe(sys[0],sys[1],sys[2],sys[3],sys[4],0.05)[1]
        self.assertEqual(A_rank,A_b_rank) 
        self.assertEqual(A_rank,sys[0].shape[1])
        self.assertNotEqual(status,"single solution")

    def test_bigger_one(self):
        sys= self.sys_bigger_one       
        A_b=np.concatenate((sys[0],sys[1]),1)
        A_rank=np.linalg.matrix_rank(sys[0])
        A_b_rank=np.linalg.matrix_rank(A_b)        
        status=get_recipe(sys[0],sys[1],sys[2],sys[3],sys[4],0.05)[1]
        self.assertEqual(A_rank,A_b_rank) 
        self.assertEqual(A_rank,sys[0].shape[1])
        self.assertNotEqual(status,"single solution")
        
    def test_single_solution(self):
        sys= self.sys_single_solution      
        A_b=np.concatenate((sys[0],sys[1]),1)
        A_rank=np.linalg.matrix_rank(sys[0])
        A_b_rank=np.linalg.matrix_rank(A_b)        
        status=get_recipe(sys[0],sys[1],sys[2],sys[3],sys[4],0.05)[1]
        solution=get_recipe(sys[0],sys[1],sys[2],sys[3],sys[4],0.05)[0]
        self.assertEqual(A_rank,A_b_rank) 
        self.assertEqual(A_rank,sys[0].shape[1])
        self.assertEqual(status,"single solution")
        self.assertTrue(np.allclose([0.4,0.4,0.2],solution,0.001))
        
    def test_underdetermined_max_ispossible(self):
        sys= self.sys_under_max      
        A_b=np.concatenate((sys[0],sys[1]),1)
        A_rank=np.linalg.matrix_rank(sys[0])
        A_b_rank=np.linalg.matrix_rank(A_b)        
        status=get_recipe(sys[0],sys[1],sys[2],sys[3],sys[4],0.05)[1]
        fsugar_fraction=sugarwise_test("test under.csv","ingredients under.csv","lab_under_max.csv",0.05)
        self.assertEqual(A_rank,A_b_rank) 
        self.assertTrue(A_rank<sys[0].shape[1])
        self.assertEqual(status,"underdetermined")
        self.assertAlmostEquals(fsugar_fraction,0.25,places=3)
        
    def test_underdetermined_zero_fsugar(self):
        sys= self.sys_under_zero      
        A_b=np.concatenate((sys[0],sys[1]),1)
        A_rank=np.linalg.matrix_rank(sys[0])
        A_b_rank=np.linalg.matrix_rank(A_b)        
        status=get_recipe(sys[0],sys[1],sys[2],sys[3],sys[4],0.05)[1]
        fsugar_fraction=sugarwise_test("test under_zero.csv","ingredients under.csv","lab_under_zero.csv",0.05)
        self.assertEqual(A_rank,A_b_rank) 
        self.assertTrue(A_rank<sys[0].shape[1])
        self.assertEqual(status,"underdetermined")
        self.assertAlmostEquals(fsugar_fraction,0,places=3)
        
    def test_under_negative(self):
        sys= self.sys_under_negative      
        A_b=np.concatenate((sys[0],sys[1]),1)
        A_rank=np.linalg.matrix_rank(sys[0])
        A_b_rank=np.linalg.matrix_rank(A_b)        
        status=get_recipe(sys[0],sys[1],sys[2],sys[3],sys[4],0.05)[1]
        self.assertEqual(A_rank,A_b_rank) 
        self.assertTrue(A_rank<sys[0].shape[1])
        self.assertNotEqual(status,"underdetermined")

    def test_under_bigger1(self):
        sys= self.sys_under_bigger1     
        A_b=np.concatenate((sys[0],sys[1]),1)
        A_rank=np.linalg.matrix_rank(sys[0])
        A_b_rank=np.linalg.matrix_rank(A_b)        
        status=get_recipe(sys[0],sys[1],sys[2],sys[3],sys[4],0.05)[1]
        self.assertEqual(A_rank,A_b_rank) 
        self.assertTrue(A_rank<sys[0].shape[1])
        self.assertNotEqual(status,"underdetermined")  
        
    def test_overdetermined(self):
        sys=self.sys_over
        lst_sq_sol=np.linalg.lstsq(sys[2],sys[1])[0].transpose()[0]
        sq_b=np.dot(sys[2],lst_sq_sol)        
        tol_lst=[0.05,0.15,0.25]
        sugar_amt_lst=[]        
        for tol in tol_lst:
            result=get_recipe(sys[0],sys[1],sys[2],sys[3],sys[4],tol)
            self.assertEqual(result[1],"overdetermined")
            b_ov=np.dot(sys[2],result[0])
            self.assertTrue((np.linalg.norm(sys[1].transpose()[0]-b_ov))-0.001<tol)
            self.assertTrue(np.linalg.norm(sys[1].transpose()[0]-b_ov)>np.linalg.norm(sys[1].transpose()[0]-sq_b))
            fsugar_fraction=sugarwise_test("test over.csv","ingredients_over.csv","lab_over.csv",tol)
            sugar_amt_lst.append(fsugar_fraction)
        for j in range(len(sugar_amt_lst)-1):
            self.assertTrue(sugar_amt_lst[j+1]>sugar_amt_lst[j])


unittest.main()
