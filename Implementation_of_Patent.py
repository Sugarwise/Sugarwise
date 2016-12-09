# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:51:45 2016

@author: Vinicius Thaddeu dos Santos Ferreira: Sugarwise Chief Scientist
"""
import numpy as np
import csv
import scipy as sp
from scipy import optimize as op
import copy
import cvxopt as cvx

# BELOW IS A COLLECTION OF WORDS THAT I USE TO PARSE INGREDIENT NAMES, AND SEARCH FOR KEY WORDS THAT INDICATE THAT THE INGREDIENT IS A FREE SUGAR CONTAINING INGREDIENT
##IT IS COLLECTED IN DICTIONARY FORMAT: THE KEYS ARE THE KEY WORDS THAT INDICATE THE INGREDIENT IS POSSIBLY FREE SUGAR CONTAINING, AND THE VALUES ARE A LIST OF WORDS THAT...
##... MUST EITHER BE PRESENT OR NOT PRESENT IN THE INGREDIENT FOR IT TO BE CONSIDERED A FREE SUGAR. FOR PALM AND EXTRACT, THE VALUE IS A LIST OF WORDS THAT MUST ALSO BE PRESENT IN THE...
##.. INGREDIENT NAME FOR THE INGREDIENT TO BE LABELED FREE SUGAR CONTAINING. FOR ALL OTHER KEY WORDS, THE VALUE IN THE DICTIONARY ARE WORDS THAT MUST NOT BE IN THE INGREDIENT NAME...
##... FOR IT TO BE CONSIDERED A FREE SUGAR CONTAINING INGREDIENT. THIS DISTINCTION IS USED IN THE SCRIPT FOR THE INGREDIENT CLASS 


fsugar_dic = {};
fsugar_dic["honey"] = set(
    ["mushrooms", "flavour", "flavouring", "flavourings", "flavours", "artificial", "flavour^", "fungus"]);
fsugar_dic["agave"] = set([]);
fsugar_dic["molasses"] = set(["flavouring", "salt"]);
fsugar_dic["sugar"] = set(["milk", "fat", "nibs"]);
fsugar_dic["syrup"] = set(["malittol", "milk", "oligofructose", "maltitol", "sorbitol", "skimmed", "scribitol", ]);
fsugar_dic["juice"] = set(
    ["chicken", "celery", "vera", "iceberg", "hibiscus", "sauerkraut", "lime", "beef", "flavouring", "mushroom", "cod",
     "garlic", "cabbage", "tomato", "onion", "truffle", "dill", "coconut", "cashew", "carrot-", ]);
fsugar_dic["nectar"] = set([]);
fsugar_dic["jam"] = set([]);
fsugar_dic["jelly"] = set(["bonestock", "pork"]);
fsugar_dic["concentrate"] = set(
    ["safflower", "fennel", "chicken", "milk", "celery", "dairy", "coffee", "spirulin", "purple", "pepper", "algae",
     "chilli", "soybean", "soy", "lime", "beef", "shiitake", "parsley", "butter", "paprika",
     "protein", "carrot-juice", "lamb", "lemon", "mushroom", "cashew", "vegetable", "leek", "garlic", "carrot-",
     "watercress", "oyster", "fermented", "spirulina", "nettle", "turmeric", "egg"]);
fsugar_dic["glucose"] = set(["maize", "flavouring", "oxidase"]);
fsugar_dic["sucrose"] = set(["isobutyrate"]);
fsugar_dic["maltose"] = set([]);
fsugar_dic["dextrose"] = set([]);
fsugar_dic["fructose"] = set([]);
fsugar_dic["sugars"] = set([]);
fsugar_dic["syrups"] = set([]);
fsugar_dic["juices"] = set(
    ["pork:", "mushroom", "fish", "citric", "carrot", "chicken", "lime", "lamb", "lemon", "beef", "onion", "meat",
     "aronia"]);
fsugar_dic["palm"] = set(["nectar", "treacle", "sugar"]);
fsugar_dic["extract"] = set(
    ["orange", "mango", "grape", "malted", "marnier", "apple", "liqueur", "irish", "beer", "beet", "blueberry", "malt",
     "honey", "agave", "hop", "redcurrant", "berry", "fruit", "cola", "fig", "cointreau", "tiramisu", "barley",
     "bourbon", "syrup", "elderberry", "blackcurrant"])


class Nutrient:
    """Nutrient Class: Made to store information of nutrient name and  if the nutrient is a sugar, the calories of the nutrient, and it's column index on the csv file"""

    def __init__(self, name, i):
        self.name = name
        if name in ["maltose", "glucose", "sucrose", "fructose", "dextrose", "lactose", "galactose", "sugar"]:
            self.sugar_boolean = True
        else:
            self.sugar_boolean = False
        if name in ["maltose", "glucose", "sucrose", "fructose", "dextrose", "lactose", "sugar", "protein",
                    "carbohydrate"]:
            self.calories = 4
        elif name == "fat":
            self.calories = 9
        else:
            self.calories = 0
        self.row_index = i  # this is the row index for the nutrient in the databse file, NOT the index of the nutrient in the linear system
        self.index = -1  # to be changed later in the "build_linear_system" function to the nutrient's index in the linear system


class Ingredient:
    """Ingredient Class: Made to store information of ingredient name and if the ingredient is free sugar containing or not (via parsing), it's nutrient profile, and it's index on the linear system generated"""

    def __init__(self, name):
        self.name = name
        self.nutrient_dic = {}  # to be filled later in the "parser" function, keys=Nutrient object, values=nutrient amount per 100g of ingredient
        self.index = -1  # to be changed later in the "build_linear_system" function to the ingredient's index in the linear system
        name_edit = name.replace(',',
                                 ' ')  # takes commas out of the ingredients names for better parsing, replaces them with double spaces
        name_edit = name_edit.replace('  ',
                                      ' ')  # replaces double spaces in ingredient names with single spaces, for better parsing
        name_split = name_edit.split(' ')  # splits ingredient's name into a list with it's component words
        for i in range(0, len(name_split)):  # this loop makes everything lowercase
            name_split[i] = name_split[i].lower()
        self.fsugar_boolean = False  # initializes the free sugar boolean as false
        # this loop parses the ingredient's name (which is now split in a list format) and labels it as free sugar containing or not accordingly, as described above
        for word in fsugar_dic.keys():
            if word == "palm" and word in name_split and len(fsugar_dic[word].intersection(name_split)) > 0:
                self.fsugar_boolean = True
                break
            elif word == "extract" and word in name_split and len(fsugar_dic[word].intersection(name_split)) > 0:
                self.fsugar_boolean = True
                break
            elif word != "palm" and word != "extract" and word in name_split and len(
                    fsugar_dic[word].intersection(name_split)) == 0:
                self.fsugar_boolean = True
                break


def parser(db_filename, ingredient_lst_filename, lab_filename):
    """FOR THE DATABASE:Takes as input the database file of which contains the nutritional profile of ingredients, and a file with the ingredient list.
    outputs a list which contains Ingredient Objects FOR ALL INGREDIENTS IN THE LIST
    FOR THE LAB DATA:Takes as input a text file with the lab information, and outputs a dictionary which contains: keys=Nutrient object, values=nutrient amount per 100g
    NOTE: THESE THREE FILES MUST HAVE THE SAME EXACT FORMAT AS THE FILES I HAVE SUPPLIED INITIALLY
    NOTE: IT MUST BE MADE SURE THAT THE INGREDIENT NAMES IN THE INGREDIENT LIST MATCH INGREDIENT NAMES IN THE DATABASE"""

    # initializations below
    product_lst = []
    lab_dic = {}
    db_dic = {}
    lab_parsed = []
    db_parsed = []
    # parsing of lab file into the dictionary described in docs
    with open(lab_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lab_parsed.append(row)
    for i in range(1, len(lab_parsed[
                              0])):  # the loop starts at i=1 because the first word of the row is "Name" for the first row, and "Lab Amount" for the second row
        if lab_parsed[1][
            i] != '':  # skips blank values in the lab data csv file, which later on in the code signifies that that nutrient will not be considered in subsequent analysis
            nutrient = Nutrient(lab_parsed[0][i].lower(),
                                i)  # makes a nutrient object with name and appropriate row index
            if i == 4:  # i=4 here is the row index for carbohydrates, and i=5 is the row index for total sugars, WHICH IS HOW THE CSV FILES I SUPPLIED IS FORMATTED. A CHANGE IN FORMAT MUST BE ACCOMPANIED BY A CHANGE HERE
                lab_dic[nutrient] = float(lab_parsed[1][i]) - float(lab_parsed[1][
                                                                        i + 1])  # here, I am distinguinshing between sugars and carbohydrates by SUBTRACTING the amount of sugars from the carbohydrate amount
            else:  # or else, the number in the lab csv file is the nutrient content
                lab_dic[nutrient] = float(lab_parsed[1][i])
    # parsing of the database information into a dictionary: KEYS: ingredient name, VALUES: nutrient dictionary where KEYS:nutrient object, and VALUES: nutrient amount per 100g of ingredient
    with open(db_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            db_parsed.append(row)
    for row in db_parsed:
        ingredient_name = row[0]
        nutrient_dic = {}
        for nutrient in lab_dic.keys():
            try:  # this try and except clause is for cases when the database doesn't have a numerical amount for a certain nutrient, this catches the error, and the except clause sets the amount as 0
                if nutrient.row_index == 4:
                    nutrient_amount = float(row[nutrient.row_index]) - float(row[
                                                                                 nutrient.row_index + 1])  # same thing as before, subtracting the total sugar amount from the carbohydrate amount
                else:
                    nutrient_amount = float(row[nutrient.row_index])
                nutrient_dic[nutrient] = nutrient_amount
            except ValueError:
                nutrient_dic[
                    nutrient] = 0  # so if there isn't a numerical amount for the nutrient in the databse, we assume it is zero
        db_dic[ingredient_name] = nutrient_dic
    # the loop below parses the ingredient list
    with open(ingredient_lst_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ingredient_name_lst = row
    # the lines below make Ingredient objects out of the ingredient names from the parsed ingredient list, and makes the object's nutrient_dic the same dictionary from the parsed database for that ingredient
    product_lst = [Ingredient(name) for name in ingredient_name_lst]
    for ingredient in product_lst:
        ingredient.nutrient_dic = db_dic[ingredient.name]
    product_lst = sorted(product_lst,
                         key=lambda ingredient: ingredient.name)  # reorders the ingredients by alphabetical order
    return product_lst, lab_dic


def build_lin_system(product_lst, lab_dic):
    """Takes as input the dictionary of the parsed lab data and a list of Ingredient objects of the food product's ingredients, and outputs:
    matrix A, the vector b,the matrix A', a modified product list (with sugars as ingredients), and the mass normalizer used to calculate all values (as outlined in the patent)"""
    # the lines below delete nutrient information contained in the product_lst for all nutrients that were not lab tested, by checking if the nutrient names of the nutrient_dic are among the nutrient names in lab_dic
    lab_nutrient_lst = []  # initiates a list that will be the list of nutrient names
    for nutrient in lab_dic.keys():
        lab_nutrient_lst.append(nutrient.name)
    for ingredient in product_lst:
        for nutrient in ingredient.nutrient_dic:
            if nutrient.name not in lab_nutrient_lst:
                del ingredient.nutrient_dic[
                    nutrient]  # delets nutrients from the dic whose names are not among the nutrient's names in lab_dic
    # below is calculation of A and b as outlined in the patent
    normalizer = sum(lab_dic.values())
    b = np.zeros([len(lab_dic.keys()), 1])  # initializes b
    for i in range(0, len(
            b)):  # this loop gets the i_th nutrient from a list of sorted lab_dic keys, gets its amount and divides it by the normalizer
        nutrient_mass = lab_dic[sorted(lab_dic.keys(), key=lambda nutrient: nutrient.name)[i]]
        b[i] = nutrient_mass / normalizer
    A = np.zeros([len(lab_dic.keys()), len(
        product_lst)])  # initializes A according to patent: M is number of tested nutrients, N is number of ingredients
    # the lines of code below are to make the A' matrix and the modified list of ingredients, according to the patent    
    product_lst_prime = copy.deepcopy(product_lst)
    extra_sugar_ingredients = []  # intializes a list of new sugar ingredients
    for nutrient in lab_dic.keys():
        if nutrient.sugar_boolean and nutrient.name not in [ingredient.name for ingredient in
                                                            product_lst]:  # this checks if the nutrient is a sugar, and if it ISN'T already a ingredient
            new_ingredient = Ingredient(nutrient.name)
            for key in lab_dic.keys():  # this loop makes the nutrient amounts for the new sugar "ingredient" all zero
                new_ingredient.nutrient_dic[key] = 0
            new_ingredient.nutrient_dic[
                nutrient] = 100  # and this makes the nutrient amount for the sugar ITSELF 100g per 100g
            extra_sugar_ingredients.append(copy.deepcopy(new_ingredient))
    product_lst_prime = product_lst_prime + extra_sugar_ingredients  # appending the extra_ingredients_lst to the product_lst ensures that all the new sugar "ingredients" have an index between N and N'
    A_prime = np.zeros([len(lab_dic.keys()), len(product_lst_prime)])  # this initializes A'
    # the loops below acually calculate A and A' given all the parsed data
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            # lines below define the j and i indexes for each nutrient, as outlined in the patent
            ingredient_j = product_lst[j]
            nutrient_i = sorted(ingredient_j.nutrient_dic, key=lambda nutrient: nutrient.name)[
                i]  # this makes sure that indexes are distributed to the ingredient according to alphabetical order
            A[i, j] = ingredient_j.nutrient_dic[
                          nutrient_i] / normalizer  # calculates the matrix elements as defined in the patent
    for i in range(0, A_prime.shape[0]):
        for j in range(0, A_prime.shape[1]):
            ingredient_j = product_lst_prime[j]
            ingredient_j.index = j
            nutrient_i = sorted(ingredient_j.nutrient_dic, key=lambda nutrient: nutrient.name)[i]
            nutrient_i.index = i
            A_prime[i, j] = ingredient_j.nutrient_dic[nutrient_i] / normalizer
    # below I used the information contained in the ingredient and nutrient objects to get the coefficients for the free sugar functional, as outlined in the patent
    fsugar_functional_numerator_coeff = []
    for i in range(0, A_prime.shape[
        1]):  # looping through indices will make the order of the coefficients the same as the order of the ingredients in the matrix
        fsugar_amt = 0
        for nutrient in product_lst_prime[
            i].nutrient_dic:  # this loop will just picks out the amount of total sugar in the ingredient and adds it to the fsugar functional coefficients
            if product_lst_prime[i].fsugar_boolean and nutrient.name == "sugar":
                fsugar_amt = nutrient.calories * (product_lst_prime[i].nutrient_dic[nutrient] / normalizer)
        fsugar_functional_numerator_coeff.append(fsugar_amt)
    fsugar_functional_denominator_coeff = []
    for i in range(0, A_prime.shape[1]):
        calories_counter = 0  # initializes counter of calories
        for nutrient in product_lst_prime[i].nutrient_dic:
            calories_counter = calories_counter + (
            nutrient.calories * (product_lst_prime[i].nutrient_dic[nutrient] / normalizer))
        fsugar_functional_denominator_coeff.append(calories_counter)
    return A, b, A_prime, fsugar_functional_numerator_coeff, fsugar_functional_denominator_coeff


def get_recipe(A, b, A_prime, fsugar_functional_numerator_coeff, fsugar_functional_denominator_coeff, tol):
    """this function takes as input all components of the linear system and the free sugar functional, and it follows the patent flowchart to output a
    solution vector that a) for the single solution case IS the solution, or b) for the overdetermined or underdtermined case is the solution vector 
    that gives the highest possible free sugar content, according to the appropriate constraints outlined in the patent"""
    A_b = np.concatenate((A, b), 1)
    A_rank = np.linalg.matrix_rank(A)
    A_b_rank = np.linalg.matrix_rank(A_b)

    # single solution case below
    if A_rank == A_b_rank and A_rank == A.shape[1]:
        status = "single solution"
        solution = np.linalg.lstsq(A, b)[0].transpose()[
            0]  # we solve the system by least squares because, that the if statement is true, it is GUARANTEED that the least squares solution is the real solution, and this algorithim does not rely on matrix inversion(for which a square matrix is needed)
        # the lines below check if the solution is physical
        if solution.sum() > 1.001:  # the 0.001 is to account for floating point error
            solution = "overdetermined"
        for element in solution:
            if element < 0:
                solution = "overdetermined"
                break

    # over or under determined case below: I attempt to maximize the free sugar functional given the mass conservation constraints and the A'x'==b constraint
    # given that this is a fractional program problem, because the free sugar functional is a ratio of free sugar calories to calories, I make the transformation into a linear program by adding an extra variable and constraint, and solve it...
    ##using the Simplex Algorithim as handled by scipy. If the problem is unfeasible, then that means the system is either inconsistent or unphysical, in which case 
    ##we go to scenario 3, as indicated in the patent
    if (A_rank == A_b_rank and A_rank < A.shape[1]) or A_b_rank > A_rank or solution == "overdetermined":
        A_prime_eq = np.concatenate((A_prime, -b),
                                    1)  # given that, because of the transformation, Ax=b-->Ay=bt-->Ay-bt=0, we can write this as (A|-b)y_t=0
        extra_fractional_constraint = np.array(fsugar_functional_denominator_coeff + [
            0])  # we add zero to account for the fact that, in the transformed linear program, the coefficient for the t variable is 0 in the objective function's denominator
        A_prime_eq = np.concatenate((A_prime_eq, [extra_fractional_constraint]),
                                    0)  # we concatenate the denominator constraint onto A_prime_eq because linprog takes in as input a matrix with all equality constraints
        b_eq = np.zeros([1, A_prime.shape[0] + 1])[0];
        b_eq[
            -1] = 1  # this vector contains the zeros for all the equations expressed in the constraint Ay-bt=0, AND the 1 for the extra_fractiononal_constraint=1
        A_prime_ub = np.array([[1] * A_prime.shape[1] + [
            -1]])  # this includes the coefficients to the express the constraint: sum(x)<1 --> (sum(y))-t<0
        b_ub = np.array([0])  # the zero in the constraint sum(x)<1 --> (sum(y))-t<0
        minimizing_function = -np.array(fsugar_functional_numerator_coeff + [0])
        linprog_result = op.linprog(minimizing_function, A_eq=A_prime_eq, b_eq=b_eq, A_ub=A_prime_ub, b_ub=b_ub,
                                    bounds=(
                                    0, np.inf))  # bounds no less than zero because we are looking for positive quantities
        if linprog_result.status == 2:  # linprog_result==2 means the linear system is inconsistent
            solution = "overdetermined"
        else:
            solution = linprog_result.x[:-1] / linprog_result.x[-1]
            status = "underdetermined"
    # overdetermined case below: if the system is inconsistent, I attempt to maximize the free sugar functional given the conservation of mass constraints, and the...
    # ...the constraint that ||Ax-b||<error_tolerance. Given that this is a fractional program problem, because the free sugar functional is a ratio of free sugar
    # ...calories to calories, I make the transformation into a second order cone program by adding an extra variable and constraint, and solve it...
    # using the SOCP solver algorithim by CVXOPT. This requires making special matrices and vectors to put the problem into canonical form for the solver, as discussed in:
    # https://pwnetics.wordpress.com/2010/12/18/second-order-cone-programming-with-cvxopt/
    # If the problem is infeasible, then that means that the lab data is inconsistent with the ingredient list by an unaceptable amount.
    if solution == "overdetermined":
        # the Gq matrix below will be the matrix related to the second order constraint ||Ax-b||<tol-->||(A|-b)*y_t||<tol*t. The Gq matrix is  [[-c.T],-(A|-b)] according to SOCP theory. Because there are no constant terms, hq=[00...0]
        G0_q = np.array([[0] * A_prime.shape[1] + [
            -tol]])  # this is the c.Transpose row vector, which for this problem, is [00...tol], since only the variable t has a coefficient
        G1_q = -np.concatenate((A_prime, -b),
                               1)  # Note that tol and (A|-b) have a negative sign in front of it, according to the thoery
        G_normq = np.concatenate((G0_q, G1_q), 0)
        G_normq = cvx.matrix(
            G_normq.astype(np.double))  # the number type is cast as double because that is what CVXOPT accepts
        Gq = [G_normq]  # the SOCP solvers requires a list of the second order constraints
        hq = [cvx.matrix([0.0] * (G1_q.shape[0] + 1))]
        # the Gl matrix below will be for componentwise inequality constraints, i.e., the linear inequality constraint sum(x)<1 --> (sum(y))-t<0 and the positivity constraints x>0-->y>0
        Gl = np.array(A_prime_ub)  # this takes care of the conservation of mass constraint
        for i in range(G1_q.shape[
                           1]):  # this loop concatenates [00...1...0] row vectors, which signify the positivity constraints. G1_q.shape[1] is just A_prime.shape[1]+1
            ineq = [0] * G1_q.shape[1]
            ineq[i] = -1
            Gl = np.concatenate((Gl, np.array([ineq])), 0)
        Gl = cvx.matrix(Gl.astype(np.double))
        hl = cvx.matrix([0.0] + [0] * G1_q.shape[1])  # all the constant terms for all the inequalities is zero
        # the A matrix and b vector below will hold the only equality constraint, which is the extra constraint from the denominator of the objective function: d.Transpose=1
        A = cvx.matrix(np.array([extra_fractional_constraint]))
        b = cvx.matrix([1.0])
        minimizing_function_cvx = -cvx.matrix(fsugar_functional_numerator_coeff + [
            0.0])  # CVXOPT only does minimization, so we must use the negative of the numerator of the objective function
        cvx.solvers.options['show_progress'] = False
        sol = cvx.solvers.socp(minimizing_function_cvx, Gq=Gq, hq=hq, Gl=Gl, hl=hl, A=A, b=b)
        if sol['status'] != "optimal":
            solution = "infeasible"
            status = "infeasible"
        else:
            solution = np.array(sol['x'])[:-1] / np.array(sol['x'])[-1]
            solution = solution.transpose()[0]
            status = "overdetermined"
    return solution, status


def sugarwise_test(db_filename, ingredient_lst_filename, lab_filename, tol):
    '''Main function which puts together all the functions above and does the final free sugar calculation, given a solution recipe vector, or informs the user the problem is infeasible'''
    parsed_data = parser(db_filename, ingredient_lst_filename, lab_filename)
    lin_system = build_lin_system(parsed_data[0], parsed_data[1])
    solution = get_recipe(lin_system[0], lin_system[1], lin_system[2], lin_system[3], lin_system[4], tol)[0]
    if solution != "infeasible":
        fsugar_functional_numerator = 0  # initializes numerator value of free sugar functional
        fsugar_functional_denominator = 0  # initializes denominator value of free sugar functional
        for i in range(len(
                solution)):  # this loop calculates the fraction of calories derived from free sugars from the calculated solution vector
            fsugar_functional_numerator = fsugar_functional_numerator + (solution[i] * lin_system[3][i])
            fsugar_functional_denominator = fsugar_functional_denominator + (solution[i] * lin_system[4][i])
        fsugar_percentage = fsugar_functional_numerator / fsugar_functional_denominator
        return fsugar_percentage
    else:
        return "problem is infeasible, manufacturer of lab-supplied information is egregiously wrong"
