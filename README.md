# Final Project: Course parallel programming in supercomputer

Task 2 - Numerical Integration using method Newton-Cotes

Student: 
==========
* Фамиля: Эспинола Ривера
* имя: Хольгер Элиас
* программа: Исскуственный Интеллект и Машинное обучение
* Группа: 3540201/ 20301

Content: 
=========

The project have:

1. Newton-cotes algorithm implemented in MPI + python
   file: newton_cotes_final1.py
2. Newton-cotes algorithm implemented in MPI + C
   file: newton_cotes_vc14.c
3. Library tinyexpr: library used for interpreter math-string functions and convert to 
   mathematic function to make calculations (in C)
   files: tinyexpr.c, tinyexpr.h
   * this 2 files need to use together with the file newton_cotes_vc14.c
4. Folder with experiments
   Have 4 different experiments, involved 4 functions with different levels of complexity 
   for compute the integral
   folder: exp1, exp2, exp3, exp4 ==> need to use for algorithm MPI + python
   folder: exp1_c, exp2_c, exp3_c, exp4_c ==> need to use for algorithm MPI + C
   files inside the folders, contains the same parameters, except the value of quadrature k.
   can do 5 different simulations for 5 different levels of quadrature.
5. Report of project
   file: ОТЧЁТ_ПАРАЛЛЕЛЬНОЕ_ПРОГРАММИРОВАНИЕ_СК.docx
   
