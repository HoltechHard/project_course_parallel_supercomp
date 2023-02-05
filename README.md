# Final Project: Course parallel programming in supercomputer

Task 2 - Numerical Integration using method Newton-Cotes  
LINK VIDEO-EXPERIMENTS:   
https://drive.google.com/drive/folders/1iEgp8xX0qo6JIcwQgWbtVyN-KFJAtkyO?usp=share_link  

Student: 
==========
* Фамиля: Эспинола Ривера
* имя: Хольгер Элиас
* программа: Исскуственный Интеллект и Машинное обучение
* Группа: 3540201/ 20301

Content: 
=========

The project have:

1. Parallel version of Newton-cotes algorithm implemented in MPI + python    
   file: newton_cotes_final1.py  
2. Parallel version of Newton-cotes algorithm implemented in MPI + C  
   file: newton_cotes_vc14.c  
3. Parallel version of Newton-cotes algorithm implemented in Pthreads + C  
   file: ncotes_pthreads_vc12.c  
4. Parallel version of Newton-cotes algorithm implemented in Open-MP + C  
   file: ncotes_openmp_vc2.c  
5. Sequential version of Newton-cotes algorithm implemented in python
   file: alg_newton_cotes_seq.py
5. Library tinyexpr: library used for interpreter math-string functions and convert to   
   mathematic function which calculate numerical value of math-function given any input  
   (** necessary for implementations in C)  
   files: tinyexpr.c, tinyexpr.h  
   * this 2 files need to use together with the files:  
      - newton_cotes_vc14.c  
      - ncotes_pthreads_vc12.c  
      - ncotes_openmp_vc2.c  
6. Folder with experiments  
   Have 4 different experiments, involved 4 functions with different levels of complexity  
   for compute the integral  
   folder: exp1, exp2, exp3, exp4 ==> need to use for algorithm MPI + python  
   folder: exp1_c, exp2_c, exp3_c, exp4_c ==> need to use for algorithms MPI + C / Pthreads + C / Open-MP + C  
   files inside the folders, contains the same parameters, except the value of quadrature k.  
   can do 5 different simulations for 5 different levels of quadrature.  
7. Report of project  
   file: Final_ОТЧЁТ_ПАРАЛЛЕЛЬНОЕ_ПРОГРАММИРОВАНИЕ.docx  
         Final_ОТЧЁТ_ПАРАЛЛЕЛЬНОЕ_ПРОГРАММИРОВАНИЕ.pdf  
8. Jupyter notebook with graphics of experiments  
   file: graphics_exp.ipynb
   
