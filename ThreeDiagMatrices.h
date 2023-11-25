#ifndef DISTRIBUTEDCOMPUTING_THREEDIAGMATRICES_H
#define DISTRIBUTEDCOMPUTING_THREEDIAGMATRICES_H

#endif //DISTRIBUTEDCOMPUTING_THREEDIAGMATRICES_H
#include "vector"

using namespace std;

/**
    Метод прогонки (англ. tridiagonal matrix algorithm)
    или алгоритм Томаса (англ. Thomas algorithm) используется для решения систем линейных уравнений вида Ax = f, где
    A — трёхдиагональная матрица. Представляет собой вариант метода последовательного исключения неизвестных. 
    @param a - коэффициенты на нижней диагонали матрицы СЛАУ, взятые со знаком минус
    @param b - коэффициенты на главной диагонали матрицы СЛАУ
    @param c - коэффициенты на верхней диагонали матрицы СЛАУ, взятые со знаком минус
    @param f - правая часть уравнения
    @param n - кол-во неизвестных
    @return x - решение уравнения
 */
std::vector<double> ThomasAlgorithm(std::vector<double> a,
                                    std::vector<double> b,
                                    std::vector<double> c,
                                    std::vector<double> f,
                                    int n);

/**
    Метод прогонки (англ. tridiagonal matrix algorithm)
    или алгоритм Томаса (англ. Thomas algorithm) используется для решения систем линейных уравнений вида Ax = f, где
    A — трёхдиагональная матрица. Представляет собой вариант метода последовательного исключения неизвестных.
    @param a - коэффициенты на нижней диагонали матрицы СЛАУ, взятые со знаком минус
    @param b - коэффициенты на главной диагонали матрицы СЛАУ
    @param c - коэффициенты на верхней диагонали матрицы СЛАУ, взятые со знаком минус
    @param f - правая часть уравнения
    @param n - кол-во неизвестных
    @return x - решение уравнения
 */
double* BaseAlgorithm(const double *a,
                      const double *b,
                      const double *c,
                      const double *f,
                      int n);

/**
    Метод циклической редукции
    @param a - коэффициенты на нижней диагонали матрицы СЛАУ, взятые со знаком минус
    @param b - коэффициенты на главной диагонали матрицы СЛАУ
    @param c - коэффициенты на верхней диагонали матрицы СЛАУ, взятые со знаком минус
    @param f - правая часть уравнения
    @param n - кол-во неизвестных
    @return x - решение уравнения
 */
std::vector<double> CycleReduction(const vector<double> &a,
                                   const vector<double> &b,
                                   const vector<double> &c,
                                   const vector<double> &f,
                                   int n, int num_workers);

double* CycleReduction(const double *a,
                       const double *b,
                       const double *c,
                       const double *f,
                       int n, int num_workers);

