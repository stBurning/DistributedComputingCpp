#ifndef DISTRIBUTEDCOMPUTING_THREEDIAGMATRICES_H
#define DISTRIBUTEDCOMPUTING_THREEDIAGMATRICES_H

#endif //DISTRIBUTEDCOMPUTING_THREEDIAGMATRICES_H
#include "vector"

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
    Метод циклической редукции
    @param a - коэффициенты на нижней диагонали матрицы СЛАУ, взятые со знаком минус
    @param b - коэффициенты на главной диагонали матрицы СЛАУ
    @param c - коэффициенты на верхней диагонали матрицы СЛАУ, взятые со знаком минус
    @param f - правая часть уравнения
    @param n - кол-во неизвестных
    @return x - решение уравнения
 */
std::vector<double> CycleReductionAlgorithm(std::vector<double> a,
                                            std::vector<double> b,
                                            std::vector<double> c,
                                            std::vector<double> f,
                                            int n);