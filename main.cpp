#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>
#include <omp.h>
#include "ThreeDiagMatrices.h"
#include <chrono>
#include <thread>


using namespace std;

double q(double x){
    return sin(x);
}

double f(double x){
    return (9 + sin(x))*sin(3*x);
}

// Target function
double u(double x){
    return sin(3*x);
}



/** Split range on points
 * @param a left range border
 * @param b right range border
 * @param n count of sub...
 * @return [a, a+1/n, a+2/n, ..., b]
 * */
std::vector<double> build_nodes_vec(double a, double b, int n){
    std::vector<double> x(n+1);// Результат разбиения
    auto h = (b - a) / n; // Шаг разбиения
    for (int i = 0; i <= n; i++)
        x[i] = a + i * h;
    return x;
}

double* build_nodes(double a, double b, int n){
    auto x = new double [n+1];   // Результат разбиения
    auto h = (b - a) / n; // Шаг разбиения
    for (int i = 0; i <= n; i++)
        x[i] = a + i * h;
    return x;
}

template <typename T>
std::vector<T> array2vec(const T &a, int size){
    return std::vector<T>(a, a + size);
}

template <typename T>
T* vec2arr(const vector<T> vec, int size){
    auto arr = new T[size];
    for (int i = 0; i < size; i++)
        arr[i] = vec[i];
    return arr;
}
/**
Max mean error
 * */
double mae(const double* x, const double* y, int n){
    double max = 0;
    for (int i = 0; i < n; ++i){
        auto value = abs(x[i] - y[i]);
        if (value > max)
            max = value;
    }
    return max;
}


int main(){
    // Начальные условия
    int N = 1024*1024*32;                 // Число разбиений отрезка

    cout << "N: " << N << endl;
    auto a = 0;                        // Левый край отрезка
    auto b = std::numbers::pi; // Правый край отрезка
    double h = double(b - a) / N;      // Размер шага

    std::cout << "[a, b] = [" << a << "; " << b <<"]" << std::endl;

    auto x = build_nodes(a, b, N);

    // Фактические значения
    auto y = new double[N+1];
    for (int i = 0; i <= N; ++i) {
        y[i] = u(x[i]);
    }

    // Преобразование задачи
    auto lower_diag = new double [N+1];  // -a_i
    auto middle_diag = new double [N+1]; // b_i
    auto upper_diag = new double [N+1];  // -c_i
    auto f_values = new double [N+1];    // f_i

    f_values[0] = u(a);
    f_values[N] = u(b);

    upper_diag[0] = 0;  // c_0 = 0
    middle_diag[0] = 1; // b_0 = 1
    middle_diag[N] = 1; // b_n = 1
    lower_diag[N] = 0;  // a_n = 0

    for (int i = 1; i < N; i++){
        upper_diag[i] = 1;
        lower_diag[i] = 1;
        middle_diag[i] = 2 + h*h*q(x[i]);   // b_i = 2 + h^2 * q(x_i)
        f_values[i] = h*h*f(x[i]);          // h^2 * f(x_i)
    }

    // Метод циклической редукции
    std::chrono::steady_clock::time_point begin_cr_2 = std::chrono::steady_clock::now();
    auto u_2 = CycleReduction(lower_diag,
                             middle_diag,
                             upper_diag,
                             f_values, N, 2);
    std::chrono::steady_clock::time_point end_cr_2 = std::chrono::steady_clock::now();
    auto cr_time_2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cr_2 - begin_cr_2).count();
    std::cout << "[CycleReductionAlgorithm] Time difference = " << cr_time_2 << "[ns]" << std::endl;
    std::cout << "[CycleReductionAlgorithm] MAE:" << mae(y, u_2, N+1) << endl;

    // Метод циклической редукции
    std::chrono::steady_clock::time_point begin_cr_4 = std::chrono::steady_clock::now();
    auto u_4 = CycleReduction(lower_diag,
                             middle_diag,
                             upper_diag,
                             f_values, N, 4);
    std::chrono::steady_clock::time_point end_cr_4 = std::chrono::steady_clock::now();
    auto cr_time_4 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cr_4 - begin_cr_4).count();
    std::cout << "[CycleReductionAlgorithm] Time difference = " << cr_time_4 << "[ns]" << std::endl;
    std::cout << "[CycleReductionAlgorithm] MAE:" << mae(y, u_4, N+1) << endl;


    // Метод прогонки
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto w_ = BaseAlgorithm(lower_diag,
                            middle_diag,
                            upper_diag,
                            f_values, N);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto ta_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    std::cout << "[BaseAlgorithm] Time difference = " << ta_time << "[ns]" << std::endl;
    std::cout << "[BaseAlgorithm] MAE:" << mae(y, w_, N+1) << endl;

    std::cout << "[BaseAlgorithm] Time difference = " << ta_time << "[ns]" << std::endl;
    std::cout << "MAE2:" << mae(w_, u_2, N+1) << endl;
    std::cout << "[T2] Time difference = " << cr_time_2 << "[ns]" << std::endl;
    std::cout << "Ratio2: " << (float)ta_time / (float)cr_time_2 << std::endl;

    std::cout << "MAE4:" << mae(w_, u_2, N+1) << endl;
    std::cout << "[T4] Time difference = " << cr_time_4 << "[ns]" << std::endl;
    std::cout << "Ratio4: " << (float)ta_time / (float)cr_time_4 << std::endl;
//    std::cout << "Cycle Reduction:";
//    for (int i = 0; i <= N; i++) {
//        std::cout << u_[i] << " " << u(x[i]) << "\n";
//    }
    return 0;
}