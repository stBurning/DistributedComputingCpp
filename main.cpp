#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>
#include "ThreeDiagMatrices.h"


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
std::vector<double> build_nodes(double a, double b, int n){

    std::vector<double> x(n+1);   // Результат разбиения
    auto h = (b - a) / n; // Шаг разбиения
    for (int i = 0; i <= n; i++) {
        x[i] = a + i * h;
    }
    return x;
}

int main(){
    // Начальные условия
    int N = 64;                       // Число разбиений отрезка
    auto a = 0;                       // Левый край отрезка
    auto b = std::numbers::pi; // Правый край отрезка
    auto h = (b - a) / N;      // Размер шага

    std::cout << "[a, b] = [" << a << "; " << b <<"]" << std::endl;

    auto x = build_nodes(a, b, N);
    std::cout << "x: ";
    for (int i = 0; i < N; i++){
        std::cout << x[i] << ", ";
    }
    std::cout<<x[N]<<std::endl;

    std::cout << "u(x): ";
    for (int i = 0; i < N; i++){
        std::cout << u(x[i]) << ", ";
    }
    std::cout<<u(x[N])<<std::endl;

    // Преобразование задачи
    std::vector<double> lower_diag(N+1);  // -a_i
    std::vector<double> middle_diag (N+1); // b_i
    std::vector<double> upper_diag(N+1);  // -c_i
    std::vector<double> f_values(N+1);    // f_i

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

    auto w_ = ThomasAlgorithm(lower_diag,
                                            middle_diag,
                                            upper_diag,
                                            f_values, N);

    std::cout << "Baseline:  ";
    for (int i = 0; i <= N; i++) {
        std::cout << w_[i] << " " << u(x[i]) << "\n";
    }

    auto u_ = CycleReductionAlgorithm(lower_diag,
                                                    middle_diag,
                                                    upper_diag,
                                                    f_values, N);

    std::cout << "Cycle Reduction:  ";
    for (int i = 0; i <= N; i++) {
        std::cout << u_[i] << " " << u(x[i]) << "\n";
    }

    return 0;
}